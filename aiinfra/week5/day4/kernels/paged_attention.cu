// paged_attention.cu —— PagedAttention 最小化实现（block table + 分块 KV cache attention）
// 编译命令: nvcc -o paged_attention paged_attention.cu -O3 -arch=sm_120
// 运行命令: ./paged_attention
//
// 演示 PagedAttention 的三大核心机制：
//   1. KV cache 按 block 分块存储（物理 block 可不连续）
//   2. block table 维护 逻辑 block → 物理 block 映射
//   3. attention kernel 通过 block table 间接寻址读取 KV
//
// 简化：单 head、单 sequence、decode 场景（1 个 query 对 N 个历史 key）

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)
#define KV_BLOCK_SIZE 16    // 每个 KV cache block 容纳 16 个 token（vLLM 默认）

// ---------- 块归约 ----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, o);
    }
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v; __syncthreads();
    if (wid == 0) { v = (lane < NUM_WARPS) ? sh[lane] : 0.f; v = warp_reduce_sum(v); if (lane==0) sh[0]=v; }
    __syncthreads(); return sh[0];
}

// ---------- PagedAttention kernel（decode：1 query 对 N 历史 key）----------
// kv_cache_pool: 物理 block 池，布局 [num_blocks, KV_BLOCK_SIZE, d]
//                block i 的数据在 kv_cache_pool[i * KV_BLOCK_SIZE * d ...]
// block_table:   [max_num_blocks_per_seq]，block_table[l] = 第 l 个逻辑 block 的物理 block 号
// q:             [d]，当前 query 向量
// output:        [d]，attention 输出
// seq_len:       历史 key 数量（已缓存的 token 数）
// d:             head_dim
// num_blocks:    该 sequence 使用的逻辑 block 数 = ceil(seq_len / KV_BLOCK_SIZE)
__global__ void paged_attention_kernel(
    const float* __restrict__ kv_cache_pool,   // K 和 V 各一个 pool，这里简化：k_pool 在前，v_pool 在后
    const float* __restrict__ v_cache_pool,
    const int*   __restrict__ block_table,     // 逻辑 → 物理 block 映射
    const float* __restrict__ q,
    float*       __restrict__ output,
    int seq_len, int d, int max_blocks_per_seq) {

    __shared__ float q_shm[256];               // query 向量（d ≤ 256）
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)d);

    // ① 载入 query 到 shared
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        q_shm[t] = q[t];
    }
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float o_local = 0.f;

    // ② 遍历所有历史 key（按逻辑 block 顺序，通过 block_table 找物理 block）
    int num_logical_blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    for (int lb = 0; lb < num_logical_blocks; ++lb) {
        int physical_block = block_table[lb];          // ★ 核心：逻辑→物理映射
        const float* k_block = kv_cache_pool + (size_t)physical_block * KV_BLOCK_SIZE * d;
        const float* v_block = v_cache_pool  + (size_t)physical_block * KV_BLOCK_SIZE * d;

        // 遍历该 block 内的 token（最后一块可能不满）
        int tokens_in_block = min(KV_BLOCK_SIZE, seq_len - lb * KV_BLOCK_SIZE);
        for (int s = 0; s < tokens_in_block; ++s) {
            const float* k_vec = k_block + s * d;
            const float* v_vec = v_block + s * d;

            // 点积 score = q · k / √d（block 归约）
            float part = 0.f;
            for (int t = tid; t < d; t += BLOCK_SIZE) {
                part += q_shm[t] * k_vec[t];
            }
            float s_k = block_reduce_sum(part, red) * scale;
            if (tid == 0) s_k_shm = s_k;
            __syncthreads(); s_k = s_k_shm;

            // online softmax 三公式
            if (tid == 0) {
                float m_new = fmaxf(m, s_k);
                float alpha = expf(m - m_new);
                float p     = expf(s_k - m_new);
                float l_new = l * alpha + p;
                alpha_shm = (l * alpha) / l_new;
                beta_shm  = p / l_new;
                m = m_new; l = l_new;
            }
            __syncthreads();

            // 累加输出
            for (int t = tid; t < d; t += BLOCK_SIZE) {
                o_local = o_local * alpha_shm + beta_shm * v_vec[t];
            }
            __syncthreads();
        }
    }
    // ③ 写回
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        output[t] = o_local;
    }
}

// ---------- CPU 参考（连续 KV cache，用于验证 paged 版正确性）----------
void attention_cpu(const float* K, const float* V, const float* q,
                   float* out, int seq_len, int d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> sc(seq_len);
    float mx = -INFINITY;
    for (int s = 0; s < seq_len; ++s) {
        float dot = 0.f;
        for (int t = 0; t < d; ++t) {
            dot += q[t] * K[s * d + t];
        }
        sc[s] = dot * scale; mx = fmaxf(mx, sc[s]);
    }
    float sum = 0.f;
    for (int s = 0; s < seq_len; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
    for (int t = 0; t < d; ++t) {
        float acc = 0.f;
        for (int s = 0; s < seq_len; ++s) {
            acc += sc[s] * V[s * d + t];
        }
        out[t] = acc / sum;
    }
}

int main() {
    const int d = 64;
    const int seq_len = 50;                          // 50 个历史 token
    const int num_logical_blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;  // = 4
    const int total_physical_blocks = 16;            // 物理 block 池大小

    printf("=== PagedAttention Test ===\n");
    printf("d=%d, seq_len=%d, KV_BLOCK_SIZE=%d, num_logical_blocks=%d\n",
           d, seq_len, KV_BLOCK_SIZE, num_logical_blocks);

    // ---- 构造 block table：故意打乱物理 block 顺序，证明逻辑连续≠物理连续 ----
    // 逻辑 block 0..3 → 物理 block 7, 1, 12, 3（不连续！）
    std::vector<int> block_table = {7, 1, 12, 3};
    printf("block_table (logical→physical): ");
    for (int i = 0; i < num_logical_blocks; ++i) {
        printf("%d→%d  ", i, block_table[i]);
    }
    printf("\n");

    // ---- 构造物理 block 池 ----
    // k_pool[total_physical_blocks * KV_BLOCK_SIZE * d], v_pool 同
    size_t pool_size = (size_t)total_physical_blocks * KV_BLOCK_SIZE * d;
    std::vector<float> h_k_pool(pool_size, -999.f), h_v_pool(pool_size, -999.f);
    // 按逻辑顺序生成连续的 K/V 数据，再按 block_table 写入物理 pool
    std::vector<float> k_contig((size_t)seq_len * d), v_contig((size_t)seq_len * d);
    srand(42);
    for (int i = 0; i < seq_len * d; ++i) {
        k_contig[i] = ((rand() % 2000) - 1000) / 100.f;
        v_contig[i] = ((rand() % 2000) - 1000) / 100.f;
    }
    for (int lb = 0; lb < num_logical_blocks; ++lb) {
        int pb = block_table[lb];
        int tokens = min(KV_BLOCK_SIZE, seq_len - lb * KV_BLOCK_SIZE);
        for (int s = 0; s < tokens; ++s) {
            for (int t = 0; t < d; ++t) {
                h_k_pool[(size_t)pb * KV_BLOCK_SIZE * d + s * d + t] = k_contig[(lb * KV_BLOCK_SIZE + s) * d + t];
                h_v_pool[(size_t)pb * KV_BLOCK_SIZE * d + s * d + t] = v_contig[(lb * KV_BLOCK_SIZE + s) * d + t];
            }
        }
    }

    // query
    std::vector<float> h_q(d);
    for (int t = 0; t < d; ++t) {
        h_q[t] = ((rand() % 2000) - 1000) / 100.f;
    }

    // ---- device 分配 ----
    float *d_k_pool, *d_v_pool, *d_q, *d_out;
    int   *d_block_table;
    cudaMalloc(&d_k_pool, pool_size * sizeof(float));
    cudaMemcpy(d_k_pool, h_k_pool.data(), pool_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_v_pool, pool_size * sizeof(float));
    cudaMemcpy(d_v_pool, h_v_pool.data(), pool_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_block_table, num_logical_blocks * sizeof(int));
    cudaMemcpy(d_block_table, block_table.data(), num_logical_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_q, d * sizeof(float));
    cudaMemcpy(d_q, h_q.data(), d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, d * sizeof(float));

    // ---- 运行 PagedAttention kernel ----
    paged_attention_kernel<<<1, BLOCK_SIZE>>>(
        d_k_pool, d_v_pool, d_block_table, d_q, d_out,
        seq_len, d, num_logical_blocks);
    cudaDeviceSynchronize();

    std::vector<float> h_out(d);
    cudaMemcpy(h_out.data(), d_out, d * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- CPU 参考（连续布局）----
    std::vector<float> h_ref(d);
    attention_cpu(k_contig.data(), v_contig.data(), h_q.data(), h_ref.data(), seq_len, d);

    // ---- 验证 ----
    float max_diff = 0.f;
    for (int t = 0; t < d; ++t) {
        max_diff = fmaxf(max_diff, fabsf(h_out[t] - h_ref[t]));
    }
    printf("max diff (paged vs contiguous): %.2e (%s)\n",
           max_diff, max_diff < 1e-4f ? "PASS" : "FAIL");

    // ---- 内存利用率对比 ----
    // 静态分配：预分配 max_seq_len（假设 128）→ 浪费 (128-50)/128 = 61%
    // PagedAttention：只分配 ceil(50/16)=4 个 block → 无浪费
    int max_seq_len = 128;
    float static_waste = 100.0f * (max_seq_len - seq_len) / max_seq_len;
    float paged_used = 100.0f * num_logical_blocks * KV_BLOCK_SIZE / max_seq_len;
    printf("\n[Memory utilization]\n");
    printf("  Static alloc (max=%d): waste %.0f%% (allocated %d, used %d)\n",
           max_seq_len, static_waste, max_seq_len, seq_len);
    printf("  PagedAttention:        use %.0f%% of static (%d blocks × %d tok = %d slots, %d actual)\n",
           paged_used, num_logical_blocks, KV_BLOCK_SIZE,
           num_logical_blocks * KV_BLOCK_SIZE, seq_len);
    printf("  PagedAttention 的物理 block 可不连续（本例 7,1,12,3），逻辑连续由 block table 保证\n");

    cudaFree(d_k_pool); cudaFree(d_v_pool); cudaFree(d_block_table);
    cudaFree(d_q); cudaFree(d_out);
    return 0;
}
