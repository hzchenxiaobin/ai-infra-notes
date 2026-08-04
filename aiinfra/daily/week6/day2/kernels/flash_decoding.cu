// flash_decoding.cu —— FlashDecoding 最小化实现（KV 按 block 切分 + 跨 block 合并）
// 编译命令: nvcc -o flash_decoding flash_decoding.cu -O3 -arch=sm_120
// 运行命令: ./flash_decoding
//
// 演示 FlashDecoding 的三大核心机制：
//   1. Decode 阶段（M=1）：单 query 对 N 个历史 key
//   2. KV sequence 按 block 切分到不同 SM，每个 block 独立计算 partial attention
//   3. 跨 block 合并：用 online softmax 的 rescaling factor 合并 partial max/sum/output
//
// 简化：单 head、单 sequence、FP32

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)
#define MAX_D 256

// ---------- 块内归约 ----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}
__inline__ __device__ float block_reduce_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- Phase 1: FlashDecoding kernel ----------
// 每个 block 处理 KV sequence 的一段 [kv_start, kv_end)
// 输出 partial: partial_o[block_id][d], partial_m[block_id], partial_l[block_id]
//
// q:        [d]           当前 query 向量
// k_cache:  [seq_len, d]  历史 key（简化为连续布局）
// v_cache:  [seq_len, d]  历史 value
// partial_o:[num_blocks, d]  每 block 的 partial output（未归一化）
// partial_m:[num_blocks]     每 block 的 partial max（score 最大值）
// partial_l:[num_blocks]     每 block 的 partial sum（exp score 之和）
// seq_len, d: 维度
// tokens_per_block: 每 block 处理的 KV token 数
__global__ void flash_decoding_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int seq_len, int d, int tokens_per_block)
{
    __shared__ float q_shm[MAX_D];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_score_shm, alpha_shm, p_shm;

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int kv_start = block_id * tokens_per_block;
    int kv_end = min(kv_start + tokens_per_block, seq_len);
    const float scale = 1.0f / sqrtf((float)d);

    for (int t = tid; t < d; t += THREADS_PER_BLOCK)
        q_shm[t] = q[t];
    __syncthreads();

    float m_local = -INFINITY;
    float l_local = 0.f;
    // Each thread accumulates its slice of the d-dim output vector
    // tid handles elements: tid, tid+THREADS_PER_BLOCK, tid+2*THREADS_PER_BLOCK, ...
    const int elems_per_thread = (d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float o_local[8]; // assume d <= 8 * THREADS_PER_BLOCK (d=64, THREADS=64 → 1 elem/thread)
    for (int i = 0; i < elems_per_thread && (tid + i * THREADS_PER_BLOCK) < d; i++)
        o_local[i] = 0.f;

    for (int s = kv_start; s < kv_end; ++s) {
        const float* k_vec = k_cache + (size_t)s * d;
        const float* v_vec = v_cache + (size_t)s * d;

        float score = 0.f;
        for (int t = tid; t < d; t += THREADS_PER_BLOCK)
            score += q_shm[t] * k_vec[t];
        score = block_reduce_sum(score, red) * scale;
        if (tid == 0) s_score_shm = score;
        __syncthreads();
        score = s_score_shm;

        float m_new = fmaxf(m_local, score);
        float alpha = expf(m_local - m_new);
        float p = expf(score - m_new);
        float l_new = l_local * alpha + p;

        if (tid == 0) { alpha_shm = alpha; p_shm = p; }
        __syncthreads();

        for (int i = 0; i < elems_per_thread; i++) {
            int t = tid + i * THREADS_PER_BLOCK;
            if (t < d)
                o_local[i] = o_local[i] * alpha_shm + p_shm * v_vec[t];
        }
        __syncthreads();

        m_local = m_new;
        l_local = l_new;
    }

    float* my_o = partial_o + (size_t)block_id * d;
    for (int i = 0; i < elems_per_thread; i++) {
        int t = tid + i * THREADS_PER_BLOCK;
        if (t < d)
            my_o[t] = o_local[i];
    }
    if (tid == 0) {
        partial_m[block_id] = m_local;
        partial_l[block_id] = l_local;
    }
}

// ---------- Phase 2: 合并 kernel ----------
// 用 online softmax 合并所有 block 的 partial 结果
// 最终 output: O = Σ_j (exp(m_j - m_global) * l_j * partial_o_j) / Σ_j (exp(m_j - m_global) * l_j)
__global__ void flash_decoding_merge_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ output,
    int num_blocks, int d)
{
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float global_max_shm, global_sum_shm;
    int tid = threadIdx.x;

    // global_max: all threads read the same partial_m[], so value is identical
    // but we still use block_reduce_max for correctness across configurations
    float global_max = -INFINITY;
    for (int j = 0; j < num_blocks; ++j)
        global_max = fmaxf(global_max, partial_m[j]);
    global_max = block_reduce_max(global_max, red);
    if (tid == 0) global_max_shm = global_max;
    __syncthreads();

    // global_sum: all threads compute the same value, so only tid==0 writes
    // (no block_reduce_sum needed — that would multiply by THREADS_PER_BLOCK)
    if (tid == 0) {
        float global_sum = 0.f;
        for (int j = 0; j < num_blocks; ++j)
            global_sum += expf(partial_m[j] - global_max_shm) * partial_l[j];
        global_sum_shm = global_sum;
    }
    __syncthreads();

    for (int t = tid; t < d; t += THREADS_PER_BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < num_blocks; ++j) {
            float w = expf(partial_m[j] - global_max_shm);
            acc += w * partial_o[(size_t)j * d + t];
        }
        output[t] = acc / global_sum_shm;
    }
}

// ---------- CPU 参考实现 ----------
void cpu_attention(const float* q, const float* k_cache, const float* v_cache,
                   float* output, int seq_len, int d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(seq_len);
    float max_score = -INFINITY;
    for (int s = 0; s < seq_len; ++s) {
        float dot = 0.f;
        for (int t = 0; t < d; ++t)
            dot += q[t] * k_cache[s * d + t];
        scores[s] = dot * scale;
        max_score = fmaxf(max_score, scores[s]);
    }
    float sum_exp = 0.f;
    for (int s = 0; s < seq_len; ++s) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    for (int t = 0; t < d; ++t) output[t] = 0.f;
    for (int s = 0; s < seq_len; ++s)
        for (int t = 0; t < d; ++t)
            output[t] += scores[s] * v_cache[s * d + t];
    for (int t = 0; t < d; ++t)
        output[t] /= sum_exp;
}

// ---------- 主函数 ----------
int main() {
    const int d = 64;
    const int seq_len = 1024;
    const int tokens_per_block = 64;
    const int num_blocks = (seq_len + tokens_per_block - 1) / tokens_per_block;

    printf("=== FlashDecoding Test ===\n");
    printf("d=%d, seq_len=%d, tokens_per_block=%d, num_blocks=%d\n\n",
           d, seq_len, tokens_per_block, num_blocks);

    std::vector<float> h_q(d), h_k(seq_len * d), h_v(seq_len * d);
    for (int i = 0; i < d; ++i) h_q[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < seq_len * d; ++i) {
        h_k[i] = (float)(rand() % 100) / 100.0f;
        h_v[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_q, *d_k, *d_v, *d_partial_o, *d_partial_m, *d_partial_l, *d_output;
    cudaMalloc(&d_q, d * sizeof(float));
    cudaMalloc(&d_k, (size_t)seq_len * d * sizeof(float));
    cudaMalloc(&d_v, (size_t)seq_len * d * sizeof(float));
    cudaMalloc(&d_partial_o, (size_t)num_blocks * d * sizeof(float));
    cudaMalloc(&d_partial_m, num_blocks * sizeof(float));
    cudaMalloc(&d_partial_l, num_blocks * sizeof(float));
    cudaMalloc(&d_output, d * sizeof(float));

    cudaMemcpy(d_q, h_q.data(), d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), (size_t)seq_len * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), (size_t)seq_len * d * sizeof(float), cudaMemcpyHostToDevice);

    flash_decoding_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_q, d_k, d_v, d_partial_o, d_partial_m, d_partial_l,
        seq_len, d, tokens_per_block);
    flash_decoding_merge_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_partial_o, d_partial_m, d_partial_l, d_output, num_blocks, d);
    cudaDeviceSynchronize();

    std::vector<float> h_output(d), h_ref(d);
    cudaMemcpy(h_output.data(), d_output, d * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_attention(h_q.data(), h_k.data(), h_v.data(), h_ref.data(), seq_len, d);

    float max_diff = 0.f;
    for (int t = 0; t < d; ++t)
        max_diff = fmaxf(max_diff, fabsf(h_output[t] - h_ref[t]));
    printf("max diff (FlashDecoding vs CPU ref): %.4e\n", max_diff);
    printf("result: %s\n\n", max_diff < 1e-3f ? "PASS" : "FAIL");

    printf("[Parallelism analysis]\n");
    printf("  Standard decode: 1 block handles entire KV (seq_len=%d)\n", seq_len);
    printf("  FlashDecoding:   %d blocks handle %d tokens each\n", num_blocks, tokens_per_block);
    printf("  Parallelism:     %dx improvement (utilizing idle SMs)\n", num_blocks);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_partial_o); cudaFree(d_partial_m); cudaFree(d_partial_l); cudaFree(d_output);
    return 0;
}
