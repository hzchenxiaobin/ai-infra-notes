// attention_naive.cu —— 标准 Attention Forward（物化 S 和 P，用于 IO 分析）
// 编译命令: nvcc -o attention_naive kernels/attention_naive.cu -O3 -arch=sm_120 -lineinfo
// 运行命令: ./attention_naive

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 复用 Week 2 Day 1 / Day 16 的 warp reduce 原语
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// ============================================================
// 标准 Attention Forward Kernel（物化 S 和 P 到 HBM）
// 一个 block 处理一行 query（qrow）
// 三步：S=QK^T → P=softmax(S) → O=PV，S/P 全部写回 HBM
// ============================================================
__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ S,
                                         float* __restrict__ P,
                                         float* __restrict__ O,
                                         int N, int d) {
    int qrow = blockIdx.x;
    if (qrow >= N) return;

    __shared__ float smem[32];
    __shared__ float row_max;
    __shared__ float row_sum;

    int tid = threadIdx.x;
    float scale = 1.0f / sqrtf((float)d);

    // Step 1: S[qrow][j] = sum_d Q[qrow][d] * K[j][d] * scale
    //         物化 S 到 HBM（这就是 O(N²) 写入的来源）
    for (int j = tid; j < N; j += blockDim.x) {
        float s_val = 0.0f;
        for (int dd = 0; dd < d; dd++) {
            s_val += Q[qrow * d + dd] * K[j * d + dd];
        }
        S[qrow * N + j] = s_val * scale;
    }
    __syncthreads();

    // Step 2: P[qrow][j] = softmax(S[qrow][:])
    //         读 S（O(N²) 读），写 P（O(N²) 写）
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, S[qrow * N + j]);
    }
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0) row_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float p_val = expf(S[qrow * N + j] - row_max);
        P[qrow * N + j] = p_val;
        local_sum += p_val;
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_sum = local_sum;
    __syncthreads();

    // Step 3: O[qrow][dd] = sum_j P[qrow][j] * V[j][dd]
    //         读 P（O(N²) 读），读 V，写 O
    float inv_sum = 1.0f / row_sum;
    for (int dd = tid; dd < d; dd += blockDim.x) {
        float o_val = 0.0f;
        for (int j = 0; j < N; j++) {
            o_val += (P[qrow * N + j] * inv_sum) * V[j * d + dd];
        }
        O[qrow * d + dd] = o_val;
    }
}

// ============================================================
// CPU 参考（用于验证）
// ============================================================
void cpuAttention(const float* Q, const float* K, const float* V,
                  float* O, int N, int d) {
    float* S = (float*)malloc(N * N * sizeof(float));
    float* P = (float*)malloc(N * N * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int dd = 0; dd < d; dd++) s += Q[i*d+dd] * K[j*d+dd];
            S[i*N+j] = s * scale;
        }
        float mx = S[i*N];
        for (int j = 1; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) { P[i*N+j] = expf(S[i*N+j]-mx); sm += P[i*N+j]; }
        for (int j = 0; j < N; j++) P[i*N+j] /= sm;
        for (int dd = 0; dd < d; dd++) {
            float o = 0.0f;
            for (int j = 0; j < N; j++) o += P[i*N+j] * V[j*d+dd];
            O[i*d+dd] = o;
        }
    }
    free(S); free(P);
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    bool ok = maxDiff < eps;
    printf("  maxDiff = %.2e (%s)\n", maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    // 测试多个 seq_len，观察 IO 随 N² 增长
    int seqLens[] = {256, 512, 1024, 2048};
    int d = 64;
    int threads = 256;

    printf("=== Standard Attention Forward (naive, materialize S/P) ===\n");
    printf("%-8s %-14s %-16s %-12s %-10s\n",
           "N", "S/P size(MB)", "HBM IO(MB)", "Time(ms)", "Check");
    printf("------------------------------------------------------------------\n");

    for (int si = 0; si < 4; si++) {
        int N = seqLens[si];
        size_t bytesQKV = N * d * sizeof(float);
        size_t bytesSP = (size_t)N * N * sizeof(float);

        float *h_Q = (float*)malloc(bytesQKV), *h_K = (float*)malloc(bytesQKV);
        float *h_V = (float*)malloc(bytesQKV), *h_O = (float*)malloc(bytesQKV);
        float *h_O_cpu = (float*)malloc(bytesQKV);
        initData(h_Q, N*d); initData(h_K, N*d); initData(h_V, N*d);

        float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
        cudaMalloc(&d_Q, bytesQKV); cudaMalloc(&d_K, bytesQKV);
        cudaMalloc(&d_V, bytesQKV); cudaMalloc(&d_O, bytesQKV);
        cudaMalloc(&d_S, bytesSP);  cudaMalloc(&d_P, bytesSP);
        cudaMemcpy(d_Q, h_Q, bytesQKV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, bytesQKV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, bytesQKV, cudaMemcpyHostToDevice);

        // 理论 HBM IO：读 Q,K,V（3Nd）+ 读/写 S,P（各 2N²）+ 写 O（Nd）
        // = 4Nd + 4N²（简化，每元素 4 bytes）
        double hbmIO = (4.0 * N * d + 4.0 * N * N) * sizeof(float) / (1024.0*1024.0);
        double spSize = (double)bytesSP / (1024.0*1024.0);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        attention_naive_kernel<<<N, threads>>>(d_Q, d_K, d_V, d_S, d_P, d_O, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_O, d_O, bytesQKV, cudaMemcpyDeviceToHost);
        cpuAttention(h_Q, h_K, h_V, h_O_cpu, N, d);
        bool ok = checkResult(h_O, h_O_cpu, N*d, 1e-3f);

        printf("%-8d %-14.2f %-16.2f %-12.3f %-10s\n",
               N, spSize, hbmIO, ms, ok ? "PASS" : "FAIL");

        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_cpu);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S);
        cudaFree(d_P); cudaFree(d_O);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n观察要点：\n");
    printf("1. S/P size 随 N² 增长（N 翻倍 → size 4x）\n");
    printf("2. HBM IO 随 N² 增长（N 翻倍 → IO 4x）\n");
    printf("3. Time 近似随 N² 增长（长序列下 O(N²) IO 主导）\n");
    printf("4. 用 ncu 验证 dram__bytes_read.sum + dram__bytes_write.sum ≈ 理论 HBM IO\n");
    return 0;
}
