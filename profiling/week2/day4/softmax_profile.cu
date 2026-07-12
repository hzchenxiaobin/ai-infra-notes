// softmax_profile.cu —— Softmax 三遍扫描（ncu profiling 版，Day 4 任务 5）
// 编译命令: nvcc -o softmax_profile softmax_profile.cu -O3 -arch=sm_120 -g -lineinfo
// 运行命令: ./softmax_profile

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 朴素三遍扫描 Softmax（每线程遍历整个数组，O(N²) 访存）
__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // Pass 1: 求 max
    float max_val = -INFINITY;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Pass 2: 求 sum(exp(x - max))
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max_val);
    }

    // Pass 3: 归一化
    output[idx] = expf(input[idx] - max_val) / sum;
}

// 行级 Softmax（batch 版本，每行一个 block）
__global__ void softmax_row_kernel(const float* input, float* output, int M, int N) {
    __shared__ float s_sum;
    __shared__ float s_max;
    int row = blockIdx.x;
    if (row >= M)
        return;
    int tid = threadIdx.x;

    // Pass 1: 求 max
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * N + i]);
    }

    // block reduce max（简化版：用 shared memory）
    __shared__ float smem[32];
    int lane = tid % 32;
    int wid = tid / 32;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if (lane == 0)
        smem[wid] = local_max;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    if (wid == 0) {
        local_max = (lane < numWarps) ? smem[lane] : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
        if (lane == 0)
            s_max = local_max;
    }
    __syncthreads();
    float row_max = s_max;

    // Pass 2: 求 sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[row * N + i] - row_max);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0)
        smem[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        local_sum = (lane < numWarps) ? smem[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0)
            s_sum = local_sum;
    }
    __syncthreads();
    float row_sum = s_sum;

    // Pass 3: 归一化
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < N; i += blockDim.x) {
        output[row * N + i] = expf(input[row * N + i] - row_max) * inv_sum;
    }
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
    }
}

void cpuSoftmax(const float* in, float* out, int N) {
    float mx = in[0];
    for (int i = 1; i < N; i++) {
        mx = fmaxf(mx, in[i]);
    }
    float s = 0.0f;
    for (int i = 0; i < N; i++) {
        out[i] = expf(in[i] - mx);
        s += out[i];
    }
    for (int i = 0; i < N; i++) {
        out[i] /= s;
    }
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) {
        maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    }
    bool ok = maxDiff < eps;
    printf("  maxDiff = %.2e (%s)\n", maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    // 测试行级 Softmax（更实用的版本）
    const int M = 128;  // batch 行数
    const int N = 1024; // 每行元素数
    const int threads = 256;

    printf("=== Softmax Profiling (row-level, M=%d, N=%d) ===\n\n", M, N);

    size_t bytes = (size_t)M * N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    initData(h_in, M * N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 行级 Softmax
    cudaEventRecord(start);
    softmax_row_kernel<<<M, threads>>>(d_in, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    // 验证第一行
    cpuSoftmax(h_in, h_ref, N);
    printf("[softmax_row_kernel]\n");
    checkResult(h_out, h_ref, N, 1e-5f);
    printf("  Time: %.3f ms\n", ms);

    printf("\n=== ncu 分析命令 ===\n");
    printf("# 行级 Softmax\n");
    printf("ncu --kernel-name regex:softmax_row_kernel \\\n");
    printf("  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  ./softmax_profile\n\n");
    printf("# 完整报告\n");
    printf("ncu --set full --kernel-name regex:softmax_row_kernel -o softmax_report ./softmax_profile\n\n");
    printf("# nsys 时间线\n");
    printf("nsys profile -o softmax_timeline ./softmax_profile\n");

    free(h_in);
    free(h_out);
    free(h_ref);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
