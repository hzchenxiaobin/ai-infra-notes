#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 实验 1：测量 coalesced vs stride 内存访问的有效带宽
//
// 说明：
//   - coalesced_copy：线程 idx 访问 in[idx]，地址连续，可合并访问
//   - stride_copy：线程 idx 访问 in[(idx * stride) % n]，地址间隔大，产生 stride access
//
// 使用 cudaEvent 测量 kernel 执行时间，并估算有效带宽：
//   effective_bandwidth_GB/s = (read_bytes + write_bytes) / elapsed_seconds / 1e9

// 连续访问：每个 warp 的 32 个线程读取连续地址，合并为少量 transaction
__global__ void coalesced_copy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// 间隔访问：线程 idx 以 stride 为步长读取，warp 内地址分散在不同 sector
__global__ void stride_copy(const float* in, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 通过取模让索引落在 [0, n) 内，保证读取位置合法
        out[idx] = in[(static_cast<long long>(idx) * stride) % n];
    }
}

void fill_array(float* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// 运行指定 kernel 多次，返回平均耗时（毫秒）
template <typename Kernel, typename... Args>
float benchmark_kernel(Kernel kernel, int threads_per_block, int n, int warmup, int repeats, Args... args) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    // warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads_per_block>>>(args...);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        kernel<<<blocks, threads_per_block>>>(args...);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeats;
}

int main() {
    // 数组大小：256 MB float 数组（约 6710 万元素）
    const int n = 1 << 26;
    const size_t bytes = n * sizeof(float);

    printf("=== Coalesced vs Stride Bandwidth Benchmark ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", n, bytes / (1024.0 * 1024.0));

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    fill_array(h_in, n);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int warmup = 5;
    const int repeats = 20;

    // 1. Coalesced copy
    float ms_coalesced = benchmark_kernel(coalesced_copy, threads_per_block, n, warmup, repeats, d_in, d_out, n);

    // 2. Stride copy with different strides
    int strides[] = {1, 2, 4, 8, 16, 32};
    const int num_strides = sizeof(strides) / sizeof(strides[0]);

    printf("Kernel                    | Elapsed (ms) | Effective Bandwidth (GB/s)\n");
    printf("-------------------------|--------------|----------------------------\n");

    double total_bytes = 2.0 * bytes; // read + write
    double bw_coalesced = total_bytes / (ms_coalesced / 1000.0) / 1e9;
    printf("coalesced_copy           | %12.4f | %26.2f\n", ms_coalesced, bw_coalesced);

    for (int i = 0; i < num_strides; ++i) {
        int stride = strides[i];
        float ms_stride = benchmark_kernel(stride_copy, threads_per_block, n, warmup, repeats, d_in, d_out, n, stride);
        double bw_stride = total_bytes / (ms_stride / 1000.0) / 1e9;
        printf("stride_copy(stride=%2d)  | %12.4f | %26.2f\n", stride, ms_stride, bw_stride);
    }

    // 简单正确性检查：coalesced_copy 输出应与输入相同
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        if (h_out[i] != h_in[i]) {
            ok = false;
        }
    }
    printf("\nCoalesced copy correctness: %s\n", ok ? "PASS" : "FAIL");

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
