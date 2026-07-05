#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 实验 2：对比不同 TILE_DIM 下 shared memory 转置的性能
//
// 分别测试 8x8、16x16、32x32 三种 tile 大小：
//   - 8x8：每个 block 64 个线程，约 256 字节 shared memory，占用资源最小
//   - 16x16：每个 block 256 个线程，约 1 KB shared memory，平衡配置
//   - 32x32：每个 block 1024 个线程，约 4 KB shared memory（含 padding），
//            是 CUDA 单个 block 允许的最大线程数，常用配置
//
// 注：64x64 的 tile 若对应 64x64 线程会超过 CUDA 单个 block 1024 线程的限制，
//     实现更大 tile 需要让每个线程处理多个元素或采用更复杂的线程映射。
//
// 用模板参数把 TILE_DIM 编译为常量，使 __shared__ 数组维度可在编译期确定。

template <int TILE_DIM>
__global__ void transpose_tiled(const float* in, float* out, int width, int height) {
    // +1 padding 消除 shared memory bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Transpose block coordinates for coalesced write
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void fill_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// 检查 tiled 转置结果是否与 CPU 转置一致
template <int TILE_DIM>
bool run_and_check(const float* d_in, float* d_out, float* h_out,
                   const float* h_ref, int width, int height, int size,
                   float* elapsed_ms) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 多次重复测量，取平均
    const int repeats = 20;

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        transpose_tiled<TILE_DIM><<<grid, block>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    *elapsed_ms = ms_total / repeats;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int y = 0; y < height && ok; ++y) {
        for (int x = 0; x < width; ++x) {
            if (h_out[x * height + y] != h_ref[y * width + x]) {
                ok = false;
                break;
            }
        }
    }
    return ok;
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height;

    printf("=== Transpose with Different Tile Sizes ===\n");
    printf("Matrix: %d x %d\n\n", width, height);

    float* h_in = (float*)malloc(size * sizeof(float));
    float* h_out = (float*)malloc(size * sizeof(float));
    fill_matrix(h_in, size);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    float ms = 0.0f;
    double total_bytes = 2.0 * size * sizeof(float);

    printf("Tile Size | Correctness | Avg Time (ms) | Effective Bandwidth (GB/s)\n");
    printf("----------|-------------|---------------|----------------------------\n");

    // 8x8
    bool ok8 = run_and_check<8>(d_in, d_out, h_out, h_in, width, height, size, &ms);
    printf("8 x 8     | %s        | %13.4f | %26.2f\n",
           ok8 ? "PASS" : "FAIL", ms, total_bytes / (ms / 1000.0) / 1e9);

    // 16x16
    bool ok16 = run_and_check<16>(d_in, d_out, h_out, h_in, width, height, size, &ms);
    printf("16 x 16   | %s        | %13.4f | %26.2f\n",
           ok16 ? "PASS" : "FAIL", ms, total_bytes / (ms / 1000.0) / 1e9);

    // 32x32
    bool ok32 = run_and_check<32>(d_in, d_out, h_out, h_in, width, height, size, &ms);
    printf("32 x 32   | %s        | %13.4f | %26.2f\n",
           ok32 ? "PASS" : "FAIL", ms, total_bytes / (ms / 1000.0) / 1e9);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
