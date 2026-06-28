#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

// Naive: coalesced read, strided write
__global__ void transpose_naive(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

// Optimized: shared memory tile with padding to avoid bank conflict
__global__ void transpose_optimized(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Transpose block coordinates for write
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write from shared memory to global memory
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void fill_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height;

    float *h_in = (float*)malloc(size * sizeof(float));
    float *h_out_naive = (float*)malloc(size * sizeof(float));
    float *h_out_opt = (float*)malloc(size * sizeof(float));
    fill_matrix(h_in, size);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    cudaMemcpy(h_out_naive, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    transpose_optimized<<<grid, block>>>(d_in, d_out, width, height);
    cudaMemcpy(h_out_opt, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Simple correctness check: compare with CPU transpose
    bool ok = true;
    for (int y = 0; y < height && ok; ++y) {
        for (int x = 0; x < width; ++x) {
            float ref = h_in[y * width + x];
            if (h_out_naive[x * height + y] != ref || h_out_opt[x * height + y] != ref) {
                ok = false;
                break;
            }
        }
    }
    printf("Transpose correctness: %s\n", ok ? "PASS" : "FAIL");

    free(h_in);
    free(h_out_naive);
    free(h_out_opt);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
