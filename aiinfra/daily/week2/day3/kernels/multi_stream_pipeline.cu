// multi_stream_pipeline.cu —— 多 Stream 重叠流水线完整实现
// 编译命令: nvcc -o multi_stream multi_stream_pipeline.cu -O3 -arch=sm_120
// 运行命令: ./multi_stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = A[i] + B[i];
        for (int j = 0; j < 10000; j++) {
            sum = sum * 0.999f + 0.001f;
        }
        C[i] = sum;
    }
}

// 顺序版本（baseline）
float sequentialVersion(float* h_A, float* h_B, float* h_C, float* d_A, float* d_B, float* d_C, int totalSize,
                        int chunkSize) {
    int numChunks = (totalSize + chunkSize - 1) / chunkSize;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;
        int currSize = (offset + chunkSize <= totalSize) ? chunkSize : (totalSize - offset);
        size_t bytes = currSize * sizeof(float);

        cudaMemcpy(d_A + offset, h_A + offset, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + offset, h_B + offset, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (currSize + threads - 1) / threads;
        vecAdd<<<blocks, threads>>>(d_A + offset, d_B + offset, d_C + offset, currSize);

        cudaMemcpy(h_C + offset, d_C + offset, bytes, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Multi-Stream 重叠版本
float multiStreamVersion(float* h_A, float* h_B, float* h_C, float* d_A, float* d_B, float* d_C, int totalSize,
                         int chunkSize, int nStreams) {
    int numChunks = (totalSize + chunkSize - 1) / chunkSize;
    cudaStream_t* streams = new cudaStream_t[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < numChunks; i++) {
        int streamIdx = i % nStreams;
        int offset = i * chunkSize;
        int currSize = (offset + chunkSize <= totalSize) ? chunkSize : (totalSize - offset);
        size_t bytes = currSize * sizeof(float);

        cudaMemcpyAsync(d_A + offset, h_A + offset, bytes, cudaMemcpyHostToDevice, streams[streamIdx]);
        cudaMemcpyAsync(d_B + offset, h_B + offset, bytes, cudaMemcpyHostToDevice, streams[streamIdx]);

        int threads = 256;
        int blocks = (currSize + threads - 1) / threads;
        vecAdd<<<blocks, threads, 0, streams[streamIdx]>>>(d_A + offset, d_B + offset, d_C + offset, currSize);

        cudaMemcpyAsync(h_C + offset, d_C + offset, bytes, cudaMemcpyDeviceToHost, streams[streamIdx]);
    }

    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int totalSize = 1 << 24; // 16,777,216 个元素
    const int chunkSize = 1 << 18; // 262,144 个元素 per chunk
    const int nStreams = 4;

    printf("=== Multi-Stream Overlap Pipeline ===\n");
    printf("Total size: %d (%.2f MB)\n", totalSize, totalSize * sizeof(float) / (1024.0 * 1024.0));
    printf("Chunk size: %d (%.2f MB)\n", chunkSize, chunkSize * sizeof(float) / (1024.0 * 1024.0));
    printf("Num chunks: %d, Num streams: %d\n\n", (totalSize + chunkSize - 1) / chunkSize, nStreams);

    size_t totalBytes = totalSize * sizeof(float);
    float *h_A, *h_B, *h_C_seq, *h_C_multi;
    cudaMallocHost(&h_A, totalBytes);
    cudaMallocHost(&h_B, totalBytes);
    cudaMallocHost(&h_C_seq, totalBytes);
    cudaMallocHost(&h_C_multi, totalBytes);

    srand(42);
    for (int i = 0; i < totalSize; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, totalBytes);
    cudaMalloc(&d_B, totalBytes);
    cudaMalloc(&d_C, totalBytes);

    printf("Running sequential version...\n");
    float seqMs = sequentialVersion(h_A, h_B, h_C_seq, d_A, d_B, d_C, totalSize, chunkSize);
    printf("Sequential: %.3f ms\n\n", seqMs);

    printf("Running multi-stream version (nStreams=%d)...\n", nStreams);
    float multiMs = multiStreamVersion(h_A, h_B, h_C_multi, d_A, d_B, d_C, totalSize, chunkSize, nStreams);
    printf("Multi-Stream: %.3f ms\n\n", multiMs);

    bool correct = true;
    for (int i = 0; i < totalSize; i++) {
        if (fabs(h_C_seq[i] - h_C_multi[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    float speedup = seqMs / multiMs;
    printf("=== Performance Summary ===\n");
    printf("Sequential: %.3f ms\n", seqMs);
    printf("Multi-Stream: %.3f ms\n", multiMs);
    printf("Speedup: %.2fx\n", speedup);
    printf("Result check: %s\n", correct ? "PASS" : "FAIL");

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_seq);
    cudaFreeHost(h_C_multi);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
