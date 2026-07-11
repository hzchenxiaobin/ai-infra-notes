#include <stdio.h>
__global__ void hello_gpu() {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("block=(%d, %d, %d), thread=(%d, %d, %d), global_tid=%d\n",
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, global_tid
    );
}

int main() {
    dim3 grid(2, 2, 1);
    dim3 block(4, 2, 1);

    printf("Launching kernel: grid=(%d, %d, %d), block=(%d, %d, %d)\n",
    grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("Total threads: %d\n",  grid.x * grid.y * grid.z * block.x * block.y * block.z);

    hello_gpu<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}