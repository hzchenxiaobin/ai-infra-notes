#include <cuda_runtime.h>
#include <stdio.h>

// 本程序用于验证 Day 3 occupancy 手算练习的结果。
// 它会打印当前 GPU 的硬件参数，并对几个不同资源用量的 kernel：
// 1. 用 cudaFuncGetAttributes 查询寄存器/共享内存用量
// 2. 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 获取运行时 active blocks
// 3. 按课程中的公式手工计算 active blocks / occupancy
//
// 编译运行：
//   nvcc -o occupancy_verify occupancy_verify.cu
//   ./occupancy_verify

struct OccProps {
    int maxThreadsPerSM;
    int maxBlocksPerSM;
    int maxWarpsPerSM;
    int regsPerSM;
    int smemPerSM;
    int warpSize;
};

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline int min4(int a, int b, int c, int d) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    if (d < m) m = d;
    return m;
}

// 按 A100 (CC 8.0) 及同代架构常用的粒度建模：
// 寄存器按 256 个/block 对齐，共享内存按 1024 bytes/block 对齐。
// 若在其他架构上运行，granularity 可能有差异，结果仅供理解原理。
void calculate_occupancy(const OccProps& p, int blockSize, int regsPerThread,
                         int smemPerBlock, int& activeBlocks, int& activeWarps,
                         float& occupancy) {
    const int regGranularity = 256;
    const int smemGranularity = 1024;

    int warpsPerBlock = ceil_div(blockSize, p.warpSize);

    int blocksFromThreads = p.maxThreadsPerSM / blockSize;

    int regsPerBlock = ceil_div(blockSize * regsPerThread, regGranularity) * regGranularity;
    int blocksFromRegs = (regsPerBlock > 0) ? (p.regsPerSM / regsPerBlock) : p.maxBlocksPerSM;

    int smemPerBlockAligned = ceil_div(smemPerBlock, smemGranularity) * smemGranularity;
    int blocksFromSmem = (smemPerBlockAligned > 0) ? (p.smemPerSM / smemPerBlockAligned)
                                                   : p.maxBlocksPerSM;

    activeBlocks = min4(blocksFromThreads, blocksFromRegs, blocksFromSmem, p.maxBlocksPerSM);
    activeWarps = activeBlocks * warpsPerBlock;
    occupancy = (float)activeWarps / p.maxWarpsPerSM * 100.0f;
}

void print_device_props(const cudaDeviceProp& prop) {
    printf("=== Device: %s (Compute Capability %d.%d) ===\n", prop.name, prop.major, prop.minor);
    printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    printf("  Max threads / SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max blocks / SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Max warps / SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("  Registers / SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory / SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("  Warp size: %d\n\n", prop.warpSize);
}

// ------------------------------------------------------------------
// 几个资源用量不同的示例 kernel
// ------------------------------------------------------------------

// 轻量 kernel：寄存器和共享内存都很少
__global__ void kernel_light(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// 中等寄存器压力：使用多个累加变量
__global__ void kernel_medium(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    float e = 0.0f, f = 0.0f, g = 0.0f, h = 0.0f;
    if (idx < n) {
        float v = in[idx];
        a += v;       b += v * 2.0f; c += v * 3.0f; d += v * 4.0f;
        e += v * 5.0f; f += v * 6.0f; g += v * 7.0f; h += v * 8.0f;
    }
    if (idx < n) {
        out[idx] = a + b + c + d + e + f + g + h;
    }
}

// 使用静态共享内存：静态分配 1024 bytes (256 floats)
__global__ void kernel_smem(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < n) {
        sdata[tid] = in[idx];
    }
    __syncthreads();
    if (idx < n) {
        out[idx] = sdata[tid] * 2.0f;
    }
}

// 使用 launch_bounds 提示编译器：每个 SM 至少驻留 2 个 block
__launch_bounds__(256, 2)
__global__ void kernel_launch_bounds(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        acc += in[(idx + i) % n] * 1.5f;
    }
    if (idx < n) {
        out[idx] = acc;
    }
}

// ------------------------------------------------------------------
// 对单个 kernel + block size + 动态共享内存做分析
// ------------------------------------------------------------------
void analyze_kernel(const OccProps& p, const char* name, const void* kernelFunc,
                    int blockSize, int dynamicSmem = 0) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernelFunc);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaFuncGetAttributes failed for %s: %s\n\n",
               name, cudaGetErrorString(err));
        return;
    }

    int activeBlocksCUDA = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksCUDA, kernelFunc, blockSize, dynamicSmem);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaOccupancyMaxActiveBlocksPerMultiprocessor failed for %s: %s\n\n",
               name, cudaGetErrorString(err));
        return;
    }

    // 手算时把静态共享内存和动态共享内存相加
    int totalSmemPerBlock = (int)attr.sharedSizeBytes + dynamicSmem;

    int activeBlocksHand = 0, activeWarpsHand = 0;
    float occupancyHand = 0.0f;
    calculate_occupancy(p, blockSize, attr.numRegs, totalSmemPerBlock,
                        activeBlocksHand, activeWarpsHand, occupancyHand);

    int warpsPerBlock = ceil_div(blockSize, p.warpSize);
    int activeWarpsCUDA = activeBlocksCUDA * warpsPerBlock;
    float occupancyCUDA = (float)activeWarpsCUDA / p.maxWarpsPerSM * 100.0f;

    printf("--- %s (blockSize=%d, dynamicSmem=%d) ---\n", name, blockSize, dynamicSmem);
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory per block: %zu bytes (static) + %d bytes (dynamic) = %d bytes\n",
           attr.sharedSizeBytes, dynamicSmem, totalSmemPerBlock);
    printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);
    printf("  Warps per block: %d\n", warpsPerBlock);
    printf("  CUDA API   -> active blocks / SM: %d, active warps / SM: %d, occupancy: %.1f%%\n",
           activeBlocksCUDA, activeWarpsCUDA, occupancyCUDA);
    printf("  Hand calc  -> active blocks / SM: %d, active warps / SM: %d, occupancy: %.1f%%\n",
           activeBlocksHand, activeWarpsHand, occupancyHand);
    printf("\n");
}

int main() {
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    print_device_props(prop);

    OccProps p;
    p.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    p.maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    p.maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    p.regsPerSM = prop.regsPerMultiprocessor;
    p.smemPerSM = prop.sharedMemPerMultiprocessor;
    p.warpSize = prop.warpSize;

    printf("=== Occupancy Analysis for Sample Kernels ===\n\n");

    analyze_kernel(p, "kernel_light", (const void*)kernel_light, 256);
    analyze_kernel(p, "kernel_medium", (const void*)kernel_medium, 256);
    analyze_kernel(p, "kernel_smem", (const void*)kernel_smem, 256);
    analyze_kernel(p, "kernel_launch_bounds", (const void*)kernel_launch_bounds, 256);

    printf("=== Varying Block Size for kernel_medium ===\n\n");
    int blockSizes[] = {128, 256, 512, 1024};
    for (int i = 0; i < 4; ++i) {
        analyze_kernel(p, "kernel_medium", (const void*)kernel_medium, blockSizes[i]);
    }

    printf("=== Varying Dynamic Shared Memory for kernel_light ===\n\n");
    int smemSizes[] = {0, 4096, 8192, 16384};
    for (int i = 0; i < 4; ++i) {
        analyze_kernel(p, "kernel_light", (const void*)kernel_light, 256, smemSizes[i]);
    }

    return 0;
}
