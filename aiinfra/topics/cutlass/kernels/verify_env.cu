// verify_env.cu —— 验证 GPU 环境是否支持 CUTLASS
// 编译: nvcc -o verify_env verify_env.cu -arch=sm_90a -std=c++17
// 运行: ./verify_env

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== GPU 环境验证 ===\n");
    printf("设备名称:        %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM 数量:         %d\n", prop.multiProcessorCount);
    printf("共享内存/SM:     %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("寄存器/SM:       %d\n", prop.regsPerMultiprocessor);
    printf("全局内存:        %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Warp Size:       %d\n", prop.warpSize);

    int major = prop.major;
    printf("\n=== Tensor Core 支持 ===\n");
    if (major >= 8) {
        printf("✅ Ampere+ Tensor Core (mma.m16n8k16) 可用\n");
    } else if (major >= 7) {
        printf("✅ Volta/Turing Tensor Core (wmma) 可用\n");
    } else {
        printf("❌ 不支持 Tensor Core，CUTLASS 3.x 需要 CC >= 8.0\n");
        return -1;
    }

    if (major >= 9) {
        printf("✅ Hopper TMA + WGMMA 可用\n");
    }
    if (major >= 12) {
        printf("✅ Blackwell 下一代 Tensor Core 可用\n");
    }

    printf("\n✅ 环境验证通过，可以开始 CUTLASS 学习\n");
    return 0;
}
