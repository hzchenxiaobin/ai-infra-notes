# cuda_vs_ascend_comparison.py —— CUDA vs Ascend CANN 对比速查表
# 运行命令: python cuda_vs_ascend_comparison.py
# 依赖: 仅标准库

ARCHITECTURE = [
    ("计算单元", "SM (Streaming Multiprocessor)", "AI Core (Cube+Vector+Scalar+MTE)"),
    ("执行模型", "SIMT (warp=32 thread)", "指令级并行 (三单元流水)"),
    ("调度粒度", "warp (32 thread)", "tiling (数据块)"),
    ("矩阵加速", "Tensor Core (WMMA/mma)", "Cube Unit (matmul, 16x16x16)"),
    ("数据搬运", "thread 自带 load/store", "MTE 异步 DMA"),
    ("延迟隐藏", "warp 切换 (硬件调度)", "三单元流水 (软件 pipe)"),
]

PROGRAMMING_MODEL = [
    ("线程层次", "grid > block > thread", "grid > block > tiling"),
    ("片上共享存储", "shared memory (~100KB/SM)", "Unified Buffer (UB)"),
    ("warp/块内通信", "warp shuffle (__shfl_*)", "UB 间 DataCopy (vector copy)"),
    ("块内同步", "__syncthreads()", "sync barrier (pipe_barrier)"),
    ("向量化加载", "float4 (128-bit load)", "DataCopy (burst length)"),
    ("矩阵加速", "Tensor Core (WMMA)", "Cube Unit (matmul)"),
    ("kernel 语言", "CUDA C++ (.cu)", "Ascend C++ (.cpp)"),
]

MEMORY_HIERARCHY = [
    ("Register", "~0 cycle", "~0 cycle"),
    ("片上共享", "shared memory ~20-30 cycles", "Unified Buffer ~数十 cycles"),
    ("L1/L2 Cache", "L1 ~20c, L2 ~200c", "L1 ~数十c, L2 ~百c"),
    ("Global Memory", "GDDR7/HBM ~400-800c", "HBM ~数百 cycles"),
]

TOOLCHAIN = [
    ("编译器", "nvcc", "Ascend 编译器 (aoe/msauccomp)"),
    ("Kernel Profiler", "Nsight Compute (ncu)", "msprof / Ascend Profiler"),
    ("System Profiler", "Nsight Systems (nsys)", "msprof (system trace)"),
    ("运行时 API", "CUDA Runtime (cudaXxx)", "ACL (Ascend Computing Language)"),
    ("数学库", "cuBLAS / cuDNN", "ACL BLAS / ACL NN"),
    ("集合通信", "NCCL", "HCCL"),
    ("框架集成", "PyTorch CUDA", "torch_npu (PyTorch Ascend)"),
]

OPTIMIZATION = [
    ("访存合并", "coalesced access", "contiguous access"),
    ("bank 冲突", "shared memory 32 bank", "UB bank conflict"),
    ("数据复用", "shared memory tiling", "UB tiling + L1 cache"),
    ("矩阵加速单元", "Tensor Core (WMMA)", "Cube Unit (matmul)"),
    ("向量化", "float4 (128-bit)", "DataCopy (burst)"),
    ("双缓冲", "double buffering (ping-pong)", "DoubleBuffer (pipe TQue)"),
    ("延迟隐藏", "warp 切换 (多 warp 并发)", "三单元流水 (MTE+Cube+Vector)"),
    ("自动调优", "auto-tuning (参数搜索)", "tiling 参数自动搜索"),
]

MIGRATION = [
    ("tiling 思想", "shared memory tile", "UB tile", "可迁移"),
    ("访存合并", "coalesced", "contiguous", "可迁移"),
    ("矩阵加速思路", "Tensor Core 用法", "Cube Unit 用法", "可迁移"),
    ("双缓冲", "double buffer", "pipe DoubleBuffer", "可迁移"),
    ("warp 级优化", "warp shuffle / 32 对齐", "无 warp 概念", "重新设计"),
    ("thread 寄存器", "per-thread reg", "tiling 级无 thread", "重新设计"),
    ("block size", "由 reg/SMem 决定", "由 UB+Cube 尺寸", "重新设计"),
    ("同步原语", "__syncthreads", "sync barrier", "语义迁移"),
]

SECTIONS = [
    ("1. 架构对比 (GPU vs NPU)", ARCHITECTURE, ("维度", "NVIDIA CUDA", "Ascend CANN")),
    ("2. 编程模型对比", PROGRAMMING_MODEL, ("维度", "NVIDIA CUDA", "Ascend CANN")),
    ("3. 存储层次对比", MEMORY_HIERARCHY, ("层级", "NVIDIA CUDA", "Ascend CANN")),
    ("4. 工具链对比", TOOLCHAIN, ("工具", "NVIDIA", "Ascend")),
    ("5. 优化技术对比", OPTIMIZATION, ("优化点", "CUDA", "Ascend")),
    ("6. 迁移策略 (CUDA -> Ascend)", MIGRATION, ("概念", "CUDA", "Ascend", "迁移性")),
]


def print_table(title, rows, headers):
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def main():
    print("=" * 72)
    print("  CUDA vs Ascend CANN 多硬件对比速查表")
    print("=" * 72)
    for title, rows, headers in SECTIONS:
        print_table(title, rows, headers)
    print("\n" + "=" * 72)
    print("  口诀：架构不同原理通——tiling/合并/矩阵加速/双缓冲可迁移，")
    print("        warp/thread 级优化需重新设计，延迟隐藏范式根本不同。")
    print("=" * 72)


if __name__ == "__main__":
    main()
