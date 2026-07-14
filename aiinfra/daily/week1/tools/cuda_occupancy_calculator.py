#!/usr/bin/env python3
"""
CUDA Occupancy Calculator (Python version)

NVIDIA 官方的 Excel 版 CUDA Occupancy Calculator 已经停止维护。
这个脚本提供了等效的计算能力：根据 GPU 计算能力、kernel 资源使用情况、
block 大小，计算理论 occupancy。

用法：
    python3 cuda_occupancy_calculator.py --cc 8.0 --registers 32 --block-size 256
    python3 cuda_occupancy_calculator.py --cc 7.5 --registers 64 --smem 8192 --block-size 128
"""

import argparse
import math


# GPU 架构参数表
# 数据来源：CUDA C Programming Guide, Compute Capabilities
GPU_ARCHS = {
    # 计算能力: (每 SM 最大线程数, 每 SM 最大 warp 数, 每 SM 最大 block 数,
    #           每 SM 寄存器数量, 每 SM 共享内存字节数, 每 block 最大共享内存,
    #           warp 分配粒度, 寄存器分配粒度)
    "5.0":  (2048, 64, 32, 65536, 65536,  49152, 2, 256),
    "5.2":  (2048, 64, 32, 65536, 98304,  49152, 2, 256),
    "5.3":  (2048, 64, 32, 65536, 65536,  49152, 2, 256),
    "6.0":  (2048, 64, 32, 65536, 65536,  49152, 2, 256),
    "6.1":  (2048, 64, 32, 65536, 98304,  49152, 4, 256),
    "6.2":  (2048, 64, 32, 65536, 98304,  49152, 4, 256),
    "7.0":  (2048, 64, 32, 65536, 98304,  49152, 4, 256),
    "7.2":  (2048, 64, 32, 65536, 98304,  49152, 4, 256),
    "7.5":  (2048, 64, 16, 65536, 65536,  49152, 4, 256),
    "8.0":  (2048, 64, 32, 65536, 167936, 49152, 4, 256),
    "8.6":  (1536, 48, 16, 65536, 167936, 49152, 4, 256),
    "8.7":  (2048, 64, 32, 65536, 167936, 49152, 4, 256),
    "8.9":  (1536, 48, 24, 65536, 99000,  49152, 4, 256),
    "9.0":  (2048, 64, 32, 65536, 228000, 49152, 4, 256),
    "10.0": (2048, 64, 32, 65536, 228000, 49152, 4, 256),
    "10.1": (2048, 64, 32, 65536, 228000, 49152, 4, 256),
    "12.0": (2048, 64, 32, 65536, 228000, 49152, 4, 256),
    "12.1": (2048, 64, 32, 65536, 228000, 49152, 4, 256),
}


def ceil_div(a, b):
    return (a + b - 1) // b


def calculate_occupancy(cc, registers_per_thread, shared_mem_per_block,
                        block_size, blocks_per_grid=1):
    """
    计算 CUDA kernel 的理论 occupancy。

    参数：
        cc: 计算能力字符串，如 "8.0"
        registers_per_thread: 每个线程的寄存器数
        shared_mem_per_block: 每个 block 的共享内存字节数
        block_size: 每个 block 的线程数
        blocks_per_grid: grid 中 block 的数量（用于计算需要的 SM 数）
    """
    if cc not in GPU_ARCHS:
        raise ValueError(f"不支持的计算能力: {cc}。支持: {list(GPU_ARCHS.keys())}")

    (max_threads_per_sm, max_warps_per_sm, max_blocks_per_sm,
     total_registers_per_sm, total_smem_per_sm, max_smem_per_block,
     warp_alloc_granularity, reg_alloc_granularity) = GPU_ARCHS[cc]

    # warp 数按粒度向上取整
    warps_per_block = ceil_div(block_size, 32)
    warps_per_block = ((warps_per_block + warp_alloc_granularity - 1)
                       // warp_alloc_granularity * warp_alloc_granularity)

    # 1. 受线程数限制
    blocks_by_threads = max_threads_per_sm // block_size

    # 2. 受 warp 数限制
    blocks_by_warps = max_warps_per_sm // warps_per_block

    # 3. 受 block 数限制
    blocks_by_block_limit = max_blocks_per_sm

    # 4. 受寄存器限制
    # 每个 block 需要的寄存器按粒度向上取整
    regs_per_block = (registers_per_thread * block_size)
    regs_per_block = ((regs_per_block + reg_alloc_granularity - 1)
                      // reg_alloc_granularity * reg_alloc_granularity)
    blocks_by_registers = total_registers_per_sm // regs_per_block

    # 5. 受共享内存限制
    blocks_by_smem = total_smem_per_sm // shared_mem_per_block if shared_mem_per_block > 0 else float('inf')

    # 实际每 SM 能同时运行的 block 数取最小值
    active_blocks_per_sm = min(blocks_by_threads, blocks_by_warps,
                               blocks_by_block_limit, blocks_by_registers,
                               blocks_by_smem)

    # 实际每 SM 的 active warp 数
    active_warps_per_sm = active_blocks_per_sm * warps_per_block

    # occupancy
    occupancy = active_warps_per_sm / max_warps_per_sm

    return {
        "compute_capability": cc,
        "block_size": block_size,
        "warps_per_block": warps_per_block,
        "registers_per_thread": registers_per_thread,
        "shared_mem_per_block": shared_mem_per_block,
        "max_blocks_per_sm": max_blocks_per_sm,
        "blocks_by_threads": blocks_by_threads,
        "blocks_by_warps": blocks_by_warps,
        "blocks_by_registers": blocks_by_registers,
        "blocks_by_smem": blocks_by_smem if shared_mem_per_block > 0 else "N/A",
        "active_blocks_per_sm": active_blocks_per_sm,
        "active_warps_per_sm": active_warps_per_sm,
        "max_warps_per_sm": max_warps_per_sm,
        "occupancy": occupancy,
    }


def print_report(result):
    print("=" * 60)
    print("CUDA Occupancy Calculator (Python)")
    print("=" * 60)
    print(f"计算能力 (Compute Capability): {result['compute_capability']}")
    print(f"Block 大小:                   {result['block_size']}")
    print(f"每 Block Warp 数:             {result['warps_per_block']}")
    print(f"每线程寄存器数:               {result['registers_per_thread']}")
    print(f"每 Block 共享内存:            {result['shared_mem_per_block']} bytes")
    print("-" * 60)
    print("每 SM 限制分析:")
    print(f"  受最大 block 数限制:        {result['max_blocks_per_sm']}")
    print(f"  受线程数限制:               {result['blocks_by_threads']}")
    print(f"  受 warp 数限制:             {result['blocks_by_warps']}")
    print(f"  受寄存器限制:               {result['blocks_by_registers']}")
    smem_limit = result['blocks_by_smem']
    print(f"  受共享内存限制:             {smem_limit}")
    print("-" * 60)
    print(f"实际每 SM 活跃 block 数:      {result['active_blocks_per_sm']}")
    print(f"实际每 SM 活跃 warp 数:       {result['active_warps_per_sm']} / {result['max_warps_per_sm']}")
    print(f"理论 Occupancy:               {result['occupancy'] * 100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Occupancy Calculator (Python version)")
    parser.add_argument("--cc", required=True,
                        help="GPU 计算能力，如 8.0、7.5、6.1")
    parser.add_argument("--registers", type=int, required=True,
                        help="每个线程的寄存器数")
    parser.add_argument("--smem", type=int, default=0,
                        help="每个 block 的共享内存字节数 (默认 0)")
    parser.add_argument("--block-size", type=int, required=True,
                        help="每个 block 的线程数")
    parser.add_argument("--blocks", type=int, default=1,
                        help="grid 中 block 数量 (默认 1)")

    args = parser.parse_args()

    result = calculate_occupancy(
        cc=args.cc,
        registers_per_thread=args.registers,
        shared_mem_per_block=args.smem,
        block_size=args.block_size,
        blocks_per_grid=args.blocks
    )

    print_report(result)


if __name__ == "__main__":
    main()
