"""dist_allreduce_demo.py —— torch.distributed 单机双进程 all-reduce 实测 demo

用 torchrun --nproc_per_node=2 启动两个进程，演示：
  1. init_process_group（gloo/NCCL 后端）
  2. 每个 rank 各持有一份 tensor，做 all-reduce 后所有 rank 得到总和
  3. 与 ring all-reduce 的通信量推导互证（2(P-1)·N·sizeof(dtype)）

运行（单卡 GPU 用 gloo，多卡用 NCCL）：

  # 单卡环境（gloo，CPU tensor，演示分布式语义）
  torchrun --nproc_per_node=2 --rdzv_backend=c10d \\
      --rdzv_endpoint=localhost:29500 dist_allreduce_demo.py

  # 多卡环境（NCCL，GPU tensor，真通信）
  torchrun --nproc_per_node=2 --rdzv_backend=c10d \\
      --rdzv_endpoint=localhost:29500 dist_allreduce_demo.py

  # 显式指定后端
  TORCH_DISTRIBUTED_BACKEND=gloo torchrun --nproc_per_node=2 ...

注意：NCCL 后端要求每个 rank 绑定不同的 GPU（rank i → cuda:i）。
单卡环境跑 NCCL 会报 "Duplicate GPU detected"，此时用 gloo 演示语义，
多卡环境才能真正测 NCCL 的 GPU 间通信带宽。

依赖: pip install torch
"""

import os
import sys
import time
import torch
import torch.distributed as dist


def main():
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
    if backend == "nccl" and torch.cuda.device_count() < 2:
        print(f"[INFO] GPU 数量={torch.cuda.device_count()}, NCCL 需多卡, 自动切到 gloo")
        backend = "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切到 gloo 后端")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if backend == "nccl":
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}/{world_size}] backend={backend}, device={device}")

    # 每个 rank 各持有一份 tensor: rank i 的值 = i+1
    tensor = torch.tensor([rank + 1, rank + 1, rank + 1, rank + 1],
                          dtype=torch.float32, device=device)

    print(f"[Rank {rank}] before all-reduce: {tensor.tolist()}")

    # all-reduce: 所有 rank 的 tensor 求和，结果广播到所有 rank
    # P=2 时: rank0=[1,1,1,1] + rank1=[2,2,2,2] → [3,3,3,3]
    dist.barrier()
    t0 = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()

    print(f"[Rank {rank}] after  all-reduce: {tensor.tolist()}")
    print(f"[Rank {rank}] all-reduce 耗时: {(t1-t0)*1000:.3f} ms")

    # 验证正确性
    expected = sum(i + 1 for i in range(world_size))
    assert tensor.tolist() == [expected] * 4, \
        f"all-reduce 结果错误: 期望 {[expected]*4}, 得到 {tensor.tolist()}"
    print(f"[Rank {rank}] 验证通过（期望 {expected}）")

    # ---- 通信量推导（与 ring all-reduce 互证）----
    # ring all-reduce 通信量 = 2(P-1)·N·sizeof(dtype)
    # P=2, N=4, float32(4B): 2×1×4×4 = 32 bytes
    N = 4
    dtype_size = 4  # float32
    comm_bytes = 2 * (world_size - 1) * N * dtype_size
    print(f"\n[Rank {rank}] ---- 通信量推导 ----")
    print(f"  ring all-reduce 通信量 = 2(P-1)·N·sizeof(dtype)")
    print(f"  P={world_size}, N={N}, sizeof={dtype_size}B")
    print(f"  通信量 = 2×{world_size-1}×{N}×{dtype_size} = {comm_bytes} bytes")
    print(f"  注: 实际 wall-clock 含 launch overhead + NCCL 内部优化,")
    print(f"  小 tensor 的 wall-clock 主要被 overhead 主导, 大 tensor 才看带宽")

    # ---- 大 tensor 带宽实测 ----
    big = torch.randn(1 << 20, device=device, dtype=torch.float32)  # 4MB
    # warmup
    for _ in range(3):
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    dist.all_reduce(big, op=dist.ReduceOp.SUM)
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    big_bytes = 2 * (world_size - 1) * (1 << 20) * 4
    elapsed = t1 - t0
    bandwidth = big_bytes / elapsed / 1e9  # GB/s
    print(f"\n[Rank {rank}] ---- 大 tensor 带宽实测 ----")
    print(f"  tensor 大小: {(1<<20)*4/1e6:.1f} MB")
    print(f"  通信量: {big_bytes/1e6:.1f} MB")
    print(f"  耗时: {elapsed*1000:.3f} ms")
    print(f"  有效带宽: {bandwidth:.2f} GB/s")
    if backend == "gloo":
        print(f"  (gloo 走 CPU/共享内存, 带宽远低于 NCCL 的 GPU 互联; 多卡 NCCL 才看真实通信带宽)")

    dist.destroy_process_group()
    print(f"[Rank {rank}] done")


if __name__ == "__main__":
    main()

