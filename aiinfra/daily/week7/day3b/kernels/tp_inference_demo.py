# tp_inference_demo.py —— 2-GPU Tensor Parallelism 推理 Demo（单卡模拟）
# 运行命令: python tp_inference_demo.py
# 依赖: torch（CPU 或单 GPU 均可运行）
# 说明: 用 torch.chunk 模拟权重切分，用 torch.sum 模拟 all-reduce
#       真实多卡场景将 chunk 换成 torch.distributed.all_reduce / all_gather 即可
"""
模拟 2-GPU Tensor Parallelism（TP=2）：
  - ColumnParallelLinear（QKV 投影）：权重按 output dim 切分
      单卡 W:[out, in]  -> rank_i 持有 W_i:[out/2, in]
      输入 X:[B, in] 各 rank 相同；输出 Y_i:[B, out/2]，按 head 天然并行，无需通信
  - RowParallelLinear（Output 投影）：权重按 input dim 切分
      单卡 W:[out, in]  -> rank_i 持有 W_i:[out, in/2]
      输入 X_i:[B, in/2] 各 rank 不同；输出 Y_i:[B, out] 为部分和
      需 all-reduce(sum) 聚合得最终 Y
  - TP Attention Block = ColumnParallel(QKV) + Attention + RowParallel(Output)
      一个 block 仅需 1 次 all-reduce（在 Output 投影处）
"""

import time
import torch
import torch.nn as nn

TP_SIZE = 2


class ColumnParallelLinear(nn.Module):
    """列并行线性层：W 按 output dim 切分，各 rank 输出 [B, out/tp]，拼接即完整输出"""

    def __init__(self, in_features, out_features, tp_size=TP_SIZE):
        super().__init__()
        assert out_features % tp_size == 0
        self.tp_size = tp_size
        self.shard = out_features // tp_size
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.shard, in_features) * 0.02)
            for _ in range(tp_size)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.shard))
            for _ in range(tp_size)
        ])

    def forward(self, x):
        # 每个 rank 独立计算 Y_i = x @ W_i^T + b_i
        shards = [x @ w.t() + b for w, b in zip(self.weights, self.bias)]
        # 模拟"拼接"：实际 TP 中下游按 head 消费，常无需显式 concat
        return torch.cat(shards, dim=-1)

    def single_gpu_weight(self):
        W = torch.cat([w for w in self.weights], dim=0)
        b = torch.cat([b for b in self.bias], dim=0)
        return W, b


class RowParallelLinear(nn.Module):
    """行并行线性层：W 按 input dim 切分，各 rank 输出部分和，all-reduce 聚合"""

    def __init__(self, in_features, out_features, tp_size=TP_SIZE):
        super().__init__()
        assert in_features % tp_size == 0
        self.tp_size = tp_size
        self.shard = in_features // tp_size
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_features, self.shard) * 0.02)
            for _ in range(tp_size)
        ])
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 输入按 input dim 切分（模拟各 rank 持有不同输入片）
        x_shards = torch.chunk(x, self.tp_size, dim=-1)
        # 各 rank 计算部分和 Y_i = X_i @ W_i^T
        partial = [x_i @ w.t() for x_i, w in zip(x_shards, self.weights)]
        # 模拟 all-reduce(sum)：真实场景用 torch.distributed.all_reduce(y)
        y = torch.stack(partial, dim=0).sum(dim=0) + self.bias
        return y

    def single_gpu_weight(self):
        W = torch.cat([w for w in self.weights], dim=1)
        return W, self.bias


class TPAttentionBlock(nn.Module):
    """简化 TP Attention Block：ColumnParallel(QKV) + RowParallel(Output)
    这是 TP 的经典 pattern——前层 column 切 QKV（按 head），后层 row 聚合输出
    中间 attention 各 head 独立计算，无需跨 rank 通信
    """

    def __init__(self, d_model=512, n_heads=8, tp_size=TP_SIZE):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert n_heads % tp_size == 0
        self.qkv = ColumnParallelLinear(d_model, 3 * d_model, tp_size)
        self.out = RowParallelLinear(d_model, d_model, tp_size)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        scale = self.d_head ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out(ctx)

    def single_gpu_forward(self, x):
        B, N, D = x.shape
        Wqkv, bqkv = self.qkv.single_gpu_weight()
        qkv = torch.nn.functional.linear(x, Wqkv, bqkv)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        scale = self.d_head ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        Wout, bout = self.out.single_gpu_weight()
        return torch.nn.functional.linear(ctx, Wout, bout)


def correctness_test():
    print("=" * 64)
    print("  正确性测试：TP=2 模拟 vs 单卡等价")
    print("=" * 64)
    torch.manual_seed(42)
    block = TPAttentionBlock(d_model=512, n_heads=8, tp_size=2).eval()
    x = torch.randn(2, 128, 512)
    with torch.no_grad():
        y_tp = block.forward(x)
        y_single = block.single_gpu_forward(x)
    max_diff = (y_tp - y_single).abs().max().item()
    print(f"  output shape : TP={tuple(y_tp.shape)}, single={tuple(y_single.shape)}")
    print(f"  max abs diff : {max_diff:.2e}")
    print(f"  result       : {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    return max_diff < 1e-5


def communication_pattern():
    print("\n" + "=" * 64)
    print("  通信模式分析（TP=2，一个 Attention Block）")
    print("=" * 64)
    print("  ColumnParallelLinear (QKV 投影):")
    print("    输入 X[B, in]   各 rank 相同（broadcast 或各 rank 自有）")
    print("    权重 W_i[out/2, in]  各 rank 不同（按 output dim 切分）")
    print("    输出 Y_i[B, out/2]   各 rank 不同（按 head 天然并行）")
    print("    通信量: 0（无需 all-reduce，下游 attention 按 head 消费）")
    print("  RowParallelLinear (Output 投影):")
    print("    输入 X_i[B, in/2]    各 rank 不同（来自上一层 column 切分）")
    print("    权重 W_i[out, in/2]  各 rank 不同（按 input dim 切分）")
    print("    输出 Y_i[B, out]     各 rank 为部分和")
    print("    通信量: all-reduce(sum) [B, out] 元素，FP32 每元素 4B")
    print("  => 一个 Attention Block 共 1 次 all-reduce（在 output 投影处）")


def timing_comparison():
    print("\n" + "=" * 64)
    print("  性能对比（单卡模拟，仅验证通信 pattern，非真实多卡加速）")
    print("=" * 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    torch.manual_seed(0)
    block = TPAttentionBlock(d_model=1024, n_heads=16, tp_size=2).to(device).eval()
    x = torch.randn(8, 512, 1024, device=device)
    with torch.no_grad():
        for _ in range(3):
            block(x); block.single_gpu_forward(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            block(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t_tp = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            block.single_gpu_forward(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t_single = (time.perf_counter() - t0) / 20 * 1000
    print(f"  TP=2 forward   : {t_tp:.3f} ms")
    print(f"  single forward : {t_single:.3f} ms")
    print(f"  注: 单卡模拟无法体现真实多卡加速，仅验证正确性与通信 pattern")
    print(f"  真实 2-GPU TP: GEMM 算力 x2，代价是 1 次 all-reduce（约 1-2 ms / 1KB 量级）")


if __name__ == "__main__":
    ok = correctness_test()
    communication_pattern()
    timing_comparison()
    print("\n" + "=" * 64)
    print(f"  Demo 完成！TP 推理模拟 {'通过' if ok else '失败'}")
    print("=" * 64)
