# cuda_graph_capture.py —— PyTorch CUDA Graph 捕获 Demo（Mini Engine Decode 单步）
# 运行命令: python cuda_graph_capture.py
# 依赖: torch + CUDA（单 GPU 即可）
# Profiling: nsys profile -o cuda_graph --trace=cuda python cuda_graph_capture.py
"""
演示 CUDA Graph 消除 kernel launch overhead：
  - Mini decode step = Embedding + LayerNorm + QKV Linear + Attention + Out Linear + Sampling
  - Eager 模式：每步逐个 launch 所有 kernel（每次 launch ~5-10μs CPU 开销）
  - Graph 模式：先 capture 整个 step 为一张图，replay 时一次 launch 全部 kernel
  - 对比：torch.cuda.Event 计时 eager vs graph replay，验证正确性一致
注意：decode 阶段 M=1，kernel 极快（μs 级），launch overhead 占比可达 50%+，
      是 CUDA Graph 收益最大的场景（vLLM/TensorRT-LLM 在 decode 路径广泛使用）
"""

import torch
import torch.nn as nn

D_MODEL = 512
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS
VOCAB = 32000
ITERS = 50


class MiniDecodeStep(nn.Module):
    """简化单步 decode：embedding → LN → QKV → attn → out_proj → lm_head → argmax"""

    def __init__(self, d=D_MODEL, h=N_HEADS, v=VOCAB):
        super().__init__()
        self.h, self.dh = h, d // h
        self.embed = nn.Embedding(v, d)
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, v)

    def forward(self, tok, past_kv):
        B = tok.shape[0]
        x = self.embed(tok)
        x = self.ln(x)
        qkv = self.qkv(x).reshape(B, 3, self.h, self.dh)
        q, k, v = qkv.unbind(dim=1)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=1)
            v = torch.cat([pv, v], dim=1)
        scale = self.dh ** -0.5
        attn = torch.softmax((q * scale) @ k.transpose(-1, -2), dim=-1)
        ctx = (attn @ v).reshape(B, self.h * self.dh)
        x = self.out(ctx)
        return self.lm_head(x).argmax(dim=-1), (k, v)


def measure(fn, n=ITERS, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA 环境（单 GPU 即可）")
        return
    torch.manual_seed(42)
    dev = "cuda"
    step = MiniDecodeStep().to(dev).eval()
    tok0 = torch.tensor([1], device=dev)

    def eager():
        with torch.no_grad():
            step(tok0, None)

    # ---- CUDA Graph 捕获 ----
    static_tok = tok0.clone()
    static_kv = None
    # 1) warmup（必须在 capture 前跑一次以初始化 lazy cuBLAS/cache）
    for _ in range(3):
        with torch.no_grad():
            eager()
    torch.cuda.synchronize()
    # 2) capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin()
        with torch.no_grad():
            static_out, static_kv = step(static_tok, None)
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)

    def replay():
        static_tok.copy_(tok0)
        g.replay()

    eager_ms = measure(eager)
    graph_ms = measure(replay)

    print("=" * 64)
    print("  CUDA Graph Capture Demo（Mini Decode 单步）")
    print("=" * 64)
    print(f"  d_model={D_MODEL}, heads={N_HEADS}, vocab={VOCAB}")
    print(f"\n  eager (逐 kernel launch) : {eager_ms:.3f} ms / step")
    print(f"  graph (一次 replay)      : {graph_ms:.3f} ms / step")
    print(f"  launch overhead 降低     : {(1 - graph_ms / eager_ms) * 100:.1f}%")
    print(f"  加速比                   : {eager_ms / graph_ms:.2f}x")

    # ---- 正确性校验：graph 输出 == eager 输出 ----
    with torch.no_grad():
        y_eager, _ = step(tok0, None)
    replay()
    torch.cuda.synchronize()
    y_graph = static_out
    match = torch.equal(y_eager, y_graph)
    print(f"\n  正确性: eager==graph ? {'PASS' if match else 'FAIL'}")
    print(f"    eager token : {y_eager.item()}, graph token : {y_graph.item()}")
    print("\n  nsys 可视化:")
    print("    nsys profile -o cuda_graph --trace=cuda python cuda_graph_capture.py")
    print("    # timeline 中 graph replay 段 kernel 间隙几乎消失")


if __name__ == "__main__":
    main()
