# shape_bucketing.py —— 动态 batch 的 Shape Bucketing CUDA Graph
# 运行命令: python shape_bucketing.py
# 依赖: torch + CUDA（单 GPU 即可）
"""
处理动态 batch size 下的 CUDA Graph：
  - CUDA Graph 要求静态 shape，但推理时 batch size 随请求变化
  - Shape Bucketing：为 {1,2,4,8,16} 每个 bucket 预捕获一张 graph
  - 运行时选最近 bucket（向上取整），把输入 copy 到静态 buffer，replay，截取有效部分
  - 变长序列：pad 到统一 max_seq_len，attention 用 mask 屏蔽 padding
注意：bucket 越多显存占用越大（每 bucket 一套静态 buffer），需在覆盖度与显存间权衡
"""

import torch
import torch.nn as nn

BUCKETS = [1, 2, 4, 8, 16]
D_MODEL = 512
MAX_SEQ = 128


class MiniBlock(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)

    def forward(self, x, mask):
        h = self.ln(x)
        h = self.fc(h) * mask.unsqueeze(-1)
        return x + h


class BucketedGraphRunner:
    """为每个 bucket 预捕获 graph，运行时按 batch 选最近 bucket 回放"""

    def __init__(self, model, buckets, max_seq, d):
        self.model = model
        self.buckets = buckets
        self.graphs, self.sin, self.smask, self.sout = {}, {}, {}, {}
        for b in buckets:
            self._capture(b, max_seq, d)

    def _capture(self, b, max_seq, d):
        sin = torch.zeros(b, max_seq, d, device="cuda")
        smask = torch.zeros(b, max_seq, device="cuda")
        for _ in range(3):
            with torch.no_grad():
                self.model(sin, smask)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g.capture_begin()
            with torch.no_grad():
                sout = self.model(sin, smask)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        self.graphs[b] = g
        self.sin[b], self.smask[b], self.sout[b] = sin, smask, sout

    def _pick(self, b):
        for bk in self.buckets:
            if bk >= b:
                return bk
        return self.buckets[-1]

    def run(self, x, mask):
        b, n = x.shape[0], x.shape[1]
        bk = self._pick(b)
        sin, smask = self.sin[bk], self.smask[bk]
        sin[:b, :n] = x
        smask[:b, :n] = mask
        self.graphs[bk].replay()
        torch.cuda.synchronize()
        return self.sout[bk][:b, :n]


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA 环境（单 GPU 即可）")
        return
    torch.manual_seed(0)
    model = MiniBlock().to("cuda").eval()
    runner = BucketedGraphRunner(model, BUCKETS, MAX_SEQ, D_MODEL)
    print("=" * 64)
    print("  Shape Bucketing CUDA Graph Demo")
    print("=" * 64)
    print(f"  buckets={BUCKETS}, max_seq={MAX_SEQ}, d={D_MODEL}")

    import time
    for b in [1, 3, 7, 12, 16]:
        x = torch.randn(b, MAX_SEQ, D_MODEL, device="cuda")
        m = torch.ones(b, MAX_SEQ, device="cuda")
        with torch.no_grad():
            y_ref = model(x, m)
        y = runner.run(x, m)
        diff = (y - y_ref).abs().max().item()
        bk = runner._pick(b)
        for _ in range(3):
            runner.run(x, m)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            runner.run(x, m)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"  batch={b:2d} → bucket={bk:2d} | graph {ms:.3f} ms/step | max_diff={diff:.2e}")

    print("\n  动态 batch 无需重捕获：运行时选最近 bucket，copy 输入后 replay")


if __name__ == "__main__":
    main()
