"""benchmark_triton.py —— Triton vs CUDA vs PyTorch 三方 Benchmark（Week 4 Day 5）

对比 Day 4 的三个 Triton kernel（Softmax / GEMM / FlashAttention）与：
  - CUDA 手写版：naive CUDA kernel 内嵌在本脚本中，经 torch.utils.cpp_extension.load_inline
    现场编译（需要 nvcc；无 nvcc 时 CUDA 列显示 N/A，不影响 Triton vs PyTorch 对比）
  - PyTorch 原生：torch.matmul（cuBLAS）/ torch.softmax / naive attention；
    FlashAttention 另对比官方 flash_attn 包（若已安装，否则该列 N/A）

运行: python3 kernels/benchmark_triton.py
依赖: pip install torch triton；flash-attn 可选；CUDA 列需要 nvcc
 profiling: ncu --kernel-name regex:"gemm_kernel|softmax" python3 kernels/benchmark_triton.py（Day 6 任务 1/3）

⚠️ 本机无 GPU，本脚本未实测——README 中"预期输出"表为预估口径，实测数据待 GPU 环境回填。
"""

import os
import shutil
import sys

_here = os.path.dirname(os.path.abspath(__file__))
# 跨 day 导入 Day 4 的 Triton kernel（sys.path 惯例参照 week10/day2/kernels/mini_engine_v2.py）
sys.path.insert(0, os.path.join(_here, "..", "..", "day4", "kernels"))


def _fail(msg):
    print(f"[benchmark_triton] 错误: {msg}", file=sys.stderr)
    sys.exit(1)


try:
    import torch
except ImportError:
    _fail("未安装 PyTorch。请先 pip install torch（CUDA 版）。")

if not torch.cuda.is_available():
    _fail("未检测到可用的 CUDA 设备。本脚本需要 NVIDIA GPU（课程目标环境为 RTX 5090 / sm_120）。")

try:
    import triton
except ImportError:
    _fail("未安装 Triton。请先 pip install triton。")

from triton_gemm import triton_gemm  # noqa: E402
from triton_softmax import triton_softmax  # noqa: E402
from triton_flash_attention import triton_flash_attention, naive_attention  # noqa: E402

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


# ============================================================
# CUDA 手写版（naive，代表 Day 1-2 的手写 CUDA 水平，无 Tensor Core）
# ============================================================
CUDA_DECL = """
#include <torch/extension.h>
torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor softmax_cuda(torch::Tensor x);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define TILE 16

// 手写 smem tiling GEMM（FP16 输入/输出，FP32 累加；对应 Day 1 手写 CUDA 口径）
__global__ void gemm_cuda_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                 __half* __restrict__ C, int M, int N, int K) {
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options());
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_cuda_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()), M, N, K);
    return c;
}

// 手写 Softmax：一行一个 block，max/sum 两级 reduce（对应 Day 2 手写 CUDA 口径）
__global__ void softmax_cuda_kernel(const float* __restrict__ x, float* __restrict__ y, int n_cols) {
    const float* xr = x + (size_t)blockIdx.x * n_cols;
    float* yr = y + (size_t)blockIdx.x * n_cols;
    int tid = threadIdx.x;
    int lane = tid % 32, wid = tid / 32;
    __shared__ float smem[32];
    __shared__ float row_max, row_sum;

    float m = -INFINITY;
    for (int i = tid; i < n_cols; i += blockDim.x) m = fmaxf(m, xr[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) smem[wid] = m;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    m = (lane < nw) ? smem[lane] : -INFINITY;
    if (wid == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
        if (lane == 0) row_max = m;
    }
    __syncthreads();

    float s = 0.0f;
    for (int i = tid; i < n_cols; i += blockDim.x) s += expf(xr[i] - row_max);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    s = (lane < nw) ? smem[lane] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) row_sum = s;
    }
    __syncthreads();

    float inv = 1.0f / row_sum;
    for (int i = tid; i < n_cols; i += blockDim.x) yr[i] = expf(xr[i] - row_max) * inv;
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    int n_rows = x.size(0), n_cols = x.size(1);
    auto y = torch::empty_like(x);
    int threads = 256;
    softmax_cuda_kernel<<<n_rows, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n_cols);
    return y;
}
"""


def build_cuda_ops():
    """编译内嵌的 naive CUDA kernel；无 nvcc 或编译失败时返回 None（CUDA 列 N/A）。"""
    if shutil.which("nvcc") is None:
        print("[提示] 未找到 nvcc，跳过 CUDA 手写版编译，三方表的 CUDA(ms) 列显示 N/A。\n")
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        return load_inline(
            name="w4d5_cuda_ops",
            cpp_sources=CUDA_DECL,
            cuda_sources=CUDA_SRC,
            functions=["gemm_cuda", "softmax_cuda"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        print(f"[提示] CUDA 手写版编译失败（{e}），CUDA(ms) 列显示 N/A。\n")
        return None


def measure(fn, *args):
    """计时（ms）：优先 triton.testing.do_bench，失败时退回 torch.cuda.Event 等效计时。"""
    try:
        from triton.testing import do_bench

        return do_bench(lambda: fn(*args), warmup=25, rep=100)
    except Exception:
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 50


def _fmt(ms):
    return f"{ms:.3f}" if ms is not None else "N/A"


def _check(max_diff, tol):
    return "PASS" if max_diff < tol else "FAIL"


# ============================================================
# 1. GEMM：Triton vs CUDA 手写 vs cuBLAS（FP16）
# ============================================================
@torch.no_grad()
def bench_gemm(cuda_ops):
    sizes = [512, 1024, 2048, 4096]
    dev = torch.cuda.get_device_name()
    print(f"=== Triton vs CUDA vs PyTorch GEMM Benchmark ({dev}, FP16->FP32) ===")
    header = (
        f"{'M=N=K':<10}| {'Triton(ms)':<12}{'CUDA(ms)':<12}{'cuBLAS(ms)':<12}"
        f"| {'Triton%':<9}{'CUDA%':<9}{'Triton/CUDA':<13}| {'max_diff':<10}{'check':<6}"
    )
    print(header)
    print("-" * len(header))
    for size in sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        y_ref = torch.matmul(a, b)
        y_triton = triton_gemm(a, b)
        max_diff = (y_ref.float() - y_triton.float()).abs().max().item()

        t_triton = measure(triton_gemm, a, b)
        t_cublas = measure(torch.matmul, a, b)
        if cuda_ops is not None:
            y_cuda = cuda_ops.gemm_cuda(a, b)
            cuda_diff = (y_ref.float() - y_cuda.float()).abs().max().item()
            t_cuda = measure(cuda_ops.gemm_cuda, a, b)
            cuda_pct = f"{t_cublas / t_cuda * 100:.0f}%"
            ratio = f"{t_cuda / t_triton:.1f}x"
            assert cuda_diff < 1e-2, f"CUDA GEMM 正确性失败: max_diff={cuda_diff:.2e}"
        else:
            t_cuda = None
            cuda_pct = "N/A"
            ratio = "N/A"

        triton_pct = f"{t_cublas / t_triton * 100:.0f}%"
        check = _check(max_diff, 1e-2)
        print(
            f"{size:<10}| {_fmt(t_triton):<12}{_fmt(t_cuda):<12}{_fmt(t_cublas):<12}"
            f"| {triton_pct:<9}{cuda_pct:<9}{ratio:<13}| {max_diff:<10.2e}{check:<6}"
        )
    print()


# ============================================================
# 2. Softmax：Triton vs CUDA 手写 vs PyTorch（FP32）
# ============================================================
@torch.no_grad()
def bench_softmax(cuda_ops):
    shapes = [(1024, 1024), (4096, 1024), (4096, 4096)]
    print("=== Softmax Benchmark (FP32) ===")
    header = (
        f"{'M×D':<12}| {'Triton(ms)':<12}{'CUDA(ms)':<12}{'PyTorch(ms)':<12}"
        f"| {'Triton/PyTorch':<16}| {'max_diff':<10}{'check':<6}"
    )
    print(header)
    print("-" * len(header))
    for M, N in shapes:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        y_ref = torch.softmax(x, dim=-1)
        y_triton = triton_softmax(x)
        max_diff = (y_ref - y_triton).abs().max().item()

        t_triton = measure(triton_softmax, x)
        t_torch = measure(lambda t: torch.softmax(t, dim=-1), x)
        t_cuda = measure(cuda_ops.softmax_cuda, x) if cuda_ops is not None else None
        if cuda_ops is not None:
            cuda_diff = (y_ref - cuda_ops.softmax_cuda(x)).abs().max().item()
            assert cuda_diff < 1e-5, f"CUDA Softmax 正确性失败: max_diff={cuda_diff:.2e}"

        ratio = f"{t_torch / t_triton:.2f}x"
        check = _check(max_diff, 1e-5)
        print(
            f"{f'{M}×{N}':<12}| {_fmt(t_triton):<12}{_fmt(t_cuda):<12}{_fmt(t_torch):<12}"
            f"| {ratio:<16}| {max_diff:<10.2e}{check:<6}"
        )
    print()


# ============================================================
# 3. FlashAttention（causal, FP16）：Triton vs PyTorch naive vs 官方 flash-attn
#    注：CUDA 手写 FA 的对比见 Week 5 Day 3，本脚本不重复实现
# ============================================================
@torch.no_grad()
def bench_flash_attention():
    cases = [(2, 4, 512, 64), (1, 8, 1024, 64), (1, 8, 2048, 64)]
    print("=== FlashAttention Benchmark (causal, FP16) ===")
    if flash_attn_func is None:
        print("[提示] 未安装 flash-attn，官方(ms) 列显示 N/A。")
    header = (
        f"{'(B,H,N,D)':<18}| {'Triton(ms)':<12}{'naive(ms)':<12}{'官方FA(ms)':<12}"
        f"| {'Triton/naive':<14}{'Triton/官方':<12}| {'max_diff':<10}{'check':<6}"
    )
    print(header)
    print("-" * len(header))
    for B, H, N, D in cases:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        y_ref = naive_attention(q, k, v)
        y_triton = triton_flash_attention(q, k, v)
        max_diff = (y_ref.float() - y_triton.float()).abs().max().item()

        t_triton = measure(triton_flash_attention, q, k, v)
        t_naive = measure(naive_attention, q, k, v)
        if flash_attn_func is not None:
            # 官方接口为 (B, N, H, D) 布局
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y_official = flash_attn_func(qh, kh, vh, causal=True).transpose(1, 2)
            official_diff = (y_ref.float() - y_official.float()).abs().max().item()
            assert official_diff < 1e-2, f"官方 FA 正确性失败: max_diff={official_diff:.2e}"
            t_official = measure(flash_attn_func, qh, kh, vh, causal=True)
            ratio_official = f"{t_official / t_triton * 100:.0f}%"
        else:
            t_official = None
            ratio_official = "N/A"

        ratio_naive = f"{t_naive / t_triton:.2f}x"
        check = _check(max_diff, 1e-2)
        print(
            f"{str((B, H, N, D)):<18}| {_fmt(t_triton):<12}{_fmt(t_naive):<12}{_fmt(t_official):<12}"
            f"| {ratio_naive:<14}{ratio_official:<12}| {max_diff:<10.2e}{check:<6}"
        )
    print()


def main():
    torch.manual_seed(42)
    dev = torch.cuda.get_device_name()
    print(f"[benchmark_triton] device = {dev}, torch = {torch.__version__}, triton = {triton.__version__}\n")

    cuda_ops = build_cuda_ops()

    bench_gemm(cuda_ops)
    bench_softmax(cuda_ops)
    bench_flash_attention()

    print("[说明] Triton% = cuBLAS(ms)/Triton(ms)，即 Triton 达 cuBLAS 性能的百分比；")
    print("       Triton/CUDA = CUDA(ms)/Triton(ms)，即 Triton 相对手写 CUDA 的加速比。")
    print("       CUDA 手写 FA 的三方对比见 Week 5 Day 3；本表数字以实测为准（README 表为预估口径）。")


if __name__ == "__main__":
    main()
