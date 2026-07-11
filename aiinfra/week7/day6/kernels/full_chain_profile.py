# full_chain_profile.py —— Mini 系统全链路 Profiling
# 运行命令: python full_chain_profile.py
# 依赖: 仅标准库（模拟引擎，无需 GPU/PyTorch）
#
# 本文件是 Week7 Day6 的核心产出：对完整 Mini AI Infra 系统进行全链路 profiling。
#   1. 阶段计时：submit / schedule / forward / result 各占多少时间
#   2. 系统级指标：吞吐、延迟分布（P50/P99）、GPU 利用率模拟
#   3. 瓶颈定位：识别 top3 瓶颈（调度开销/内存分配/kernel launch）
#   4. vLLM 对比模拟：同条件下的差距分析
#
# 有 GPU 时可配合 nsys/ncu 采集真实时间线：
#   nsys profile -o mini_system_profile python full_chain_profile.py
#   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
#       python full_chain_profile.py

import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. 阶段计时器
# ============================================================

class PhaseTimer:
    """记录每个阶段的耗时，支持嵌套和聚合。"""

    def __init__(self):
        self.phases: Dict[str, List[float]] = defaultdict(list)
        self._starts: Dict[str, float] = {}

    def start(self, phase: str):
        self._starts[phase] = time.perf_counter()

    def end(self, phase: str):
        if phase in self._starts:
            elapsed = time.perf_counter() - self._starts[phase]
            self.phases[phase].append(elapsed)
            del self._starts[phase]

    def stats(self) -> Dict[str, dict]:
        result = {}
        for phase, times in self.phases.items():
            result[phase] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times) if times else 0,
                "p50": statistics.median(times) if times else 0,
                "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else (times[0] if times else 0),
                "pct": 0.0,  # 占比，后面计算
            }
        # 计算占比
        total_all = sum(r["total"] for r in result.values())
        if total_all > 0:
            for r in result.values():
                r["pct"] = r["total"] / total_all * 100
        return result


# ============================================================
# 2. 模拟 Mini 引擎（带阶段计时）
# ============================================================

@dataclass
class ProfileRequest:
    req_id: int
    prompt: str
    max_new_tokens: int
    submit_time: float
    schedule_time: Optional[float] = None
    forward_times: List[float] = field(default_factory=list)
    result_time: Optional[float] = None
    result: List[str] = field(default_factory=list)


class ProfiledMiniEngine:
    """带全链路计时的模拟推理引擎。"""

    def __init__(self, forward_time=0.02, schedule_time=0.001,
                 submit_time=0.0005, use_custom_kernel=True,
                 num_layers=4, d_model=512):
        self.forward_time = forward_time
        self.schedule_time = schedule_time
        self.submit_time = submit_time
        self.use_custom_kernel = use_custom_kernel
        self.num_layers = num_layers
        self.d_model = d_model
        self.timer = PhaseTimer()
        self._req_counter = 0

        # 模拟 kernel 列表（用于 kernel 级 profiling）
        self.kernel_breakdown = {
            "layernorm": 0.002,
            "qkv_gemm": 0.005,
            "flash_attention": 0.006,
            "out_proj_gemm": 0.003,
            "ffn_gemm_1": 0.004,
            "ffn_gemm_2": 0.004,
        }
        if use_custom_kernel:
            # 自定义 kernel 融合后更快
            self.kernel_breakdown["layernorm"] *= 0.6
            self.kernel_breakdown["flash_attention"] *= 0.7

    def _submit(self, prompt: str, max_new_tokens: int) -> ProfileRequest:
        self.timer.start("submit")
        time.sleep(self.submit_time)
        self._req_counter += 1
        req = ProfileRequest(
            req_id=self._req_counter,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            submit_time=time.perf_counter(),
        )
        self.timer.end("submit")
        return req

    def _schedule(self, req: ProfileRequest):
        self.timer.start("schedule")
        time.sleep(self.schedule_time)
        req.schedule_time = time.perf_counter()
        self.timer.end("schedule")

    def _forward(self, req: ProfileRequest):
        self.timer.start("forward")

        # 模拟逐层 forward
        for layer in range(self.num_layers):
            for kernel_name, kernel_time in self.kernel_breakdown.items():
                self.timer.start(f"  kernel:{kernel_name}")
                time.sleep(kernel_time)
                self.timer.end(f"  kernel:{kernel_name}")

            # 模拟 kernel launch overhead
            self.timer.start("  kernel:launch_overhead")
            time.sleep(0.0002)
            self.timer.end("  kernel:launch_overhead")

        token = f"r{req.req_id}_tok{len(req.result)}"
        req.result.append(token)
        req.forward_times.append(time.perf_counter())

        self.timer.end("forward")

    def _result(self, req: ProfileRequest):
        self.timer.start("result")
        time.sleep(0.0003)
        req.result_time = time.perf_counter()
        self.timer.end("result")

    def run_single(self, prompt: str, max_new_tokens: int = 10) -> ProfileRequest:
        """单请求全链路：submit → schedule → forward×N → result。"""
        req = self._submit(prompt, max_new_tokens)
        self._schedule(req)
        for _ in range(max_new_tokens):
            self._forward(req)
        self._result(req)
        return req

    def run_batch(self, prompts: List[str], max_new_tokens: int = 10) -> List[ProfileRequest]:
        """批量请求（模拟 continuous batching）。"""
        reqs = [self._submit(p, max_new_tokens) for p in prompts]
        for req in reqs:
            self._schedule(req)
        # 模拟每轮 decode 所有 running 请求
        for step in range(max_new_tokens):
            for req in reqs:
                self._forward(req)
        for req in reqs:
            self._result(req)
        return reqs


# ============================================================
# 3. Profiling 报告
# ============================================================

def generate_report(engine: ProfiledMiniEngine, reqs: List[ProfileRequest]):
    """生成全链路 profiling 报告。"""
    print("\n" + "=" * 70)
    print("全链路 Profiling 报告")
    print("=" * 70)

    # --- 阶段时间分解 ---
    print("\n📊 1. 阶段时间分解")
    print("-" * 50)
    stats = engine.timer.stats()
    print(f"{'阶段':<20} {'次数':>6} {'总时间(s)':>10} {'平均(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} {'占比':>6}")
    print("-" * 72)
    for phase in ["submit", "schedule", "forward", "result"]:
        if phase in stats:
            s = stats[phase]
            print(f"{phase:<20} {s['count']:>6} {s['total']:>10.3f} {s['avg']*1000:>10.3f} "
                  f"{s['p50']*1000:>10.3f} {s['p99']*1000:>10.3f} {s['pct']:>5.1f}%")

    # --- Kernel 级分解 ---
    print("\n📊 2. Kernel 级时间分解（每层 forward）")
    print("-" * 50)
    kernel_stats = {k: v for k, v in stats.items() if k.startswith("  kernel:")}
    print(f"{'Kernel':<25} {'次数':>6} {'总时间(s)':>10} {'平均(ms)':>10} {'占比':>6}")
    print("-" * 57)
    total_kernel = sum(s["total"] for s in kernel_stats.values())
    for kname in sorted(kernel_stats.keys()):
        s = kernel_stats[kname]
        short_name = kname.replace("  kernel:", "")
        print(f"{short_name:<25} {s['count']:>6} {s['total']:>10.3f} {s['avg']*1000:>10.3f} {s['pct']:>5.1f}%")

    # --- 系统级指标 ---
    print("\n📊 3. 系统级指标")
    print("-" * 50)
    total_tokens = sum(len(r.result) for r in reqs)
    total_time = sum(r.result_time - r.submit_time for r in reqs)
    latencies = [(r.result_time - r.submit_time) * 1000 for r in reqs]

    print(f"  Total requests:     {len(reqs)}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total time:         {total_time:.3f}s")
    print(f"  Throughput:         {total_tokens / total_time:.1f} tokens/s")
    print(f"  Avg latency:        {sum(latencies)/len(latencies):.1f} ms")
    print(f"  P50 latency:        {statistics.median(latencies):.1f} ms")
    print(f"  P99 latency:        {sorted(latencies)[int(len(latencies)*0.99)]:.1f} ms")

    # --- 瓶颈分析 ---
    print("\n📊 4. 瓶颈分析（Top 3）")
    print("-" * 50)
    all_phases = {**{k: v for k, v in stats.items() if not k.startswith("  ")}}
    sorted_phases = sorted(all_phases.items(), key=lambda x: x[1]["total"], reverse=True)
    for i, (phase, s) in enumerate(sorted_phases[:3]):
        print(f"  #{i+1}: {phase} — {s['total']:.3f}s ({s['pct']:.1f}%)")

    # --- 优化建议 ---
    print("\n📊 5. 优化建议")
    print("-" * 50)
    top_phase = sorted_phases[0][0] if sorted_phases else ""
    if top_phase == "forward":
        print("  • forward 是主瓶颈 → 优化 kernel（向量化、融合、Tensor Core）")
        launch_pct = kernel_stats.get("  kernel:launch_overhead", {}).get("pct", 0)
        if launch_pct > 5:
            print(f"  • launch overhead 占 {launch_pct:.1f}% → 考虑 CUDA Graph")
        attn_time = kernel_stats.get("  kernel:flash_attention", {}).get("total", 0)
        if attn_time > total_kernel * 0.3:
            print("  • flash_attention 占比高 → 用官方 FlashAttention-2")
    elif top_phase == "schedule":
        print("  • schedule 是主瓶颈 → C++ 重写调度器、简化调度逻辑")
    elif top_phase == "submit":
        print("  • submit 是主瓶颈 → 减少锁竞争、批量提交")
    elif top_phase == "result":
        print("  • result 是主瓶颈 → 减少 callback 开销、异步返回")

    if engine.use_custom_kernel:
        print("  • 自定义 kernel 已启用 → 对比 PyTorch 版本确认收益")
    else:
        print("  • 未启用自定义 kernel → Day4 的 kernel 融合可降低 forward 时间")

    return stats


# ============================================================
# 4. vLLM 对比模拟
# ============================================================

def compare_with_vllm():
    """模拟与 vLLM 的同条件对比。"""
    print("\n" + "=" * 70)
    print("vLLM 对比（模拟）")
    print("=" * 70)

    # 模拟 Mini 系统
    print("\n📊 Mini AI Infra（教学版）：")
    mini_engine = ProfiledMiniEngine(
        forward_time=0.02, use_custom_kernel=True, num_layers=4
    )
    mini_reqs = mini_engine.run_batch(["test prompt"] * 20, max_new_tokens=10)
    mini_stats = generate_report(mini_engine, mini_reqs)

    # 模拟 vLLM（更快：优化 kernel + CUDA Graph + PagedAttention）
    print("\n" + "=" * 70)
    print("📊 vLLM（模拟优化版）：")
    vllm_engine = ProfiledMiniEngine(
        forward_time=0.008,          # 优化 kernel，快 2.5x
        schedule_time=0.0002,        # C++ 调度器
        submit_time=0.0001,          # 批量提交
        use_custom_kernel=True,
        num_layers=4,
    )
    # vLLM 的 kernel 更快
    vllm_engine.kernel_breakdown = {
        "layernorm": 0.001,
        "qkv_gemm": 0.002,
        "flash_attention": 0.002,    # FlashAttention-2
        "out_proj_gemm": 0.001,
        "ffn_gemm_1": 0.0015,
        "ffn_gemm_2": 0.0015,
    }
    vllm_reqs = vllm_engine.run_batch(["test prompt"] * 20, max_new_tokens=10)
    vllm_stats = generate_report(vllm_engine, vllm_reqs)

    # 对比
    print("\n" + "=" * 70)
    print("📊 差距分析")
    print("-" * 50)
    mini_total = sum(s["total"] for s in mini_stats.values() if not s.get("count", 0) == 0)
    vllm_total = sum(s["total"] for s in vllm_stats.values() if not s.get("count", 0) == 0)

    mini_fwd = mini_stats.get("forward", {}).get("total", 0)
    vllm_fwd = vllm_stats.get("forward", {}).get("total", 0)

    print(f"  {'指标':<25} {'Mini':>12} {'vLLM':>12} {'差距':>10}")
    print(f"  {'-'*59}")
    print(f"  {'forward 总时间':<25} {mini_fwd:>10.3f}s {vllm_fwd:>10.3f}s {mini_fwd/vllm_fwd:>8.1f}x")
    print(f"  {'schedule 时间':<25} {mini_stats.get('schedule',{}).get('total',0):>10.3f}s {vllm_stats.get('schedule',{}).get('total',0):>10.3f}s")

    mini_lat = [(r.result_time - r.submit_time) for r in mini_reqs]
    vllm_lat = [(r.result_time - r.submit_time) for r in vllm_reqs]
    print(f"  {'avg latency':<25} {sum(mini_lat)/len(mini_lat)*1000:>10.1f}ms {sum(vllm_lat)/len(vllm_lat)*1000:>10.1f}ms {sum(mini_lat)/sum(vllm_lat):>8.1f}x")

    mini_tokens = sum(len(r.result) for r in mini_reqs)
    vllm_tokens = sum(len(r.result) for r in vllm_reqs)
    mini_tp = mini_tokens / sum(mini_lat)
    vllm_tp = vllm_tokens / sum(vllm_lat)
    print(f"  {'throughput (tok/s)':<25} {mini_tp:>10.1f} {vllm_tp:>10.1f} {vllm_tp/mini_tp:>8.1f}x")

    print(f"\n  差距来源：")
    print(f"    1. Kernel 优化：vLLM 用 FlashAttention-2 + 高度优化的 GEMM")
    print(f"    2. CUDA Graph：减少 kernel launch overhead")
    print(f"    3. C++ Scheduler：调度逻辑比 Python 快 10x")
    print(f"    4. PagedAttention：KV Cache 管理更高效")
    print(f"    5. torch.compile：自动算子融合")


# ============================================================
# 5. nsys/ncu 采集指南
# ============================================================

def print_profiling_guide():
    """打印 nsys/ncu 采集命令指南。"""
    print("\n" + "=" * 70)
    print("nsys / ncu 采集指南（有 GPU 环境时使用）")
    print("=" * 70)

    print("""
# 1. Nsight Systems — 系统级时间线
nsys profile -o mini_system_profile \\
    --trace=cuda,nvtx \\
    python full_chain_profile.py

# 查看时间线
nsys-ui mini_system_profile.nsys-rep

# 查看 kernel 统计
nsys stats -t cuda_gpu_kern_sum mini_system_profile.nsys-rep

# 2. Nsight Compute — Kernel 级指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum \\
    --kernel-name regex:"gemm|flash_attention|softmax|layernorm" \\
    python full_chain_profile.py

# 3. PyTorch Profiler — 算子级分解
# (在有 PyTorch 的环境中)
# from torch.profiler import profile, ProfilerActivity
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     engine.run(...)
# print(prof.key_averages().table(sort_by="cuda_time_total"))

# 4. 昇腾 msprof（CANN 环境）
# msprof --output=./prof_data python full_chain_profile.py
""")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Mini AI Infra 全链路 Profiling")
    print("=" * 70)

    # 1. 单请求 profiling
    print("\n🔬 实验 1：单请求全链路 profiling")
    engine1 = ProfiledMiniEngine(forward_time=0.02, use_custom_kernel=True, num_layers=4)
    req1 = engine1.run_single("hello world this is a test", max_new_tokens=5)
    generate_report(engine1, [req1])

    # 2. 批量 profiling
    print("\n" + "=" * 70)
    print("🔬 实验 2：批量请求 profiling（20 请求 × 10 tokens）")
    engine2 = ProfiledMiniEngine(forward_time=0.02, use_custom_kernel=True, num_layers=4)
    prompts = [f"prompt number {i} for batching test" for i in range(20)]
    reqs2 = engine2.run_batch(prompts, max_new_tokens=10)
    generate_report(engine2, reqs2)

    # 3. vLLM 对比
    compare_with_vllm()

    # 4. 采集指南
    print_profiling_guide()

    print("\n" + "=" * 70)
    print("Profiling 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
