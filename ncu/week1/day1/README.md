# Week 1 Day 1 — hello_gpu Profiling

> 若在 WSL2 中运行 `ncu` 遇到 `ERR_NVGPUCTRPERM`，请参考 [`ncu/README.md`](../README.md) 中的“环境准备与常见故障”章节，在 Windows 宿主开放 GPU Performance Counters 权限。

```bash
make
./hello_gpu

# nsys timeline
nsys profile -o hello_gpu_timeline ./hello_gpu

# ncu 基础指标
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./hello_gpu
```
