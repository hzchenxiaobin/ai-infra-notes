# Week 1 Day 1 — hello_gpu Profiling

```bash
make
./hello_gpu

# nsys timeline
nsys profile -o hello_gpu_timeline ./hello_gpu

# ncu 基础指标
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./hello_gpu
```
