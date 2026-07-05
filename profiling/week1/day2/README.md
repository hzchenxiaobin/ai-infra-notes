# Week 1 Day 2 — Occupancy Profiling

```bash
make
./occupancy_test

# ncu occupancy / register 分析
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  launch__shared_mem_per_block_dynamic,\
  launch__shared_mem_per_block_static \
  ./occupancy_test
```

进阶：修改源码中的 `#pragma unroll`、局部变量或 `__launch_bounds__` 后重新 `make` 并对比。
