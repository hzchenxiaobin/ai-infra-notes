# Week 1 Day 6 — 综合 Nsight Profiling

本目录复用 Week 1 前 5 天的 kernel，直接到对应目录编译后执行完整报告采集。

```bash
# Day 1
make -C ../day1
ncu --set full -o day6_hello_gpu ../day1/hello_gpu

# Day 2
make -C ../day2
ncu --set full -o day6_occupancy_test ../day2/occupancy_test

# Day 4
make -C ../day4
ncu --set full -o day6_transpose ../day4/transpose

# Day 5
make -C ../day5
ncu --set full -o day6_bank_conflict ../day5/bank_conflict

# GUI 查看
ncu-ui day6_transpose.ncu-rep

# nsys 系统级 trace
nsys profile -o day6_full_timeline --trace cuda,nvtx,osrt ../day4/transpose
```
