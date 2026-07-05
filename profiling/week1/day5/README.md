# Week 1 Day 5 — Bank Conflict Profiling

```bash
make
./bank_conflict

# 整体对比
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
  sm__cycles_elapsed.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./bank_conflict

# 单独 kernel 精确对比
ncu --kernel-name regex:conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./bank_conflict

ncu --kernel-name regex:no_conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./bank_conflict
```
