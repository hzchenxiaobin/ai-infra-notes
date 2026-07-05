NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -g -lineinfo

TARGETS = register_gemm softmax_profile

all: $(TARGETS)

# Register Blocking GEMM（需要 cuBLAS）
register_gemm: register_blocking_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublas

# Softmax profiling
softmax_profile: softmax_profile.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ===== GEMM ncu profiling（任务 1-4）=====

profile-gemm: register_gemm
	ncu \
	  --kernel-name regex:gemmRegisterBlocking \
	  -o gemm_profile_report \
	  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__warps_eligible.sum.per_cycle,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  ./register_gemm 2>&1 | tee ncu_output.txt

profile-gemm-csv: gemm_profile_report.ncu-rep
	ncu --csv --page details -i gemm_profile_report.ncu-rep > gemm_profile.csv

profile-gemm-full: register_gemm
	ncu --set full --kernel-name regex:gemmRegisterBlocking -o gemm_full ./register_gemm

# ===== Softmax ncu profiling（任务 5）=====

profile-softmax: softmax_profile
	ncu --kernel-name regex:softmax_row_kernel \
	  --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  ./softmax_profile

profile-softmax-full: softmax_profile
	ncu --set full --kernel-name regex:softmax_row_kernel -o softmax_report ./softmax_profile

# ===== nsys 时间线（实验 1）=====

nsys-gemm: register_gemm
	nsys profile -o timeline_gemm ./register_gemm

nsys-softmax: softmax_profile
	nsys profile -o timeline_softmax ./softmax_profile

clean:
	rm -f $(TARGETS) *.ncu-rep *.nsys-rep *.qdrep *.csv ncu_output.txt

.PHONY: all clean \
        profile-gemm profile-gemm-csv profile-gemm-full \
        profile-softmax profile-softmax-full \
        nsys-gemm nsys-softmax
