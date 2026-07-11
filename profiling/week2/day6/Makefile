NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -g -lineinfo

TARGETS = integrated_gemm histogram

all: $(TARGETS)

# integrated_gemm 需要 cuBLAS
integrated_gemm: integrated_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublas

histogram: histogram.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ===== GEMM ncu profiling =====

profile-gemm: integrated_gemm
	ncu --kernel-name regex:gemmIntegrated \
	  -o integrated_profile \
	  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  ./integrated_gemm

profile-gemm-full: integrated_gemm
	ncu --set full --kernel-name regex:gemmIntegrated -o integrated_full ./integrated_gemm

# ===== Histogram ncu profiling =====

profile-hist-global: histogram
	ncu --kernel-name regex:histogram_global \
	  --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	  ./histogram

profile-hist-shared: histogram
	ncu --kernel-name regex:histogram_shared \
	  --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	  ./histogram

profile-hist-hbm: histogram
	ncu --kernel-name regex:"histogram_global|histogram_shared" \
	  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
	  ./histogram

# ===== nsys 时间线 =====

nsys-gemm: integrated_gemm
	nsys profile -o gemm_timeline ./integrated_gemm

nsys-hist: histogram
	nsys profile -o hist_timeline ./histogram

clean:
	rm -f $(TARGETS) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean \
        profile-gemm profile-gemm-full \
        profile-hist-global profile-hist-shared profile-hist-hbm \
        nsys-gemm nsys-hist
