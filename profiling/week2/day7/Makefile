NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -g -lineinfo

TARGETS = block_reduce gemm_timed

all: $(TARGETS)

block_reduce: block_reduce_timed.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

gemm_timed: gemm_timed.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublas

# ===== Block Reduce ncu profiling（任务 1）=====

profile-reduce: block_reduce
	ncu --kernel-name regex:blockReduceSum \
	  --metrics \
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  ./block_reduce

profile-reduce-full: block_reduce
	ncu --set full --kernel-name regex:blockReduceSum -o reduce_report ./block_reduce

nsys-reduce: block_reduce
	nsys profile -o reduce_timeline ./block_reduce

# ===== GEMM ncu profiling（任务 2）=====

profile-gemm: gemm_timed
	ncu --kernel-name regex:gemmRegisterBlocking \
	  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  ./gemm_timed

profile-gemm-full: gemm_timed
	ncu --set full --kernel-name regex:gemmRegisterBlocking -o gemm_report ./gemm_timed

nsys-gemm: gemm_timed
	nsys profile -o gemm_timeline ./gemm_timed

clean:
	rm -f $(TARGETS) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean \
        profile-reduce profile-reduce-full nsys-reduce \
        profile-gemm profile-gemm-full nsys-gemm
