NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -lineinfo

TARGETS = softmax_layernorm sl_dscan

all: $(TARGETS)

softmax_layernorm: softmax_layernorm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sl_dscan: softmax_layernorm_dscan.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ===== 任务 3：ncu 验证 memory-bound =====

profile: softmax_layernorm
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
	  --kernel-name regex:"softmax_kernel|layernorm_kernel" \
	  ./softmax_layernorm

profile-full: softmax_layernorm
	ncu --set full --kernel-name regex:"softmax_kernel|layernorm_kernel" \
	  -o sl_report ./softmax_layernorm

profile-stall: softmax_layernorm
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  --kernel-name regex:"softmax_kernel|layernorm_kernel" \
	  ./softmax_layernorm

# ===== 实验 1：D-scan 性能尺度律 =====

dscan: sl_dscan
	./sl_dscan

profile-dscan: sl_dscan
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  --kernel-name regex:"softmax_kernel|layernorm_kernel" \
	  ./sl_dscan

# ===== nsys 时间线 =====

nsys: softmax_layernorm
	nsys profile -o sl_timeline ./softmax_layernorm

clean:
	rm -f $(TARGETS) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean profile profile-full profile-stall dscan profile-dscan nsys
