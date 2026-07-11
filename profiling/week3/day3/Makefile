NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -lineinfo

TARGETS = softmax_layernorm_opt warp_vs_block

all: $(TARGETS)

softmax_layernorm_opt: softmax_layernorm_opt.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

warp_vs_block: warp_vs_block_dscan.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ===== 任务 3：ncu 对比优化前后 DRAM/SM Throughput =====

# 优化版 kernel（warp softmax + float4 layernorm）
profile-opt: softmax_layernorm_opt
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
	  --kernel-name regex:"softmax_warp_kernel|layernorm_float4_kernel" \
	  ./softmax_layernorm_opt

# 基准版 kernel（block softmax + scalar layernorm）
profile-base: softmax_layernorm_opt
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
	  --kernel-name regex:"softmax_block_kernel|layernorm_scalar_kernel" \
	  ./softmax_layernorm_opt

# 一次 profile 全部 4 个 kernel
profile-all: softmax_layernorm_opt
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
	  --kernel-name regex:"softmax" \
	  ./softmax_layernorm_opt

profile-full: softmax_layernorm_opt
	ncu --set full -o sl_opt_report ./softmax_layernorm_opt

# ===== 实验 2：warp vs block 不同 D 值对比 =====

dscan: warp_vs_block
	./warp_vs_block

profile-dscan: warp_vs_block
	ncu --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	  --kernel-name regex:"softmax_warp_kernel|softmax_block_kernel" \
	  ./warp_vs_block

# ===== 实验 3：ncu stall 原因分析 =====

profile-stall: softmax_layernorm_opt
	ncu --metrics \
smsp__average_warps_issue_stalled_long_scoreboard.pct,\
smsp__average_warps_issue_stalled_membar.pct \
	  --kernel-name regex:"layernorm" \
	  ./softmax_layernorm_opt

# ===== nsys 时间线 =====

nsys: softmax_layernorm_opt
	nsys profile -o sl_opt_timeline ./softmax_layernorm_opt

clean:
	rm -f $(TARGETS) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean \
        profile-opt profile-base profile-all profile-full \
        dscan profile-dscan profile-stall nsys
