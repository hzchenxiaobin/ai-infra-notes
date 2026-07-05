NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -lineinfo

# 所有可编译的 LeetGPU profiling 程序
TARGETS = vector_add relu matrix_transpose reduction histogram \
          matrix_addition matmul softmax argmax

# 源文件映射
vector_add: vector-add.cu
relu: relu.cu
matrix_transpose: matrix-transpose.cu
reduction: reduction.cu
histogram: histogram.cu
matrix_addition: matrix-addition.cu
matmul: matrix-multiplication.cu
softmax: softmax.cu
argmax: argmax.cu

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ===== 各题 ncu profiling 命令 =====

profile-vector-add: vector_add
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./vector_add

profile-relu: relu
	ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./relu

profile-matrix-transpose: matrix_transpose
	ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./matrix_transpose

profile-reduction: reduction
	ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./reduction

profile-histogram: histogram
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./histogram

profile-matrix-addition: matrix_addition
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread ./matrix_addition

profile-matmul: matmul
	ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./matmul

profile-softmax: softmax
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
sm__warp_issue_stalled_long_scoreboard_per_warp_active.pct ./softmax

profile-argmax: argmax
	ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread ./argmax

# 一次性编译所有
profile-all: all
	@echo "=== 运行所有 profiling ==="
	@for t in vector_add relu matrix_transpose reduction histogram matrix_addition matmul softmax argmax; do \
		echo "--- $$t ---"; ./$$t; echo; \
	done

clean:
	rm -f $(TARGETS) *.ncu-rep

.PHONY: all clean profile-all \
        profile-vector-add profile-relu profile-matrix-transpose \
        profile-reduction profile-histogram profile-matrix-addition \
        profile-matmul profile-softmax profile-argmax
