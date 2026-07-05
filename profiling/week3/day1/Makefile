NVCC = nvcc
PYTHON = python3

# Python 脚本，不需要编译
all:
	@echo "Run: make profile     (torch.profiler)"
	@echo "Run: make nsys        (nsys timeline)"
	@echo "Run: make ncu         (ncu kernel analysis)"

# ===== torch.profiler 分析 =====
profile:
	$(PYTHON) trace_transformer.py

# ===== nsys 系统级时间线（练习 2）=====
nsys:
	nsys profile -o transformer_trace $(PYTHON) trace_transformer.py

# nsys 查看 kernel 统计
nsys-stats:
	nsys stats -t cuda_gpu_kern_sum transformer_trace.nsys-rep

# ===== ncu kernel 级分析 =====
ncu:
	ncu --kernel-name regex:"gemm|softmax|layer_norm|gelu" \
	    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	    $(PYTHON) trace_transformer.py

# ncu 只分析 GEMM kernel
ncu-gemm:
	ncu --kernel-name regex:"gemm" \
	    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
	    $(PYTHON) trace_transformer.py

# ncu 只分析 softmax kernel
ncu-softmax:
	ncu --kernel-name regex:"softmax" \
	    --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
	    $(PYTHON) trace_transformer.py

clean:
	rm -f trace_*.json transformer_trace.nsys-rep transformer_trace.qdrep *.qdstrm

.PHONY: all profile nsys nsys-stats ncu ncu-gemm ncu-softmax clean
