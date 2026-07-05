NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -lineinfo

TARGETS = hello_gpu vector_add_blocksize

all: $(TARGETS)

hello_gpu: hello_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add_blocksize: vector_add_blocksize.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

profile: vector_add_blocksize
	ncu --metrics \
	  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
	  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
	  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
	  launch__registers_per_thread \
	  --kernel-name regex:vector_add \
	  ./vector_add_blocksize

clean:
	rm -f $(TARGETS)

.PHONY: all clean profile
