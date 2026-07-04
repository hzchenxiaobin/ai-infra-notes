NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

TARGETS = hello_gpu

all: $(TARGETS)

hello_gpu: hello_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
