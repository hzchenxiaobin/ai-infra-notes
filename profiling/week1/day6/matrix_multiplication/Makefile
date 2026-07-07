NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

TARGETS = matmul

all: $(TARGETS)

matmul: matrix_multiplication.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
