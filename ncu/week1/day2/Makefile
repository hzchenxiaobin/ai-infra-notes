NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

TARGETS = occupancy_test

all: $(TARGETS)

occupancy_test: occupancy_test.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
