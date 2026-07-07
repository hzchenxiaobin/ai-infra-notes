NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

TARGETS = transpose bandwidth transpose_tiles

all: $(TARGETS)

transpose: transpose.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

bandwidth: bandwidth.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

transpose_tiles: transpose_tiles.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
