NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

TARGETS = bank_conflict

all: $(TARGETS)

bank_conflict: bank_conflict.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
