NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -lineinfo

TARGET = warp_reduce

all: $(TARGET)

$(TARGET): warp_reduce.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

profile: $(TARGET)
	ncu --metrics \
	  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
	  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
	  launch__registers_per_thread,\
	  smsp__average_warps_issue_stalled_long_scoreboard.pct \
	  --kernel-name regex:blockReduceSum \
	  ./warp_reduce

profile-full: $(TARGET)
	ncu --set full --kernel-name regex:blockReduceSum ./warp_reduce

nsys: $(TARGET)
	nsys profile -o warp_reduce_timeline ./warp_reduce

clean:
	rm -f $(TARGET) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean profile profile-full nsys
