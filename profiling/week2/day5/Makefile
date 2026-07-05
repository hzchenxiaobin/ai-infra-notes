NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -g -lineinfo

TARGET = flash_attention

all: $(TARGET)

$(TARGET): flash_attention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# FlashAttention ncu profiling
profile-flash: $(TARGET)
	ncu --kernel-name regex:flashAttentionFwd \
	    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	    ./flash_attention

# Standard Attention ncu profiling（对比）
profile-standard: $(TARGET)
	ncu --kernel-name regex:standardAttentionFwd \
	    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
	    ./flash_attention

# HBM 读写量对比（FlashAttention vs Standard）
profile-hbm: $(TARGET)
	ncu --kernel-name regex:"flashAttentionFwd|standardAttentionFwd" \
	    --metrics \
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
	    ./flash_attention

# 完整报告
profile-full: $(TARGET)
	ncu --set full --kernel-name regex:flashAttentionFwd -o flash_attn_report ./flash_attention

# nsys 时间线
nsys: $(TARGET)
	nsys profile -o flash_attn_timeline ./flash_attention

clean:
	rm -f $(TARGET) *.ncu-rep *.nsys-rep *.qdrep

.PHONY: all clean profile-flash profile-standard profile-hbm profile-full nsys
