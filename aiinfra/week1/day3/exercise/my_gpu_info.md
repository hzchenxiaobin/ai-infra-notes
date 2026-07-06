# GPU 信息：NVIDIA GeForce RTX 5090

## 1. deviceQuery 原始输出

```text
./Samples/1_Utilities/deviceQuery/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 5090"
  CUDA Driver Version / Runtime Version          13.0 / 12.8
  CUDA Capability Major/Minor version number:    12.0
  Total amount of global memory:                 32109 MBytes (33668857856 bytes)
  (170) Multiprocessors, (128) CUDA Cores/MP:    21760 CUDA Cores
  GPU Max Clock rate:                            2407 MHz (2.41 GHz)
  Memory Clock rate:                             14001 Mhz
  Memory Bus Width:                              512-bit
  L2 Cache Size:                                 100663296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.0, CUDA Runtime Version = 12.8, NumDevs = 1
Result = PASS
```

## 2. 关键参数汇总

| 参数 | 数值 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 5090 |
| CUDA Driver Version | 13.0 |
| CUDA Runtime Version | 12.8 |
| CUDA Capability | 12.0 (sm_120) |
| 显存容量 | 32109 MB (~32 GB) |
| SM 数量 | 170 |
| CUDA Cores / SM | 128 |
| CUDA Cores 总数 | 21760 |
| GPU Max Clock | 2407 MHz (2.407 GHz) |
| Memory Clock | 14001 MHz |
| Memory Bus Width | 512-bit |
| L2 Cache | 100663296 bytes (96 MB) |

## 3. 峰值算力与显存带宽计算

### 3.1 峰值 FP32 算力

公式：

```text
Peak FP32 (FLOPS) = CUDA Cores × Clock × 2
```

其中 `×2` 是因为 FMA（乘加）算 2 个 FLOP：

```text
21760 × 2.407 × 2 = 104752.64 GFLOPS
```

**≈ 104.75 TFLOPS**

> 与 RTX 5090 官方标称 ~104.8 TFLOPS 一致。

### 3.2 显存带宽

GDDR 为双倍数据速率（DDR），实际传输速率是 Memory Clock 的 2 倍：

```text
Bandwidth = 14001 × 2 × 512 / 8 / 1000 = 1792.13 GB/s
```

**≈ 1792 GB/s**

> 与 RTX 5090 官方标称 1792 GB/s 一致。

### 3.3 算力/带宽比（额外参考）

```text
104.75 TFLOPS / 1.792 TB/s ≈ 58.45 FLOPs/byte
```

这意味着：若要让 GPU 算力打满，算核的算术强度大约需要超过 **58 FLOPs/byte**。
