# CUDA 代码远程执行结果汇总

> 运行环境：ssh -p 23692 root@i-1.gpushare.com
>
> - GPU：NVIDIA GeForce RTX 5090
> - Driver：570.195.03
> - CUDA：12.8
> - nvcc：V12.8.93
> - 编译参数：`-O3 -arch=sm_120`，使用 cuBLAS 的文件附加 `-lcublas`
> - 运行方式：无特殊说明均使用默认参数（无命令行参数）

## 汇总

- 扫描到带 `main()` 的可独立运行 CUDA/C++ 源文件：**46** 个
- 全部编译成功：**46 / 46**
- 全部运行成功（含正常退出/验证通过）：**46 / 46**

### 执行概览

| 文件 | 编译耗时 (s) | 运行耗时 (s) | 状态 | 关键输出 |
|------|-------------|-------------|------|----------|
| `aiinfra/week1/day1/exercise/hello_gpu.cu` | 1.20 | 0.454 | ✅ PASS | Launching kernel: grid=(2, 2, 1), block=(4, 2, 1) |
| `aiinfra/week1/day1/kernels/hello_gpu.cu` | 1.18 | 0.374 | ✅ PASS | Launching kernel: grid=(2,2,1), block=(8,1,1), total_threads |
| `aiinfra/week1/day2/exercise/occupancy_test.cu` | 1.26 | 0.391 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/week1/day2/exercise/occupancy_test_b.cu` | 1.25 | 0.369 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/week1/day2/exercise/register_spill.cu` | 1.57 | 0.439 | ✅ PASS | === Register Spill Demo === |
| `aiinfra/week1/day2/kernels/occupancy_test.cu` | 1.24 | 0.431 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/week1/day3/exercise/mini_device_query.cu` | 1.18 | 0.135 | ✅ PASS | Detected 1 CUDA device(s) |
| `aiinfra/week1/day3/exercise/occupancy_verify.cu` | 1.27 | 0.439 | ✅ PASS | === Device: NVIDIA GeForce RTX 5090 (Compute Capability 12.0 |
| `aiinfra/week1/day4/kernels/transpose.cu` | 1.22 | 0.421 | ✅ PASS | Transpose correctness: PASS |
| `aiinfra/week1/day5/kernels/bank_conflict.cu` | 1.23 | 0.372 | ✅ PASS | Bank conflict kernels finished. Use ncu to compare metrics. |
| `aiinfra/week3/day2/kernels/softmax_layernorm.cu` | 1.37 | 0.442 | ✅ PASS | === Softmax + LayerNorm Kernel Test === |
| `aiinfra/week3/day3/kernels/softmax_layernorm_opt.cu` | 1.55 | 0.488 | ✅ PASS | === Softmax + LayerNorm Optimization Comparison === |
| `aiinfra/week3/day4/kernels/attention_naive.cu` | 1.42 | 0.784 | ✅ PASS | === Standard Attention Forward (naive, materialize S/P) === |
| `aiinfra/week3/day5/kernels/softmax_layernorm_ext.cu` | 1.36 | 0.454 | ✅ PASS | === Softmax + LayerNorm (ext version, launch wrappers) === |
| `aiinfra/week3/day6/kernels/profiling_targets.cu` | 1.32 | 0.451 | ✅ PASS | === Profiling Targets: Softmax(memory-bound) + GEMM(compute- |
| `aiinfra/week5/day2/kernels/kv_cache.cu` | 1.43 | 0.390 | ✅ PASS | === KV Cache Test === |
| `aiinfra/week5/day4/kernels/paged_attention.cu` | 1.63 | 0.374 | ✅ PASS | === PagedAttention Test === |
| `profiling/leetgpu/argmax.cu` | 1.23 | 0.400 | ✅ PASS | GPU argmax idx = 524288 (expected 524288) PASS |
| `profiling/leetgpu/histogram.cu` | 1.22 | 0.460 | ✅ PASS | Global atomic: 0.106 ms |
| `profiling/leetgpu/matrix-addition.cu` | 1.26 | 1.081 | ✅ PASS | Matrix Addition PASS |
| `profiling/leetgpu/matrix-multiplication.cu` | 1.33 | 0.447 | ✅ PASS | Naive:  0.185 ms (1447.8 GFLOPS) |
| `profiling/leetgpu/matrix-transpose.cu` | 1.23 | 0.517 | ✅ PASS | Naive:  0.176 ms (190.2 GB/s) |
| `profiling/leetgpu/reduction.cu` | 1.22 | 0.453 | ✅ PASS | GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS |
| `profiling/leetgpu/relu.cu` | 1.21 | 0.405 | ✅ PASS | block=  32  time=0.088 ms  bw=95.7 GB/s |
| `profiling/leetgpu/softmax.cu` | 1.30 | 0.442 | ✅ PASS | Three-pass: 2.725 ms |
| `profiling/leetgpu/vector-add.cu` | 1.19 | 0.415 | ✅ PASS | Result: PASS |
| `profiling/week1/day1/hello_gpu.cu` | 1.16 | 0.383 | ✅ PASS | Launching kernel: grid=(2,2,1), block=(8,1,1), total_threads |
| `profiling/week1/day1/vector_add_blocksize.cu` | 1.23 | 0.501 | ✅ PASS | === Vector Add: block size 性能对比 === |
| `profiling/week1/day2/occupancy_test.cu` | 1.24 | 0.365 | ✅ PASS | === Kernel Attributes === |
| `profiling/week1/day4/bandwidth.cu` | 1.27 | 1.913 | ✅ PASS | === Coalesced vs Stride Bandwidth Benchmark === |
| `profiling/week1/day4/transpose.cu` | 1.21 | 0.409 | ✅ PASS | Transpose correctness: PASS |
| `profiling/week1/day4/transpose_tiles.cu` | 1.27 | 0.453 | ✅ PASS | === Transpose with Different Tile Sizes === |
| `profiling/week1/day5/bank_conflict.cu` | 1.22 | 0.372 | ✅ PASS | Bank conflict kernels finished. Use ncu to compare metrics. |
| `profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu` | 1.33 | 0.389 | ✅ PASS | Naive:  0.166 ms (1618.8 GFLOPS) |
| `profiling/week2/day1/warp_reduce.cu` | 1.24 | 0.477 | ✅ PASS | === Warp Shuffle Block Reduce === |
| `profiling/week2/day4/register_blocking_gemm.cu` | 2.12 | 1.566 | ✅ PASS | === Register Blocking GEMM === |
| `profiling/week2/day4/softmax_profile.cu` | 1.32 | 0.369 | ✅ PASS | === Softmax Profiling (row-level, M=128, N=1024) === |
| `profiling/week2/day5/flash_attention.cu` | 2.62 | 0.450 | ✅ PASS | === FlashAttention Simplified Forward === |
| `profiling/week2/day6/histogram.cu` | 1.22 | 0.472 | ✅ PASS | === Histogram: Global atomic vs Shared memory === |
| `profiling/week2/day6/integrated_gemm.cu` | 2.15 | 4.304 | ✅ PASS | === Integrated GEMM (Warp Shuffle + Register Blocking + floa |
| `profiling/week2/day7/block_reduce_timed.cu` | 1.23 | 0.444 | ✅ PASS | === Block Reduce (Warp Shuffle + 两级归约) === |
| `profiling/week2/day7/gemm_timed.cu` | 2.13 | 1.567 | ✅ PASS | === Register Blocking GEMM (手撕验收) === |
| `profiling/week3/day2/softmax_layernorm.cu` | 1.35 | 0.372 | ✅ PASS | === Softmax + LayerNorm Kernel Test === |
| `profiling/week3/day2/softmax_layernorm_dscan.cu` | 1.32 | 0.391 | ✅ PASS | === Softmax + LayerNorm D-scan (memory-bound scale law) === |
| `profiling/week3/day3/softmax_layernorm_opt.cu` | 1.54 | 0.491 | ✅ PASS | === Softmax + LayerNorm Optimization Comparison === |
| `profiling/week3/day3/warp_vs_block_dscan.cu` | 1.37 | 0.589 | ✅ PASS | === Softmax: warp-level vs block-level D-scan === |

## 详细输出

### `aiinfra/week1/day1/exercise/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day1/exercise/hello_gpu /root/aiinfra_run/aiinfra/week1/day1/exercise/hello_gpu.cu`
- **编译耗时**：1.20s
- **运行耗时**：0.454s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Launching kernel: grid=(2, 2, 1), block=(4, 2, 1)
Total threads: 32
block=(0, 0, 0), thread=(0, 0, 0), global_tid=0
block=(0, 0, 0), thread=(1, 0, 0), global_tid=1
block=(0, 0, 0), thread=(2, 0, 0), global_tid=2
block=(0, 0, 0), thread=(3, 0, 0), global_tid=3
block=(0, 0, 0), thread=(0, 1, 0), global_tid=0
block=(0, 0, 0), thread=(1, 1, 0), global_tid=1
block=(0, 0, 0), thread=(2, 1, 0), global_tid=2
block=(0, 0, 0), thread=(3, 1, 0), global_tid=3
block=(1, 0, 0), thread=(0, 0, 0), global_tid=4
block=(1, 0, 0), thread=(1, 0, 0), global_tid=5
block=(1, 0, 0), thread=(2, 0, 0), global_tid=6
block=(1, 0, 0), thread=(3, 0, 0), global_tid=7
block=(1, 0, 0), thread=(0, 1, 0), global_tid=4
block=(1, 0, 0), thread=(1, 1, 0), global_tid=5
block=(1, 0, 0), thread=(2, 1, 0), global_tid=6
block=(1, 0, 0), thread=(3, 1, 0), global_tid=7
block=(0, 1, 0), thread=(0, 0, 0), global_tid=0
block=(0, 1, 0), thread=(1, 0, 0), global_tid=1
block=(0, 1, 0), thread=(2, 0, 0), global_tid=2
block=(0, 1, 0), thread=(3, 0, 0), global_tid=3
block=(0, 1, 0), thread=(0, 1, 0), global_tid=0
block=(0, 1, 0), thread=(1, 1, 0), global_tid=1
block=(0, 1, 0), thread=(2, 1, 0), global_tid=2
block=(0, 1, 0), thread=(3, 1, 0), global_tid=3
block=(1, 1, 0), thread=(0, 0, 0), global_tid=4
block=(1, 1, 0), thread=(1, 0, 0), global_tid=5
block=(1, 1, 0), thread=(2, 0, 0), global_tid=6
block=(1, 1, 0), thread=(3, 0, 0), global_tid=7
block=(1, 1, 0), thread=(0, 1, 0), global_tid=4
block=(1, 1, 0), thread=(1, 1, 0), global_tid=5
block=(1, 1, 0), thread=(2, 1, 0), global_tid=6
block=(1, 1, 0), thread=(3, 1, 0), global_tid=7
```
</details>

### `aiinfra/week1/day1/kernels/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day1/kernels/hello_gpu /root/aiinfra_run/aiinfra/week1/day1/kernels/hello_gpu.cu`
- **编译耗时**：1.18s
- **运行耗时**：0.374s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Launching kernel: grid=(2,2,1), block=(8,1,1), total_threads=32
block=(0,0,0), thread=(0,0,0), global_tid=0
block=(0,0,0), thread=(1,0,0), global_tid=1
block=(0,0,0), thread=(2,0,0), global_tid=2
block=(0,0,0), thread=(3,0,0), global_tid=3
block=(0,0,0), thread=(4,0,0), global_tid=4
block=(0,0,0), thread=(5,0,0), global_tid=5
block=(0,0,0), thread=(6,0,0), global_tid=6
block=(0,0,0), thread=(7,0,0), global_tid=7
block=(1,0,0), thread=(0,0,0), global_tid=8
block=(1,0,0), thread=(1,0,0), global_tid=9
block=(1,0,0), thread=(2,0,0), global_tid=10
block=(1,0,0), thread=(3,0,0), global_tid=11
block=(1,0,0), thread=(4,0,0), global_tid=12
block=(1,0,0), thread=(5,0,0), global_tid=13
block=(1,0,0), thread=(6,0,0), global_tid=14
block=(1,0,0), thread=(7,0,0), global_tid=15
block=(0,1,0), thread=(0,0,0), global_tid=0
block=(0,1,0), thread=(1,0,0), global_tid=1
block=(0,1,0), thread=(2,0,0), global_tid=2
block=(0,1,0), thread=(3,0,0), global_tid=3
block=(0,1,0), thread=(4,0,0), global_tid=4
block=(0,1,0), thread=(5,0,0), global_tid=5
block=(0,1,0), thread=(6,0,0), global_tid=6
block=(0,1,0), thread=(7,0,0), global_tid=7
block=(1,1,0), thread=(0,0,0), global_tid=8
block=(1,1,0), thread=(1,0,0), global_tid=9
block=(1,1,0), thread=(2,0,0), global_tid=10
block=(1,1,0), thread=(3,0,0), global_tid=11
block=(1,1,0), thread=(4,0,0), global_tid=12
block=(1,1,0), thread=(5,0,0), global_tid=13
block=(1,1,0), thread=(6,0,0), global_tid=14
block=(1,1,0), thread=(7,0,0), global_tid=15
```
</details>

### `aiinfra/week1/day2/exercise/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day2/exercise/occupancy_test /root/aiinfra_run/aiinfra/week1/day2/exercise/occupancy_test.cu`
- **编译耗时**：1.26s
- **运行耗时**：0.391s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Kernel Attributes ===
Registers per thread: 40
Shared memory per block: 0 bytes
Constant memory per block: 0 bytes
Local memory per thread: 0 bytes
Max threads per block: 1024
=========================
```
</details>

### `aiinfra/week1/day2/exercise/occupancy_test_b.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day2/exercise/occupancy_test_b /root/aiinfra_run/aiinfra/week1/day2/exercise/occupancy_test_b.cu`
- **编译耗时**：1.25s
- **运行耗时**：0.369s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Kernel Attributes ===
Registers per thread: 40
Shared memory per block: 0 bytes
Constant memory per block: 0 bytes
Local memory per thread: 0 bytes
Max threads per block: 1024
=========================
```
</details>

### `aiinfra/week1/day2/exercise/register_spill.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day2/exercise/register_spill /root/aiinfra_run/aiinfra/week1/day2/exercise/register_spill.cu`
- **编译耗时**：1.57s
- **运行耗时**：0.439s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Register Spill Demo ===
Launching: grid=8192, block=128, total_threads=1048576
Done. Use the command below to check spilling:
  nvcc -Xptxas -v week1/day2/exercise/register_spill.cu
Look for 'spill stores' and 'spill loads' in the output.
```
</details>

### `aiinfra/week1/day2/kernels/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day2/kernels/occupancy_test /root/aiinfra_run/aiinfra/week1/day2/kernels/occupancy_test.cu`
- **编译耗时**：1.24s
- **运行耗时**：0.431s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Kernel Attributes ===
Registers per thread: 40
Shared memory per block: 0 bytes
Constant memory per block: 0 bytes
Local memory per thread: 0 bytes
Max threads per block: 1024
=========================
```
</details>

### `aiinfra/week1/day3/exercise/mini_device_query.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day3/exercise/mini_device_query /root/aiinfra_run/aiinfra/week1/day3/exercise/mini_device_query.cu`
- **编译耗时**：1.18s
- **运行耗时**：0.135s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Detected 1 CUDA device(s)

Device 0: NVIDIA GeForce RTX 5090
  Compute Capability: 12.0
  Total Global Memory: 31.37 GB
  Number of SMs: 170
  Warp Size: 32
  Max Threads per Block: 1024
  Max Threads per SM: 1536
  Max Blocks per SM: 24
  Shared Memory per Block: 48 KB
  Registers per Block: 65536
  Memory Clock Rate: 14001 MHz
  Memory Bus Width: 512 bits
  GPU Clock Rate: 2407 MHz
```
</details>

### `aiinfra/week1/day3/exercise/occupancy_verify.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day3/exercise/occupancy_verify /root/aiinfra_run/aiinfra/week1/day3/exercise/occupancy_verify.cu`
- **编译耗时**：1.27s
- **运行耗时**：0.439s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Device: NVIDIA GeForce RTX 5090 (Compute Capability 12.0) ===
  Number of SMs: 170
  Max threads / SM: 1536
  Max blocks / SM: 24
  Max warps / SM: 48
  Registers / SM: 65536
  Shared memory / SM: 102400 bytes
  Warp size: 32

=== Occupancy Analysis for Sample Kernels ===

--- kernel_light (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 10
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_medium (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 12
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_smem (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 8
  Shared memory per block: 1024 bytes (static) + 0 bytes (dynamic) = 1024 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_launch_bounds (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 33
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 256
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

=== Varying Block Size for kernel_medium ===

--- kernel_medium (blockSize=128, dynamicSmem=0) ---
  Registers per thread: 12
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 4
  CUDA API   -> active blocks / SM: 12, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 12, active warps / SM: 48, occupancy: 100.0%

--- kernel_medium (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 12
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_medium (blockSize=512, dynamicSmem=0) ---
  Registers per thread: 12
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 16
  CUDA API   -> active blocks / SM: 3, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 3, active warps / SM: 48, occupancy: 100.0%

--- kernel_medium (blockSize=1024, dynamicSmem=0) ---
  Registers per thread: 12
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 32
  CUDA API   -> active blocks / SM: 1, active warps / SM: 32, occupancy: 66.7%
  Hand calc  -> active blocks / SM: 1, active warps / SM: 32, occupancy: 66.7%

=== Varying Dynamic Shared Memory for kernel_light ===

--- kernel_light (blockSize=256, dynamicSmem=0) ---
  Registers per thread: 10
  Shared memory per block: 0 bytes (static) + 0 bytes (dynamic) = 0 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_light (blockSize=256, dynamicSmem=4096) ---
  Registers per thread: 10
  Shared memory per block: 0 bytes (static) + 4096 bytes (dynamic) = 4096 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_light (blockSize=256, dynamicSmem=8192) ---
  Registers per thread: 10
  Shared memory per block: 0 bytes (static) + 8192 bytes (dynamic) = 8192 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%

--- kernel_light (blockSize=256, dynamicSmem=16384) ---
  Registers per thread: 10
  Shared memory per block: 0 bytes (static) + 16384 bytes (dynamic) = 16384 bytes
  Max threads per block: 1024
  Warps per block: 8
  CUDA API   -> active blocks / SM: 5, active warps / SM: 40, occupancy: 83.3%
  Hand calc  -> active blocks / SM: 6, active warps / SM: 48, occupancy: 100.0%
```
</details>

### `aiinfra/week1/day4/kernels/transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day4/kernels/transpose /root/aiinfra_run/aiinfra/week1/day4/kernels/transpose.cu`
- **编译耗时**：1.22s
- **运行耗时**：0.421s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Transpose correctness: PASS
```
</details>

### `aiinfra/week1/day5/kernels/bank_conflict.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week1/day5/kernels/bank_conflict /root/aiinfra_run/aiinfra/week1/day5/kernels/bank_conflict.cu`
- **编译耗时**：1.23s
- **运行耗时**：0.372s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```
</details>

### `aiinfra/week3/day2/kernels/softmax_layernorm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week3/day2/kernels/softmax_layernorm /root/aiinfra_run/aiinfra/week3/day2/kernels/softmax_layernorm.cu`
- **编译耗时**：1.37s
- **运行耗时**：0.442s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm Kernel Test ===
Config: M=128, D=1024, threads=256

[Softmax]
  Softmax vs CPU: maxDiff = 4.19e-09 (PASS)
  Time: 0.140 ms
[LayerNorm]
  LayerNorm vs CPU: maxDiff = 1.07e-06 (PASS)
  Time: 0.026 ms
```
</details>

### `aiinfra/week3/day3/kernels/softmax_layernorm_opt.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week3/day3/kernels/softmax_layernorm_opt /root/aiinfra_run/aiinfra/week3/day3/kernels/softmax_layernorm_opt.cu`
- **编译耗时**：1.55s
- **运行耗时**：0.488s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm Optimization Comparison ===
Config: M=1024, D=1024 (D must be multiple of 4 for float4)

[Softmax: block-level (Day16) vs warp-level (optimized)]
  warp-level correctness: maxDiff = 6.52e-09 (PASS)
  block-level (Day16): 0.0062 ms
  warp-level (optim) : 0.0044 ms
  speedup            : 1.42x

[LayerNorm: scalar load (Day16) vs float4 vectorized]
  float4 correctness: maxDiff = 1.67e-06 (PASS)
  scalar (Day16) : 0.0063 ms
  float4 (optim) : 0.0061 ms
  speedup        : 1.03x

=== ncu 验证命令 ===
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__time_duration.sum \
  --kernel-name regex:"softmax_warp_kernel|layernorm_float4_kernel" \
  ./softmax_layernorm_opt
```
</details>

### `aiinfra/week3/day4/kernels/attention_naive.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week3/day4/kernels/attention_naive /root/aiinfra_run/aiinfra/week3/day4/kernels/attention_naive.cu`
- **编译耗时**：1.42s
- **运行耗时**：0.784s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Standard Attention Forward (naive, materialize S/P) ===
N        S/P size(MB)   HBM IO(MB)       Time(ms)     Check     
------------------------------------------------------------------
  maxDiff = 1.02e-08 (PASS)
256      0.25           1.25             0.155        PASS      
  maxDiff = 1.30e-08 (PASS)
512      1.00           4.50             0.080        PASS      
  maxDiff = 1.26e-08 (PASS)
1024     4.00           17.00            0.276        PASS      
  maxDiff = 1.12e-08 (PASS)
2048     16.00          66.00            0.811        PASS      

观察要点：
1. S/P size 随 N² 增长（N 翻倍 → size 4x）
2. HBM IO 随 N² 增长（N 翻倍 → IO 4x）
3. Time 近似随 N² 增长（长序列下 O(N²) IO 主导）
4. 用 ncu 验证 dram__bytes_read.sum + dram__bytes_write.sum ≈ 理论 HBM IO
```
</details>

### `aiinfra/week3/day5/kernels/softmax_layernorm_ext.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week3/day5/kernels/softmax_layernorm_ext /root/aiinfra_run/aiinfra/week3/day5/kernels/softmax_layernorm_ext.cu`
- **编译耗时**：1.36s
- **运行耗时**：0.454s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm (ext version, launch wrappers) ===
Config: M=128, D=1024

[Softmax]
  Softmax vs CPU: maxDiff = 4.19e-09 (PASS)
[LayerNorm]
  LayerNorm vs CPU: maxDiff = 1.07e-06 (PASS)

这两个 launch wrapper 就是 PyTorch C++ Extension 要调用的入口。
集成方式见 mini_engine.py 的 load_inline 调用。
```
</details>

### `aiinfra/week3/day6/kernels/profiling_targets.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week3/day6/kernels/profiling_targets /root/aiinfra_run/aiinfra/week3/day6/kernels/profiling_targets.cu`
- **编译耗时**：1.32s
- **运行耗时**：0.451s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Profiling Targets: Softmax(memory-bound) + GEMM(compute-bound) ===
Softmax: M=256, D=1024
GEMM:    M=512, N=512, K=512

[Softmax] time=0.101 ms  maxDiff=4.42e-09  (PASS)
[GEMM]    time=0.091 ms  TFLOPS=2.94  (naive, no tiling)

--- ncu 分析指引 ---
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct,\
  gpu__time_duration.sum \
  --kernel-name regex:"softmax_kernel|gemm_kernel" ./profiling_targets

预期：softmax_kernel -> DRAM% >> SM% (memory-bound)
      gemm_kernel    -> SM%   >> DRAM% (compute-bound)
```
</details>

### `aiinfra/week5/day2/kernels/kv_cache.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week5/day2/kernels/kv_cache /root/aiinfra_run/aiinfra/week5/day2/kernels/kv_cache.cu`
- **编译耗时**：1.43s
- **运行耗时**：0.390s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== KV Cache Test ===
Config: layers=2, batch=1, heads=8, max_len=1024, d_head=64
After Round 1 (len=10): seq_len=10
After Round 2 (len=5): seq_len=15
After Round 3 (len=8): seq_len=23
PASS: seq_len = 23 (expected 23)
Data verification (Round 1 K in cache): max_diff = 9.50e-02 (FAIL)
KV Cache bytes per token: 8192
Max memory usage: 8 MB

[LLaMA-7B reference] bytes per token: 524288 (512.0 KB)
[LLaMA-7B reference] 4096 tokens: 2048 MB
[LLaMA-7B reference] batch=16, 4096 tokens: 32 GB
```
</details>

### `aiinfra/week5/day4/kernels/paged_attention.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/week5/day4/kernels/paged_attention /root/aiinfra_run/aiinfra/week5/day4/kernels/paged_attention.cu`
- **编译耗时**：1.63s
- **运行耗时**：0.374s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== PagedAttention Test ===
d=64, seq_len=50, KV_BLOCK_SIZE=16, num_logical_blocks=4
block_table (logical→physical): 0→7  1→1  2→12  3→3  
max diff (paged vs contiguous): 9.54e-07 (PASS)

[Memory utilization]
  Static alloc (max=128): waste 61% (allocated 128, used 50)
  PagedAttention:        use 50% of static (4 blocks × 16 tok = 64 slots, 50 actual)
  PagedAttention 的物理 block 可不连续（本例 7,1,12,3），逻辑连续由 block table 保证
```
</details>

### `profiling/leetgpu/argmax.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/argmax /root/aiinfra_run/profiling/leetgpu/argmax.cu`
- **编译耗时**：1.23s
- **运行耗时**：0.400s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
GPU argmax idx = 524288 (expected 524288) PASS
```
</details>

### `profiling/leetgpu/histogram.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/histogram /root/aiinfra_run/profiling/leetgpu/histogram.cu`
- **编译耗时**：1.22s
- **运行耗时**：0.460s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Global atomic: 0.106 ms
Shared privat: 0.020 ms
Speedup: 5.34x
```
</details>

### `profiling/leetgpu/matrix-addition.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-addition /root/aiinfra_run/profiling/leetgpu/matrix-addition.cu`
- **编译耗时**：1.26s
- **运行耗时**：1.081s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Matrix Addition PASS
```
</details>

### `profiling/leetgpu/matrix-multiplication.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-multiplication /root/aiinfra_run/profiling/leetgpu/matrix-multiplication.cu`
- **编译耗时**：1.33s
- **运行耗时**：0.447s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Naive:  0.185 ms (1447.8 GFLOPS)
Tiled:  0.052 ms (5181.4 GFLOPS)
Speedup: 3.58x
Tiled (no bank conflict): 0.068 ms (3962.5 GFLOPS)
Speedup vs Tiled: 0.76x
```
</details>

### `profiling/leetgpu/matrix-transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-transpose /root/aiinfra_run/profiling/leetgpu/matrix-transpose.cu`
- **编译耗时**：1.23s
- **运行耗时**：0.517s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Naive:  0.176 ms (190.2 GB/s)
Shared: 0.027 ms (1238.0 GB/s)
Speedup: 6.51x
```
</details>

### `profiling/leetgpu/reduction.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/reduction /root/aiinfra_run/profiling/leetgpu/reduction.cu`
- **编译耗时**：1.22s
- **运行耗时**：0.453s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS
Time: 0.131 ms (127.7 GB/s)
```
</details>

### `profiling/leetgpu/relu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/relu /root/aiinfra_run/profiling/leetgpu/relu.cu`
- **编译耗时**：1.21s
- **运行耗时**：0.405s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
block=  32  time=0.088 ms  bw=95.7 GB/s
block=  64  time=0.014 ms  bw=621.2 GB/s
block= 128  time=0.011 ms  bw=738.4 GB/s
block= 256  time=0.009 ms  bw=960.2 GB/s
block= 512  time=0.008 ms  bw=1110.8 GB/s
block=1024  time=0.008 ms  bw=1024.0 GB/s
Result: PASS
```
</details>

### `profiling/leetgpu/softmax.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/softmax /root/aiinfra_run/profiling/leetgpu/softmax.cu`
- **编译耗时**：1.30s
- **运行耗时**：0.442s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Three-pass: 2.725 ms
Online:      2.746 ms
Speedup:     0.99x
```
</details>

### `profiling/leetgpu/vector-add.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/vector-add /root/aiinfra_run/profiling/leetgpu/vector-add.cu`
- **编译耗时**：1.19s
- **运行耗时**：0.415s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Result: PASS
Time: 0.062 ms (203.84 GB/s bandwidth)
```
</details>

### `profiling/week1/day1/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day1/hello_gpu /root/aiinfra_run/profiling/week1/day1/hello_gpu.cu`
- **编译耗时**：1.16s
- **运行耗时**：0.383s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Launching kernel: grid=(2,2,1), block=(8,1,1), total_threads=32
block=(0,0,0), thread=(0,0,0), global_tid=0
block=(0,0,0), thread=(1,0,0), global_tid=1
block=(0,0,0), thread=(2,0,0), global_tid=2
block=(0,0,0), thread=(3,0,0), global_tid=3
block=(0,0,0), thread=(4,0,0), global_tid=4
block=(0,0,0), thread=(5,0,0), global_tid=5
block=(0,0,0), thread=(6,0,0), global_tid=6
block=(0,0,0), thread=(7,0,0), global_tid=7
block=(1,0,0), thread=(0,0,0), global_tid=8
block=(1,0,0), thread=(1,0,0), global_tid=9
block=(1,0,0), thread=(2,0,0), global_tid=10
block=(1,0,0), thread=(3,0,0), global_tid=11
block=(1,0,0), thread=(4,0,0), global_tid=12
block=(1,0,0), thread=(5,0,0), global_tid=13
block=(1,0,0), thread=(6,0,0), global_tid=14
block=(1,0,0), thread=(7,0,0), global_tid=15
block=(0,1,0), thread=(0,0,0), global_tid=0
block=(0,1,0), thread=(1,0,0), global_tid=1
block=(0,1,0), thread=(2,0,0), global_tid=2
block=(0,1,0), thread=(3,0,0), global_tid=3
block=(0,1,0), thread=(4,0,0), global_tid=4
block=(0,1,0), thread=(5,0,0), global_tid=5
block=(0,1,0), thread=(6,0,0), global_tid=6
block=(0,1,0), thread=(7,0,0), global_tid=7
block=(1,1,0), thread=(0,0,0), global_tid=8
block=(1,1,0), thread=(1,0,0), global_tid=9
block=(1,1,0), thread=(2,0,0), global_tid=10
block=(1,1,0), thread=(3,0,0), global_tid=11
block=(1,1,0), thread=(4,0,0), global_tid=12
block=(1,1,0), thread=(5,0,0), global_tid=13
block=(1,1,0), thread=(6,0,0), global_tid=14
block=(1,1,0), thread=(7,0,0), global_tid=15
```
</details>

### `profiling/week1/day1/vector_add_blocksize.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day1/vector_add_blocksize /root/aiinfra_run/profiling/week1/day1/vector_add_blocksize.cu`
- **编译耗时**：1.23s
- **运行耗时**：0.501s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Vector Add: block size 性能对比 ===
N = 1048576 (4.0 MB per array)

block_size   grid_size    time(ms)       bandwidth(GB/s)
--------------------------------------------------------
32           32768        0.0167         751.4         
64           16384        0.0103         1226.1        
128          8192         0.0062         2033.7        
256          4096         0.0042         3025.9        
512          2048         0.0041         3084.8        
1024         1024         0.0043         2948.3        

正确性: PASS

=== ncu 分析命令 ===
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread \
  --kernel-name regex:vector_add \
  ./vector_add_blocksize
```
</details>

### `profiling/week1/day2/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day2/occupancy_test /root/aiinfra_run/profiling/week1/day2/occupancy_test.cu`
- **编译耗时**：1.24s
- **运行耗时**：0.365s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Kernel Attributes ===
Registers per thread: 40
Shared memory per block: 0 bytes
Constant memory per block: 0 bytes
Local memory per thread: 0 bytes
Max threads per block: 1024
=========================
```
</details>

### `profiling/week1/day4/bandwidth.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/bandwidth /root/aiinfra_run/profiling/week1/day4/bandwidth.cu`
- **编译耗时**：1.27s
- **运行耗时**：1.913s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Coalesced vs Stride Bandwidth Benchmark ===
Array size: 67108864 elements (256.00 MB)

Kernel                    | Elapsed (ms) | Effective Bandwidth (GB/s)
-------------------------|--------------|----------------------------
coalesced_copy           |       0.3506 |                    1531.22
stride_copy(stride= 1)  |       0.3511 |                    1528.98
stride_copy(stride= 2)  |       0.5123 |                    1048.05
stride_copy(stride= 4)  |       0.8372 |                     641.25
stride_copy(stride= 8)  |       1.4995 |                     358.02
stride_copy(stride=16)  |       1.4636 |                     366.81
stride_copy(stride=32)  |       1.5740 |                     341.08

Coalesced copy correctness: FAIL
```
</details>

### `profiling/week1/day4/transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/transpose /root/aiinfra_run/profiling/week1/day4/transpose.cu`
- **编译耗时**：1.21s
- **运行耗时**：0.409s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Transpose correctness: PASS
```
</details>

### `profiling/week1/day4/transpose_tiles.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/transpose_tiles /root/aiinfra_run/profiling/week1/day4/transpose_tiles.cu`
- **编译耗时**：1.27s
- **运行耗时**：0.453s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Transpose with Different Tile Sizes ===
Matrix: 1024 x 1024

Tile Size | Correctness | Avg Time (ms) | Effective Bandwidth (GB/s)
----------|-------------|---------------|----------------------------
8 x 8     | PASS        |        0.0172 |                     487.17
16 x 16   | PASS        |        0.0056 |                    1497.54
32 x 32   | PASS        |        0.0058 |                    1438.38
```
</details>

### `profiling/week1/day5/bank_conflict.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day5/bank_conflict /root/aiinfra_run/profiling/week1/day5/bank_conflict.cu`
- **编译耗时**：1.22s
- **运行耗时**：0.372s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```
</details>

### `profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day6/matrix_multiplication/matrix_multiplication /root/aiinfra_run/profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu`
- **编译耗时**：1.33s
- **运行耗时**：0.389s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
Naive:  0.166 ms (1618.8 GFLOPS)
Tiled:  0.053 ms (5071.7 GFLOPS)
Speedup: 3.13x
Tiled (no bank conflict): 0.069 ms (3914.4 GFLOPS)
Speedup vs Tiled: 0.77x
```
</details>

### `profiling/week2/day1/warp_reduce.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day1/warp_reduce /root/aiinfra_run/profiling/week2/day1/warp_reduce.cu`
- **编译耗时**：1.24s
- **运行耗时**：0.477s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Warp Shuffle Block Reduce ===
Array size: 4194304 (16.00 MB)
GPU Sum: 20971.052734
CPU Sum: 20971.052734
Diff:    0.000000 (PASS)
Time:    0.160 ms (104.71 GB/s bandwidth)

=== ncu 分析命令 ===
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct \
  --kernel-name regex:blockReduceSum \
  ./warp_reduce
```
</details>

### `profiling/week2/day4/register_blocking_gemm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day4/register_blocking_gemm /root/aiinfra_run/profiling/week2/day4/register_blocking_gemm.cu -lcublas`
- **编译耗时**：2.12s
- **运行耗时**：1.566s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Register Blocking GEMM ===
Parameters: BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
M          N          K          Our(ms)      cuBLAS(ms)   Percent   
------------------------------------------------------------
1024       1024       1024       0.267        0.093        34.8     % PASS
2048       2048       2048       0.669        0.301        45.1     % PASS
4096       4096       4096       6.122        2.141        35.0     % PASS
```
</details>

### `profiling/week2/day4/softmax_profile.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day4/softmax_profile /root/aiinfra_run/profiling/week2/day4/softmax_profile.cu`
- **编译耗时**：1.32s
- **运行耗时**：0.369s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax Profiling (row-level, M=128, N=1024) ===

[softmax_row_kernel]
  maxDiff = 1.40e-09 (PASS)
  Time: 0.115 ms

=== ncu 分析命令 ===
# 行级 Softmax
ncu --kernel-name regex:softmax_row_kernel \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./softmax_profile

# 完整报告
ncu --set full --kernel-name regex:softmax_row_kernel -o softmax_report ./softmax_profile

# nsys 时间线
nsys profile -o softmax_timeline ./softmax_profile
```
</details>

### `profiling/week2/day5/flash_attention.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day5/flash_attention /root/aiinfra_run/profiling/week2/day5/flash_attention.cu`
- **编译耗时**：2.62s
- **运行耗时**：0.450s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== FlashAttention Simplified Forward ===
Config: N=256, D=64, batch=1, heads=1
SRAM usage per block: 28.00 KB
Grid: (8, 1, 1), Block: (32, 4)
FlashAttention GPU Time: 0.647 ms
Result check: PASS
Standard Attention GPU Time: 2.980 ms
Result check: PASS
Speedup: 4.61x

=== ncu 分析命令 ===
# FlashAttention
ncu --kernel-name regex:flashAttentionFwd \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
    ./flash_attention

# Standard Attention（对比）
ncu --kernel-name regex:standardAttentionFwd \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
    ./flash_attention

# HBM 读写量对比
ncu --kernel-name regex:"flashAttentionFwd|standardAttentionFwd" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    ./flash_attention
```
</details>

### `profiling/week2/day6/histogram.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day6/histogram /root/aiinfra_run/profiling/week2/day6/histogram.cu`
- **编译耗时**：1.22s
- **运行耗时**：0.472s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Histogram: Global atomic vs Shared memory ===
N = 1048576, B = 256, blocks = 1024, threads = 256

Mismatch at bin 0: global=1071, shared=4173
Global atomic:  0.153 ms
Shared memory:   0.027 ms
Speedup:         5.79x
Correctness:     FAIL

=== ncu 分析命令 ===
# Global atomic
ncu --kernel-name regex:histogram_global \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  ./histogram

# Shared memory privatization
ncu --kernel-name regex:histogram_shared \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  ./histogram

# 对比两个 kernel 的 HBM 读写量
ncu --kernel-name regex:"histogram_global|histogram_shared" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  ./histogram
```
</details>

### `profiling/week2/day6/integrated_gemm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day6/integrated_gemm /root/aiinfra_run/profiling/week2/day6/integrated_gemm.cu -lcublas`
- **编译耗时**：2.15s
- **运行耗时**：4.304s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===
BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256

M        N        K        Our(ms)    cuBLAS(ms) GFLOPS     Percent 
----------------------------------------------------------------
1024     1024     1024     0.145      0.098      14847.1    67.6   % PASS
2048     2048     2048     0.428      0.303      40118.9    70.8   % PASS
4096     4096     4096     3.256      2.152      42210.2    66.1   % PASS
8192     8192     8192     23.114     16.051     47568.3    69.4   % PASS

=== ncu 分析命令 ===
ncu --kernel-name regex:gemmIntegrated \
  -o integrated_profile \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./integrated_gemm
```
</details>

### `profiling/week2/day7/block_reduce_timed.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day7/block_reduce_timed /root/aiinfra_run/profiling/week2/day7/block_reduce_timed.cu`
- **编译耗时**：1.23s
- **运行耗时**：0.444s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Block Reduce (Warp Shuffle + 两级归约) ===
N = 4194304 (16.00 MB)

GPU Sum: 2094860.1250
CPU Sum: 2094860.1250
Diff:    0.000000 (PASS)
Time:    0.010 ms (1726.96 GB/s bandwidth)

=== ncu 分析命令 ===
ncu --kernel-name regex:blockReduceSum \
  --metrics \
    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./block_reduce
```
</details>

### `profiling/week2/day7/gemm_timed.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day7/gemm_timed /root/aiinfra_run/profiling/week2/day7/gemm_timed.cu -lcublas`
- **编译耗时**：2.13s
- **运行耗时**：1.567s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Register Blocking GEMM (手撕验收) ===
BM=128 BN=128 BK=8 TM=8 TN=8 Threads=256

N        Our(ms)    cuBLAS(ms) GFLOPS     Percent 
------------------------------------------------
512      0.078      0.049      3454.9     62.5   % PASS
1024     0.135      0.076      15868.7    56.3   % PASS
2048     0.461      0.300      37243.9    65.0   % PASS
4096     3.167      2.163      43392.7    68.3   % PASS

=== ncu 分析命令 ===
ncu --kernel-name regex:gemmRegisterBlocking \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./gemm_timed
```
</details>

### `profiling/week3/day2/softmax_layernorm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day2/softmax_layernorm /root/aiinfra_run/profiling/week3/day2/softmax_layernorm.cu`
- **编译耗时**：1.35s
- **运行耗时**：0.372s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm Kernel Test ===
Config: M=128, D=1024, threads=256

[Softmax]
  Softmax vs CPU: maxDiff = 4.19e-09 (PASS)
  Time: 0.150 ms
[LayerNorm]
  LayerNorm vs CPU: maxDiff = 1.07e-06 (PASS)
  Time: 0.028 ms
```
</details>

### `profiling/week3/day2/softmax_layernorm_dscan.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day2/softmax_layernorm_dscan /root/aiinfra_run/profiling/week3/day2/softmax_layernorm_dscan.cu`
- **编译耗时**：1.32s
- **运行耗时**：0.391s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm D-scan (memory-bound scale law) ===
M=128, threads=256

D        SM time(ms)    LN time(ms)    SM BW(GB/s)    LN BW(GB/s)   
------------------------------------------------------------------------
256      0.0060         0.0044         65.7           88.7          
512      0.0034         0.0041         232.6          192.9         
768      0.0034         0.0041         350.4          284.6         
1024     0.0033         0.0042         471.5          371.1         
2048     0.0045         0.0062         706.0          505.2         
4096     0.0063         0.0083         993.2          756.8         

观察要点：
1. D 翻倍 → 时间接近翻倍（memory-bound，时间 ≈ Bytes/Bandwidth）
2. 带宽利用率应相对稳定（受 DRAM 带宽限制）
3. ncu 验证：DRAM Throughput >> SM Throughput → memory-bound

=== ncu 命令（指定 D=4096）===
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct \
  --kernel-name regex:"softmax_kernel|layernorm_kernel" \
  ./sl_dscan
```
</details>

### `profiling/week3/day3/softmax_layernorm_opt.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day3/softmax_layernorm_opt /root/aiinfra_run/profiling/week3/day3/softmax_layernorm_opt.cu`
- **编译耗时**：1.54s
- **运行耗时**：0.491s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax + LayerNorm Optimization Comparison ===
Config: M=1024, D=1024 (D must be multiple of 4 for float4)

[Softmax: block-level (Day16) vs warp-level (optimized)]
  warp-level correctness: maxDiff = 6.52e-09 (PASS)
  block-level (Day16): 0.0062 ms
  warp-level (optim) : 0.0044 ms
  speedup            : 1.41x

[LayerNorm: scalar load (Day16) vs float4 vectorized]
  float4 correctness: maxDiff = 1.67e-06 (PASS)
  scalar (Day16) : 0.0063 ms
  float4 (optim) : 0.0061 ms
  speedup        : 1.02x

=== ncu 验证命令 ===
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__time_duration.sum \
  --kernel-name regex:"softmax_warp_kernel|layernorm_float4_kernel" \
  ./softmax_layernorm_opt
```
</details>

### `profiling/week3/day3/warp_vs_block_dscan.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day3/warp_vs_block_dscan /root/aiinfra_run/profiling/week3/day3/warp_vs_block_dscan.cu`
- **编译耗时**：1.37s
- **运行耗时**：0.589s
- **状态**：✅ 运行成功
- **运行输出**：
<details><summary>点击展开</summary>

```text
=== Softmax: warp-level vs block-level D-scan ===
M=1024

D        Block(ms)        Warp(ms)         Speedup     
--------------------------------------------------------
256      0.0041           0.0034           1.20        
512      0.0042           0.0033           1.26        
1024     0.0062           0.0043           1.45        
2048     0.0083           0.0075           1.12        
4096     0.0145           0.0146           0.99        

观察要点：
1. D<=1024 时 warp 级通常更快（无 __syncthreads 开销）
2. D=4096 时 warp 级每 lane 处理 128 元素，并行度下降
3. 这就是 PyTorch 用 D=1024 做 dispatch 分界的原因

=== ncu 命令 ===
# 对比 warp vs block 在特定 D 下的指标
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:"softmax_warp_kernel|softmax_block_kernel" \
  ./warp_vs_block
```
</details>

## 未独立运行的 kernel 文件

以下文件是 LeetGPU 平台提交用的 kernel 实现，没有 `main()`，需要平台 starter 或额外 host 代码调用，因此未在上面独立执行：

- `leetgpu/week2/day1/pre.cu`
- `leetgpu/week2/day1/prefix_sum_inclusive.cu`
- `leetgpu/week2/day1/presum.cu`
