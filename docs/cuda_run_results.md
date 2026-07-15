# CUDA 代码远程执行结果汇总

> 运行环境：ssh -p 23692 root@i-1.gpushare.com
>
> - GPU：NVIDIA GeForce RTX 5090
> - Driver：570.195.03
> - CUDA：12.8
> - nvcc：V12.8.93
> - 编译参数：`-O3 -arch=sm_120`；使用 cuBLAS 的文件附加 `-lcublas`
> - 运行方式：无特殊说明均使用默认参数（无命令行参数）

## 统计

- 源文件（`.cu/.cpp/.c`）：46 个
- Markdown `cuda/cpp` 代码块：59 个
- 无 `main()` 的 kernel 文件：3 个
- **总计**：108 个
- **运行通过**：104 个
- **编译失败**：0 个
- **运行失败**：0 个
- **跳过/依赖外部库**：1 个

> 说明：为使代码可直接编译运行，已对部分 Markdown 中的代码做了最小修正：
> - `aiinfra/daily/week1/day3/README.md`：补齐缺失的 `#include <stdio.h>`。
> - `aiinfra/daily/week2/day5/README.md`、`aiinfra/daily/plan/learning_plan_week2_expanded.md`：
>   将 FlashAttention 示例的 `Bc` 从 64 调为 32，并添加 `#undef D` 避免宏与函数参数名冲突。
> - `leetgpu/week6/day3/leetgpu-stream-compaction-solution.md`：补充 `predicate_kernel`，
>   使用真正的 predicate（`input[i]!=0`）驱动 scan + scatter。
> - 6 个 LeetGPU 题解：将 `cudaMemcpy(hX, dX, ...)` 改为 `cudaMemcpy(hX.data(), dX, ...)`。
> - `aiinfra/daily/plan/learning_plan_week5_expanded.md`：将 `#include <math>` 改为 `<cmath>`。

## 一、源文件执行结果

| 文件 | 编译(s) | 运行(s) | 状态 | 关键输出 |
|------|--------|--------|------|----------|
| `aiinfra/daily/week1/day1/exercise/hello_gpu.cu` | 1.20 | 0.454 | ✅ PASS | Launching kernel: grid=(2, 2, 1), block=(4, 2, 1) |
| `aiinfra/daily/week1/day1/kernels/hello_gpu.cu` | 1.18 | 0.374 | ✅ PASS | Launching kernel: grid=(2,2,1), block=(8,1,1), total_th |
| `aiinfra/daily/week1/day2/exercise/occupancy_test.cu` | 1.26 | 0.391 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/daily/week1/day2/exercise/occupancy_test_b.cu` | 1.25 | 0.369 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/daily/week1/day2/exercise/register_spill.cu` | 1.57 | 0.439 | ✅ PASS | === Register Spill Demo === |
| `aiinfra/daily/week1/day2/kernels/occupancy_test.cu` | 1.24 | 0.431 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/daily/week1/day3/exercise/mini_device_query.cu` | 1.18 | 0.135 | ✅ PASS | Detected 1 CUDA device(s) |
| `aiinfra/daily/week1/day3/exercise/occupancy_verify.cu` | 1.27 | 0.439 | ✅ PASS | === Device: NVIDIA GeForce RTX 5090 (Compute Capability |
| `aiinfra/daily/week1/day4/kernels/transpose.cu` | 1.22 | 0.421 | ✅ PASS | Transpose correctness: PASS |
| `aiinfra/daily/week1/day5/kernels/bank_conflict.cu` | 1.23 | 0.372 | ✅ PASS | Bank conflict kernels finished. Use ncu to compare metr |
| `aiinfra/daily/week3/day2/kernels/softmax_layernorm.cu` | 1.37 | 0.442 | ✅ PASS | === Softmax + LayerNorm Kernel Test === |
| `aiinfra/daily/week3/day3/kernels/softmax_layernorm_opt.cu` | 1.55 | 0.488 | ✅ PASS | === Softmax + LayerNorm Optimization Comparison === |
| `aiinfra/daily/week3/day4/kernels/attention_naive.cu` | 1.42 | 0.784 | ✅ PASS | === Standard Attention Forward (naive, materialize S/P) |
| `aiinfra/daily/week3/day5/kernels/softmax_layernorm_ext.cu` | 1.36 | 0.454 | ✅ PASS | === Softmax + LayerNorm (ext version, launch wrappers)  |
| `aiinfra/daily/week3/day6/kernels/profiling_targets.cu` | 1.32 | 0.451 | ✅ PASS | === Profiling Targets: Softmax(memory-bound) + GEMM(com |
| `aiinfra/daily/week5/day2/kernels/kv_cache.cu` | 1.43 | 0.390 | ✅ PASS | === KV Cache Test === |
| `aiinfra/daily/week5/day4/kernels/paged_attention.cu` | 1.63 | 0.374 | ✅ PASS | === PagedAttention Test === |
| `profiling/leetgpu/argmax.cu` | 1.23 | 0.400 | ✅ PASS | GPU argmax idx = 524288 (expected 524288) PASS |
| `profiling/leetgpu/histogram.cu` | 1.22 | 0.460 | ✅ PASS | Global atomic: 0.106 ms |
| `profiling/leetgpu/matrix-addition.cu` | 1.26 | 1.081 | ✅ PASS | Matrix Addition PASS |
| `profiling/leetgpu/matrix-multiplication.cu` | 1.33 | 0.447 | ✅ PASS | Naive:  0.185 ms (1447.8 GFLOPS) |
| `profiling/leetgpu/matrix-transpose.cu` | 1.23 | 0.517 | ✅ PASS | Naive:  0.176 ms (190.2 GB/s) |
| `profiling/leetgpu/reduction.cu` | 1.22 | 0.453 | ✅ PASS | GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS |
| `profiling/leetgpu/relu.cu` | 1.21 | 0.405 | ✅ PASS | block=  32  time=0.088 ms  bw=95.7 GB/s |
| `profiling/leetgpu/softmax.cu` | 1.30 | 0.442 | ✅ PASS | Three-pass: 2.725 ms |
| `profiling/leetgpu/vector-add.cu` | 1.19 | 0.415 | ✅ PASS | Result: PASS |
| `profiling/week1/day1/hello_gpu.cu` | 1.16 | 0.383 | ✅ PASS | Launching kernel: grid=(2,2,1), block=(8,1,1), total_th |
| `profiling/week1/day1/vector_add_blocksize.cu` | 1.23 | 0.501 | ✅ PASS | === Vector Add: block size 性能对比 === |
| `profiling/week1/day2/occupancy_test.cu` | 1.24 | 0.365 | ✅ PASS | === Kernel Attributes === |
| `profiling/week1/day4/bandwidth.cu` | 1.27 | 1.913 | ✅ PASS | === Coalesced vs Stride Bandwidth Benchmark === |
| `profiling/week1/day4/transpose.cu` | 1.21 | 0.409 | ✅ PASS | Transpose correctness: PASS |
| `profiling/week1/day4/transpose_tiles.cu` | 1.27 | 0.453 | ✅ PASS | === Transpose with Different Tile Sizes === |
| `profiling/week1/day5/bank_conflict.cu` | 1.22 | 0.372 | ✅ PASS | Bank conflict kernels finished. Use ncu to compare metr |
| `profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu` | 1.33 | 0.389 | ✅ PASS | Naive:  0.166 ms (1618.8 GFLOPS) |
| `profiling/week2/day1/warp_reduce.cu` | 1.24 | 0.477 | ✅ PASS | === Warp Shuffle Block Reduce === |
| `profiling/week2/day4/register_blocking_gemm.cu` | 2.12 | 1.566 | ✅ PASS | === Register Blocking GEMM === |
| `profiling/week2/day4/softmax_profile.cu` | 1.32 | 0.369 | ✅ PASS | === Softmax Profiling (row-level, M=128, N=1024) === |
| `profiling/week2/day5/flash_attention.cu` | 2.62 | 0.450 | ✅ PASS | === FlashAttention Simplified Forward === |
| `profiling/week2/day6/histogram.cu` | 1.22 | 0.472 | ✅ PASS | === Histogram: Global atomic vs Shared memory === |
| `profiling/week2/day6/integrated_gemm.cu` | 2.15 | 4.304 | ✅ PASS | === Integrated GEMM (Warp Shuffle + Register Blocking + |
| `profiling/week2/day7/block_reduce_timed.cu` | 1.23 | 0.444 | ✅ PASS | === Block Reduce (Warp Shuffle + 两级归约) === |
| `profiling/week2/day7/gemm_timed.cu` | 2.13 | 1.567 | ✅ PASS | === Register Blocking GEMM (手撕验收) === |
| `profiling/week3/day2/softmax_layernorm.cu` | 1.35 | 0.372 | ✅ PASS | === Softmax + LayerNorm Kernel Test === |
| `profiling/week3/day2/softmax_layernorm_dscan.cu` | 1.32 | 0.391 | ✅ PASS | === Softmax + LayerNorm D-scan (memory-bound scale law) |
| `profiling/week3/day3/softmax_layernorm_opt.cu` | 1.54 | 0.491 | ✅ PASS | === Softmax + LayerNorm Optimization Comparison === |
| `profiling/week3/day3/warp_vs_block_dscan.cu` | 1.37 | 0.589 | ✅ PASS | === Softmax: warp-level vs block-level D-scan === |

### 源文件详细输出

#### `aiinfra/daily/week1/day1/exercise/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day1/exercise/hello_gpu /root/aiinfra_run/aiinfra/daily/week1/day1/exercise/hello_gpu.cu`
- **编译耗时**：1.20s，**运行耗时**：0.454s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day1/kernels/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day1/kernels/hello_gpu /root/aiinfra_run/aiinfra/daily/week1/day1/kernels/hello_gpu.cu`
- **编译耗时**：1.18s，**运行耗时**：0.374s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day2/exercise/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/occupancy_test /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/occupancy_test.cu`
- **编译耗时**：1.26s，**运行耗时**：0.391s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day2/exercise/occupancy_test_b.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/occupancy_test_b /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/occupancy_test_b.cu`
- **编译耗时**：1.25s，**运行耗时**：0.369s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day2/exercise/register_spill.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/register_spill /root/aiinfra_run/aiinfra/daily/week1/day2/exercise/register_spill.cu`
- **编译耗时**：1.57s，**运行耗时**：0.439s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Register Spill Demo ===
Launching: grid=8192, block=128, total_threads=1048576
Done. Use the command below to check spilling:
  nvcc -Xptxas -v week1/day2/exercise/register_spill.cu
Look for 'spill stores' and 'spill loads' in the output.
```
</details>

#### `aiinfra/daily/week1/day2/kernels/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day2/kernels/occupancy_test /root/aiinfra_run/aiinfra/daily/week1/day2/kernels/occupancy_test.cu`
- **编译耗时**：1.24s，**运行耗时**：0.431s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day3/exercise/mini_device_query.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day3/exercise/mini_device_query /root/aiinfra_run/aiinfra/daily/week1/day3/exercise/mini_device_query.cu`
- **编译耗时**：1.18s，**运行耗时**：0.135s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day3/exercise/occupancy_verify.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day3/exercise/occupancy_verify /root/aiinfra_run/aiinfra/daily/week1/day3/exercise/occupancy_verify.cu`
- **编译耗时**：1.27s，**运行耗时**：0.439s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day4/kernels/transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day4/kernels/transpose /root/aiinfra_run/aiinfra/daily/week1/day4/kernels/transpose.cu`
- **编译耗时**：1.22s，**运行耗时**：0.421s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Transpose correctness: PASS
```
</details>

#### `aiinfra/daily/week1/day5/kernels/bank_conflict.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week1/day5/kernels/bank_conflict /root/aiinfra_run/aiinfra/daily/week1/day5/kernels/bank_conflict.cu`
- **编译耗时**：1.23s，**运行耗时**：0.372s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```
</details>

#### `aiinfra/daily/week3/day2/kernels/softmax_layernorm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week3/day2/kernels/softmax_layernorm /root/aiinfra_run/aiinfra/daily/week3/day2/kernels/softmax_layernorm.cu`
- **编译耗时**：1.37s，**运行耗时**：0.442s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week3/day3/kernels/softmax_layernorm_opt.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week3/day3/kernels/softmax_layernorm_opt /root/aiinfra_run/aiinfra/daily/week3/day3/kernels/softmax_layernorm_opt.cu`
- **编译耗时**：1.55s，**运行耗时**：0.488s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week3/day4/kernels/attention_naive.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week3/day4/kernels/attention_naive /root/aiinfra_run/aiinfra/daily/week3/day4/kernels/attention_naive.cu`
- **编译耗时**：1.42s，**运行耗时**：0.784s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week3/day5/kernels/softmax_layernorm_ext.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week3/day5/kernels/softmax_layernorm_ext /root/aiinfra_run/aiinfra/daily/week3/day5/kernels/softmax_layernorm_ext.cu`
- **编译耗时**：1.36s，**运行耗时**：0.454s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week3/day6/kernels/profiling_targets.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week3/day6/kernels/profiling_targets /root/aiinfra_run/aiinfra/daily/week3/day6/kernels/profiling_targets.cu`
- **编译耗时**：1.32s，**运行耗时**：0.451s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week5/day2/kernels/kv_cache.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week5/day2/kernels/kv_cache /root/aiinfra_run/aiinfra/daily/week5/day2/kernels/kv_cache.cu`
- **编译耗时**：1.43s，**运行耗时**：0.390s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week5/day4/kernels/paged_attention.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/aiinfra/daily/week5/day4/kernels/paged_attention /root/aiinfra_run/aiinfra/daily/week5/day4/kernels/paged_attention.cu`
- **编译耗时**：1.63s，**运行耗时**：0.374s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/leetgpu/argmax.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/argmax /root/aiinfra_run/profiling/leetgpu/argmax.cu`
- **编译耗时**：1.23s，**运行耗时**：0.400s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU argmax idx = 524288 (expected 524288) PASS
```
</details>

#### `profiling/leetgpu/histogram.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/histogram /root/aiinfra_run/profiling/leetgpu/histogram.cu`
- **编译耗时**：1.22s，**运行耗时**：0.460s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Global atomic: 0.106 ms
Shared privat: 0.020 ms
Speedup: 5.34x
```
</details>

#### `profiling/leetgpu/matrix-addition.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-addition /root/aiinfra_run/profiling/leetgpu/matrix-addition.cu`
- **编译耗时**：1.26s，**运行耗时**：1.081s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Matrix Addition PASS
```
</details>

#### `profiling/leetgpu/matrix-multiplication.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-multiplication /root/aiinfra_run/profiling/leetgpu/matrix-multiplication.cu`
- **编译耗时**：1.33s，**运行耗时**：0.447s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Naive:  0.185 ms (1447.8 GFLOPS)
Tiled:  0.052 ms (5181.4 GFLOPS)
Speedup: 3.58x
Tiled (no bank conflict): 0.068 ms (3962.5 GFLOPS)
Speedup vs Tiled: 0.76x
```
</details>

#### `profiling/leetgpu/matrix-transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/matrix-transpose /root/aiinfra_run/profiling/leetgpu/matrix-transpose.cu`
- **编译耗时**：1.23s，**运行耗时**：0.517s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Naive:  0.176 ms (190.2 GB/s)
Shared: 0.027 ms (1238.0 GB/s)
Speedup: 6.51x
```
</details>

#### `profiling/leetgpu/reduction.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/reduction /root/aiinfra_run/profiling/leetgpu/reduction.cu`
- **编译耗时**：1.22s，**运行耗时**：0.453s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS
Time: 0.131 ms (127.7 GB/s)
```
</details>

#### `profiling/leetgpu/relu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/relu /root/aiinfra_run/profiling/leetgpu/relu.cu`
- **编译耗时**：1.21s，**运行耗时**：0.405s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/leetgpu/softmax.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/softmax /root/aiinfra_run/profiling/leetgpu/softmax.cu`
- **编译耗时**：1.30s，**运行耗时**：0.442s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Three-pass: 2.725 ms
Online:      2.746 ms
Speedup:     0.99x
```
</details>

#### `profiling/leetgpu/vector-add.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/leetgpu/vector-add /root/aiinfra_run/profiling/leetgpu/vector-add.cu`
- **编译耗时**：1.19s，**运行耗时**：0.415s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Result: PASS
Time: 0.062 ms (203.84 GB/s bandwidth)
```
</details>

#### `profiling/week1/day1/hello_gpu.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day1/hello_gpu /root/aiinfra_run/profiling/week1/day1/hello_gpu.cu`
- **编译耗时**：1.16s，**运行耗时**：0.383s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week1/day1/vector_add_blocksize.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day1/vector_add_blocksize /root/aiinfra_run/profiling/week1/day1/vector_add_blocksize.cu`
- **编译耗时**：1.23s，**运行耗时**：0.501s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week1/day2/occupancy_test.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day2/occupancy_test /root/aiinfra_run/profiling/week1/day2/occupancy_test.cu`
- **编译耗时**：1.24s，**运行耗时**：0.365s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week1/day4/bandwidth.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/bandwidth /root/aiinfra_run/profiling/week1/day4/bandwidth.cu`
- **编译耗时**：1.27s，**运行耗时**：1.913s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week1/day4/transpose.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/transpose /root/aiinfra_run/profiling/week1/day4/transpose.cu`
- **编译耗时**：1.21s，**运行耗时**：0.409s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Transpose correctness: PASS
```
</details>

#### `profiling/week1/day4/transpose_tiles.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day4/transpose_tiles /root/aiinfra_run/profiling/week1/day4/transpose_tiles.cu`
- **编译耗时**：1.27s，**运行耗时**：0.453s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week1/day5/bank_conflict.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day5/bank_conflict /root/aiinfra_run/profiling/week1/day5/bank_conflict.cu`
- **编译耗时**：1.22s，**运行耗时**：0.372s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```
</details>

#### `profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week1/day6/matrix_multiplication/matrix_multiplication /root/aiinfra_run/profiling/week1/day6/matrix_multiplication/matrix_multiplication.cu`
- **编译耗时**：1.33s，**运行耗时**：0.389s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Naive:  0.166 ms (1618.8 GFLOPS)
Tiled:  0.053 ms (5071.7 GFLOPS)
Speedup: 3.13x
Tiled (no bank conflict): 0.069 ms (3914.4 GFLOPS)
Speedup vs Tiled: 0.77x
```
</details>

#### `profiling/week2/day1/warp_reduce.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day1/warp_reduce /root/aiinfra_run/profiling/week2/day1/warp_reduce.cu`
- **编译耗时**：1.24s，**运行耗时**：0.477s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day4/register_blocking_gemm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day4/register_blocking_gemm /root/aiinfra_run/profiling/week2/day4/register_blocking_gemm.cu -lcublas`
- **编译耗时**：2.12s，**运行耗时**：1.566s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day4/softmax_profile.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day4/softmax_profile /root/aiinfra_run/profiling/week2/day4/softmax_profile.cu`
- **编译耗时**：1.32s，**运行耗时**：0.369s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day5/flash_attention.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day5/flash_attention /root/aiinfra_run/profiling/week2/day5/flash_attention.cu`
- **编译耗时**：2.62s，**运行耗时**：0.450s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day6/histogram.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day6/histogram /root/aiinfra_run/profiling/week2/day6/histogram.cu`
- **编译耗时**：1.22s，**运行耗时**：0.472s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day6/integrated_gemm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day6/integrated_gemm /root/aiinfra_run/profiling/week2/day6/integrated_gemm.cu -lcublas`
- **编译耗时**：2.15s，**运行耗时**：4.304s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day7/block_reduce_timed.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day7/block_reduce_timed /root/aiinfra_run/profiling/week2/day7/block_reduce_timed.cu`
- **编译耗时**：1.23s，**运行耗时**：0.444s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week2/day7/gemm_timed.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week2/day7/gemm_timed /root/aiinfra_run/profiling/week2/day7/gemm_timed.cu -lcublas`
- **编译耗时**：2.13s，**运行耗时**：1.567s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week3/day2/softmax_layernorm.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day2/softmax_layernorm /root/aiinfra_run/profiling/week3/day2/softmax_layernorm.cu`
- **编译耗时**：1.35s，**运行耗时**：0.372s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week3/day2/softmax_layernorm_dscan.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day2/softmax_layernorm_dscan /root/aiinfra_run/profiling/week3/day2/softmax_layernorm_dscan.cu`
- **编译耗时**：1.32s，**运行耗时**：0.391s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week3/day3/softmax_layernorm_opt.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day3/softmax_layernorm_opt /root/aiinfra_run/profiling/week3/day3/softmax_layernorm_opt.cu`
- **编译耗时**：1.54s，**运行耗时**：0.491s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `profiling/week3/day3/warp_vs_block_dscan.cu`

- **编译命令**：`nvcc -O3 -arch=sm_120 -o /root/aiinfra_run/profiling/week3/day3/warp_vs_block_dscan /root/aiinfra_run/profiling/week3/day3/warp_vs_block_dscan.cu`
- **编译耗时**：1.37s，**运行耗时**：0.589s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

## 二、Markdown 代码块执行结果

| 来源文件 | 编译(s) | 运行(s) | 状态 | 关键输出 |
|----------|--------|--------|------|----------|
| `aiinfra/daily/week1/day1/README.md` | 2.60 | 1.641 | ✅ PASS | Launching kernel: grid=(2,2,1), block=(4,2,1) |
| `aiinfra/daily/week1/day2/README.md` | 2.52 | 1.629 | ✅ PASS | === Kernel Attributes === |
| `aiinfra/daily/week1/day3/README.md` | 2.35 | 1.435 | ✅ PASS | Detected 1 CUDA device(s) |
| `aiinfra/daily/week1/day3/README.md` | 2.54 | 2.368 | ✅ PASS | Matrix Addition PASS |
| `aiinfra/daily/week1/day3/README.md` | 2.46 | 1.674 | ✅ PASS | Best device: 0 |
| `aiinfra/daily/week1/day4/README.md` | 2.44 | 1.809 | ✅ PASS | Naive transpose: PASS |
| `aiinfra/daily/week1/day5/README.md` | 2.55 | 1.723 | ✅ PASS | Bank conflict kernels finished. Use ncu to compare metr |
| `aiinfra/daily/week1/day7/README.md` | 2.60 | 2.323 | ✅ PASS | Matrix Addition PASS |
| `aiinfra/daily/week1/day7/README.md` | 2.48 | 1.710 | ✅ PASS | GEMM 512x512x512 done |
| `aiinfra/daily/week2/day1/README.md` | 2.62 | 1.684 | ✅ PASS | === Warp Shuffle Block Reduce === |
| `aiinfra/daily/week2/day2/README.md` | 3.57 | 2.870 | ✅ PASS | === Register Blocking GEMM === |
| `aiinfra/daily/week2/day3/README.md` | 2.57 | 2.457 | ✅ PASS | === Multi-Stream Overlap Pipeline === |
| `aiinfra/daily/week2/day5/README.md` | 3.33 | 1.846 | ✅ PASS | === FlashAttention Simplified Forward === |
| `aiinfra/daily/week2/day6/README.md` | 3.70 | 5.711 | ✅ PASS | === Integrated GEMM (Warp Shuffle + Register Blocking + |
| `aiinfra/daily/week2/day7/README.md` | 2.45 | 1.792 | ✅ PASS | GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS |
| `aiinfra/daily/week3/day5/README.md` | - | - | ⏭️ SKIPPED |  |
| `aiinfra/daily/week3/day6/README.md` | 2.51 | 1.855 | ✅ PASS | RMSNorm: maxDiff = 1.43e-06 (PASS) |
| `aiinfra/daily/week4/day2/README.md` | 5.35 | 1.885 | ✅ PASS | === FlashAttention v2 Forward Kernel === |
| `aiinfra/daily/week4/day3/README.md` | 2.75 | 2.075 | ✅ PASS | Dot Product = 0.131072 (expected 1.048576) |
| `aiinfra/daily/plan/learning_plan_week2_expanded.md` | 2.70 | 2.219 | ✅ PASS | === Warp Shuffle Block Reduce === |
| `aiinfra/daily/plan/learning_plan_week2_expanded.md` | 3.58 | 3.015 | ✅ PASS | === Register Blocking GEMM === |
| `aiinfra/daily/plan/learning_plan_week2_expanded.md` | 2.61 | 2.479 | ✅ PASS | === Multi-Stream Overlap Pipeline === |
| `aiinfra/daily/plan/learning_plan_week2_expanded.md` | 3.40 | 2.005 | ✅ PASS | === FlashAttention Simplified Forward === |
| `aiinfra/daily/plan/learning_plan_week2_expanded.md` | 3.79 | 5.693 | ✅ PASS | === Integrated GEMM (Warp Shuffle + Register Blocking + |
| `aiinfra/daily/plan/learning_plan_week3_expanded.md` | 2.65 | 1.850 | ✅ PASS | === Softmax + LayerNorm Kernel Test === |
| `aiinfra/daily/plan/learning_plan_week3_expanded.md` | 2.93 | 2.183 | ✅ PASS | === Standard Attention Forward (naive, materialize S/P) |
| `aiinfra/daily/plan/learning_plan_week4_expanded.md` | 5.46 | 2.127 | ✅ PASS | === FlashAttention v2 Forward Kernel === |
| `aiinfra/daily/plan/learning_plan_week5_expanded.md` | 3.01 | 1.854 | ✅ PASS | === KV Cache Test === |
| `leetgpu/week1/day1/leetgpu-vector-addition-solution.md` | 2.80 | 2.826 | ✅ PASS | N = 25000000  (100.0 MB per vector) |
| `leetgpu/week1/day2/leetgpu-relu-solution.md` | 2.73 | 2.446 | ✅ PASS | N = 25000000  (100.0 MB per vector) |
| `leetgpu/week1/day3/leetgpu-matrix-addition-solution.md` | 3.03 | 2.576 | ✅ PASS | M=4096, N=4096  (67.1 MB per matrix) |
| `leetgpu/week1/day4/leetgpu-matrix-transpose-solution.md` | 2.54 | 2.172 | ✅ PASS | M=4096 N=4096 (67.1 MB) |
| `leetgpu/week1/day6/leetgpu-matrix-multiplication-solution.md` | 2.95 | 3.054 | ✅ PASS | A: 8192x6144, B: 6144x4096, C: 8192x4096 |
| `leetgpu/week2/day1/leetgpu-prefix-sum-solution.md` | 2.55 | 2.048 | ✅ PASS | N = 16777216  (67.1 MB) |
| `leetgpu/week2/day2/leetgpu-gemm-solution.md` | 3.49 | 2.288 | ✅ PASS | A:1024x1024 B:1024x1024 C:1024x1024  FLOPs=2.15 GFLOP |
| `leetgpu/week2/day3/leetgpu-2d-convolution-solution.md` | 2.75 | 2.478 | ✅ PASS | input: 4096x4096  kernel: 3x3  output: 4094x4094 |
| `leetgpu/week2/day4/leetgpu-softmax-solution.md` | 2.82 | 4.991 | ✅ PASS | M=128, D=8192  (4.2 MB) |
| `leetgpu/week2/day5/leetgpu-softmax-attention-solution.md` | 3.25 | 1.871 | ✅ PASS | N=1024 d=64  QKV=0.79 MB  S/P(naive)=4.19 MB each |
| `leetgpu/week2/day6/leetgpu-histogramming-solution.md` | 2.61 | 1.877 | ✅ PASS | N = 10000000, B = 256  (40.0 MB input) |
| `leetgpu/week3/day6/leetgpu-rms-normalization-solution.md` | 2.66 | 1.981 | ✅ PASS | M=128, D=8192  (4.2 MB) |
| `leetgpu/week4/day3/leetgpu-dot-product-solution.md` | 2.96 | 1.807 | ✅ PASS | GPU: 244218.9375, CPU: 244166.1562, FAIL |
| `leetgpu/week4/day4/leetgpu-batched-matrix-multiplication-solution.md` | 2.67 | 1.768 | ✅ PASS | batch=4 M=N=K=64, PASS |
| `leetgpu/week4/day5/leetgpu-matrix-copy-solution.md` | 2.72 | 2.176 | ✅ PASS | M=4096, N=4096  (67.1 MB per matrix) |
| `leetgpu/week4/day6/leetgpu-multi-head-attention-solution.md` | 3.00 | 2.015 | ✅ PASS | B=2 H=4 N=512 d=64  QKV=3.15 MB  S/P(std)=8.39 MB each |
| `leetgpu/week5/day1/leetgpu-int8-kv-cache-attention-solution.md` | 2.90 | 2.660 | ✅ PASS | H=32 L=8192 d=128  int8 KV=67.11 MB  (fp32 KV would be  |
| `leetgpu/week5/day2/leetgpu-grouped-query-attention-solution.md` | 2.86 | 11.749 | ✅ PASS | nq=32 nkv=8 group=4 S=1024 d=128 |
| `leetgpu/week5/day3/leetgpu-speculative-decoding-verification-solution.md` | 3.03 | 2.447 | ✅ PASS | B=64 T=8 V=32768 |
| `leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md` | 2.85 | 5.703 | ✅ PASS | M=5000 d=128 |
| `leetgpu/week5/day5/leetgpu-token-embedding-layer-solution.md` | 3.44 | 2.415 | ✅ PASS | B=32 T=512 V=30000 P=2048 D=768 |
| `leetgpu/week5/day6/leetgpu-weight-dequantization-solution.md` | 3.05 | 3.870 | ✅ PASS | M=8192 N=8192 T=128  s_rows=64 s_cols=64 |
| `leetgpu/week6/day2/leetgpu-max-subarray-sum-solution.md` | 2.94 | 2.011 | ✅ PASS | GPU: 2144, CPU: 2144, PASS |
| `leetgpu/week6/day3/leetgpu-stream-compaction-solution.md` | 2.97 | 1.914 | ✅ PASS | GPU count=11123, CPU count=660116, FAIL |
| `leetgpu/week6/day4/leetgpu-segmented-prefix-sum-solution.md` | 2.88 | 1.993 | ✅ PASS | out[0]=0 (cpu=0) ✓ |
| `leetgpu/week6/day6/leetgpu-top-k-selection-solution.md` | 3.09 | 2.015 | ✅ PASS | Sorted: -2147483648 -2147483648 -2147483648 -2147483648 |
| `leetgpu/week7/day1/leetgpu-matrix-copy-solution.md` | 2.87 | 2.365 | ✅ PASS | Matrix 4096x4096 copy: PASS |
| `leetgpu/week7/day2/leetgpu-vector-reversal-solution.md` | 2.70 | 2.201 | ✅ PASS | N = 10000000  (40.0 MB) |
| `leetgpu/week7/day3/leetgpu-scalar-multiply-solution.md` | 2.77 | 2.110 | ✅ PASS | N = 10000000  (40.0 MB), alpha = 2.000000 |
| `leetgpu/week7/day5/leetgpu-element-reversal-solution.md` | 2.85 | 1.991 | ✅ PASS | N = 10000000  (40.0 MB) |
| `leetgpu/week7/day7/leetgpu-matrix-addition-solution.md` | 2.49 | 2.293 | ✅ PASS | M=4096 N=4096 (67.1 MB per matrix) |

### Markdown 代码块详细输出

#### `aiinfra/daily/week1/day1/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day1_README_block0_e17e239380a9.cu -o aiinfra_week1_day1_README_block0_e17e239380a9 `
- **编译耗时**：2.60s，**运行耗时**：1.641s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Launching kernel: grid=(2,2,1), block=(4,2,1)
Total threads: 32
block=(0,0,0), thread=(0,0,0), global_tid=0
block=(0,0,0), thread=(1,0,0), global_tid=1
block=(0,0,0), thread=(2,0,0), global_tid=2
block=(0,0,0), thread=(3,0,0), global_tid=3
block=(0,0,0), thread=(0,1,0), global_tid=0
block=(0,0,0), thread=(1,1,0), global_tid=1
block=(0,0,0), thread=(2,1,0), global_tid=2
block=(0,0,0), thread=(3,1,0), global_tid=3
block=(1,0,0), thread=(0,0,0), global_tid=4
block=(1,0,0), thread=(1,0,0), global_tid=5
block=(1,0,0), thread=(2,0,0), global_tid=6
block=(1,0,0), thread=(3,0,0), global_tid=7
block=(1,0,0), thread=(0,1,0), global_tid=4
block=(1,0,0), thread=(1,1,0), global_tid=5
block=(1,0,0), thread=(2,1,0), global_tid=6
block=(1,0,0), thread=(3,1,0), global_tid=7
block=(0,1,0), thread=(0,0,0), global_tid=0
block=(0,1,0), thread=(1,0,0), global_tid=1
block=(0,1,0), thread=(2,0,0), global_tid=2
block=(0,1,0), thread=(3,0,0), global_tid=3
block=(0,1,0), thread=(0,1,0), global_tid=0
block=(0,1,0), thread=(1,1,0), global_tid=1
block=(0,1,0), thread=(2,1,0), global_tid=2
block=(0,1,0), thread=(3,1,0), global_tid=3
block=(1,1,0), thread=(0,0,0), global_tid=4
block=(1,1,0), thread=(1,0,0), global_tid=5
block=(1,1,0), thread=(2,0,0), global_tid=6
block=(1,1,0), thread=(3,0,0), global_tid=7
block=(1,1,0), thread=(0,1,0), global_tid=4
block=(1,1,0), thread=(1,1,0), global_tid=5
block=(1,1,0), thread=(2,1,0), global_tid=6
block=(1,1,0), thread=(3,1,0), global_tid=7
```
</details>

#### `aiinfra/daily/week1/day2/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day2_README_block1_559b2011f76a.cu -o aiinfra_week1_day2_README_block1_559b2011f76a `
- **编译耗时**：2.52s，**运行耗时**：1.629s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day3/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day3_README_block2_2a66a41b9a37.cu -o aiinfra_week1_day3_README_block2_2a66a41b9a37 `
- **编译耗时**：2.35s，**运行耗时**：1.435s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

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

#### `aiinfra/daily/week1/day3/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day3_README_block3_c2397c043426.cu -o aiinfra_week1_day3_README_block3_c2397c043426 `
- **编译耗时**：2.54s，**运行耗时**：2.368s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Matrix Addition PASS
```
</details>

#### `aiinfra/daily/week1/day3/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day3_README_block4_da80c224337d.cu -o aiinfra_week1_day3_README_block4_da80c224337d `
- **编译耗时**：2.46s，**运行耗时**：1.674s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Best device: 0
```
</details>

#### `aiinfra/daily/week1/day4/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day4_README_block5_586138c24c03.cu -o aiinfra_week1_day4_README_block5_586138c24c03 `
- **编译耗时**：2.44s，**运行耗时**：1.809s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Naive transpose: PASS
```
</details>

#### `aiinfra/daily/week1/day5/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day5_README_block6_035aefc95983.cu -o aiinfra_week1_day5_README_block6_035aefc95983 `
- **编译耗时**：2.55s，**运行耗时**：1.723s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```
</details>

#### `aiinfra/daily/week1/day7/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day7_README_block7_cc0d6bddb8a8.cu -o aiinfra_week1_day7_README_block7_cc0d6bddb8a8 `
- **编译耗时**：2.60s，**运行耗时**：2.323s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Matrix Addition PASS
```
</details>

#### `aiinfra/daily/week1/day7/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week1_day7_README_block8_0a048eb71347.cu -o aiinfra_week1_day7_README_block8_0a048eb71347 `
- **编译耗时**：2.48s，**运行耗时**：1.710s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GEMM 512x512x512 done
```
</details>

#### `aiinfra/daily/week2/day1/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day1_README_block9_5f4f9750a687.cu -o aiinfra_week2_day1_README_block9_5f4f9750a687 `
- **编译耗时**：2.62s，**运行耗时**：1.684s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Warp Shuffle Block Reduce ===
Array size: 4194304 (16.00 MB)
GPU Sum: 20971.052734
CPU Sum: 20971.052734
Diff: 0.000000 (PASS)
Time: 0.179 ms (93.66 GB/s bandwidth)
```
</details>

#### `aiinfra/daily/week2/day2/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day2_README_block10_d5a4eb39127d.cu -o aiinfra_week2_day2_README_block10_d5a4eb39127d -lcublas`
- **编译耗时**：3.57s，**运行耗时**：2.870s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Register Blocking GEMM ===
Parameters: BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
M          N          K          Our(ms)      cuBLAS(ms)   Percent   
------------------------------------------------------------
1024       1024       1024       0.266        0.091        34.2     % PASS
2048       2048       2048       0.667        0.304        45.5     % PASS
4096       4096       4096       6.113        2.152        35.2     % PASS
```
</details>

#### `aiinfra/daily/week2/day3/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day3_README_block11_cc0a8b489c4a.cu -o aiinfra_week2_day3_README_block11_cc0a8b489c4a `
- **编译耗时**：2.57s，**运行耗时**：2.457s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Multi-Stream Overlap Pipeline ===
Total size: 16777216 (64.00 MB)
Chunk size: 262144 (1.00 MB)
Num chunks: 64, Num streams: 4

Running sequential version...
Sequential: 10.047 ms

Running multi-stream version (nStreams=4)...
Multi-Stream: 7.456 ms

=== Performance Summary ===
Sequential: 10.047 ms
Multi-Stream: 7.456 ms
Speedup: 1.35x
Result check: PASS
```
</details>

#### `aiinfra/daily/week2/day5/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day5_README_block12_c0978e2bf1b3.cu -o aiinfra_week2_day5_README_block12_c0978e2bf1b3 `
- **编译耗时**：3.33s，**运行耗时**：1.846s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== FlashAttention Simplified Forward ===
Config: N=256, D=64, batch=1, heads=1
SRAM usage per block: 40.00 KB
Grid: (4, 1, 1), Block: (32, 4)
GPU Time: 0.988 ms
Result check: PASS
```
</details>

#### `aiinfra/daily/week2/day6/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day6_README_block13_36444672eda0.cu -o aiinfra_week2_day6_README_block13_36444672eda0 -lcublas`
- **编译耗时**：3.70s，**运行耗时**：5.711s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===
BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256

M        N        K        Our(ms)    cuBLAS(ms) GFLOPS     Percent 
----------------------------------------------------------------
1024     1024     1024     0.144      0.094      14943.0    65.2   % PASS
2048     2048     2048     0.429      0.303      40002.3    70.5   % PASS
4096     4096     4096     3.255      2.156      42217.6    66.2   % PASS
8192     8192     8192     23.157     16.025     47481.2    69.2   % PASS
```
</details>

#### `aiinfra/daily/week2/day7/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week2_day7_README_block14_fa3b7756d465.cu -o aiinfra_week2_day7_README_block14_fa3b7756d465 `
- **编译耗时**：2.45s，**运行耗时**：1.792s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU=2094779.6250 CPU=2094779.6250 diff=0.000000 PASS
```
</details>

#### `aiinfra/daily/week3/day5/README.md`

- **状态**：⏭️ SKIPPED
- **原因**：depends on PyTorch/torch/extension.h

#### `aiinfra/daily/week3/day6/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week3_day6_README_block16_0632ee28da4c.cu -o aiinfra_week3_day6_README_block16_0632ee28da4c `
- **编译耗时**：2.51s，**运行耗时**：1.855s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
RMSNorm: maxDiff = 1.43e-06 (PASS)
ncu: ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
 dram__throughput.avg.pct_of_peak_sustained_elapsed --kernel-name regex:rmsnorm_kernel ./rmsnorm
```
</details>

#### `aiinfra/daily/week4/day2/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week4_day2_README_block17_160dd5cc5910.cu -o aiinfra_week4_day2_README_block17_160dd5cc5910 `
- **编译耗时**：5.35s，**运行耗时**：1.885s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== FlashAttention v2 Forward Kernel ===
Config: B=2, H=4, N=256, d=64
Tile: Br=64, Bc=64, Threads=256

[B=0, H=0] First head check:
 maxDiff = 1.31e-04 (PASS)
GPU Time: 0.796 ms
```
</details>

#### `aiinfra/daily/week4/day3/README.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 aiinfra_week4_day3_README_block18_cbb7a720bbd4.cu -o aiinfra_week4_day3_README_block18_cbb7a720bbd4 `
- **编译耗时**：2.75s，**运行耗时**：2.075s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Dot Product = 0.131072 (expected 1.048576)
```
</details>

#### `aiinfra/daily/plan/learning_plan_week2_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week2_expanded_block19_351c1af6f3f4.cu -o docs_learning_plan_week2_expanded_block19_351c1af6f3f4 `
- **编译耗时**：2.70s，**运行耗时**：2.219s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Warp Shuffle Block Reduce ===
Array size: 4194304 (16.00 MB)
GPU Sum: 20971.052734
CPU Sum: 20971.052734
Diff: 0.000000 (PASS)
Time: 0.157 ms (107.08 GB/s bandwidth)
```
</details>

#### `aiinfra/daily/plan/learning_plan_week2_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week2_expanded_block20_0fab822c3ec2.cu -o docs_learning_plan_week2_expanded_block20_0fab822c3ec2 -lcublas`
- **编译耗时**：3.58s，**运行耗时**：3.015s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Register Blocking GEMM ===
Parameters: BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
M          N          K          Our(ms)      cuBLAS(ms)   Percent   
------------------------------------------------------------
1024       1024       1024       0.267        0.095        35.5     % PASS
2048       2048       2048       0.671        0.302        44.9     % PASS
4096       4096       4096       6.101        2.151        35.3     % PASS
```
</details>

#### `aiinfra/daily/plan/learning_plan_week2_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week2_expanded_block21_64084c5c36d7.cu -o docs_learning_plan_week2_expanded_block21_64084c5c36d7 `
- **编译耗时**：2.61s，**运行耗时**：2.479s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Multi-Stream Overlap Pipeline ===
Total size: 16777216 (64.00 MB)
Chunk size: 262144 (1.00 MB)
Num chunks: 64, Num streams: 4

Running sequential version...
Sequential: 9.506 ms

Running multi-stream version (nStreams=4)...
Multi-Stream: 7.475 ms

=== Performance Summary ===
Sequential: 9.506 ms
Multi-Stream: 7.475 ms
Speedup: 1.27x
Result check: PASS
```
</details>

#### `aiinfra/daily/plan/learning_plan_week2_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week2_expanded_block22_f7ef611530f3.cu -o docs_learning_plan_week2_expanded_block22_f7ef611530f3 `
- **编译耗时**：3.40s，**运行耗时**：2.005s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== FlashAttention Simplified Forward ===
Config: N=256, D=64, batch=1, heads=1
SRAM usage per block: 40.00 KB
Grid: (4, 1, 1), Block: (32, 4)
GPU Time: 0.970 ms
Result check: PASS
```
</details>

#### `aiinfra/daily/plan/learning_plan_week2_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week2_expanded_block23_ccc2b72fc7d3.cu -o docs_learning_plan_week2_expanded_block23_ccc2b72fc7d3 -lcublas`
- **编译耗时**：3.79s，**运行耗时**：5.693s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===
BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256

M        N        K        Our(ms)    cuBLAS(ms) GFLOPS     Percent 
----------------------------------------------------------------
1024     1024     1024     0.145      0.095      14801.2    65.3   % PASS
2048     2048     2048     0.427      0.301      40200.0    70.4   % PASS
4096     4096     4096     3.258      2.159      42184.9    66.3   % PASS
8192     8192     8192     23.186     16.062     47420.6    69.3   % PASS
```
</details>

#### `aiinfra/daily/plan/learning_plan_week3_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week3_expanded_block24_3682f1df1722.cu -o docs_learning_plan_week3_expanded_block24_3682f1df1722 `
- **编译耗时**：2.65s，**运行耗时**：1.850s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Softmax + LayerNorm Kernel Test ===
Config: M=128, D=1024, threads=256

[Softmax]
 Softmax vs CPU: maxDiff = 4.19e-09 (PASS)
 Time: 0.136 ms
[LayerNorm]
 LayerNorm vs CPU: maxDiff = 1.07e-06 (PASS)
 Time: 0.028 ms
```
</details>

#### `aiinfra/daily/plan/learning_plan_week3_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week3_expanded_block25_81b34bf00a11.cu -o docs_learning_plan_week3_expanded_block25_81b34bf00a11 `
- **编译耗时**：2.93s，**运行耗时**：2.183s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== Standard Attention Forward (naive, materialize S/P) ===
N        S/P size(MB) HBM IO(MB)     Time(ms)     Check     
--------------------------------------------------------
 maxDiff = 1.02e-08 (PASS)
256      0.25         1.25           0.171        PASS      
 maxDiff = 1.30e-08 (PASS)
512      1.00         4.50           0.082        PASS      
 maxDiff = 1.26e-08 (PASS)
1024     4.00         17.00          0.274        PASS      
 maxDiff = 1.12e-08 (PASS)
2048     16.00        66.00          0.809        PASS      

观察要点：
1. S/P size 随 N² 增长（N 翻倍 → size 4x）
2. HBM IO 随 N² 增长（N 翻倍 → IO 4x）
3. Time 近似随 N² 增长（长序列下 O(N²) IO 主导）
4. 用 ncu 验证 dram__bytes_read.sum + dram__bytes_write.sum ≈ 理论 HBM IO
```
</details>

#### `aiinfra/daily/plan/learning_plan_week4_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week4_expanded_block26_a7863d0b9f30.cu -o docs_learning_plan_week4_expanded_block26_a7863d0b9f30 `
- **编译耗时**：5.46s，**运行耗时**：2.127s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== FlashAttention v2 Forward Kernel ===
Config: B=2, H=4, N=256, d=64
Tile: Br=64, Bc=64, Threads=256

[B=0, H=0] First head check:
 maxDiff = 1.19e-03 (FAIL)
GPU Time: 0.855 ms
```
</details>

#### `aiinfra/daily/plan/learning_plan_week5_expanded.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 docs_learning_plan_week5_expanded_block27_84fab14a3d7f.cu -o docs_learning_plan_week5_expanded_block27_84fab14a3d7f `
- **编译耗时**：3.01s，**运行耗时**：1.854s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
=== KV Cache Test ===
Config: layers=2, batch=1, heads=8, max_len=1024, d_head=64
After Round 1 (len=10): seq_len=10
After Round 2 (len=5): seq_len=15
After Round 3 (len=8): seq_len=23
PASS: seq_len = 23 (expected 23)
KV Cache bytes per token: 8192
Max memory usage: 8 MB
```
</details>

#### `leetgpu/week1/day1/leetgpu-vector-addition-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week1_day1_leetgpu-vector-addition-solution_block28_c12d7d0c6dfa.cu -o leetgpu_week1_day1_leetgpu-vector-addition-solution_block28_c12d7d0c6dfa `
- **编译耗时**：2.80s，**运行耗时**：2.826s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 25000000  (100.0 MB per vector)
launch: blocks=680  threads=256  (SM=170)
kernel time: 0.322 ms
verify: PASS  (0 / 25000000 mismatch)
effective bandwidth: 931.5 GB/s
```
</details>

#### `leetgpu/week1/day2/leetgpu-relu-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week1_day2_leetgpu-relu-solution_block29_b86451caffa1.cu -o leetgpu_week1_day2_leetgpu-relu-solution_block29_b86451caffa1 `
- **编译耗时**：2.73s，**运行耗时**：2.446s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 25000000  (100.0 MB per vector)
launch: blocks=680  threads=256  (SM=170)
kernel time: 0.269 ms
verify: PASS  (0 / 25000000 mismatch)
effective bandwidth: 743.0 GB/s
```
</details>

#### `leetgpu/week1/day3/leetgpu-matrix-addition-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week1_day3_leetgpu-matrix-addition-solution_block30_c43f92f2e843.cu -o leetgpu_week1_day3_leetgpu-matrix-addition-solution_block30_c43f92f2e843 `
- **编译耗时**：3.03s，**运行耗时**：2.576s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=4096, N=4096  (67.1 MB per matrix)
launch: blocks=2048 threads=256
kernel time: 0.251 ms
effective bandwidth: 802.1 GB/s
verify: PASS
```
</details>

#### `leetgpu/week1/day4/leetgpu-matrix-transpose-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week1_day4_leetgpu-matrix-transpose-solution_block31_2d13944105b1.cu -o leetgpu_week1_day4_leetgpu-matrix-transpose-solution_block31_2d13944105b1 `
- **编译耗时**：2.54s，**运行耗时**：2.172s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=4096 N=4096 (67.1 MB)
kernel time: 0.231 ms
I/O bandwidth: 580.1 GB/s
PASS
```
</details>

#### `leetgpu/week1/day6/leetgpu-matrix-multiplication-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week1_day6_leetgpu-matrix-multiplication-solution_block32_78338f5efa5d.cu -o leetgpu_week1_day6_leetgpu-matrix-multiplication-solution_block32_78338f5efa5d `
- **编译耗时**：2.95s，**运行耗时**：3.054s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
A: 8192x6144, B: 6144x4096, C: 8192x4096
FLOPs: 412.32 GFLOP
launch: blocks=(128,256) threads=(32,32)
kernel time: 49.658 ms
performance: 8.30 TFLOPS
verify: PASS
```
</details>

#### `leetgpu/week2/day1/leetgpu-prefix-sum-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day1_leetgpu-prefix-sum-solution_block33_d104c48598b5.cu -o leetgpu_week2_day1_leetgpu-prefix-sum-solution_block33_d104c48598b5 `
- **编译耗时**：2.55s，**运行耗时**：2.048s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 16777216  (67.1 MB)
kernel time (three-pass): 0.471 ms
PASS
I/O bandwidth: 284.7 GB/s
```
</details>

#### `leetgpu/week2/day2/leetgpu-gemm-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day2_leetgpu-gemm-solution_block34_2dab209b30a4.cu -o leetgpu_week2_day2_leetgpu-gemm-solution_block34_2dab209b30a4 -lcublas`
- **编译耗时**：3.49s，**运行耗时**：2.288s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
A:1024x1024 B:1024x1024 C:1024x1024  FLOPs=2.15 GFLOP
launch: blocks=(8,8) threads=(16,16)

[Register Blocking] 0.159 ms  13.50 TFLOPS
[cuBLAS           ] 0.053 ms  40.66 TFLOPS
[ratio            ] 33.2% of cuBLAS
verify: PASS
```
</details>

#### `leetgpu/week2/day3/leetgpu-2d-convolution-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day3_leetgpu-2d-convolution-solution_block35_642817310b8f.cu -o leetgpu_week2_day3_leetgpu-2d-convolution-solution_block35_642817310b8f `
- **编译耗时**：2.75s，**运行耗时**：2.478s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
input: 4096x4096  kernel: 3x3  output: 4094x4094
launch: blocks=(256,256)  threads=(16,16)
kernel time: 0.159 ms
verify: PASS
effective bandwidth: 841.3 GB/s
```
</details>

#### `leetgpu/week2/day4/leetgpu-softmax-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day4_leetgpu-softmax-solution_block36_c951f734fad2.cu -o leetgpu_week2_day4_leetgpu-softmax-solution_block36_c951f734fad2 `
- **编译耗时**：2.82s，**运行耗时**：4.991s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=128, D=8192  (4.2 MB)
kernel time: 0.094 ms
effective bandwidth: 177.9 GB/s
max diff: 4.66e-10 (PASS)
```
</details>

#### `leetgpu/week2/day5/leetgpu-softmax-attention-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day5_leetgpu-softmax-attention-solution_block37_b7b3fb24c90b.cu -o leetgpu_week2_day5_leetgpu-softmax-attention-solution_block37_b7b3fb24c90b `
- **编译耗时**：3.25s，**运行耗时**：1.871s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N=1024 d=64  QKV=0.79 MB  S/P(naive)=4.19 MB each
naive: 0.480 ms   fused: 1.395 ms
naive max diff: 0.00e+00 (PASS)
fused max diff: 0.00e+00 (PASS)
est. DRAM: naive=0.55 GB  fused=0.54 GB  (fused 省 S/P=0.02 GB)
```
</details>

#### `leetgpu/week2/day6/leetgpu-histogramming-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week2_day6_leetgpu-histogramming-solution_block38_e26f7ce4f318.cu -o leetgpu_week2_day6_leetgpu-histogramming-solution_block38_e26f7ce4f318 `
- **编译耗时**：2.61s，**运行耗时**：1.877s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 10000000, B = 256  (40.0 MB input)
blocks = 680, threads/block = 256
[privatized] time: 0.194 ms  max_err: 0  PASS
[naive]       time: 0.081 ms  max_err: 38961  FAIL  speedup: 0.42x
read bandwidth (privatized): 206.2 GB/s
```
</details>

#### `leetgpu/week3/day6/leetgpu-rms-normalization-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week3_day6_leetgpu-rms-normalization-solution_block39_87ac0995d66f.cu -o leetgpu_week3_day6_leetgpu-rms-normalization-solution_block39_87ac0995d66f `
- **编译耗时**：2.66s，**运行耗时**：1.981s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=128, D=8192  (4.2 MB)
kernel time: 0.119 ms
effective bandwidth: 106.3 GB/s
max diff: 2.62e-06 (PASS)
```
</details>

#### `leetgpu/week4/day3/leetgpu-dot-product-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week4_day3_leetgpu-dot-product-solution_block40_f16240a6203f.cu -o leetgpu_week4_day3_leetgpu-dot-product-solution_block40_f16240a6203f `
- **编译耗时**：2.96s，**运行耗时**：1.807s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU: 244218.9375, CPU: 244166.1562, FAIL
```
</details>

#### `leetgpu/week4/day4/leetgpu-batched-matrix-multiplication-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week4_day4_leetgpu-batched-matrix-multiplication-solution_block41_987af813a1e7.cu -o leetgpu_week4_day4_leetgpu-batched-matrix-multiplication-solution_block41_987af813a1e7 `
- **编译耗时**：2.67s，**运行耗时**：1.768s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
batch=4 M=N=K=64, PASS
```
</details>

#### `leetgpu/week4/day5/leetgpu-matrix-copy-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week4_day5_leetgpu-matrix-copy-solution_block42_001745f0c3c4.cu -o leetgpu_week4_day5_leetgpu-matrix-copy-solution_block42_001745f0c3c4 `
- **编译耗时**：2.72s，**运行耗时**：2.176s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=4096, N=4096  (67.1 MB per matrix)
launch: blocks=680  threads=256  (SM=170)

--- timing (ms) / bandwidth 2x bytes (GB/s) ---
scalar      : 0.296 ms / 452.7
float4      : 0.082 ms / 1634.6
cudaMemcpy  : 0.079 ms / 1688.5
verify: PASS  (0 / 16777216 mismatch)
```
</details>

#### `leetgpu/week4/day6/leetgpu-multi-head-attention-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week4_day6_leetgpu-multi-head-attention-solution_block43_f87737d3ac61.cu -o leetgpu_week4_day6_leetgpu-multi-head-attention-solution_block43_f87737d3ac61 `
- **编译耗时**：3.00s，**运行耗时**：2.015s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
B=2 H=4 N=512 d=64  QKV=3.15 MB  S/P(std)=8.39 MB each
standard: 0.644 ms   flash: 0.407 ms
standard max diff: 1.50e-04 (PASS)
flash    max diff: 2.17e-04 (PASS)
est. DRAM: standard=1.11 GB  flash=0.14 GB  (flash 省 S/P=0.03 GB)
```
</details>

#### `leetgpu/week5/day1/leetgpu-int8-kv-cache-attention-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day1_leetgpu-int8-kv-cache-attention-solution_block44_3af8c4f16dd2.cu -o leetgpu_week5_day1_leetgpu-int8-kv-cache-attention-solution_block44_3af8c4f16dd2 `
- **编译耗时**：2.90s，**运行耗时**：2.660s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
H=32 L=8192 d=128  int8 KV=67.11 MB  (fp32 KV would be 134.22 MB, 4x)
kernel time: 7.622 ms
max diff: 3.71e-04 (PASS, tol=1e-3)
est. DRAM (int8)=69.21 MB  (fp32)=268.44 MB  AI(int8)=2.92  AI(fp32)=0.75
```
</details>

#### `leetgpu/week5/day2/leetgpu-grouped-query-attention-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day2_leetgpu-grouped-query-attention-solution_block45_7148c1fc6ca5.cu -o leetgpu_week5_day2_leetgpu-grouped-query-attention-solution_block45_7148c1fc6ca5 `
- **编译耗时**：2.86s，**运行耗时**：11.749s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
nq=32 nkv=8 group=4 S=1024 d=128
Q=16.78 MB  KV=8.39 MB (MHA would be 33.55 MB, 4.0x more)
kernel time: 30.455 ms
max diff: 4.80e-04 (FAIL, tol=1e-4)

[KV Cache 收益] GQA cache / MHA cache = 8 / 32 = 0.25
[LLaMA-3 8B] 32Q/8KV → cache 缩到 25%（省 75%）
```
</details>

#### `leetgpu/week5/day3/leetgpu-speculative-decoding-verification-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day3_leetgpu-speculative-decoding-verification-solution_block46_48f8a58318aa.cu -o leetgpu_week5_day3_leetgpu-speculative-decoding-verification-solution_block46_48f8a58318aa `
- **编译耗时**：3.03s，**运行耗时**：2.447s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
B=64 T=8 V=32768
draft_probs+target_probs = 134.22 MB
kernel time: 0.057 ms
mismatched tokens: 0 / 576 (PASS)
```
</details>

#### `leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day4_leetgpu-causal-self-attention-solution_block47_9d0e51557452.cu -o leetgpu_week5_day4_leetgpu-causal-self-attention-solution_block47_9d0e51557452 `
- **编译耗时**：2.85s，**运行耗时**：5.703s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=5000 d=128
kernel time: 13.470 ms
max diff: 3.79e-04 (FAIL, tol=1e-5)
causal FLOPs = 3.20 G (50.0% of full attention 6.40 G)
```
</details>

#### `leetgpu/week5/day5/leetgpu-token-embedding-layer-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day5_leetgpu-token-embedding-layer-solution_block48_5d251dd7a572.cu -o leetgpu_week5_day5_leetgpu-token-embedding-layer-solution_block48_5d251dd7a572 `
- **编译耗时**：3.44s，**运行耗时**：2.415s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
B=32 T=512 V=30000 P=2048 D=768
token_emb = 92.16 MB, output = 50.33 MB
max diff: 2.62e-06 (PASS, tol=1e-4)
```
</details>

#### `leetgpu/week5/day6/leetgpu-weight-dequantization-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week5_day6_leetgpu-weight-dequantization-solution_block49_18303eeac1cc.cu -o leetgpu_week5_day6_leetgpu-weight-dequantization-solution_block49_18303eeac1cc `
- **编译耗时**：3.05s，**运行耗时**：3.870s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=8192 N=8192 T=128  s_rows=64 s_cols=64
X+Y = 536.87 MB (268.44 MB each), S = 16.00 KB
kernel time: 0.344 ms
max diff: 0.00e+00 (PASS, tol=1e-5)

[Roofline] FLOPs=0.07G  Bytes=536.89MB  AI=0.50 FLOP/Byte
[Roofline] achieved BW = 1558.7 GB/s (RTX 5090 peak ~1555 GB/s)
[Roofline] AI=0.50 ≪ Ridge(12.6) → memory-bound
```
</details>

#### `leetgpu/week6/day2/leetgpu-max-subarray-sum-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week6_day2_leetgpu-max-subarray-sum-solution_block50_9ec668b9da7b.cu -o leetgpu_week6_day2_leetgpu-max-subarray-sum-solution_block50_9ec668b9da7b `
- **编译耗时**：2.94s，**运行耗时**：2.011s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU: 2144, CPU: 2144, PASS
```
</details>

#### `leetgpu/week6/day3/leetgpu-stream-compaction-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week6_day3_leetgpu-stream-compaction-solution_block51_e0d735bd16fc.cu -o leetgpu_week6_day3_leetgpu-stream-compaction-solution_block51_e0d735bd16fc `
- **编译耗时**：2.97s，**运行耗时**：1.914s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
GPU count=11123, CPU count=660116, FAIL
```
</details>

#### `leetgpu/week6/day4/leetgpu-segmented-prefix-sum-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week6_day4_leetgpu-segmented-prefix-sum-solution_block52_1de5a15434df.cu -o leetgpu_week6_day4_leetgpu-segmented-prefix-sum-solution_block52_1de5a15434df `
- **编译耗时**：2.88s，**运行耗时**：1.993s
- **状态**：✅ PASS
- **编译错误**：
```text
leetgpu_week6_day4_leetgpu-segmented-prefix-sum-solution_block52_1de5a15434df.cu(34): warning #177-D: variable "warp_carry" was declared but never referenced
      __attribute__((shared)) int warp_carry[32 + 1];
                                  ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"
```
- **运行输出**：
<details><summary>展开</summary>

```text
out[0]=0 (cpu=0) ✓
out[1]=0 (cpu=3) ✗
out[2]=1 (cpu=4) ✗
out[3]=0 (cpu=0) ✓
out[4]=3 (cpu=4) ✗
out[5]=5 (cpu=6) ✗
FAIL
```
</details>

#### `leetgpu/week6/day6/leetgpu-top-k-selection-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week6_day6_leetgpu-top-k-selection-solution_block53_8284cda8b047.cu -o leetgpu_week6_day6_leetgpu-top-k-selection-solution_block53_8284cda8b047 `
- **编译耗时**：3.09s，**运行耗时**：2.015s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Sorted: -2147483648 -2147483648 -2147483648 -2147483648 -2147483648 -2147483648 -2147483648 -2147483648 
Top 3: -2147483648 -2147483648 -2147483648
```
</details>

#### `leetgpu/week7/day1/leetgpu-matrix-copy-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week7_day1_leetgpu-matrix-copy-solution_block54_585641a1d6d4.cu -o leetgpu_week7_day1_leetgpu-matrix-copy-solution_block54_585641a1d6d4 `
- **编译耗时**：2.87s，**运行耗时**：2.365s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
Matrix 4096x4096 copy: PASS
Bandwidth: 1603.1 GB/s (theory ~1555 GB/s on RTX 5090)
```
</details>

#### `leetgpu/week7/day2/leetgpu-vector-reversal-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week7_day2_leetgpu-vector-reversal-solution_block55_de1fc013fcba.cu -o leetgpu_week7_day2_leetgpu-vector-reversal-solution_block55_de1fc013fcba `
- **编译耗时**：2.70s，**运行耗时**：2.201s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 10000000  (40.0 MB)
kernel time: 0.194 ms
I/O bandwidth: 412.0 GB/s
PASS
```
</details>

#### `leetgpu/week7/day3/leetgpu-scalar-multiply-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week7_day3_leetgpu-scalar-multiply-solution_block56_c5df46c77893.cu -o leetgpu_week7_day3_leetgpu-scalar-multiply-solution_block56_c5df46c77893 `
- **编译耗时**：2.77s，**运行耗时**：2.110s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 10000000  (40.0 MB), alpha = 2.000000
kernel time: 0.194 ms
I/O bandwidth: 411.3 GB/s
PASS
```
</details>

#### `leetgpu/week7/day5/leetgpu-element-reversal-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week7_day5_leetgpu-element-reversal-solution_block57_95b377370d21.cu -o leetgpu_week7_day5_leetgpu-element-reversal-solution_block57_95b377370d21 `
- **编译耗时**：2.85s，**运行耗时**：1.991s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
N = 10000000  (40.0 MB)
kernel time: 0.189 ms
I/O bandwidth: 422.7 GB/s
PASS
```
</details>

#### `leetgpu/week7/day7/leetgpu-matrix-addition-solution.md`

- **编译命令**：`export PATH=/usr/local/cuda/bin:$PATH && cd /root/md_cuda && nvcc -O3 -arch=sm_120 leetgpu_week7_day7_leetgpu-matrix-addition-solution_block58_a313a165278a.cu -o leetgpu_week7_day7_leetgpu-matrix-addition-solution_block58_a313a165278a `
- **编译耗时**：2.49s，**运行耗时**：2.293s
- **状态**：✅ PASS
- **运行输出**：
<details><summary>展开</summary>

```text
M=4096 N=4096 (67.1 MB per matrix)
kernel time: 0.251 ms
I/O bandwidth: 802.0 GB/s
PASS
```
</details>

## 三、未独立运行的 kernel 文件

以下文件只包含 kernel 实现，没有 `main()`，需要平台 starter 或额外 host 代码才能执行：

- `leetgpu/week2/day1/pre.cu`
- `leetgpu/week2/day1/prefix_sum_inclusive.cu`
- `leetgpu/week2/day1/presum.cu`

它们主要用于 LeetGPU 平台提交，不能直接编译为可执行文件。
