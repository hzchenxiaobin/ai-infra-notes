# Day 3 练习题：手动计算 GPU Occupancy

> 本练习以 **NVIDIA A100 (Compute Capability 8.0)** 为例，所有计算均按 A100 的硬件上限进行：
>
> | 资源 | A100 上限 |
> |------|----------|
> | Max Threads / SM | 2048 |
> | Max Warps / SM | 64 |
> | Max Blocks / SM | 32 |
> | Register File / SM | 256 KB = **65536 个 32-bit 寄存器** |
> | 寄存器分配粒度 | 每 block 按 **256 个寄存器** 对齐 |
> | Shared Memory / SM | 164 KB = **167936 bytes** |
> | Shared Memory 分配粒度 | 每 block 按 **1024 bytes** 对齐 |
> | Warp Size | 32 |

## 核心公式回顾

```text
warps_per_block = ceil(threads_per_block / 32)

blocks_from_threads = floor(max_threads_per_sm / threads_per_block)
blocks_from_regs    = floor(max_regs_per_sm / (ceil(threads_per_block * regs_per_thread / 256) * 256))
blocks_from_smem    = floor(max_smem_per_sm / (ceil(smem_per_block / 1024) * 1024))

active_blocks = min(blocks_from_threads, blocks_from_regs, blocks_from_smem, max_blocks_per_sm)
active_warps  = active_blocks * warps_per_block
occupancy     = active_warps / max_warps_per_sm * 100%
```

> 💡 **注意**：`ceil(x / g) * g` 表示按粒度 `g` 向上取整。如果 `x` 已经是 `g` 的整数倍，则保持不变。

---

## 题目 1：寄存器约束

**参数**：
- Block size：256 threads
- Registers per thread：64
- Shared memory per block：0

**问题**：
1. 每个 block 需要多少寄存器？
2. SM 的寄存器限制允许同时驻留多少个 block？
3. SM 的 thread 限制允许同时驻留多少个 block？
4. 最终 active blocks、active warps、occupancy 分别是多少？
5. 瓶颈资源是什么？

<details>
<summary>点击查看答案</summary>

1. `regs_per_block = ceil(256 * 64 / 256) * 256 = 16384`
2. `blocks_from_regs = floor(65536 / 16384) = 4`
3. `blocks_from_threads = floor(2048 / 256) = 8`
4. `active_blocks = min(8, 4, ∞, 32) = 4`  
   `warps_per_block = ceil(256 / 32) = 8`  
   `active_warps = 4 * 8 = 32`  
   `occupancy = 32 / 64 = 50%`
5. **瓶颈资源：Registers per thread**

</details>

---

## 题目 2：共享内存约束

**参数**：
- Block size：256 threads
- Registers per thread：32
- Shared memory per block：24 KB = 24576 bytes

**问题**：
1. 分别计算 thread、register、shared memory 三种限制下的 block 数量。
2. 最终 active blocks、active warps、occupancy 分别是多少？
3. 瓶颈资源是什么？

<details>
<summary>点击查看答案</summary>

1. 
   - `blocks_from_threads = floor(2048 / 256) = 8`
   - `regs_per_block = ceil(256 * 32 / 256) * 256 = 8192`  
     `blocks_from_regs = floor(65536 / 8192) = 8`
   - `smem_per_block = ceil(24576 / 1024) * 1024 = 24576`  
     `blocks_from_smem = floor(167936 / 24576) = 6`
2. `active_blocks = min(8, 8, 6, 32) = 6`  
   `warps_per_block = 8`  
   `active_warps = 6 * 8 = 48`  
   `occupancy = 48 / 64 = 75%`
3. **瓶颈资源：Shared memory per block**

</details>

---

## 题目 3：Thread 数量约束

**参数**：
- Block size：1024 threads
- Registers per thread：32
- Shared memory per block：0

**问题**：
1. 分别计算 thread、register 限制下的 block 数量。
2. 最终 active blocks、active warps、occupancy 分别是多少？
3. 瓶颈资源是什么？

<details>
<summary>点击查看答案</summary>

1. 
   - `blocks_from_threads = floor(2048 / 1024) = 2`
   - `regs_per_block = ceil(1024 * 32 / 256) * 256 = 32768`  
     `blocks_from_regs = floor(65536 / 32768) = 2`
2. `active_blocks = min(2, 2, ∞, 32) = 2`  
   `warps_per_block = ceil(1024 / 32) = 32`  
   `active_warps = 2 * 32 = 64`  
   `occupancy = 64 / 64 = 100%`
3. **Thread 数量和寄存器同时达到上限**（两者都是瓶颈）

</details>

---

## 题目 4：高寄存器压力

**参数**：
- Block size：256 threads
- Registers per thread：128
- Shared memory per block：0

**问题**：
1. 分别计算 thread、register 限制下的 block 数量。
2. 最终 active blocks、active warps、occupancy 分别是多少？
3. 如果要将 occupancy 提升到 50%，在不改代码的前提下，最简单的调整是什么？

<details>
<summary>点击查看答案</summary>

1. 
   - `blocks_from_threads = floor(2048 / 256) = 8`
   - `regs_per_block = ceil(256 * 128 / 256) * 256 = 32768`  
     `blocks_from_regs = floor(65536 / 32768) = 2`
2. `active_blocks = min(8, 2, ∞, 32) = 2`  
   `warps_per_block = 8`  
   `active_warps = 2 * 8 = 16`  
   `occupancy = 16 / 64 = 25%`
3. **增大 block size 到 1024**：此时 `regs_per_block = 1024 * 128 = 131072 > 65536`，超过 SM 寄存器上限，**不行**。  
   正确做法：**减小 block size 到 128**：`regs_per_block = ceil(128 * 128 / 256) * 256 = 16384`，`blocks_from_regs = 4`，`blocks_from_threads = 16`，`active_blocks = 4`，`active_warps = 4 * 4 = 16`，occupancy 仍是 25%。  
   因此，**仅调整 block size 无法把 occupancy 提升到 50%**；必须降低每个线程的寄存器用量（例如通过 `__launch_bounds__` 或优化代码）。

</details>

---

## 题目 5：综合约束

**参数**：
- Block size：128 threads
- Registers per thread：64
- Shared memory per block：16 KB = 16384 bytes

**问题**：
1. 分别计算 thread、register、shared memory 三种限制下的 block 数量。
2. 最终 active blocks、active warps、occupancy 分别是多少？
3. 瓶颈资源是什么？

<details>
<summary>点击查看答案</summary>

1. 
   - `blocks_from_threads = floor(2048 / 128) = 16`
   - `regs_per_block = ceil(128 * 64 / 256) * 256 = 8192`  
     `blocks_from_regs = floor(65536 / 8192) = 8`
   - `smem_per_block = ceil(16384 / 1024) * 1024 = 16384`  
     `blocks_from_smem = floor(167936 / 16384) = 10`
2. `active_blocks = min(16, 8, 10, 32) = 8`  
   `warps_per_block = ceil(128 / 32) = 4`  
   `active_warps = 8 * 4 = 32`  
   `occupancy = 32 / 64 = 50%`
3. **瓶颈资源：Registers per thread**

</details>

---

## 题目 6：block size 调优

一个 kernel 当前配置如下：
- Block size：512 threads
- Registers per thread：96
- Shared memory per block：0

**问题**：
1. 当前 occupancy 是多少？
2. 如果保持 96 registers/thread，仅把 block size 改成 256，occupancy 会变化吗？为什么？
3. 如果通过 `__launch_bounds__` 把 registers/thread 降到 64，保持 block size = 512，occupancy 是多少？

<details>
<summary>点击查看答案</summary>

1. 当前配置：
   - `blocks_from_threads = floor(2048 / 512) = 4`
   - `regs_per_block = ceil(512 * 96 / 256) * 256 = 49152`  
     `blocks_from_regs = floor(65536 / 49152) = 1`
   - `active_blocks = min(4, 1, ∞, 32) = 1`  
     `warps_per_block = ceil(512 / 32) = 16`  
     `active_warps = 16`  
     `occupancy = 16 / 64 = 25%`
2. block size 改成 256：
   - `blocks_from_threads = floor(2048 / 256) = 8`
   - `regs_per_block = ceil(256 * 96 / 256) * 256 = 24576`  
     `blocks_from_regs = floor(65536 / 24576) = 2`
   - `active_blocks = 2`，`warps_per_block = 8`，`active_warps = 16`  
     **occupancy 仍然是 25%**。  
     原因：虽然 active blocks 翻倍，但每个 block 的 warp 数减半，总 active warps 不变。
3. registers/thread 降到 64，block size = 512：
   - `blocks_from_threads = 4`
   - `regs_per_block = ceil(512 * 64 / 256) * 256 = 32768`  
     `blocks_from_regs = floor(65536 / 32768) = 2`
   - `active_blocks = 2`，`warps_per_block = 16`，`active_warps = 32`  
     `occupancy = 32 / 64 = 50%`

</details>

---

## 题目 7：用 CUDA API 验证你的计算

请编译并运行同目录下的 [occupancy_verify.cu](occupancy_verify.cu)：

```bash
nvcc -std=c++11 -o occupancy_verify occupancy_verify.cu
./occupancy_verify
```

程序会输出：
1. 当前 GPU 的关键硬件参数
2. 几个不同 kernel 的 `cudaFuncAttributes`
3. CUDA 运行时通过 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算出的 active blocks
4. 程序按本练习公式手工计算出的理论 occupancy

**任务**：
1. 对比手算结果与程序输出是否一致。
2. 如果程序运行在 A100 上，尝试修改 [occupancy_verify.cu](occupancy_verify.cu) 中的 block size 和 shared memory 大小，观察 occupancy 如何变化。
3. 思考：为什么在某些情况下，手算结果和 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 可能有细微差别？（提示：寄存器/共享内存粒度、计算能力版本差异、编译器优化）

---

## 常见错误提醒

1. **不要把整个 grid 的线程加在一起算 warp**。Occupancy 是 **per-SM** 的概念，永远先算每个 block 占多少 warp，再算每个 SM 能放多少个 block。
2. **注意寄存器粒度**。不是 `threads * regs` 直接除，而要按 256 对齐。
3. **共享内存的上限是按 SM 算，不是按 block 算**。A100 每个 block 最多用 48 KB shared memory，但 occupancy 计算里看的是 SM 总共 164 KB。
4. **`maxBlocksPerMultiprocessor` 是硬上限**。即使 thread、register、smem 都没用完，active blocks 也不能超过这个值。

---

## 扩展思考

1. 如果你的 kernel 是 **memory-bound**（大量等 global memory），提高 occupancy 是否有帮助？为什么？
2. 如果你的 kernel 是 **compute-bound**（计算密集、寄存器多），盲目降低寄存器用量是否一定更好？
3. `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 的第二个参数如何影响编译器对寄存器用量的选择？
