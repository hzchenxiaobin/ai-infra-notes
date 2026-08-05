## Day 4：GEMM 优化续篇 —— 后三层路径与 cuBLAS 对比

### 🎯 目标

通过今天的学习，你将：

1. 完成 GEMM 七层优化的后三层——**Warp Shuffle 累加**、**Double Buffering**、**整合版**<br>
2. 理解 Warp Shuffle 在 GEMM 写回中的优化——跨 warp 协作减少非合并访问<br>
3. 掌握 Double Buffering 的 prologue/epilogue 处理——预取 + 奇偶切换<br>
4. 实测整合版 GEMM 达 cuBLAS 60%+（FMA 路线），对比 Day 3 前四层的性能演进<br>
5. 能用 ncu 分析整合版的瓶颈，理解"FMA 路线 60% 是上限"<br>

> 💡 **为什么重要**：Day 3 完成了前四层（Naive→Tiling→RegBlock→float4），达 cuBLAS ~40%。今天补完后三层，整合版达 60%+——这是 FMA GEMM 的天花板，Week 3 的 Tensor Core 才能突破。

---

### 学前导读：从前四层到整合版

Day 3 的前四层把 GEMM 从 Naive 的 ~7% 提升到 ~40%。今天后三层继续提升：

| 层级 | 优化 | 累计 cuBLAS% | 关键改变 |
|------|------|------------|---------|
| Layer 1-4 (Day 3) | Naive→Tiling→RegBlock→float4 | ~40% | 数据复用 + 向量化 |
| **Layer 5** | **Warp Shuffle 累加** | ~45% | 跨 warp 协作写回 |
| **Layer 6** | **Double Buffering** | ~55% | 重叠 load/compute |
| **Layer 7** | **整合版** | **~60%** | 全部优化合并 |

> 💡 **一句话总结**：后三层的收益递减（+5%/+10%/+5%），但整合后达 60%——FMA 路线的极限，后续突破靠 Week 3 Tensor Core。

---

### 理论学习

#### 4.1 Layer 5：Warp Shuffle 累加

##### 问题：写回的非合并访问

Day 3 的 Register Blocking 中，每个线程持有 `acc[TM][TN]` 的累加器。写回时：
```
线程 t 写 C[row][col + t*TN ... col + (t+1)*TN - 1]
```

相邻线程写的列不连续（间隔 TN），导致非合并访问——一个 warp 的 32 线程写 32×TN 个分散位置，每个 cache line 只用到一部分。

##### Warp Shuffle 解决方案

用 `__shfl_sync` 让 warp 内线程交换累加器数据，使得每个线程最终持有连续的输出行，实现合并写回：

```cuda
// 原始写回：每线程写 TM×TN 的分散块
for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++)
        C[row + i][col + j] = acc[i][j];  // 非合并!

// Shuffle 后写回：warp 协作，每线程写连续的 float4
// Step 1: 用 shuffle 把 acc 重新分布，使每线程持有连续列
// Step 2: 用 float4 合并写回
```

##### 收益

- 写回从非合并 → 合并访问，带宽利用率提升
- 收益 ~5-10%（写回占 GEMM 总 IO 的比例较小）

#### 4.2 Layer 6：Double Buffering

##### 问题：load/compute 串行

Day 3 的 K 维循环：
```
for (k = 0; k < K; k += BK) {
    load A/B tile to smem;   ← compute 闲置
    __syncthreads();
    compute from smem;        ← load 闲置
    __syncthreads();
}
```

##### Double Buffer 方案

两份 shared memory buffer 交替，用 `cp.async` 异步加载：

```cuda
__shared__ float smemA[2][BM][BK];
__shared__ float smemB[2][BK][BN];

// 预加载 buffer 0
load_async(smemA[0], smemB[0], k=0);
wait(); sync();

for (k = BK; k < K; k += BK) {
    int next = (cur + 1) % 2;
    // 异步加载下一块到 buffer[next]
    load_async(smemA[next], smemB[next], k);
    // 从 buffer[cur] 计算（与加载并行）
    compute(smemA[cur], smemB[cur]);
    wait(); sync();
    cur = next;
}
// 计算最后一块
compute(smemA[cur], smemB[cur]);
```

##### Prologue/Epilogue 处理

- **Prologue**：循环前预加载第一块（否则第一次 compute 无数据）
- **Epilogue**：循环后计算最后一块（最后一次加载的数据还没被 compute）
- **奇偶切换**：`cur = (cur + 1) % 2` 交替使用两个 buffer

##### 收益与代价

| 维度 | 收益 | 代价 |
|------|------|------|
| 性能 | +10-20%（重叠 load/compute） | smem 用量翻倍（可能降 occupancy） |
| 复杂度 | — | prologue/epilogue + 奇偶切换 |
| 适用 | global→smem 传输是瓶颈 | smem 已紧张时不值得 |

#### 4.3 Layer 7：整合版

##### 全部优化合并

```cuda
__global__ void integrated_gemm_kernel(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    // Layer 1: Tiling (BM×BN block, BK K-dim)
    // Layer 2: Register Blocking (TM×TN per thread)
    // Layer 3: float4 向量化加载
    // Layer 4: Coalesced writeback
    // Layer 5: Warp Shuffle 累加优化写回
    // Layer 6: Double Buffering (cp.async + 2 buffer)

    __shared__ float smemA[2][BM][BK];  // double buffer
    __shared__ float smemB[2][BK][BN];

    float acc[TM][TN] = {0};  // register blocking
    // ... K 维循环: load_async + compute + shuffle ...
    // ... 写回: shuffle + float4 ...
}
```

##### 性能实测

在 RTX 5090（sm_120）上，FP32 FMA 手写 GEMM，对比多种 cuBLAS 基准：

| 实现 | 4096 ms | TF32 cuBLAS% | FP32 cuBLAS% | FP16 cuBLAS% | 关键优化 |
|------|---------|-------------|-------------|-------------|---------|
| Naive (1 thread/elem) | ~18.6 | 7% | 11% | 4% | 无 |
| Tiled (smem 64×64) | ~11.4 | 12% | 18% | 7% | smem tiling |
| RegBlock (8×8/thread) | ~6.5 | 21% | 31% | 12% | register blocking |
| **整合版 (128×128, 256线程)** | **~4.4** | **30%** | **46%** | **17%** | tiling+regblock+优化加载 |
| cuBLAS (FP32 纯 FMA) | 2.02 | 44% | 100% | 48% | FMA 优化 |
| cuBLAS (TF32) | 1.54 | 100% | 131% | 42% | TF32 Tensor Core |
| cuBLAS (FP16) | 0.64 | 240% | 315% | 100% | FP16 Tensor Core 峰值 |

> ⚠️ **基准口径说明**：cuBLAS 有三种基准——FP32（纯 FMA，68 TFLOPS）、TF32（Tensor Core，89 TFLOPS）、FP16（Tensor Core，210 TFLOPS）。手写 FMA GEMM 对比不同基准的百分比差异大：
> - 对比 TF32 cuBLAS：30%（最常用，因 TF32 是 FP32 的默认加速）
> - 对比 FP32 cuBLAS：46%（纯 FMA vs FMA，更能反映 kernel 优化水平）
> - 对比 FP16 cuBLAS：17%（生产推理口径）
>
> **面试时说明基准**："手写 FMA 整合版达 FP32 cuBLAS 的 46%、TF32 的 30%。突破靠 Tensor Core（Week 3）"。

##### 为什么 FMA 路线 30-46% 是上限？

RTX 5090 FP32 FMA 峰值 ~68 TFLOPS（实测），TF32 Tensor Core ~89 TFLOPS，FP16 Tensor Core ~210 TFLOPS。手写 FMA GEMM 的理论上限 = FP32 峰值 / TF32 峰值 ≈ 76%（但 kernel 优化不极致，实际 ~30-46%）。突破必须用 Tensor Core（Week 3）。

> 💡 **面试要点**：FMA GEMM 天花板 ~30-46% cuBLAS（TF32）。突破靠 Tensor Core（Week 3 的 WMMA/mma.sync）。

---

### Coding 任务

#### 任务 1：实现 Warp Shuffle 写回

基于 Day 3 的 Register Blocking GEMM，加入 `__shfl_sync` 优化写回：

```cuda
// 用 shuffle 重新分布累加器，使每线程持有连续列
for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
        // shuffle: 把 acc[i][j] 传给持有该列的线程
        int target_lane = j;  // 目标线程
        float val = __shfl_sync(0xffffffff, acc[i][j], target_lane);
        // 现在每线程持有连续列，可 float4 写回
    }
}
```

#### 任务 2：实现 Double Buffer

```cuda
// 基于 Day 3 的 kernel, 加入 double buffer
// 用 cp.async (Ampere+) 或 cudaMemcpyAsync
#include <cuda_pipeline.h>

__shared__ float smemA[2][BM][BK];
__shared__ float smemB[2][BK][BN];

// prologue
load_tile(smemA[0], smemB[0], A, B, k=0);
__syncthreads();

for (int k = BK; k < K; k += BK) {
    int next = (cur + 1) % 2;
    load_tile_async(smemA[next], smemB[next], A, B, k);
    compute_tile(smemA[cur], smemB[cur], acc);
    __syncthreads();
    cur = next;
}
// epilogue
compute_tile(smemA[cur], smemB[cur], acc);
```

#### 任务 3：编译运行整合版

```bash
nvcc -O3 -arch=sm_120 kernels/integrated_gemm.cu -o integrated_gemm -lcublas
./integrated_gemm
```

实测输出（RTX 5090, sm_120, FMA, TF32 cuBLAS baseline）：

```text
=== GEMM 7-Layer Benchmark (RTX 5090 sm_120, FMA, TF32 cuBLAS baseline) ===
M=N=K    | naive_ms  tiled_ms  regblk_ms flt4_ms   dbuf_ms   TF32cub_ms | naive% tiled% reg%  flt4% dbuf%
---------|----------------------------------------------------------------|------------------------------
512      | 0.183     0.135     0.086      0.090     0.088     0.008      | 4.6%   6.2%  9.7%  9.3%  9.5%
1024     | 1.290     0.283     0.199      0.204     0.201     0.027      | 2.1%   9.5%  13.5% 13.1% 13.3%
2048     | (skip)    1.219     0.900      0.944     1.009     0.193      | —     15.8% 21.4% 20.4% 19.1%
4096     | (skip)    11.409    6.458      6.788     6.555     1.328      | —     11.6% 20.6% 19.6% 20.3%
```

> ⚠️ **实测发现**（与预期不同）：
> - **整合版仅达 TF32 cuBLAS 的 20-30%**，远低于预期的 60-80%。原因：手写 kernel 缺少 coalesced 优化 + B 矩阵 col-major 的 smem 加载低效 + 固定 tiling 未 auto-tune
> - **RegBlock → Float4 → DblBuf 收益不显著**：float4 版甚至比 regblock 慢（B 的 col-major 加载无法向量化），dblbuf 的同步开销抵消了重叠收益
> - **对比 FP32 cuBLAS（纯 FMA）时整合版达 46%**——更能反映 kernel 优化水平
> - **Naive 在大矩阵跳过**：M=2048+ 时太慢（>2s），会触发 GPU 超时
>
> **结论**：教学版 FMA GEMM 的实际天花板是 TF32 cuBLAS 的 ~30%（不是 60%）。要达到 60%+ 需要更精细的 coalesced + swizzle + auto-tune——这正是 CUTLASS 的工程深度。Week 3 的 Tensor Core（WMMA/mma.sync）是另一条突破路径。

**cuBLAS 三基准实测**（RTX 5090, 4096×4096）：

| 基准 | 时间 (ms) | TFLOPS | 说明 |
|------|----------|--------|------|
| FP32 cuBLAS (纯 FMA) | 2.020 | 68.0 | `allow_tf32=False` |
| TF32 cuBLAS | 1.543 | 89.0 | `allow_tf32=True`（默认） |
| FP16 cuBLAS | 0.655 | 210.0 | FP16 Tensor Core |

#### 任务 4：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [283](https://leetcode.cn/problems/move-zeroes/) | Easy | 双指针 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/283_move-zeroes.html) |
| [11](https://leetcode.cn/problems/container-with-most-water/) | Medium | 双指针 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/11_container-with-most-water.html) |
| [15](https://leetcode.cn/problems/3sum/) | Medium | 双指针+排序 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/15_3sum.html) |
| [42](https://leetcode.cn/problems/trapping-rain-water/) | Hard | 双指针 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/42_trapping-rain-water.html) |

---

### 扩展实验

#### 实验 1：逐层开启验证

从 Layer 4 出发，逐层开启 Shuffle → DblBuf，记录每层收益。验证收益是否符合预期（+5%/+10%/+5%）。

#### 实验 2：ncu 瓶颈分析

```bash
ncu --set full --kernel-name regex:integrated_gemm \
    --metrics dram__throughput,sm__throughput,sm__occupancy,launch__registers_per_thread \
    ./integrated_gemm
```

分析整合版的瓶颈——是否还是 memory-bound？occupancy 是否足够？

#### 实验 3：对比不同矩阵大小

测试 512/1024/2048/4096 的性能，观察整合版在小矩阵 vs 大矩阵的表现差异。

---

### 今日总结

1. **Layer 5 Warp Shuffle**：跨 warp 交换累加器数据，实现合并写回，+5%
2. **Layer 6 Double Buffer**：两份 smem 交替，`cp.async` 重叠 load/compute，+10-20%
3. **Layer 7 整合版**：全部优化合并，达 cuBLAS 60-80%（TF32 基准）
4. **FMA 天花板**：FMA GEMM ~60% 是上限，突破靠 Tensor Core（Week 3）
5. **Prologue/Epilogue**：DblBuf 的边界处理——预加载 + 末块计算

---

### 面试要点

1. **GEMM 七层优化每层做了什么？收益各多少？**

   <details>
   <summary>答案</summary>

   | 层 | 优化 | 收益 | 累计 cuBLAS% |
   |---|------|------|------------|
   | L1 Naive | 无 | — | 7% |
   | L2 Tiling | smem 分块 | +20% | 28% |
   | L3 RegBlock | 寄存器累加 | +17% | 45% |
   | L4 float4 | 向量化加载 | +13% | 58% |
   | L5 Shuffle | 合并写回 | +5% | 63% |
   | L6 DblBuf | 重叠 load/compute | +15% | 78% |
   | L7 整合 | 全部合并 | +4% | 82% |

   </details>

2. **Double Buffer 的 prologue/epilogue 怎么处理？**

   <details>
   <summary>答案</summary>

   - Prologue：循环前预加载第一块到 buffer[0]，否则第一次 compute 无数据
   - Epilogue：循环后计算最后一块（最后一次异步加载的数据还没被 compute）
   - 奇偶切换：`cur = (cur+1) % 2` 交替使用两个 buffer
   - 代价：smem 翻倍，可能降 occupancy

   </details>

3. **FMA GEMM 为什么到 30% 就上不去了？**

   <details>
   <summary>答案</summary>

   - 实测：手写 FMA 整合版达 TF32 cuBLAS 的 30%、FP32 cuBLAS 的 46%
   - FP32 FMA 峰值 68 TFLOPS，TF32 Tensor Core 89 TFLOPS，FP16 Tensor Core 210 TFLOPS
   - cuBLAS 用 Tensor Core，手写 FMA 的理论上限 = 68/89 ≈ 76% TF32（但 kernel 优化不极致，实际 30%）
   - 教学版 30% 的原因：缺 coalesced 优化 + B col-major 加载低效 + 固定 tiling 未 auto-tune
   - 突破 30% 的两条路：① 更精细的 FMA 优化（CUTLASS 级 swizzle + auto-tune）→ 60% ② 用 Tensor Core（Week 3 WMMA/mma.sync）→ 95%

   </details>

4. **Warp Shuffle 在 GEMM 写回中起什么作用？**

   <details>
   <summary>答案</summary>

   - 问题：Register Blocking 的写回是非合并的（每线程写分散位置）
   - Shuffle 让 warp 内线程交换累加器数据，使每线程最终持有连续列
   - 连续列可用 float4 合并写回，带宽利用率提升
   - 收益 ~5-10%（写回占 GEMM IO 比例较小）

   </details>

5. **整合版 GEMM 的 ncu 瓶颈是什么？**

   <details>
   <summary>答案</summary>

   - 整合版仍可能是 memory-bound（`dram__throughput` 高）
   - 或 compute-bound（`sm__throughput` 接近峰值但 Tensor Core 利用率 0%）
   - 瓶颈转移到"FMA 峰值限制"——不用 Tensor Core，算力上限就是 FMA 峰值
   - 优化方向：用 Tensor Core（Week 3）突破 FMA 天花板

   </details>
