## Day 4：GEMM 优化续篇 —— Shuffle 写回、Double Buffering 深挖与 cuBLAS 三基准对比

### 🎯 目标

通过今天的学习，你将：

1. 深挖 Warp Shuffle 写回重排的机制与适用边界——为什么 Day 3 整合版最终没用 shuffle<br>
2. 理解同步双缓冲为什么实测无效，掌握 `cp.async` 真双缓冲的结构（prologue/epilogue）<br>
3. 掌握 cuBLAS 三基准口径（FP32 / TF32 / FP16），学会面试时报数字先说基准<br>
4. 复跑 Day 3 整合版，统一性能口径：4096 实测 3.178ms = FP32 cuBLAS 的 63.4%<br>
5. 能用 ncu 分析整合版的瓶颈，理解"FMA 路线 ~63% 之后必须上 Tensor Core"<br>

> 💡 **为什么重要**：Day 3 已实测完整 v1–v6 路径（整合版 4096 达 FP32 cuBLAS 63.4%）。今天不再重复实现，而是把三个面试必考点讲透：shuffle 写回机制、真双缓冲（`cp.async`）、cuBLAS 三基准口径——这是把"我做过 GEMM 优化"讲成"我懂 GEMM 优化"的关键。

---

### 学前导读：Day 3 实测回顾与今天的三个增量

Day 3 已经把 GEMM 从 Naive 一路实测到整合版 + 双缓冲（4096³，RTX 5090，FP32 cuBLAS 基准）：

| 版本 | cuBLAS%（实测） | 关键改变 |
|------|------------|---------|
| v1 Naive | 10.6% | 无 |
| v2 SM Tiling | 13.3% | K 维数据复用 |
| v3 RegBlk（Day 2） | 30.8% | 累加器驻留寄存器 |
| v4 +float4 | 64.3% | 向量化加载（最大单步增益） |
| v5 整合版 | 62.9% | + float4 合并写回 |
| v6 +同步双缓冲 | 63.8% | 收益在噪声范围内 |

> 💡 **一句话总结**：实测口径下，shuffle 写回与同步双缓冲的收益都在噪声内——"后三层"的真正增量不在这两个机制本身，而在 ① 理解它们何时才有效（shuffle 的主场是 Week 3 Tensor Core fragment 重排；真双缓冲需要 `cp.async`）② 建立 cuBLAS 三基准口径，面试报数字不含糊。

---

### 理论学习

#### 4.1 Layer 5：Warp Shuffle 累加

##### 问题：写回的非合并访问

Day 2 的 Register Blocking 中，每个线程持有 `acc[TM][TN]` 的累加器。写回时：
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
- 预估收益 ~5-10%（写回占 GEMM 总 IO 的比例较小）

> ⚠️ **实测口径（Day 3）**：本系列整合版最终**没有用 shuffle**——行优先映射 + float4 写回已接近合并，v4→v5（coalesced 写回）收益在噪声范围内（4096：64.3%→62.9%），全量 shuffle 重排还要额外 64 条 `SHFL` 指令。shuffle 写回真正值得用的场景（Tensor Core fragment 重排、加载/写回映射冲突、小位宽打包）见 Day 3 §3.2。本节的机制理解仍然必要——它是 Week 3 WMMA 写回重排的前置。

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
| 性能 | 预估 +10-20%（重叠 load/compute） | smem 用量翻倍（可能降 occupancy） |
| 复杂度 | — | prologue/epilogue + 奇偶切换 |
| 适用 | global→smem 传输是瓶颈 | smem 已紧张时不值得 |

> ⚠️ **实测口径（Day 3 v6）**：`gemm_optimization_series.cu` 的 v6 是**同步**双缓冲（`__syncthreads` 后才计算下一 tile），编译器无法自动重叠 load 与 compute，实测与 v5 持平（4096：63.8% vs 62.9%，噪声范围内）。真双缓冲必须用 `cp.async`（Ampere+）或 TMA（Hopper+）异步拷贝——这是 Week 3 CUTLASS 的核心机制。

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

##### 性能实测（统一口径：Day 3 实测系列）

在 RTX 5090（sm_120）上，FP32 FMA 手写 GEMM，对比三种 cuBLAS 基准（4096³）：

| 实现 | 4096 ms | FP32 cuBLAS% | TF32 cuBLAS% | FP16 cuBLAS% | 关键优化 |
|------|---------|-------------|-------------|-------------|---------|
| v1 Naive (1 thread/elem) | 18.936 | 10.6% | 8.1% | 3.5% | 无 |
| v2 SM Tiling | 15.107 | 13.3% | 10.2% | 4.3% | smem tiling |
| v3 RegBlock (8×8/thread, Day 2) | 6.574 | 30.8% | 23.5% | 10.0% | register blocking |
| v4 +float4 | 3.121 | 64.3% | 49.4% | 21.0% | 向量化加载 |
| **v5 整合版（128×128, 256线程）** | **3.178** | **63.4%** | **48.6%** | **20.6%** | tiling+regblk+float4+合并写回 |
| v6 +同步双缓冲 | 3.134 | 63.8% | 49.2% | 20.9% | 双缓冲（同步实现，噪声内） |
| cuBLAS (FP32 纯 FMA) | 2.015 | 100% | 76.6% | 32.5% | FMA 优化 |
| cuBLAS (TF32) | 1.543 | 130.6% | 100% | 42.5% | TF32 Tensor Core |
| cuBLAS (FP16) | 0.655 | 307.6% | 235.6% | 100% | FP16 Tensor Core 峰值 |

> ⚠️ **基准口径说明**：cuBLAS 有三种基准——FP32（纯 FMA，实测 2.015ms / 68.2 TFLOPS）、TF32（Tensor Core，1.543ms / 89.1 TFLOPS）、FP16（Tensor Core，0.655ms / 209.8 TFLOPS）。手写 FMA GEMM 对比不同基准的百分比差异很大：
> - 对比 FP32 cuBLAS：63.4%（纯 FMA vs FMA，最能反映 kernel 优化水平，**本周默认口径**）
> - 对比 TF32 cuBLAS：48.6%（TF32 是 FP32 的默认加速路径）
> - 对比 FP16 cuBLAS：20.6%（生产推理口径）
>
> **面试时说明基准**："手写 FMA 整合版达 FP32 cuBLAS 的 63%、TF32 的 49%。再往上突破靠 Tensor Core（Week 3）"。

##### 为什么 FMA 路线到 ~63% 后必须上 Tensor Core？

手写整合版实测 43.1 TFLOPS，是 FP32 cuBLAS（68.2 TFLOPS）的 63.4%；FP32 理论峰值见 [硬件参数事实源](../../reference/hardware_specs.md)（104.75 TFLOPS），无论手写还是 cuBLAS，FMA 路线都被这条算力线封死。而 TF32 / FP16 Tensor Core 的 cuBLAS 实测达 89.1 / 209.8 TFLOPS——FMA 路线继续抠工程细节（swizzle、auto-tune）最多逼近 FP32 cuBLAS，数量级突破必须换 Tensor Core（Week 3）。

> 💡 **面试要点**：FMA GEMM 的本周实测天花板是 FP32 cuBLAS 的 ~63%（TF32 基准 ~49%）。突破靠 Tensor Core（Week 3 的 WMMA/mma.sync）。

---

### Coding 任务

#### 任务 1：实现 Warp Shuffle 写回

基于 Day 2 的 Register Blocking GEMM，加入 `__shfl_sync` 优化写回：

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

> ⚠️ 本任务代码为**结构示意片段**（不落盘为独立文件）。Day 3 的实测结论是本系列不需要 shuffle 写回（见 §4.1 的实测注记）；建议在 Day 3 的 `integrated_gemm.cu` 上实验开启/关闭 shuffle 路径，用数据验证结论。

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

> ⚠️ 本任务代码为**结构示意片段**（不落盘为独立文件）。可编译的同步双缓冲实现见 Day 3 的 `gemm_optimization_series.cu` v6；`cp.async` 真双缓冲的工业级实现是 Week 3 CUTLASS 的主题。

#### 任务 3：复跑整合版，统一性能口径

整合版 kernel 在 Day 3 目录下，复跑验证数据可重现：

```bash
nvcc -O3 -arch=sm_120 ../day3/kernels/integrated_gemm.cu -o integrated_gemm -lcublas
./integrated_gemm
```

实测输出（RTX 5090, sm_120, FP32 cuBLAS baseline）——与 Day 3 任务 2 留档一致：

```text
=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===
BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256

M        N        K        Our(ms)    cuBLAS(ms) GFLOPS    Percent
----------------------------------------------------------------
1024     1024     1024     0.143      0.064      15.1      44.8%   PASS
2048     2048     2048     0.427      0.267      40.7      62.3%   PASS
4096     4096     4096     3.178      2.015      43.1      63.4%   PASS
8192     8192     8192     24.830     15.920     44.4      64.1%   PASS
```

> ⚠️ **口径提醒**：上表 Percent 是 **FP32 cuBLAS** 基准。旧版教程此处曾引用另一组"7-Layer Benchmark"（整合版 ~4.4ms / FP32 46% / TF32 20-30%），那是旧 kernel 的历史口径，已废止——本周唯一事实源是 Day 3 的 v1–v6 实测表。
>
> **结论**：教学版 FMA GEMM 的实测天花板是 FP32 cuBLAS 的 ~63%（4096）。要再往上，一条是 CUTLASS 级的 FMA 工程深度（`cp.async` + swizzle + auto-tune），一条是 Week 3 的 Tensor Core（WMMA/mma.sync）。

**cuBLAS 三基准实测**（RTX 5090, 4096×4096）：

| 基准 | 时间 (ms) | TFLOPS | 说明 |
|------|----------|--------|------|
| FP32 cuBLAS (纯 FMA) | 2.015 | 68.2 | `allow_tf32=False` |
| TF32 cuBLAS | 1.543 | 89.1 | `allow_tf32=True`（默认） |
| FP16 cuBLAS | 0.655 | 209.8 | FP16 Tensor Core |

#### 任务 4：LeetGPU 在线题目 —— GEMM（复用，cp.async 改造）

**题目链接**：<https://leetgpu.com/challenges/general-matrix-multiplication-gemm>

**与今日知识的关联**：

Day 2 已做过本题的 Register Blocking 版本；今天复用同题做进阶改造——把 global→shared 加载改成 `cp.async` 双缓冲，对比改造前后的通过耗时，亲手验证"同步双缓冲收益在噪声内、异步拷贝才是关键"的实测结论。

> 💡 完整题解见 [GEMM 题解](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gemm-solution.html)。

#### 任务 5：LeetCode 面试题（10 周计划 · 第 2 周 Day 4）

> 📅 今日题目来自 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 2 周「字符串、滑动窗口与矩阵」Day 4（字符串匹配），共 4 题。简单题快速过、中等题精做、困难题吃透；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [165. 比较版本号](https://leetcode.cn/problems/compare-version-numbers/) | 中等 | 字符串切分 + 逐段比较 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/165_比较版本号.html) |
| [8. 字符串转换整数（atoi）](https://leetcode.cn/problems/string-to-integer-atoi/) | 中等 | 模拟 + 溢出边界处理 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/8_字符串转换整数atoi.html) |
| [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) | 简单 | KMP / 内置查找 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/28_找出字符串中第一个匹配项的下标.html) |
| [468. 验证 IP 地址](https://leetcode.cn/problems/validate-ip-address/) | 中等 | 分段 + 规则校验 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/468_验证IP地址.html) |

---

### 扩展实验

#### 实验 1：逐层开启验证

以 Day 3 的 `gemm_optimization_series.cu` 为对象，逐层对比 v4 → v5 → v6，记录每层收益并与 Day 3 实测表核对（预期：v4→v5、v5→v6 的收益都在噪声范围内——用数据验证"shuffle 写回与同步双缓冲对本系列无效"的结论）。

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

1. **Shuffle 写回**：warp 内寄存器转置让写回合并；Day 3 实测整合版未采用（映射选对 + 写回占比小），真正的主场是 Tensor Core fragment 重排（Week 3）
2. **Double Buffer**：同步实现实测收益在噪声范围内（4096：62.9%→63.8%）；真双缓冲需要 `cp.async`（Ampere+）或 TMA（Hopper+）
3. **三基准口径**：整合版 4096 = FP32 cuBLAS 63.4% / TF32 48.6% / FP16 20.6%——报数字先说基准
4. **FMA 天花板**：手写 FMA 整合版 43.1 TFLOPS，本周实测到 FP32 cuBLAS 的 ~63%；再往上靠 Tensor Core（Week 3）
5. **Prologue/Epilogue**：DblBuf 的边界处理——预加载 + 末块计算

---

### 面试要点

1. **GEMM 七层优化每层做了什么？收益各多少？**

   <details>
   <summary>答案</summary>

   以 Day 3 实测为准（RTX 5090，4096³，FP32 cuBLAS 基准）：

   | 层 | 优化 | 单步收益 | 累计 cuBLAS% |
   |---|------|------|------------|
   | L1 Naive | 无 | — | 10.6% |
   | L2 Tiling | smem 分块 | +2.7% | 13.3% |
   | L3 RegBlock | 寄存器累加 | +17.5% | 30.8% |
   | L4 float4 | 向量化加载 | +33.5%（最大单步） | 64.3% |
   | L5 合并写回 | float4 coalesced 写回 | 噪声内 | 62.9% |
   | L6 DblBuf | 同步双缓冲 | 噪声内 | 63.8% |
   | L7 整合版 | 全部合并（单文件实测） | — | 63.4% |

   ⚠️ 面试中常见的"15%→45%→…→82%"是经验预估 ladder，与本机实测口径不同——先报实测，再补预估。

   </details>

2. **Double Buffer 的 prologue/epilogue 怎么处理？**

   <details>
   <summary>答案</summary>

   - Prologue：循环前预加载第一块到 buffer[0]，否则第一次 compute 无数据
   - Epilogue：循环后计算最后一块（最后一次异步加载的数据还没被 compute）
   - 奇偶切换：`cur = (cur+1) % 2` 交替使用两个 buffer
   - 代价：smem 翻倍，可能降 occupancy

   </details>

3. **FMA GEMM 为什么到 ~63% 就上不去了？**

   <details>
   <summary>答案</summary>

   - 实测：手写 FMA 整合版达 FP32 cuBLAS 的 63.4%、TF32 cuBLAS 的 48.6%（4096³，RTX 5090）
   - cuBLAS 三基准实测：FP32 68.2 / TF32 89.1 / FP16 209.8 TFLOPS；FP32 理论峰值 104.75 TFLOPS（见 [硬件参数事实源](../../reference/hardware_specs.md)）
   - FMA 路线被 FP32 算力线封死：手写 43.1 TFLOPS，继续抠工程细节（swizzle、auto-tune）最多逼近 FP32 cuBLAS 的 68.2 TFLOPS
   - 突破的两条路：① CUTLASS 级 FMA 工程（`cp.async` + swizzle + auto-tune）→ 逼近 FP32 cuBLAS ② 用 Tensor Core（Week 3 WMMA/mma.sync）→ 90%+

   </details>

4. **Warp Shuffle 在 GEMM 写回中起什么作用？**

   <details>
   <summary>答案</summary>

   - 问题：Register Blocking 的写回是非合并的（每线程写分散位置）
   - Shuffle 让 warp 内线程交换累加器数据，使每线程最终持有连续列
   - 连续列可用 float4 合并写回，带宽利用率提升
   - 预估收益 ~5-10%（写回占 GEMM IO 比例较小）；Day 3 实测本系列未采用——行优先映射 + float4 写回已接近合并（v4→v5 在噪声内：64.3%→62.9%），shuffle 的主场是 Tensor Core fragment 重排（Week 3）

   </details>

5. **整合版 GEMM 的 ncu 瓶颈是什么？**

   <details>
   <summary>答案</summary>

   - 整合版仍可能是 memory-bound（`dram__throughput` 高）
   - 或 compute-bound（`sm__throughput` 接近峰值但 Tensor Core 利用率 0%）
   - 瓶颈转移到"FMA 峰值限制"——不用 Tensor Core，算力上限就是 FMA 峰值
   - 优化方向：用 Tensor Core（Week 3）突破 FMA 天花板

   </details>
