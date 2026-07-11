## Day 7：限时 Kernel 手撕 + GitHub 整理 + 性能对比报告

### 🎯 目标

通过今天的学习，你将：

1. 用限时手撕的方式检验本周 6 天的学习成果，把「看懂」变成「写得出」
2. 在 30 分钟内手写一个带 Warp Shuffle 的 Block Reduce Kernel
3. 在 60 分钟内手写一个含 Shared Memory Tiling + Register Blocking 的 GEMM Kernel
4. 不看资料口述 FlashAttention 的完整算法流程与 Online Softmax 三公式
5. 整理本周所有产出代码与 README，形成可展示的 GitHub 仓库
6. 编写从 Naive 到 cuBLAS 的性能对比报告，量化每一层优化的收益

> 💡 **为什么重要**：面试现场就是限时手写 + 口述。能否在白板上写出 Warp Shuffle 循环、画出两级归约结构、推导 Online Softmax，直接决定 AI Infra 岗位的成败。同时，一份干净的 GitHub 仓库和性能报告，是项目深度的最佳证明。

---

### 学前导读：为什么要限时手撕

本周 Day 1–Day 6 覆盖了五大主题：Warp Shuffle、Register Blocking、CUDA Streams、Nsight Profiling、FlashAttention。但「读懂代码」和「白板写出代码」之间有一道巨大的鸿沟。

**读懂 ≠ 会写**：

| 状态 | 典型表现 | 面试结果 |
|------|---------|---------|
| 看懂 | 能解释每行代码的作用 | 被追问细节时卡壳 |
| 会写 | 闭卷能写出核心结构 | 通过手撕题 |
| 会调 | 能定位 bug 并修复 | 拿到 offer |

限时手撕的训练目的，是把知识从「短期记忆」固化到「肌肉记忆」。当你能在 30 分钟内不查资料写出 `__shfl_down_sync` 的 butterfly 循环，说明 Warp Shuffle 真正内化了。

**今日节奏建议**（全天 6 小时）：

| 时段 | 任务 | 时长 |
|------|------|------|
| 上午 | 30 分钟手撕 Reduce + 复盘 | 2h |
| 上午 | 60 分钟手撕 GEMM + 复盘 | 2h |
| 下午 | FlashAttention 口述训练 | 1h |
| 下午 | GitHub 仓库整理 + 性能报告 | 1h |

---

### 理论学习：本周知识图谱回顾

在开始手撕前，先用 10 分钟回顾本周的知识脉络，确保脑子里有完整的优化层次图。

#### 1. 本周优化层次全景

```
Level 0: Naive GEMM (~1%)
 └── 每线程算一个元素，直接访问 Global Memory
 │
Level 1: Shared Memory Tiling (~15%) ← Day 1 基础
 └── A/B tile 预取到 Shared Memory，K 维复用
 └── 关键：协作加载 + __syncthreads
 │
Level 2: Register Blocking (~45%) ← Day 2
 └── 每线程算 TM×TN 子块，acc 驻留寄存器
 └── 关键：threadRow/threadCol 映射 + r_A/r_B
 │
Level 3: Vectorized Load (~55%) ← Day 6
 └── float4 做 128-bit Global Memory 加载
 └── 关键：地址 16 字节对齐
 │
Level 4: Warp-level Optimize (~60%) ← Day 1+6
 └── Warp Shuffle 协作 + 优化写回
 └── 关键：__shfl_xor_sync 减少非合并访问
 │
Level 5: Double Buffering (~70%) ← Day 6
 └── 两份 Shared Memory 交替，计算掩盖传输
 └── 关键：软件流水线 + __syncthreads 位置
 │
Level 6: Tensor Core / CUTLASS (~90%+) ← 超出本周范围
 └── WMMA 指令调用 Tensor Core
```

#### 2. 三大核心数据结构回忆

手撕前必须默写出来的三个结构：

| 结构 | 代码 | 出处 |
|------|------|------|
| Warp Reduce | `for (offset=16; offset>0; offset>>=1) val += __shfl_down_sync(0xFFFFFFFF, val, offset);` | Day 1 |
| Register 累加器 | `float acc[TM][TN] = {0}; float r_A[TM], r_B[TN];` | Day 2 |
| Online Softmax | `m_new=max(m,mj); l_new=l*exp(m-m_new)+Σexp(xj-m_new);` | Day 5 |

#### 3. 本周面试高频题自测

开始手撕前，先快速自测能否口答以下问题（每题不超过 30 秒）：

1. `__shfl_down_sync(0xFFFFFFFF, val, 16)` 四个参数含义？
2. 两级归约中，第二级为什么由 Warp 0 做？
3. Register Blocking 的 register 用量怎么算？TM=TN=8 是多少？
4. Default Stream 有什么坑？
5. FlashAttention 为什么比标准 Attention 快？（用 HBM 访问次数回答）

> 如果以上任何一题卡壳，先回看对应 Day 的「面试要点」再开始手撕。

---

### Coding 任务

#### 任务 1：30 分钟手写 Block Reduce Kernel

##### 模拟规则

- **条件**：关闭所有参考资料，打开一个空的 `.cu` 文件
- **时间**：30 分钟（含编译调试）
- **要求**：
 - [ ] 包含 `warpReduceSum` 函数（使用 `__shfl_down_sync`）
 - [ ] 包含 `blockReduceSum` Kernel（Warp 级 + Shared Memory + Warp 0 二级归约）
 - [ ] 包含 Host 端的 grid-stride 调用（两次 kernel launch 汇总多 block）
 - [ ] 代码能编译运行，结果与 CPU 对比误差 < 1e-3

##### 评分标准

| 项目 | 分值 | 评分要点 |
|------|------|---------|
| `__shfl_down_sync` 正确使用 | 30 | mask=0xFFFFFFFF、butterfly 循环 offset=16→8→4→2→1 |
| 两级归约结构 | 30 | Warp 级 → Shared Memory 中转 → Warp 0 最终归约 |
| `__syncthreads()` 位置 | 20 | Shared Memory 写后 sync、Warp 0 reduce 前已 sync |
| grid-stride 循环 | 10 | `for (i=tid; i<n; i+=gridDim.x*blockDim.x)` |
| 代码整洁度 | 10 | 命名规范、无内存泄漏 |

##### 参考答案（复盘时对比）

```cuda
// block_reduce_timed.cu —— 30 分钟手撕参考实现
// 编译: nvcc -o block_reduce block_reduce_timed.cu -O3 -arch=sm_80
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
 #pragma unroll
 for (int offset = 16; offset > 0; offset >>= 1) {
 val += __shfl_down_sync(0xFFFFFFFF, val, offset);
 }
 return val;
}

__global__ void blockReduceSum(const float* in, float* out, int n) {
 __shared__ float warpSums[32];
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 int lane = threadIdx.x & 31;
 int wid = threadIdx.x >> 5;

 // Step 1: grid-stride 累加
 float sum = 0.0f;
 for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
 sum += in[i];
 }

 // Step 2: Warp 级归约
 sum = warpReduceSum(sum);

 // Step 3: lane 0 写入 Shared Memory
 if (lane == 0) warpSums[wid] = sum;
 __syncthreads();

 // Step 4: Warp 0 做最终归约
 if (wid == 0) {
 int numWarps = (blockDim.x + 31) >> 5;
 sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
 sum = warpReduceSum(sum);
 if (lane == 0) out[blockIdx.x] = sum;
 }
}

int main() {
 const int N = 1 << 22;
 float *h_in = (float*)malloc(N * sizeof(float));
 for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) * 0.001f;

 float *d_in, *d_tmp, *d_out;
 cudaMalloc(&d_in, N * sizeof(float));
 cudaMalloc(&d_tmp, 1024 * sizeof(float));
 cudaMalloc(&d_out, sizeof(float));
 cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

 int threads = 256;
 int blocks = min((N + threads - 1) / threads, 1024);
 blockReduceSum<<<blocks, threads>>>(d_in, d_tmp, N);
 blockReduceSum<<<1, 256>>>(d_tmp, d_out, blocks);

 float gpuSum;
 cudaMemcpy(&gpuSum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

 double cpuSum = 0.0;
 for (int i = 0; i < N; i++) cpuSum += h_in[i];

 printf("GPU=%.4f CPU=%.4f diff=%.6f %s\n",
 gpuSum, (float)cpuSum, fabs(gpuSum - (float)cpuSum),
 fabs(gpuSum - (float)cpuSum) < 1e-3 ? "PASS" : "FAIL");

 free(h_in);
 cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
 return 0;
}
```

##### 复盘要点

手撕后对照参考实现，重点检查这些易错点：

| 易错点 | 现象 | 正确做法 |
|--------|------|---------|
| 忘记 `__syncthreads()` | Warp 0 读到脏数据 | Shared Memory 写入后、Warp 0 读取前必须 sync |
| `numWarps` 算错 | block 不是 32 整数倍时漏 warp | `numWarps = (blockDim.x + 31) / 32` |
| 第二级归约不用 Shuffle | 性能差 | Warp 0 有 32 lane，正好处理 32 个 warp 的部分和 |
| grid-stride 步长错 | 越界或漏元素 | 步长 = `gridDim.x * blockDim.x`（总线程数） |
| `lane == 0` 写回遗漏 | 多 block 结果丢失 | 只有 lane 0 写 `out[blockIdx.x]` |

---

#### 任务 2：60 分钟手写 Register Blocking GEMM

##### 模拟规则

- **条件**：关闭所有参考资料，空文件
- **时间**：60 分钟（含编译调试）
- **要求**：
 - [ ] 包含 Shared Memory Tiling（`s_A[BM][BK]`, `s_B[BK][BN]`）
 - [ ] 包含 Register Blocking（`acc[TM][TN]` 累加器、`r_A[TM]`、`r_B[TN]`）
 - [ ] 包含协作加载 Global → Shared
 - [ ] 包含正确的线程到输出 tile 的二维映射
 - [ ] 三重循环结构正确（外层 bk、内层 k、最内层 m×n FMA）
 - [ ] 代码能编译运行，与 cuBLAS 结果误差 < 1e-2

##### 评分标准

| 项目 | 分值 | 评分要点 |
|------|------|---------|
| Shared Memory 声明与加载 | 25 | `s_A[BM][BK]`/`s_B[BK][BN]` 声明、协作加载逻辑 |
| Register Blocking 结构 | 25 | `acc[TM][TN]` 累加器、`r_A`/`r_B` 加载 |
| 线程映射 | 20 | `threadRow = tid/(BN/TN)`、`threadCol = tid%(BN/TN)` |
| 三重循环结构 | 15 | 外循环 `bk`、中循环 `k`、内循环 `m`×`n` |
| 写回 Global Memory | 10 | 全局索引 `cRow + threadRow*TM + m` 计算 |
| 代码整洁度 | 5 | 命名规范、`__syncthreads` 位置正确 |

##### 参考答案骨架（复盘时对比）

```cuda
// gemm_timed.cu —— 60 分钟手撕参考骨架（BM=BN=128, BK=8, TM=TN=8）
#include <cuda_runtime.h>
#include <cstdio>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM/TM)*(BN/TN)) // 256

__global__ void gemmRegisterBlocking(const float* A, const float* B, float* C,
 int M, int N, int K) {
 __shared__ float s_A[BM][BK];
 __shared__ float s_B[BK][BN];

 float r_A[TM];
 float r_B[TN];
 float acc[TM][TN] = {{0}};

 int threadRow = threadIdx.x / (BN / TN); // 0~15
 int threadCol = threadIdx.x % (BN / TN); // 0~15
 int cRow = blockIdx.y * BM;
 int cCol = blockIdx.x * BN;

 for (int bk = 0; bk < K; bk += BK) {
 // 协作加载 A tile: 256 线程加载 128*8=1024 元素，每线程 4 个
 #pragma unroll
 for (int i = 0; i < BM; i += NUM_THREADS / BK) {
 int row = threadIdx.x / BK + i;
 int col = threadIdx.x % BK;
 if (cRow + row < M && bk + col < K)
 s_A[row][col] = A[(cRow + row) * K + (bk + col)];
 else
 s_A[row][col] = 0.0f;
 }
 // 协作加载 B tile: 256 线程加载 8*128=1024 元素，每线程 4 个
 #pragma unroll
 for (int i = 0; i < BK; i += NUM_THREADS / BN) {
 int row = threadIdx.x / BN + i;
 int col = threadIdx.x % BN;
 if (bk + row < K && cCol + col < N)
 s_B[row][col] = B[(bk + row) * N + (cCol + col)];
 else
 s_B[row][col] = 0.0f;
 }
 __syncthreads();

 // Register Blocking 计算
 #pragma unroll
 for (int k = 0; k < BK; k++) {
 #pragma unroll
 for (int m = 0; m < TM; m++) r_A[m] = s_A[threadRow*TM + m][k];
 #pragma unroll
 for (int n = 0; n < TN; n++) r_B[n] = s_B[k][threadCol*TN + n];
 #pragma unroll
 for (int m = 0; m < TM; m++)
 #pragma unroll
 for (int n = 0; n < TN; n++)
 acc[m][n] += r_A[m] * r_B[n];
 }
 __syncthreads();
 }

 // 写回
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 int gRow = cRow + threadRow * TM + m;
 int gCol = cCol + threadCol * TN + n;
 if (gRow < M && gCol < N) C[gRow * N + gCol] = acc[m][n];
 }
 }
}
```

> ⚠️ 上述骨架省略了 `main` 函数和 cuBLAS 对比部分，手撕时主函数可简化（只需能跑通正确性即可），把时间留给 kernel 本身。

##### 复盘要点

| 易错点 | 现象 | 正确做法 |
|--------|------|---------|
| `threadRow`/`threadCol` 算反 | 结果矩阵错位 | `threadRow = tid/(BN/TN)`，`threadCol = tid%(BN/TN)` |
| 协作加载越界 | 段错误或脏数据 | 所有加载加 `if (gRow < M && ...)` 边界判断 |
| `__syncthreads` 缺失 | 数据竞争 | 每次 bk 迭代：加载后 sync、计算后 sync |
| `acc` 未初始化 | 结果随机 | `float acc[TM][TN] = {{0}}` |
| 内层循环顺序错 | 性能差 | 最内层是 m×n FMA，k 在中间，bk 在最外 |
| `r_A`/`r_B` 在 k 循环外加载 | 结果错误 | 必须在每个 k 迭代内重新加载 |

---

#### 任务 3：FlashAttention 口述训练

##### 模拟规则

- **条件**：不看任何资料，对着空气或录音口述
- **时间**：5 分钟口述 + 5 分钟自问自答
- **口述内容要求**：
 1. FlashAttention 解决的问题（标准 Attention 的 O(N²) HBM 访问）
 2. 分块策略（Q tile 驻留 SRAM，K/V tile 逐块滑入）
 3. Online Softmax 三公式推导（`m_new`、`l_new`、`o_new`）
 4. 复杂度分析（HBM 从 O(N²) 降到 O(Nd)）

##### Online Softmax 三公式默写

在白板上写出以下三式，并解释每一项含义：

```
m_new = max(m, max(xj))
l_new = l * exp(m - m_new) + Σ exp(xj - m_new)
o_new = o * (l * exp(m - m_new) / l_new) + (exp(xj - m_new) / l_new) * vj
```

**自问自答清单**（口述时自问自答）：

| 问题 | 参考答案 |
|------|---------|
| 为什么不用全局 softmax，非要 online 递推？ | 每个 KV tile 看不到全局 max，必须增量更新 |
| `exp(m - m_new)` 的作用？ | 统一参考点的缩放因子，把历史累加值对齐到新 max |
| FlashAttention 的加速上限是多少？ | 受限于 HBM 带宽和 SRAM 容量，无法突破 memory bound |
| 标准 Attention 的 HBM 访问次数？ | O(N²d)，S 和 P 矩阵各读写一次 |
| FlashAttention 的 HBM 访问次数？ | O(Nd)，Q/K/V 只读写一次，S/P 不落 HBM |

> 💡 **复盘标准**：能不看资料、5 分钟内完整讲清上述 5 点 + 三公式，即为通过。

#### 任务 4：LeetGPU 综合验收题 —— Max Subarray Sum

**题目链接**：<https://leetgpu.com/challenges/max-subarray-sum>

**题目概述**：给定长度为 `N` 的 `int32` 数组和窗口大小 `window_size`，求所有长度恰好为 `window_size` 的连续子数组的最大和。

**与本周知识的关联**：本题综合了 Week2 两大主线——Prefix Sum（Day1）+ Reduction（Week1 Day4/Day5）。用 prefix sum 计算窗口和，再用 warp shuffle reduction 求最大值，是一道"两阶段 kernel"的综合手撕题。适合在验收日限时完成，检验 prefix sum + reduce 的综合掌握程度。

> 💡 完整题解（含两阶段 kernel 设计、warp shuffle max 归约、atomicMax 跨 block 汇总）见 [Max Subarray Sum 题解](../../leetgpu/week2/day7/leetgpu-max-subarray-sum-solution.md)。

#### 任务 5：GitHub 仓库整理

把本周 Day 1–Day 6 的产出整理成可展示的仓库结构。建议在 `week2/` 下补充：

```
week2/
├── README.md # 本周教程（已有）
├── kernels/
│ ├── warp_reduce.cu # Day 1 产出
│ ├── register_blocking_gemm.cu # Day 2 产出
│ ├── multi_stream_pipeline.cu # Day 3 产出
│ ├── flash_attention.cu # Day 5 产出
│ └── integrated_gemm.cu # Day 6 产出
├── notes/
│ ├── nsight_profile_report.md # Day 4 报告
│ └── week2_summary.md # 本周总结
├── exercise/
│ ├── block_reduce_timed.cu # Day 7 手撕产出
│ └── gemm_timed.cu # Day 7 手撕产出
└── website/ # 静态网站（已有）
```

##### 整理 Checklist

- [ ] 每个 `.cu` 文件顶部有注释：编译命令、功能说明、对应 Day
- [ ] `notes/week2_summary.md` 记录本周学习心得、踩坑、性能数据
- [ ] `kernels/` 中所有 kernel 能独立编译运行
- [ ] 顶层 `README.md` 的 Week 2 链接可跳转

---

#### 任务 6：性能对比报告

在 `week2/day7/notes/` 下创建 `performance_report.md`，记录从 Naive 到 cuBLAS 的完整性能曲线。

##### 报告模板

```markdown
# Week 2 GEMM 性能优化报告

## 测试环境
- GPU: <你的型号，如 NVIDIA RTX 3090 / A100>
- Compute Capability: <如 8.6>
- CUDA Version: <如 12.4>
- cuBLAS 版本: <如 12.4>

## 性能对比表（M=N=K=4096）

| 版本 | 时间(ms) | GFLOPS | cuBLAS 百分比 | 关键优化点 |
|------|---------|--------|--------------|-----------|
| Naive | | | ~1-3% | 无优化 |
| Shared Memory Tiling | | | ~15% | Shared Memory 复用 |
| Register Blocking | | | ~45% | + Register 累加器 |
| + float4 向量化 | | | ~55% | + 128-bit 加载 |
| + Warp Shuffle | | | ~60% | + Warp 级协作 |
| + Double Buffering | | | ~70% | + 软件流水线 |
| cuBLAS | | | 100% | NVIDIA 官方优化 |

## 优化层次收益分析

（记录每一层优化带来的实际增益，与理论值对比，分析差异原因）

## 瓶颈诊断记录

（用 ncu 的关键指标说明每层优化前后瓶颈的变化）
```

##### GFLOPS 计算公式

```
GFLOPS = 2.0 * M * N * K / (time_ms * 1e6)
```

##### 测试矩阵尺寸建议

扫描 `512, 1024, 2048, 4096, 8192`，观察性能百分比随尺寸的变化趋势（通常尺寸越大，手写 kernel 越接近 cuBLAS，因为分块开销被摊薄）。

---

### 扩展实验

#### 实验 1：手撕 Warp Reduce Max

把任务 1 的 sum 改成 max，限时 15 分钟。关键改动：

```cuda
__inline__ __device__ float warpReduceMax(float val) {
 #pragma unroll
 for (int offset = 16; offset > 0; offset >>= 1) {
 val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
 }
 return val;
}
```

思考：为什么 max 归约的 butterfly 循环结构和 sum 完全一样？（答：因为 `max` 满足结合律和交换律，butterfly 模式只依赖这两个性质。）

#### 实验 2：手撕带 `__launch_bounds__` 的 Kernel

限时 20 分钟，写一个故意触发 register spilling 的 kernel 并用 `nvcc -Xptxas -v` 验证。参考 Day 2 的 `register_spill.cu`。

#### 实验 3：BLAS 标准接口扩展

把整合版 GEMM 扩展为 `C = alpha * A * B + beta * C`（BLAS `sgemm` 接口），限时 30 分钟。关键改动：写回时 `C[gRow*N+gCol] = alpha*acc[m][n] + beta*C[gRow*N+gCol]`。

#### 实验 4：benchmark 脚本

写一个 shell 或 Python 脚本，自动扫描矩阵尺寸 `512, 1024, 2048, 4096, 8192`，记录每个版本的性能并生成 CSV 报告。

### 验证 Checklist

- [ ] 30 分钟内完成 Reduce Kernel 手撕，结果与 CPU 误差 < 1e-3
- [ ] 60 分钟内完成 GEMM Kernel 手撕，含 Shared Memory Tiling + Register Blocking
- [ ] 能不看资料口述 FlashAttention 完整流程（5 分钟版本）
- [ ] 能默写 Online Softmax 三公式（`m_new`、`l_new`、`o_new`）
- [ ] GitHub 仓库整理完成，`week2/day*/kernels/` 下所有 `.cu` 可独立编译
- [ ] 性能对比报告完成，包含从 Naive 到 cuBLAS 的完整性能曲线
- [ ] 能回答「和 cuBLAS 的差距在哪」并给出达到 90% 的优化路径

---

### 今日总结

Day 7 是本周的收尾与验收。通过限时手撕，我们把本周五大主题从「看懂」固化到「写得出」：

1. **Reduce 手撕**：验证 Warp Shuffle butterfly 循环 + 两级归约结构的肌肉记忆
2. **GEMM 手撕**：验证 Shared Memory Tiling + Register Blocking + 线程映射的内化程度
3. **FlashAttention 口述**：验证 Online Softmax 三公式与 HBM 复杂度分析的掌握
4. **GitHub 整理**：把零散产出组织成可展示的项目，体现工程能力
5. **性能报告**：量化每一层优化的收益，形成完整的优化方法论闭环

本周从 Day 1 的 Warp Shuffle 原语，到 Day 6 的整合 GEMM 达到 cuBLAS 70%，再到 Day 7 的限时手撕验收，构成了一条完整的「CUDA 进阶优化」学习闭环。掌握这些后，你已经具备了手写高性能 kernel 并系统分析其性能瓶颈的能力，这是 AI Infra 工程师的核心竞争力。

---

### 面试要点

1. **给你 30 分钟，手写一个带 Warp Shuffle 的 Block Reduce Kernel。要求：输入 N 个元素，输出一个总和。**

<details>
<summary>点击查看答案</summary>

 参考答案要点（30 分钟内需写出的核心结构）：

 ```cuda
 // 1. warpReduceSum（~5 分钟）
 __inline__ __device__ float warpReduceSum(float val) {
 for (int offset = 16; offset > 0; offset >>= 1)
 val += __shfl_down_sync(0xFFFFFFFF, val, offset);
 return val;
 }

 // 2. blockReduceSum Kernel（~15 分钟）
 __global__ void blockReduce(const float* in, float* out, int n) {
 __shared__ float warpS[32];
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
 float sum = 0;
 for (int i = tid; i < n; i += gridDim.x * blockDim.x) sum += in[i];
 sum = warpReduceSum(sum);
 if (lane == 0) warpS[wid] = sum;
 __syncthreads();
 if (wid == 0) {
 int numWarps = (blockDim.x + 31) / 32;
 sum = (lane < numWarps) ? warpS[lane] : 0;
 sum = warpReduceSum(sum);
 if (lane == 0) out[blockIdx.x] = sum;
 }
 }

 // 3. Host 调用（~5 分钟）
 // blockReduce<<<numBlocks, 256>>>(d_in, d_tmp, n);
 // blockReduce<<<1, 256>>>(d_tmp, d_out, numBlocks);
 // 剩余时间处理边界条件和编译调试
 ```

 - **评分关键**：`__shfl_down_sync` 参数正确（30 分）、两级归约结构完整（30 分）、`__syncthreads` 位置正确（20 分）

</details>


1. **手写 GEMM 时，Register Blocking 的三重循环结构是怎样的？为什么是这个顺序？**

<details>
<summary>点击查看答案</summary>

 - 外层 `bk`（遍历 K 维度的 tile）、中层 `k`（遍历 BK 内的元素）、内层 `m`×`n`（TM×TN FMA）
 - 顺序原因：`k` 在外会让 `acc` 累加顺序错乱；`m`×`n` 在最内层是因为 FMA 是最密集的计算，放在最内层有利于指令级并行（ILP）和寄存器复用
 - `r_A`/`r_B` 必须在每个 `k` 迭代内重新加载，否则用的是上一个 k 的数据

</details>


1. **不看资料，口述 FlashAttention 为什么比标准 Attention 快。**

<details>
<summary>点击查看答案</summary>

 - **核心**：标准 Attention 把 S=QK^T 和 P=softmax(S) 写回 HBM，HBM 访问 O(N²d)；FlashAttention 用分块 + Online Softmax，S/P 不落 HBM，HBM 访问降到 O(Nd)
 - **分块**：Q tile 驻留 SRAM，K/V tile 逐块滑入，每次只算一个 block 的局部 softmax
 - **Online Softmax**：增量更新 `m`/`l`/`o`，无需等到看到全局数据再做 softmax
 - **复杂度**：HBM 从 O(N²d) → O(Nd)，长序列下加速比显著

</details>


1. **你的 GEMM Kernel 和 cuBLAS 的差距在哪里？要达到 90% 还需要做什么？**

<details>
<summary>点击查看答案</summary>

 - **当前差距**：
 1. 缺少指令级调度优化（cuBLAS 用 PTX 内联汇编精确控制指令发射）
 2. 缺少完整 Double Buffering（软件流水线）
 3. 缺少针对特定尺寸的 auto-tuning（cuBLAS 有庞大参数查找表）
 4. 缺少 Tensor Core（cuBLAS 默认用 WMMA，吞吐远超 FMA）
 - **达到 90% 的路径**：
 1. 引入 Tensor Core（`mma.sync.aligned` 等 WMMA 指令）
 2. 实现完整 Double Buffering
 3. 使用 CUTLASS 库（NVIDIA 开源高性能 GEMM 模板库）
 4. 针对目标尺寸做 exhaustive search 找最优参数

---

</details>

### 面试准备框架

面试中回答 CUDA 优化问题，建议用这个结构：

1. **先给结论**：这个 kernel 是 memory-bound 还是 compute-bound？给出 AI 估算
2. **分层次**：从 Naive 到当前优化，逐层说明每层的收益来源
3. **给数据**：用 ncu 的 SM%/DRAM% 支撑判断
4. **说局限**：和 cuBLAS 的差距在哪，还要做什么

**示例**：

> **Q：你的 GEMM 达到了 cuBLAS 70%，剩下的 30% 差在哪？**
>
> **A**：主要四个差距。第一，没用 Tensor Core，cuBLAS 默认走 WMMA 指令，吞吐远超 FMA。第二，Double Buffering 不完整，global→shared 传输没被计算完全掩盖。第三，缺少 auto-tuning，cuBLAS 有针对每种尺寸的参数查找表。第四，缺少 PTX 内联汇编做指令级调度。达到 90% 的路径是引入 CUTLASS 模板 + Tensor Core + 完整双缓冲。

---

### 常见误区澄清

| 误区 | 正确理解 |
|------|---------|
| Register Blocking 一定比 Shared Memory Tiling 快 | 只有当 TM×TN 不溢出 register（≤255）时才快；TM=TN=16 会 spill 反而暴跌 |
| Double Buffering 总是有收益 | shared memory 翻倍可能降 occupancy；数据量小时启动开销主导 |
| Occupancy 越高 GEMM 越快 | GEMM 是 compute-bound，寄存器压力大时低 occupancy 高 ILP 可能更快 |
| FlashAttention 减少了计算量 | 计算量相同，减少的是 HBM 数据移动（O(N²)→O(Nd)） |
| 多 Stream 一定能加速 | 需 Copy/Compute Engine 独立 + Pinned Memory + 非 Default Stream，缺一不可 |
| ncu 报告的带宽就是峰值 | 需对比 `dram__throughput.pct_of_peak`，实测通常 70-85% 已优秀 |

---

### Week 2 → Week 3 衔接

Week 3 我们将学习 **Transformer 执行本质与算子手写**。为了做好准备，请确保你掌握了：

1. **Warp Shuffle 原语**（Day 1）：Week 3 手写 Softmax/LayerNorm 的 reduce 基础
2. **Register Blocking + Shared Memory Tiling**（Day 2/6）：Week 3 理解 Attention 的 QK^T/PV GEMM 基础
3. **Nsight Profiling**（Day 4）：Week 3 端到端 Profiling Transformer 的工具基础
4. **FlashAttention 简化版**（Day 5）：Week 3 学完整版 FlashAttention 的算法基础
5. **Kernel Fusion 思想**（Day 6）：Week 3 算子接入与融合的工程基础

如果你对这些概念还有模糊，建议回到对应 Day 重新做实验。Week 3 会从 GPU 视角拆解 Transformer 推理流程，手写 memory-bound 算子，是 8 周计划里承上启下的关键一周。

---

### 弹性安排

根据本周完成情况，选择以下一项或多项：

- **补进度**：完成未做的限时手撕和性能对比报告
- **深入方向 1**：实现 Tensor Core 版 GEMM（WMMA 指令），对比 FMA 版性能
- **深入方向 2**：用 CUTLASS 库跑同尺寸 GEMM，对比手写版与官方模板的差距
- **深入方向 3**：阅读 FlashAttention 论文 Section 3，预习 Week 3 完整版
- **面试准备**：和同学互相模拟面试，重点练 30 分钟手撕 + FlashAttention 口述

---

## 📁 本周目录结构

```
week2/
├── README.md # Week 2 概览
├── day1/ # Day 1: Warp Shuffle + Warp/Block Reduce
│ ├── README.md
│ └── kernels/warp_reduce.cu
├── day2/ # Day 2: Register Blocking + 2D Tiling
│ ├── README.md
│ └── kernels/register_blocking_gemm.cu
├── day3/ # Day 3: Multi-Stream + 异步流水线
│ ├── README.md
│ └── kernels/multi_stream_pipeline.cu
├── day4/ # Day 4: Nsight Compute Profiling
│ ├── README.md
│ └── profiles/ # ncu 报告
├── day5/ # Day 5: FlashAttention 简化版
│ ├── README.md
│ └── kernels/flash_attention.cu
├── day6/ # Day 6: 整合优化 GEMM
│ ├── README.md
│ └── kernels/integrated_gemm.cu
├── day7/ # Day 7: 限时手撕 + 验收
│ └── README.md
└── website/ # 网站构建
 ├── build.py
 └── images/ # SVG 插图
```

---

## 🔗 推荐资源

| 资源 | 说明 |
|------|------|
| [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Warp Shuffle / Shared Memory 官方文档 |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | NVIDIA 开源高性能 GEMM 模板库 |
| [FlashAttention 论文](https://arxiv.org/abs/2205.14135) | Online Softmax + Tiling 核心论文 |
| [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) | ncu 指标详解 |
| [NVIDIA GEMM Optimization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/) | 官方 GEMM 优化指南 |

---

## ✅ Week 2 完成标准

- [ ] Warp Reduce Kernel 编译运行正确，GPU 结果与 CPU 误差 < 1e-3
- [ ] Register Blocking GEMM 达到 cuBLAS 40%+（4096 矩阵）
- [ ] 整合版 GEMM 达到 cuBLAS 65%+（含 float4 + Warp Shuffle）
- [ ] FlashAttention 简化版小尺寸测试通过（误差 < 1e-3）
- [ ] 能用 ncu 判断 kernel 是 memory-bound 还是 compute-bound
- [ ] 30 分钟内手写 Block Reduce Kernel（含 Warp Shuffle + 两级归约）
- [ ] 60 分钟内手写 Register Blocking GEMM Kernel
- [ ] 不看资料口述 FlashAttention 算法流程 + Online Softmax 三公式
- [ ] 生成性能对比报告（Naive → cuBLAS 各层 GFLOPS + 占比）
- [ ] 完成本周 LeetGPU（Prefix Sum/GEMM/Convolution/Softmax/Attention/Histogram）与 LeetCode 题目

---

> 💡 **提示**：Week 2 是从"会写 kernel"到"能优化到 cuBLAS 70%"的关键跃迁。限时手撕是面试的硬门槛，性能报告是项目深度的证明。如果 GEMM 还没到 65%，建议回到 Day 2/6 重新做 float4 + Double Buffering 实验。Week 3 会进入 Transformer 算子手写，GEMM 优化经验是理解 Attention 的基础。
