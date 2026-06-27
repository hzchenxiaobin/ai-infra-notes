## Day 6：Nsight Profiling 实战

### 🎯 目标

通过今天的学习，你将：

1. 理解 Nsight Compute 和 Nsight Systems 的定位和区别
2. 掌握常用 `ncu` 命令和关键指标
3. 掌握常用 `nsys` 命令和时间线分析
4. 能用 Roofline 模型判断 kernel 瓶颈类型
5. 能对本 week's kernel 进行系统性的 profiling
6. 能写出一份完整的 profiling 报告

> 💡 **为什么重要**：前面的理论学习告诉你"应该怎么做"，而 profiling 告诉你"实际情况是什么"。没有 profiling，优化就是盲人摸象。Nsight 是 AI Infra 工程师的"听诊器"。

---

### 学前导读：为什么需要 profiling

写 CUDA 代码时，我们经常会有各种假设：
- "这个 kernel 应该是 memory-bound"
- "加了 shared memory 应该会更快"
- "这个优化应该能提升 2 倍"

但真实 GPU 执行时，情况可能完全不同。Profiling 的作用就是：
- **验证假设**：实际瓶颈到底在哪里
- **量化性能**：用数字说话，而不是感觉
- **发现隐藏问题**：如 bank conflict、low occupancy、launch overhead 等
- **指导优化方向**：避免在无效方向上浪费时间

** profiling 的黄金法则**：
> 不要猜测，要测量。

---

### 理论学习

#### 6.1 Nsight 工具家族

![Nsight Tools Comparison](../website/images/nsight_tools_comparison.svg)

NVIDIA 提供了两个主要的 profiling 工具：

##### Nsight Compute (`ncu`)

- **粒度**：单个 kernel
- **用途**：分析 kernel 内部的详细硬件指标
- **适用场景**：
  - 判断 kernel 是 memory-bound 还是 compute-bound
  - 查看 occupancy、register usage、shared memory
  - 分析 memory throughput、compute throughput
  - 查看 bank conflict、cache hit rate
  - 生成 Roofline 图

##### Nsight Systems (`nsys`)

- **粒度**：整个应用
- **用途**：分析时间线、CPU/GPU 交互、kernel launch overhead
- **适用场景**：
  - 找到最耗时的 kernel
  - 分析 CPU 和 GPU 的并行情况
  - 查看 kernel launch overhead
  - 分析多个 stream 的并行执行
  - 端到端 latency 分析

**使用流程**：

![Profiling Workflow](../website/images/profiling_workflow.svg)

1. 先用 **nsys** 找到最耗时的 kernel
2. 再用 **ncu** 深入分析该 kernel
3. 根据分析结果优化
4. 重复 profiling 验证效果

#### 6.2 常用 `ncu` 命令

##### 基础命令

```bash
# 基本 profiling，使用默认指标集
ncu ./your_kernel

# 指定单个指标
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed ./your_kernel

# 指定多个指标
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./your_kernel

# 生成完整报告
ncu --set full -o report ./your_kernel

# 用 GUI 打开报告
ncu-ui report.ncu-rep
```

##### 常用指标分类

![ncu Metrics Overview](../website/images/ncu_metrics_overview.svg)

| 类别 | 指标 | 含义 |
|------|------|------|
| 并行度 | `sm__occupancy.avg.pct_of_peak_sustained_elapsed` | Occupancy 百分比 |
| 并行度 | `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | 活跃 warp 比例 |
| 并行度 | `launch__registers_per_thread` | 每个线程寄存器数 |
| 内存 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 显存带宽利用率 |
| 内存 | `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | Global load 字节数 |
| 内存 | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory load bank conflict |
| 计算 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 计算单元利用率 |
| 计算 | `sm__cycles_elapsed.avg` | 执行周期数 |
| 延迟 | `gpu__time_duration.avg` | Kernel 执行时间 |

#### 6.3 常用 `nsys` 命令

```bash
# 基本时间线分析
nsys profile -o timeline ./your_app

# 打开 GUI 查看
nsys-ui timeline.nsys-rep

# 同时采集 CUDA 和 NVTX
nsys profile -o timeline --trace cuda,nvtx,osrt ./your_app

# 只统计 summary
nsys profile --stats=true ./your_app
```

**nsys 时间线重点**：
- `cudaLaunchKernel`：CPU 端 launch kernel 的时间
- `Kernel`：GPU 上实际执行的时间
- `cudaMemcpy`：数据传输时间
- `cudaStreamSynchronize`：同步等待时间

#### 6.4 如何判断瓶颈类型

![Memory vs Compute Bound](../website/images/memory_compute_bound.svg)

判断一个 kernel 是 memory-bound 还是 compute-bound 的方法：

**Memory-bound 特征**：
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` 高（接近 80-100%）
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` 低
- Arithmetic Intensity 低

**Compute-bound 特征**：
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` 高
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` 低
- Arithmetic Intensity 高

**Latency-bound 特征**：
- 两者都低
- 可能是 occupancy 太低，或依赖链太长

#### 6.5 Roofline 模型

![Roofline 模型](../website/images/roofline_model.svg)

Roofline 图帮助判断 kernel 是 compute-bound 还是 memory-bound：

```
Attainable FLOP/s = min(Peak FLOP/s, AI * Peak Bandwidth)
```

- **Arithmetic Intensity (AI)** = FLOPs / bytes
- AI 低 → memory-bound（位于斜线区域）
- AI 高 → compute-bound（位于平顶区域）

**如何用 Roofline 指导优化**：
- 如果点在斜线区域：优化内存访问（coalescing、shared memory、减少读写）
- 如果点在平顶区域：优化计算（Tensor Core、指令优化）
- 如果点离平顶很远：还有很大优化空间

---

### Coding 任务：本周 kernel  profiling

#### 任务 1：profiling hello_gpu

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/hello_gpu kernels/hello_gpu.cu
nsys profile -o profiles/day1_hello_gpu_timeline ./kernels/hello_gpu
```

**观察重点**：
- `cudaLaunchKernel` 的 CPU 耗时
- kernel 在 GPU 上的实际执行时长
- 多个 block 是否并行执行

#### 任务 2：profiling occupancy_test

```bash
ncu \
  --metrics \
    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    launch__shared_mem_per_block_static,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./kernels/occupancy_test
```

**观察重点**：
- 实际 occupancy
- 每个线程寄存器数
- 计算单元利用率

#### 任务 3：profiling transpose

```bash
ncu \
  --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    sm__cycles_elapsed.avg \
  ./kernels/transpose
```

**观察重点**：
- naive 和 optimized 版本的 dram throughput 差异
- global memory read/write 数据量
- 执行 cycle 数

#### 任务 4：profiling bank_conflict

```bash
ncu \
  --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    sm__cycles_elapsed.avg \
  ./kernels/bank_conflict
```

**观察重点**：
- conflict 和 no-conflict 版本的 bank conflict 计数
- 执行 cycle 差异

#### 任务 5：LeetGPU 在线题目 —— Matrix Multiplication

**题目链接**：<https://leetgpu.com/challenges/matrix-multiplication>

**题目概述**：

给定 M×K 矩阵 A 和 K×N 矩阵 B（行优先存储），计算 C = A × B，其中 C 为 M×N 矩阵，C[i][j] = Σ(A[i][k] * B[k][j])。

**约束条件**：`1 ≤ M, N, K ≤ 1024`，矩阵元素范围 `[-1.0, 1.0]`

**难度**：中等　**标签**：CUDA、GEMM、Shared Memory Tiling、Roofline、Profiling

**与今日知识的关联**：

本题是 GEMM 的基础版，适合用 ncu 做完整 profiling。用 Day 6 学的 Nsight Compute 分析 SM throughput、memory throughput、occupancy，画出 Roofline 图，判断 kernel 是 memory-bound 还是 compute-bound。

**解题思路**：

先用 naive GEMM（每线程算一个元素）作为 baseline，然后加 Shared Memory Tiling 优化。用 ncu --set full 生成完整报告，对比两个版本的关键指标。

**参考实现**：

```cuda
#define TILE_SIZE 16

// Naive baseline
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Shared Memory Tiling 优化版
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        // 协作加载 tile
        if (row < M && bk + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}
```

> 💡 提交后在 LeetGPU 上记录通过耗时，用 ncu 对比不同 block size / tile size 的性能差异。完整题解见 [LeetGPU/leetgpu-matrix-multiplication-solution.md](../../LeetGPU/leetgpu-matrix-multiplication-solution.md)。

---

### 扩展实验

#### 实验 1：生成完整 ncu 报告

对每个 kernel 生成完整报告：

```bash
ncu --set full -o profiles/day6_hello_gpu ./kernels/hello_gpu
ncu --set full -o profiles/day6_occupancy_test ./kernels/occupancy_test
ncu --set full -o profiles/day6_transpose ./kernels/transpose
ncu --set full -o profiles/day6_bank_conflict ./kernels/bank_conflict
```

用 ncu-ui 打开，阅读每个指标的详细说明。

#### 实验 2：绘制 Roofline 图

手动计算每个 kernel 的：
- FLOPs
- Bytes
- Arithmetic Intensity

然后在 Roofline 图上标出位置。可以使用 ncu 的 Roofline 图，也可以自己用 matplotlib 绘制。

#### 实验 3：分析 kernel launch overhead

用 nsys 查看：
- `cudaLaunchKernel` 到 kernel 开始执行的时间
- 这个时间占整个应用时间的比例
- 思考如何减少 launch overhead（如 CUDA Graph）

---

### 常见错误与调试

| 问题 | 原因 | 解决 |
|------|------|------|
| ncu 报错 `Failed to profile` | 权限问题或驱动不兼容 | 用 sudo 或检查驱动版本 |
| nsys 生成的报告很大 | 采集了太多 trace | 只采集需要的 trace，如 `--trace cuda` |
| 指标返回 N/A | 该指标在当前 GPU 上不可用 | 查看 ncu 文档确认支持的指标 |
| Roofline 图无法生成 | 缺少 FLOP 指标 | 使用 `--set full` 重新采集 |
| 报告打开后看不懂 | 指标太多 | 先看 occupancy、throughput、bank conflict 这几个核心指标 |

---

### 验证 Checklist

- [ ] 生成至少 3 个 kernel 的 Nsight Compute 报告
- [ ] 生成至少 1 个 Nsight Systems 时间线报告
- [ ] 能读取 Roofline 图并定位瓶颈类型
- [ ] 能判断 kernel 是 memory-bound / compute-bound / latency-bound
- [ ] 记录各 kernel 的 throughput 和 occupancy
- [ ] 整理 profiling 结果到 [profiles/week1_profile_summary.md](../profiles/week1_profile_summary.md)

---

### 今日总结

Day 6 我们学会了用专业工具分析 GPU 性能：

1. **Nsight Systems (`nsys`)**：应用级时间线分析，找耗时 kernel
2. **Nsight Compute (`ncu`)**：kernel 级详细指标分析
3. **核心指标**：occupancy、memory throughput、compute throughput、bank conflict
4. **瓶颈判断**：memory-bound、compute-bound、latency-bound
5. **Roofline 模型**：用 arithmetic intensity 判断优化方向
6. **工作流程**：nsys 定位 → ncu 深入 → 优化 → 再验证

掌握这些后，你就不再是"凭感觉优化"，而是"用数据驱动优化"。

---

### 面试要点

1. **如何判断一个 kernel 是 memory-bound 还是 compute-bound？**
   - 看 `dram__throughput` 和 `sm__throughput`
   - Memory-bound：memory throughput 高，compute throughput 低
   - Compute-bound：compute throughput 高，memory throughput 低
   - 也可以用 Roofline 模型：AI 低则 memory-bound，AI 高则 compute-bound

2. **Nsight Compute 和 Nsight Systems 的区别？**
   - Nsight Compute：kernel 级详细指标
   - Nsight Systems：应用级时间线，看 CPU/GPU 交互和整体流程

3. **Roofline 模型如何指导优化？**
   - 如果点在斜线区域（memory-bound）：优化内存访问
   - 如果点在平顶区域（compute-bound）：优化计算
   - 目标是让点尽量接近屋顶

4. **常用的 ncu 指标有哪些？**
   - `sm__occupancy.avg.pct_of_peak_sustained_elapsed`
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
   - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`

5. **nsys 时间线能看到什么？**
   - Kernel launch overhead
   - CPU 和 GPU 的执行时间线
   - CUDA API 调用顺序
   - 多个 stream 的并行情况

---
