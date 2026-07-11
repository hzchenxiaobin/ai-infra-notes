# 第2周深度展开：CUDA进阶优化与性能分析（7天）

> **适用对象**：陈斌斌（已完成第1周学习，掌握向量加法、Naive GEMM、Shared Memory Tiling GEMM、Softmax Kernel）
> **本周目标**：掌握Warp Shuffle、Register Blocking、CUDA Stream异步执行、Nsight性能分析和FlashAttention CUDA实现
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：手写优化GEMM达到cuBLAS 70%+性能，完成简化版FlashAttention Forward Kernel

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握Warp级原语通信、寄存器阻塞优化、多流异步执行、Nsight性能分析工具和FlashAttention分块计算 |
| **核心产出** | ① Warp Reduce Kernel ② Register Blocking GEMM（cuBLAS 40%+）③ Multi-Stream重叠执行 ④ Nsight分析报告 ⑤ FlashAttention简化版Forward Kernel ⑥ 整合优化GEMM（cuBLAS 70%+） |
| **验收标准** | ① Register Blocking GEMM达到cuBLAS 40%+ ② 整合版GEMM达到cuBLAS 70%+（4096x4096）③ 能推导online softmax三公式 ④ 能独立使用Nsight Compute分析Kernel瓶颈 |
| **面试准备** | 积累10-12道进阶面试题，覆盖Warp Shuffle、Register Blocking、Stream、Profiler、FlashAttention五大主题 |

### 本周知识图谱

```
Day 8: Warp Shuffle原语 → Warp Reduce Kernel（两级归约）
 ↓
Day 9: Register Blocking + 2D Tiling → GEMM cuBLAS 40%+
 ↓
Day 10: CUDA Streams异步 → H2D/Compute/D2H重叠流水线
 ↓
Day 11: Nsight Compute → Register Blocking GEMM瓶颈分析
 ↓
Day 12: FlashAttention → Online Softmax推导 + Forward Kernel
 ↓
Day 13: 整合Warp Shuffle + Register Blocking → GEMM cuBLAS 70%+
 ↓
Day 14: 限时Kernel手撕 + GitHub整理 + 性能对比报告
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 确认GPU支持Warp Shuffle（Compute Capability >= 12.0，Blackwell及以后架构全部支持）
- [ ] `nvcc --version`正常，CUDA Toolkit >= 11.0
- [ ] `ncu --version`正常，Nsight Compute可用
- [ ] `nsys --version`正常，Nsight Systems可用
- [ ] cuBLAS库可链接：`ldconfig -p | grep cublas`

#### 验证命令
```bash
# 验证Warp Shuffle支持（Compute Capability）
nvidia-smi --query-gpu=compute_cap --format=csv
# 预期输出：8.6 或 8.0 或 7.5 等（>= 3.0即可）

# 验证Nsight Compute
ncu --version
# 预期输出：Nsight Compute 202x.x.x

# 验证cuBLAS
ls /usr/local/cuda/lib64/libcublas.so*
# 预期输出：libcublas.so.12 等文件存在
```

---

## Day 8（周一）：Warp Shuffle原语

> **今日目标**：理解Warp级线程通信原语，掌握`__shfl_*_sync`家族API，手写完整的Warp Reduce + Block Reduce两级归约Kernel。
> **面试考察度**：⭐⭐⭐⭐ 必考，Warp Shuffle是手写Reduce的标配技术

---

### 学习任务1：精读Warp Shuffle（45分钟）

#### 阅读内容

- **资源地址**：https://face2ai.com/program-blog/
- **阅读范围**：谭升博客"GPU编程（CUDA）"分类下的Warp Shuffle专题章节
- **补充阅读**：CUDA C Programming Guide → "Warp Shuffle Functions"章节（https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions）
- **具体阅读页码/段落**：
 - 谭升博客：搜索"shuffle"关键词，阅读包含`__shfl_sync`、`__shfl_down_sync`、`__shfl_xor_sync`、`__shfl_up_sync`的完整章节（约2000字）
 - CUDA官方文档：阅读 warp-level primitives 完整API列表和代码示例段

#### 核心概念笔记

**1. Warp Shuffle四大家族**

| 原语名称 | 函数签名 | 作用描述 | 使用场景 |
|---------|---------|---------|---------|
| `__shfl_sync` | `T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)` | 从`srcLane`线程读取`var`值 | 广播：线程0的结果广播给warp内所有线程 |
| `__shfl_up_sync` | `T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` | 从`threadIdx-delta`线程读取 | 前缀和：每个线程获取左侧delta距离线程的值 |
| `__shfl_down_sync` | `T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` | 从`threadIdx+delta`线程读取 | 归约：warp内折半累加 |
| `__shfl_xor_sync` | `T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)` | 从`threadIdx ^ laneMask`线程读取 | Butterfly交换：归约、位反转排序 |

**2. 四个参数详解（以`__shfl_down_sync`为例）**

```cpp
float val = __shfl_down_sync(0xFFFFFFFF, myVal, 16, 32);
// │ │ │ │ │
// │ │ │ │ └── width: 参与shuffle的宽度（默认32）
// │ │ │ └─────── delta: 向下偏移量
// │ │ └────────────── var: 要传递的变量
// │ └─────────────────────────── mask: 线程掩码，0xFFFFFFFF=全部32线程
// └────────────────────────────────────────── 返回值: 从目标线程读取的值
```

**3. 延迟对比：Shared Memory vs Warp Shuffle**

| 通信方式 | 延迟（周期） | 是否需要`__syncthreads()` | 适用场景 |
|---------|------------|-------------------------|---------|
| Shared Memory中转 | ~20-30 cycles | 是（需同步） | Block级通信（跨Warp） |
| Warp Shuffle直连 | ~1-2 cycles | 否（硬件同步） | Warp内通信（32线程） |
| Register直连（最理想） | ~0 cycles | 否 | 同一线程内复用 |

**核心洞察**：Shuffle的延迟比Shared Memory低一个数量级，因为它直接通过寄存器文件间的专用交换网络传输数据，无需经过Shared Memory的读写路径。

**4. Warp Reduce Butterfly模式**

Warp Reduce（求和）使用`__shfl_down_sync`实现折半累加：

```
Step 1 (offset=16): lane0 <- lane0+lane16, lane1 <- lane1+lane17, ... lane15 <- lane15+lane31
Step 2 (offset=8): lane0 <- lane0+lane8, lane1 <- lane1+lane9, ... lane7 <- lane7+lane15
Step 3 (offset=4): lane0 <- lane0+lane4, lane1 <- lane1+lane5, ... lane3 <- lane3+lane7
Step 4 (offset=2): lane0 <- lane0+lane2, lane1 <- lane1+lane3
Step 5 (offset=1): lane0 <- lane0+lane1
Result: lane0持有warp内32个线程的累加和
```

### 学习任务2：理解Shuffle作用与级联策略（30分钟）

#### 阅读内容
- 谭升博客中"Warp Reduce + Block Reduce"联合使用的章节（约1000字）
- CUDA Samples中`reduction`示例的`reduction_kernel.cu`文件（重点关注第6-7级优化）

#### 核心问题思考
1. **为什么需要两级归约？** 单个Warp只有32线程，一个Block可能有1024线程（32个Warp）。warp级归约只解决32线程的汇总，block级需要将32个Warp的结果再做一次汇总。
2. **谁来做第二级归约？** Warp 0的lane 0线程读取shared memory中32个warp的部分和，再执行一次warpReduce。
3. **为什么第二级归约也用warpReduce而不是shared memory循环？** 因为Warp 0有32个lane，正好处理最多32个warp的部分和，shuffle比shared memory循环更快。

### 晚间编程任务：手写Warp Reduce Kernel（1小时）

#### 完整代码

```cpp
// warp_reduce.cu —— Warp级 + Block级两级归约完整实现
// 编译命令: nvcc -o warp_reduce warp_reduce.cu -O3 -arch=sm_120
// 运行命令: ./warp_reduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// --------------------------------------------------
// Warp级归约：使用__shfl_down_sync折半累加
// --------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
 // Butterfly模式：offset=16,8,4,2,1
 // 每一步将距离为offset的两个线程的值相加
 #pragma unroll
 for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
 val += __shfl_down_sync(0xFFFFFFFF, val, offset);
 }
 return val; // 返回后，lane 0持有warp内32个线程的累加和
}

// --------------------------------------------------
// Warp级归约（XOR模式）：使用__shfl_xor_sync
// 用途：when you need reduction result in ALL lanes, not just lane 0
// --------------------------------------------------
__inline__ __device__ float warpReduceSumXor(float val) {
 #pragma unroll
 for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
 val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
 }
 return val; // 所有lane都持有相同的累加和（广播效果）
}

// --------------------------------------------------
// Block级归约：每个warp先归约，然后warp 0做最终归约
// --------------------------------------------------
__global__ void blockReduceSum(const float* input, float* output, int n) {
 // Shared memory存储每个warp的部分和（最多32个warp per block）
 __shared__ float warpSums[32]; // 32 = 1024/32，即最大warp数

 int tid = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程ID
 int lane = threadIdx.x % warpSize; // warp内lane编号(0-31)
 int wid = threadIdx.x / warpSize; // warp编号(0-31)

 // Step 1: 每个线程从global memory读取并做per-thread累加
 // 使用grid-stride loop处理n很大的情况
 float sum = 0.0f;
 #pragma unroll 4
 for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
 sum += input[i];
 }

 // Step 2: Warp级归约（每个warp的32个线程累加到lane 0）
 sum = warpReduceSum(sum);

 // Step 3: lane 0将warp的部分和写入shared memory
 if (lane == 0) {
 warpSums[wid] = sum;
 }
 __syncthreads(); // 等待所有warp完成归约

 // Step 4: Warp 0做最终归约（读取最多32个warp的部分和）
 if (wid == 0) {
 // lane < numWarps的线程读取对应warp的部分和，其余读0
 int numWarps = (blockDim.x + warpSize - 1) / warpSize;
 sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
 sum = warpReduceSum(sum);
 // lane 0将最终结果写入global memory
 if (lane == 0) {
 output[blockIdx.x] = sum;
 }
 }
}

// --------------------------------------------------
// 多block版本：如果grid有多个block，需要第二次kernel调用汇总
// --------------------------------------------------
float launchReduce(const float* d_input, float* d_temp, float* d_output,
 int n, int threadsPerBlock) {
 int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
 blocks = min(blocks, 1024); // 限制block数量便于第二次归约

 // 第一次归约：每个block输出一个部分和
 blockReduceSum<<<blocks, threadsPerBlock>>>(d_input, d_temp, n);
 cudaDeviceSynchronize();

 // 第二次归约：将所有block的部分和再归约
 blockReduceSum<<<1, 256>>>(d_temp, d_output, blocks);
 cudaDeviceSynchronize();

 float result;
 cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
 return result;
}

// --------------------------------------------------
// Host辅助函数
// --------------------------------------------------
void initData(float* data, int n) {
 srand(42);
 for (int i = 0; i < n; i++) {
 data[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f; // 小值防止累加溢出
 }
}

float cpuReduceSum(const float* data, int n) {
 double sum = 0.0; // 用double减少累加误差
 for (int i = 0; i < n; i++) {
 sum += data[i];
 }
 return static_cast<float>(sum);
}

int main() {
 const int n = 1 << 22; // 4,194,304个元素
 printf("=== Warp Shuffle Block Reduce ===\n");
 printf("Array size: %d (%.2f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

 // Host数据
 float* h_input = (float*)malloc(n * sizeof(float));
 initData(h_input, n);

 // Device内存
 float *d_input, *d_temp, *d_output;
 cudaMalloc(&d_input, n * sizeof(float));
 cudaMalloc(&d_temp, 1024 * sizeof(float)); // 最多1024个block的部分和
 cudaMalloc(&d_output, sizeof(float));
 cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

 // 运行GPU归约
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start);
 float gpuSum = launchReduce(d_input, d_temp, d_output, n, 256);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ms;
 cudaEventElapsedTime(&ms, start, stop);

 // CPU验证
 float cpuSum = cpuReduceSum(h_input, n);
 float diff = fabs(gpuSum - cpuSum);

 printf("GPU Sum: %.6f\n", gpuSum);
 printf("CPU Sum: %.6f\n", cpuSum);
 printf("Diff: %.6f (%s)\n", diff, diff < 1e-3 ? "PASS" : "FAIL");
 printf("Time: %.3f ms (%.2f GB/s bandwidth)\n",
 ms, n * sizeof(float) / (ms * 1e6));

 // 释放
 free(h_input);
 cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output);
 cudaEventDestroy(start); cudaEventDestroy(stop);

 return 0;
}
```

#### 编译运行步骤

```bash
# 编译（根据GPU架构选择arch参数）
# Blackwell (RTX 5090): sm_120
# Blackwell (RTX 5090): sm_120
# Blackwell (RTX 5090): sm_120
nvcc -o warp_reduce warp_reduce.cu -O3 -arch=sm_120

# 运行
./warp_reduce

# 预期输出
# === Warp Shuffle Block Reduce ===
# Array size: 4194304 (16.00 MB)
# GPU Sum: 20973.4xxxx
# CPU Sum: 20973.4xxxx
# Diff: 0.00xxxx (PASS)
# Time: 0.xxx ms (xx.xx GB/s bandwidth)
```

#### 练习题

**练习1（基础）**：修改`warpReduceSum`使用`__shfl_xor_sync`替代`__shfl_down_sync`，比较两者的区别。
> 提示：`__shfl_xor_sync`的mask参数是`laneMask`，不是`delta`。在reduce场景下两者功能等效，但xor模式最终所有lane都得到结果，down模式只有lane 0有结果。
> 考点：理解`__shfl_xor_sync(0xFFFFFFFF, val, 16)`和`__shfl_down_sync(0xFFFFFFFF, val, 16)`的区别。

**练习2（进阶）**：实现一个求**最大值**的Warp Reduce（`warpReduceMax`），并用它找出数组中的最大元素。
> 提示：将`+`替换为`fmaxf`，初始值设为`-INFINITY`。
> 代码框架：`val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));`

**练习3（综合）**：实现一个Block-level的**Warp Segmented Reduce**，即同一个warp内分两组（lane 0-15和lane 16-31），各自独立归约。
> 提示：使用`__shfl_down_sync`的mask参数控制参与线程。mask=0x0000FFFF表示只激活lane 0-15，mask=0xFFFF0000表示只激活lane 16-31。
> 思考：mask参数的作用是什么？在什么场景下需要部分warp参与？

---

### 今日面试题

**面试题1**：`__shfl_down_sync(0xFFFFFFFF, val, 16)`的四个参数分别是什么含义？`0xFFFFFFFF`可以换成其他值吗？（⭐⭐⭐⭐ 必考）

**参考答案要点**：
- 参数1 `mask=0xFFFFFFFF`：32位线程掩码，每一位对应一个lane（bit i = lane i是否激活）。0xFFFFFFFF表示全部32个lane参与。可以换成其他值来实现部分warp操作（如segmented reduce）
- 参数2 `val`：要传递的本线程变量值
- 参数3 `delta=16`：目标lane的偏移量，即读取`(laneId + delta) % width`线程的值
- 参数4（省略，默认32）`width`：参与shuffle的宽度，默认32（整个warp），可以设为16/8等实现子warp操作
- 注意事项：从Blackwell架构开始必须使用`_sync`后缀版本（显式mask），旧版`__shfl_down`已被弃用。部分warp操作时，inactive lane的返回值未定义

**面试题2**：为什么Warp Shuffle比Shared Memory更适合做Warp内归约？实际延迟差距有多大？（⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **延迟差距**：Shuffle延迟约1-2 cycles，Shared Memory访问延迟约20-30 cycles，差距约10-20倍
- **原因1（硬件路径）**：Shuffle通过Warp内部的专用交换网络（shuffle network）直接从源寄存器读取数据，不经过Shared Memory的读写路径
- **原因2（无需同步）**：Shuffle在warp内部是隐式同步的（SIMT执行模型保证warp内线程同步），不需要`__syncthreads()`；Shared Memory方式需要`__syncthreads()`保证数据可见性
- **原因3（指令数少）**：Shuffle是一条指令完成读+交换；Shared Memory需要至少两条指令（store到smem + load从smem）
- **局限**：Shuffle只适用于warp内（最多32线程），跨warp通信仍需Shared Memory

---

### 今日自测清单

- [ ] 能解释`__shfl_down_sync`四个参数的含义（mask, val, delta, width）
- [ ] 能画出Warp内32线程执行shuffle butterfly的5步通信图（offset=16→8→4→2→1）
- [ ] Warp Reduce代码编译运行正确，GPU结果与CPU对比误差<1e-3
- [ ] 能解释`0xFFFFFFFF`的含义：32位掩码，激活Warp内所有32个lane
- [ ] 能解释为什么两级归约需要`__syncthreads()`（warp间同步点）
- [ ] 能说出`__shfl_down_sync`和`__shfl_xor_sync`的区别（down=单向偏移，xor=蝴蝶交换）
- [ ] 理解Blackwell+架构必须使用`_sync`版本的原因（独立线程调度需要显式mask）

---

## Day 9（周二）：Register Blocking与2D Tiling

> **今日目标**：在Shared Memory Tiling基础上引入Register Blocking，实现三级数据复用（Global→Shared→Register），性能目标cuBLAS 40%+。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，"如何优化GEMM到cuBLAS 80%"的标准考点

---

### 学习任务1：Register Blocking原理（1小时）

#### 阅读内容
- **资源地址**：https://face2ai.com/program-blog/
- **阅读范围**：谭升博客中"矩阵乘法优化"系列的Register Blocking章节
- **补充阅读**：CUDA C Best Practices Guide → "Memory Optimizations" → "Register Pressure"段落
- **具体阅读重点**：
 - 理解Thread Tile概念：每个线程负责计算输出矩阵的TM×TN子块
 - 理解三级数据复用层次：Global Memory → Shared Memory → Register
 - 理解Register Blocking如何减少对Shared Memory的访问次数

#### 核心概念笔记

**1. 从Shared Memory Tiling到Register Blocking的进化**

| 优化层次 | 数据驻留位置 | 复用对象 | 性能目标 |
|---------|------------|---------|---------|
| Naive GEMM | Global Memory | 无 | ~1-3% peak |
| Shared Memory Tiling | Shared Memory | A/B tile | ~15-25% peak |
| **Register Blocking** | **Register** | **A子行/B子列 + 累加器** | **~40-60% peak** |
| Warp-level优化 | Register+Shuffle | Warp内协作 | ~60-80% peak |
| 软件流水线 | 全部 above + 双缓冲 | 计算掩盖传输 | ~80-95% peak |

**2. Register Blocking的数据流图**

```
Global Memory (A[M][K], B[K][N])
 │
 ▼ 协作加载（所有线程参与）
Shared Memory (s_A[BM][BK], s_B[BK][BN])
 │
 ├──► Register (r_A[TM]) ──┐
 │ ▼
 └──► Register (r_B[TN]) ──► FMA累加 (acc[TM][TN])
 │ │
 ◄──────────────────────────────┘
 重复BK次（内层k循环）
```

每个线程的执行流程：
1. 从Shared Memory加载TM个A元素到`r_A[TM]`
2. 从Shared Memory加载TN个B元素到`r_B[TN]`
3. 做TM×TN次FMA累加到`acc[TM][TN]`
4. 重复BK次（内层k维度循环）
5. 一个BK块处理完后，加载下一BK块，重复

**3. 关键参数定义与计算**

| 参数 | 含义 | 典型值 | 计算公式 |
|------|------|--------|---------|
| BM | Block tile的M维度 | 128 | 可调 |
| BN | Block tile的N维度 | 128 | 可调 |
| BK | Block tile的K维度 | 8 | 较小值减少shared mem占用 |
| TM | 每个线程负责的M方向输出数 | 8 | acc寄存器 = TM×TN |
| TN | 每个线程负责的N方向输出数 | 8 | acc寄存器 = TM×TN |
| 每Block线程数 | BM/TM × BN/TN | 256 | (128/8)×(128/8)=16×16=256 |

**4. Register使用量计算**

每个线程的register消耗：
- 累加器：`acc[TM][TN]` = TM × TN = 64 个float
- A加载寄存器：`r_A[TM]` = 8 个float
- B加载寄存器：`r_B[TN]` = 8 个float
- 索引/临时变量：~8 个float
- **总计**：~88个float ≈ 88个register（每个float一个寄存器）

> 注意：现代GPU每个线程最多255个寄存器（如RTX 5090），88个register在限制内。但如果TM=TN=16，累加器就有256个register，会溢出到local memory导致性能暴跌。

**5. Double Buffering（软件流水线）原理**

```
时间轴 ──────────────────────────────────────────────►

单缓冲： [Load Tile 0] ──► [Compute Tile 0] ──► [Load Tile 1] ──► [Compute Tile 1]
 ▲ 空闲等待（Load不能被Compute掩盖）

双缓冲： [Load Tile 0→Buf0] ──► [Compute Tile 0 同时 Load Tile 1→Buf1] ──► [Compute Tile 1 同时 Load Tile 2→Buf0]
 ▲ Compute和Load并行执行，掩盖传输延迟
```

实现方式：声明两份shared memory buffer，奇偶tile交替使用。

### 学习任务2：理解2D Tiling的线程映射（30分钟）

#### 核心问题
1. **如何将二维输出tile映射到一维threadIdx？**
 ```
 输出tile (BM×BN = 128×128)被划分为 (BM/TM)×(BN/TN) = 16×16 = 256 个thread tile
 每个thread tile = TM×TN = 8×8

 threadIdx.x 的范围: 0 ~ 255
 threadRow = threadIdx.x / (BN / TN) = threadIdx.x / 16 → 范围 0~15
 threadCol = threadIdx.x % (BN / TN) = threadIdx.x % 16 → 范围 0~15

 线程(threadRow, threadCol)负责输出的行范围:
 [blockIdx.y * BM + threadRow * TM, blockIdx.y * BM + (threadRow+1) * TM)
 负责输出的列范围:
 [blockIdx.x * BN + threadCol * TN, blockIdx.x * BN + (threadCol+1) * TN)
 ```

1. **协作加载：如何从Global Memory加载A/B tile到Shared Memory？**
 - 256个线程协作加载A的BM×K = 128×8 = 1024个元素 → 每个线程加载4个A元素
 - 256个线程协作加载B的K×BN = 8×128 = 1024个元素 → 每个线程加载4个B元素
 - 加载模式需要保证coalesced access（连续线程访问连续地址）

---

### 晚间编程任务：Register Blocking GEMM（1.5小时）

#### 完整代码

```cpp
// register_blocking_gemm.cu —— Register Blocking矩阵乘法完整实现
// 编译命令: nvcc -o register_gemm register_blocking_gemm.cu -O3 -arch=sm_120 -lcublas
// 运行命令: ./register_gemm

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// --------------------------------------------------
// 可调整参数
// --------------------------------------------------
#define BM 128 // Block tile M维度
#define BN 128 // Block tile N维度
#define BK 8 // Block tile K维度（较小的BK减少shared mem占用）
#define TM 8 // 每个线程负责的M方向输出数
#define TN 8 // 每个线程负责的N方向输出数

// 每Block线程数 = (BM/TM) * (BN/TN) = 16 * 16 = 256
#define NUM_THREADS ((BM / TM) * (BN / TN))

// --------------------------------------------------
// Register Blocking GEMM Kernel
// --------------------------------------------------
__global__ void gemmRegisterBlocking(const float* __restrict__ A,
 const float* __restrict__ B,
 float* __restrict__ C,
 int M, int N, int K) {
 // Shared Memory：存储A和B的tile
 __shared__ float s_A[BM][BK]; // 128×8 = 1024 floats
 __shared__ float s_B[BK][BN]; // 8×128 = 1024 floats

 // Register：累加器和加载缓冲区
 float r_A[TM]; // A的子行（8个元素）
 float r_B[TN]; // B的子列（8个元素）
 float acc[TM][TN] = {0}; // TM×TN累加器（初始化为0）

 // 线程到输出tile的二维映射
 int threadRow = threadIdx.x / (BN / TN); // 0 ~ 15
 int threadCol = threadIdx.x % (BN / TN); // 0 ~ 15

 // Block在C矩阵中的位置
 int cRow = blockIdx.y * BM; // 当前Block负责的C行起始
 int cCol = blockIdx.x * BN; // 当前Block负责的C列起始

 // 主循环：沿K维度滑动
 for (int bk = 0; bk < K; bk += BK) {
 // =====================================================
 // Step 1: 协作加载 A[BM][BK] tile 从 Global → Shared
 // 策略：每个线程加载 (BM×BK)/NUM_THREADS = 1024/256 = 4个元素
 // =====================================================
 int aRow = threadIdx.x / BK; // 0 ~ 127 (BM范围内)
 int aCol = threadIdx.x % BK; // 0 ~ 7 (BK范围内)

 #pragma unroll
 for (int i = 0; i < BM; i += NUM_THREADS / BK) {
 // NUM_THREADS/BK = 256/8 = 32，每次处理32行
 int loadRow = aRow + i;
 if (loadRow < BM && (cRow + loadRow) < M && (bk + aCol) < K) {
 s_A[loadRow][aCol] = A[(cRow + loadRow) * K + (bk + aCol)];
 } else if (loadRow < BM) {
 s_A[loadRow][aCol] = 0.0f; // 边界填充0
 }
 }

 // =====================================================
 // Step 2: 协作加载 B[BK][BN] tile 从 Global → Shared
 // 策略：每个线程加载 (BK×BN)/NUM_THREADS = 1024/256 = 4个元素
 // =====================================================
 int bRow = threadIdx.x / BN; // 0 ~ 7 (BK范围内)
 int bCol = threadIdx.x % BN; // 0 ~ 127 (BN范围内)

 #pragma unroll
 for (int i = 0; i < BK; i += NUM_THREADS / BN) {
 // NUM_THREADS/BN = 256/128 = 2，每次处理2行
 int loadRow = bRow + i;
 if (loadRow < BK && (bk + loadRow) < K && (cCol + bCol) < N) {
 s_B[loadRow][bCol] = B[(bk + loadRow) * N + (cCol + bCol)];
 } else if (loadRow < BK) {
 s_B[loadRow][bCol] = 0.0f;
 }
 }

 __syncthreads(); // 等待tile加载完成

 // =====================================================
 // Step 3: 从Shared Memory加载到Register并计算
 // 内层循环：沿BK维度展开
 // =====================================================
 #pragma unroll
 for (int k = 0; k < BK; k++) {
 // 加载TM个A元素到寄存器（A的一"子行"）
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 r_A[m] = s_A[threadRow * TM + m][k];
 }
 // 加载TN个B元素到寄存器（B的一"子列"）
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 r_B[n] = s_B[k][threadCol * TN + n];
 }
 // TM×TN次FMA累加
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 acc[m][n] += r_A[m] * r_B[n];
 }
 }
 }

 __syncthreads(); // 准备加载下一tile
 }

 // =====================================================
 // Step 4: 将acc结果写回Global Memory（C矩阵）
 // =====================================================
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 int gRow = cRow + threadRow * TM + m;
 int gCol = cCol + threadCol * TN + n;
 if (gRow < M && gCol < N) {
 C[gRow * N + gCol] = acc[m][n];
 }
 }
 }
}

// --------------------------------------------------
// cuBLAS基准（用于对比性能）
// --------------------------------------------------
float runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
 cublasHandle_t handle;
 cublasCreate(&handle);

 const float alpha = 1.0f;
 const float beta = 0.0f;

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 // 预热
 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
 &alpha, d_B, N, d_A, K, &beta, d_C, N);
 cudaDeviceSynchronize();

 cudaEventRecord(start);
 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
 &alpha, d_B, N, d_A, K, &beta, d_C, N);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ms;
 cudaEventElapsedTime(&ms, start, stop);

 cublasDestroy(handle);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 return ms;
}

// --------------------------------------------------
// Host辅助函数
// --------------------------------------------------
void initMatrix(float* mat, int rows, int cols) {
 srand(42);
 for (int i = 0; i < rows * cols; i++) {
 mat[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
 }
}

void cpuGEMM(const float* A, const float* B, float* C, int M, int N, int K) {
 for (int m = 0; m < M; m++) {
 for (int n = 0; n < N; n++) {
 float sum = 0;
 for (int k = 0; k < K; k++) {
 sum += A[m * K + k] * B[k * N + n];
 }
 C[m * N + n] = sum;
 }
 }
}

bool checkResult(const float* gpu, const float* cpu, int M, int N, float eps) {
 for (int i = 0; i < M * N; i++) {
 if (fabs(gpu[i] - cpu[i]) > eps) {
 printf("Mismatch at [%d][%d]: GPU=%.6f, CPU=%.6f\n",
 i / N, i % N, gpu[i], cpu[i]);
 return false;
 }
 }
 return true;
}

float getGFLOPS(int M, int N, int K, float ms) {
 return 2.0f * M * N * K / (ms * 1e6); // GFLOPS
}

int main() {
 // 测试矩阵尺寸
 int sizes[][3] = {
 {1024, 1024, 1024},
 {2048, 2048, 2048},
 {4096, 4096, 4096},
 };

 printf("=== Register Blocking GEMM ===\n");
 printf("Parameters: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d, Threads=%d\n",
 BM, BN, BK, TM, TN, NUM_THREADS);
 printf("%-10s %-10s %-10s %-12s %-12s %-10s\n",
 "M", "N", "K", "Our(ms)", "cuBLAS(ms)", "Percent");
 printf("------------------------------------------------------------\n");

 for (int s = 0; s < 3; s++) {
 int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];

 size_t sizeA = M * K * sizeof(float);
 size_t sizeB = K * N * sizeof(float);
 size_t sizeC = M * N * sizeof(float);

 float *h_A = (float*)malloc(sizeA);
 float *h_B = (float*)malloc(sizeB);
 float *h_C = (float*)malloc(sizeC);
 float *h_C_CPU = (float*)malloc(sizeC);
 float *h_C_ref = (float*)malloc(sizeC);

 initMatrix(h_A, M, K);
 initMatrix(h_B, K, N);

 float *d_A, *d_B, *d_C;
 cudaMalloc(&d_A, sizeA);
 cudaMalloc(&d_B, sizeB);
 cudaMalloc(&d_C, sizeC);
 cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
 cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

 // 运行Register Blocking GEMM
 dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
 dim3 blockDim(NUM_THREADS);

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 // 预热
 gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
 cudaDeviceSynchronize();

 cudaEventRecord(start);
 gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ourMs;
 cudaEventElapsedTime(&ourMs, start, stop);

 cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

 // cuBLAS基准
 float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
 cudaMemcpy(h_C_ref, d_C, sizeC, cudaMemcpyDeviceToHost);

 // 验证（与cuBLAS对比，而不是CPU，因为大矩阵CPU太慢）
 bool correct = checkResult(h_C, h_C_ref, M, N, 1e-2);
 float percent = (cublasMs / ourMs) * 100; // 我们是cuBLAS的百分之多少（越大越接近）

 printf("%-10d %-10d %-10d %-12.3f %-12.3f %-9.1f%% %s\n",
 M, N, K, ourMs, cublasMs, percent,
 correct ? "PASS" : "FAIL");

 // CPU验证（只对小矩阵）
 if (M <= 512) {
 cpuGEMM(h_A, h_B, h_C_CPU, M, N, K);
 bool cpuCorrect = checkResult(h_C, h_C_CPU, M, N, 1e-3);
 printf(" CPU verification: %s\n", cpuCorrect ? "PASS" : "FAIL");
 }

 free(h_A); free(h_B); free(h_C); free(h_C_CPU); free(h_C_ref);
 cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
 cudaEventDestroy(start); cudaEventDestroy(stop);
 }

 return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o register_gemm register_blocking_gemm.cu -O3 -arch=sm_120 -lcublas

# 运行
./register_gemm

# 预期输出（RTX 5090 GPU示例）
# === Register Blocking GEMM ===
# Parameters: BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
# M N K Our(ms) cuBLAS(ms) Percent
# ------------------------------------------------------------
# 1024 1024 1024 0.xxx 0.xxx 35.2% PASS
# 2048 2048 2048 x.xxx x.xxx 42.1% PASS
# 4096 4096 4096 xx.xxx xx.xxx 45.8% PASS
```

#### 练习题

**练习1（基础）**：修改TM和TN的值（如TM=4, TN=4或TM=16, TN=4），观察性能变化和register使用量的关系。
> 提示：使用`nvcc -Xptxas -v`查看编译器报告的register使用量。超过限制会导致spill到local memory。
> 编译命令：`nvcc -o register_gemm register_blocking_gemm.cu -O3 -arch=sm_120 -Xptxas -v,-warn-spills`

**练习2（进阶）**：实现Double Buffering版本——声明两份shared memory buffer（`s_A[2][BM][BK]`），奇偶tile交替使用。
> 提示：使用`buf_idx = (bk / BK) % 2`选择当前buffer，用`__syncthreads()`保证切换时无冲突。
> 目标：理解软件流水线如何掩盖global→shared memory的传输延迟。

**练习3（综合）**：在Register Blocking基础上，使用`__shfl_xor_sync`实现warp内线程的累加器交换。
> 提示：将相邻4个线程的TN维度结果通过shuffle汇总，减少写回global memory时的非合并访问。
> 这是从40%提升到60%的关键优化之一。

---

### 今日面试题

**面试题1**："如何把手写GEMM优化到cuBLAS 80%的性能？"请逐层展开优化策略。（⭐⭐⭐⭐⭐ 必考，顶级高频题）

**参考答案要点**（按层次展开）：
1. **Naive版本（~1%）**：每个线程计算C的一个元素，全局内存访问无复用
2. **Shared Memory Tiling（~15%）**：将A/B tile预取到shared memory，实现K维度的数据复用
3. **Register Blocking（~40%）**：每个线程计算TM×TN输出子块，数据驻留寄存器，减少对shared memory的访问
4. **Warp-level优化（~60%）**：使用Warp Shuffle在warp内协作，优化写回模式
5. **Vectorized Load（~70%）**：使用`float4`做128-bit向量化加载，提升global memory带宽利用率
6. **Double Buffering + 软件流水线（~80%）**：用双缓冲掩盖global→shared memory的传输延迟
7. **Auto-tuning（~90%+）**：针对不同矩阵尺寸auto-tune BM/BN/BK/TM/TN参数组合
- **关键指标**：每个优化层次的性能增益、数据复用率的量化提升

**面试题2**：Register Blocking中的`acc[TM][TN]`为什么要放在register中而不是shared memory？register使用量如何计算？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **放在register的原因**：寄存器访问延迟~0 cycle（已分配），shared memory延迟~20-30 cycles。TM×TN累加器被访问TM×TN×BK次（内层循环），放在register能极大减少延迟
- **Register使用量计算**：acc[TM][TN]个float + r_A[TM] + r_B[TN] + 索引变量。TM=TN=8时约88个register
- **限制**：每个线程最多 255 个 register，超过会 spill 到 local memory（性能暴跌）
- **Shared memory容量限制**：acc放在shared memory会占用BM×BN个float = 128×128×4 = 64KB，超过了一般shared memory上限（48-96KB），所以必须放register

---

### 今日自测清单

- [ ] 能解释Register Blocking相比纯Shared Memory Tiling多了一级数据复用（Global→Shared→Register）
- [ ] 能计算register usage：TM×TN个累加器 + TM + TN个加载寄存器 + 索引变量
- [ ] 代码编译运行正确，性能达到cuBLAS 40%+（4096矩阵）
- [ ] 能画出数据流图：Global Memory → Shared Memory → Register → FMA累加
- [ ] 能计算每Block线程数 = (BM/TM) × (BN/TN)
- [ ] 能理解协作加载的策略：每个线程加载多少个A/B元素到shared memory
- [ ] 能解释Double Buffering的原理（用计算掩盖数据传输延迟）

---

## Day 10（周三）：CUDA Streams与异步执行

> **今日目标**：理解CUDA Stream异步执行模型，掌握多Stream并行策略，实现H2D/Compute/D2H重叠流水线。
> **面试考察度**：⭐⭐⭐ 高频，尤其是Default Stream的隐式同步行为

---

### 学习任务1：CUDA Stream与异步执行模型（45分钟）

#### 阅读内容
- **资源地址**：https://face2ai.com/program-blog/
- **阅读范围**：谭升博客中"CUDA Stream与异步执行"专题章节
- **补充阅读**：CUDA C Programming Guide → "Asynchronous Concurrent Execution"章节
- **具体阅读重点**：
 - Default Stream（Stream 0）的行为特性
 - Explicit Stream的创建、使用和销毁
 - Kernel启动的异步特性：`<<<>>>`返回后Host不等待Kernel完成
 - `cudaMemcpyAsync` vs `cudaMemcpy`的区别

#### 核心概念笔记

**1. Stream的本质**

Stream是GPU上操作（Kernel执行、内存拷贝）的队列。同一个Stream内的操作按FIFO顺序执行，不同Stream之间的操作可以并发（只要资源允许）。

```
Stream 1: [H2D拷贝1] → [Kernel1] → [D2H拷贝1]
Stream 2: [H2D拷贝2] → [Kernel2] → [D2H拷贝2]
 ↑ H2D拷贝2可以与Kernel1并发执行（copy engine和compute unit独立）
```

**2. Default Stream的"坑"（重点理解）**

| 特性 | Default Stream (Stream 0) | Explicit Stream |
|------|-------------------------|-----------------|
| 创建方式 | 隐式存在，无需创建 | `cudaStreamCreate(&stream)` |
| 同步行为 | **隐式同步所有其他Stream** | 只同步本Stream的操作 |
| 与Host关系 | `cudaMemcpy`阻塞Host | `cudaMemcpyAsync`非阻塞Host |
| 适用场景 | 简单程序、调试 | 生产环境、性能优化 |

**Default Stream的隐式同步规则（极易出错）**：
- 规则1：Default Stream上的操作会等待所有其他Stream的先前操作完成
- 规则2：其他Stream上的操作会等待Default Stream的先前操作完成
- **后果**：即使创建了多Stream，只要在Default Stream上做一次`cudaMemcpy`，所有并发都被打断

**解决方案**：始终使用`cudaStreamPerThread`或显式Stream，避免混用Default Stream和Explicit Stream。

```cpp
// 创建不与Default Stream同步的Stream（推荐做法）
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
// 或者使用Per-Thread Default Stream编译选项
// nvcc --default-stream per-thread ...
```

**3. `cudaMemcpy` vs `cudaMemcpyAsync`对比**

| 函数 | 同步性 | 是否可指定Stream | 内存要求 | 使用场景 |
|------|--------|----------------|---------|---------|
| `cudaMemcpy` | **同步**（阻塞Host直到完成） | 否（Default Stream） | 任意内存 | 简单程序 |
| `cudaMemcpyAsync` | **异步**（立即返回） | 是 | 必须使用**pinned内存** | 多Stream并发 |

Pinned Memory（页锁定内存）：通过`cudaMallocHost`分配，不会被OS换出到磁盘，支持DMA直接传输。

**4. 多Stream并发的硬件条件**

GPU有独立的硬件引擎：
- **Copy Engine（CE）**：负责H2D和D2H数据传输，通常有2个（一个上行一个下行）
- **Compute Engine（SM）**：负责Kernel执行
- **并发条件**：Copy和Compute可以同时进行（数据拷贝与Kernel计算overlap）

```
时间轴 ────────────────────────────────────────────────────────►

无Stream（顺序）： [H2D拷贝] ──► [Kernel计算] ──► [D2H拷贝]
 总计 = H2D + Compute + D2H

Multi-Stream（重叠）：
 Stream1: [H2D chunk1] ──► [Kernel chunk1] ──► [D2H chunk1]
 Stream2: [H2D chunk2] ──► [Kernel chunk2] ──► [D2H chunk2]
 Stream3: [H2D chunk3] ──► [Kernel chunk3] ──► [D2H chunk3]
 Stream4: [H2D chunk4] ──► [Kernel chunk4] ──► [D2H chunk4]
 ↑ H2D与Kernel重叠，Kernel与D2H重叠
 总计 ≈ max(H2D + D2H, Compute) + 流水线填充
```

### 学习任务2：Multi-Stream并行策略（30分钟）

#### 核心策略

**策略1：数据并行（Data Parallelism）**
- 将大数据拆分为多个chunk，每个Stream处理一个chunk
- 适用场景：大批次推理、大数据集处理

**策略2：计算与传输重叠（Compute-Transfer Overlap）**
- 在Stream A执行Kernel的同时，Stream B执行H2D拷贝
- 利用独立的Copy Engine和Compute Engine实现硬件级并行
- **性能提升上限**：理想情况下加速比 = (H2D + Compute + D2H) / max(H2D, Compute, D2H)

**策略3：Kernel并发（Kernel Concurrent Execution）**
- 多个小Kernel可以同时占据不同SM
- 条件：资源充足（寄存器、shared memory未满），且Kernel来自不同Stream

#### 流依赖管理：`cudaEvent`

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// Stream A中记录事件
cudaEventRecord(event, streamA);

// Stream B等待该事件（Stream B的操作在event完成后才能开始）
cudaStreamWaitEvent(streamB, event, 0);
```

适用场景：Stream间存在数据依赖（如Stream1的D2H完成后Stream2才能处理）。

---

### 晚间编程任务：Multi-Stream H2D/Compute/D2H重叠流水线（1小时）

#### 完整代码

```cpp
// multi_stream_pipeline.cu —— 多Stream重叠流水线完整实现
// 编译命令: nvcc -o multi_stream multi_stream_pipeline.cu -O3 -arch=sm_120
// 运行命令: ./multi_stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// --------------------------------------------------
// 向量加法Kernel（作为计算任务示例）
// --------------------------------------------------
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) {
 // 增加计算量使Kernel运行时间更显著
 float sum = A[i] + B[i];
 for (int j = 0; j < 100; j++) {
 sum = sum * 0.999f + 0.001f;
 }
 C[i] = sum;
 }
}

// --------------------------------------------------
// 顺序版本（baseline）
// --------------------------------------------------
float sequentialVersion(float* h_A, float* h_B, float* h_C,
 float* d_A, float* d_B, float* d_C,
 int totalSize, int chunkSize) {
 int numChunks = (totalSize + chunkSize - 1) / chunkSize;

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaEventRecord(start);

 for (int i = 0; i < numChunks; i++) {
 int offset = i * chunkSize;
 int currSize = (offset + chunkSize <= totalSize) ? chunkSize : (totalSize - offset);
 size_t bytes = currSize * sizeof(float);

 // 顺序：H2D → Kernel → D2H（全部在Default Stream）
 cudaMemcpy(d_A + offset, h_A + offset, bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_B + offset, h_B + offset, bytes, cudaMemcpyHostToDevice);

 int threads = 256;
 int blocks = (currSize + threads - 1) / threads;
 vecAdd<<<blocks, threads>>>(d_A + offset, d_B + offset, d_C + offset, currSize);

 cudaMemcpy(h_C + offset, d_C + offset, bytes, cudaMemcpyDeviceToHost);
 }

 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float ms;
 cudaEventElapsedTime(&ms, start, stop);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 return ms;
}

// --------------------------------------------------
// Multi-Stream重叠版本
// --------------------------------------------------
float multiStreamVersion(float* h_A, float* h_B, float* h_C,
 float* d_A, float* d_B, float* d_C,
 int totalSize, int chunkSize, int nStreams) {
 int numChunks = (totalSize + chunkSize - 1) / chunkSize;

 // 创建多个Stream（使用NonBlocking避免与Default Stream同步）
 cudaStream_t* streams = new cudaStream_t[nStreams];
 for (int i = 0; i < nStreams; i++) {
 cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
 }

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaEventRecord(start);

 for (int i = 0; i < numChunks; i++) {
 int streamIdx = i % nStreams;
 int offset = i * chunkSize;
 int currSize = (offset + chunkSize <= totalSize) ? chunkSize : (totalSize - offset);
 size_t bytes = currSize * sizeof(float);

 // 在同一个Stream中按序执行H2D→Compute→D2H
 // 不同Stream之间自动重叠
 cudaMemcpyAsync(d_A + offset, h_A + offset, bytes,
 cudaMemcpyHostToDevice, streams[streamIdx]);
 cudaMemcpyAsync(d_B + offset, h_B + offset, bytes,
 cudaMemcpyHostToDevice, streams[streamIdx]);

 int threads = 256;
 int blocks = (currSize + threads - 1) / threads;
 vecAdd<<<blocks, threads, 0, streams[streamIdx]>>>(
 d_A + offset, d_B + offset, d_C + offset, currSize);

 cudaMemcpyAsync(h_C + offset, d_C + offset, bytes,
 cudaMemcpyDeviceToHost, streams[streamIdx]);
 }

 // 同步所有Stream
 for (int i = 0; i < nStreams; i++) {
 cudaStreamSynchronize(streams[i]);
 }

 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float ms;
 cudaEventElapsedTime(&ms, start, stop);

 for (int i = 0; i < nStreams; i++) {
 cudaStreamDestroy(streams[i]);
 }
 delete[] streams;
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 return ms;
}

// --------------------------------------------------
// Main
// --------------------------------------------------
int main() {
 const int totalSize = 1 << 24; // 16,777,216个元素
 const int chunkSize = 1 << 18; // 262,144个元素 per chunk
 const int nStreams = 4;

 printf("=== Multi-Stream Overlap Pipeline ===\n");
 printf("Total size: %d (%.2f MB)\n", totalSize, totalSize * sizeof(float) / (1024.0 * 1024.0));
 printf("Chunk size: %d (%.2f MB)\n", chunkSize, chunkSize * sizeof(float) / (1024.0 * 1024.0));
 printf("Num chunks: %d, Num streams: %d\n",
 (totalSize + chunkSize - 1) / chunkSize, nStreams);
 printf("\n");

 size_t totalBytes = totalSize * sizeof(float);

 // 分配Pinned Memory（必须用于cudaMemcpyAsync）
 float *h_A, *h_B, *h_C_seq, *h_C_multi;
 cudaMallocHost(&h_A, totalBytes);
 cudaMallocHost(&h_B, totalBytes);
 cudaMallocHost(&h_C_seq, totalBytes);
 cudaMallocHost(&h_C_multi, totalBytes);

 // 初始化
 srand(42);
 for (int i = 0; i < totalSize; i++) {
 h_A[i] = static_cast<float>(rand()) / RAND_MAX;
 h_B[i] = static_cast<float>(rand()) / RAND_MAX;
 }

 // Device内存
 float *d_A, *d_B, *d_C;
 cudaMalloc(&d_A, totalBytes);
 cudaMalloc(&d_B, totalBytes);
 cudaMalloc(&d_C, totalBytes);

 // 运行顺序版本
 printf("Running sequential version...\n");
 float seqMs = sequentialVersion(h_A, h_B, h_C_seq,
 d_A, d_B, d_C, totalSize, chunkSize);
 printf("Sequential: %.3f ms\n\n", seqMs);

 // 运行Multi-Stream版本
 printf("Running multi-stream version (nStreams=%d)...\n", nStreams);
 float multiMs = multiStreamVersion(h_A, h_B, h_C_multi,
 d_A, d_B, d_C, totalSize, chunkSize, nStreams);
 printf("Multi-Stream: %.3f ms\n\n", multiMs);

 // 结果验证
 bool correct = true;
 for (int i = 0; i < totalSize; i++) {
 if (fabs(h_C_seq[i] - h_C_multi[i]) > 1e-5) {
 correct = false;
 break;
 }
 }

 // 性能报告
 float speedup = seqMs / multiMs;
 printf("=== Performance Summary ===\n");
 printf("Sequential: %.3f ms\n", seqMs);
 printf("Multi-Stream: %.3f ms\n", multiMs);
 printf("Speedup: %.2fx\n", speedup);
 printf("Result check: %s\n", correct ? "PASS" : "FAIL");

 // 释放资源
 cudaFreeHost(h_A); cudaFreeHost(h_B);
 cudaFreeHost(h_C_seq); cudaFreeHost(h_C_multi);
 cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

 return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o multi_stream multi_stream_pipeline.cu -O3 -arch=sm_120

# 运行
./multi_stream

# 预期输出
# === Multi-Stream Overlap Pipeline ===
# Total size: 16777216 (64.00 MB)
# Chunk size: 262144 (1.00 MB)
# Num chunks: 64, Num streams: 4
#
# Running sequential version...
# Sequential: xxx.xxx ms
#
# Running multi-stream version (nStreams=4)...
# Multi-Stream: xx.xxx ms
#
# === Performance Summary ===
# Sequential: xxx.xxx ms
# Multi-Stream: xx.xxx ms
# Speedup: 1.2x ~ 1.8x（取决于GPU copy engine数量）
# Result check: PASS
```

#### 练习题

**练习1（基础）**：修改代码使用`cudaStreamCreate`（不带NonBlocking标志）代替`cudaStreamCreateWithFlags`，观察性能差异。
> 提示：`cudaStreamCreate`创建的Stream会与Default Stream隐式同步。如果Default Stream上没有操作，两者效果相同；如果有Default Stream操作，NonBlocking版本能获得更多并发。

**练习2（进阶）**：实现`cudaEvent`跨Stream依赖——添加第三个处理Stream（Stream C），它必须在Stream A和Stream B的D2H都完成后才能开始。
> 提示：使用`cudaEventRecord`在Stream A和B中记录事件，然后在Stream C中使用`cudaStreamWaitEvent`等待两个事件。

**练习3（综合）**：使用`nsys`命令行工具capture多Stream版本的timeline，在Nsight Systems GUI中观察H2D/Compute/D2H的重叠情况。
> 提示：`nsys profile -o multi_stream ./multi_stream`，然后用Nsight Systems GUI打开`.nsys-rep`文件。
> 目标：在Timeline视图中看到不同Stream的操作条有重叠区域。

---

### 今日面试题

**面试题1**：CUDA的Default Stream有什么"坑"？在什么情况下会意外导致性能下降？（⭐⭐⭐ 高频）

**参考答案要点**：
- **隐式同步规则**：Default Stream上的操作会等待所有其他Stream的先前操作完成；反过来，其他Stream的操作也会等待Default Stream的先前操作完成
- **陷阱场景**：创建了多Stream做并发优化，但某处调用了`cudaMemcpy`（默认走Default Stream），导致所有Stream的并发被打断
- **解决方案**：① 全部使用Explicit Stream + `cudaMemcpyAsync` ② 使用`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)` ③ 编译时加`--default-stream per-thread`

**面试题2**：`cudaMemcpyAsync`相比`cudaMemcpy`需要什么额外条件？为什么必须使用Pinned Memory？（⭐⭐⭐ 高频）

**参考答案要点**：
- **必须使用Pinned Memory**：`cudaMemcpyAsync`要求Host端内存是page-locked（pinned），因为异步传输使用DMA引擎直接访问内存，如果内存被OS换出到磁盘，DMA无法访问
- **分配方式**：用`cudaMallocHost`或`cudaHostAlloc`代替`malloc`分配Host内存
- **性能影响**：Pinned Memory的分配和释放比pageable memory慢，不适合频繁小量分配
- **语义差异**：`cudaMemcpy`是同步的（阻塞Host），内部会自动做临时的pinned staging buffer；`cudaMemcpyAsync`是真正异步的，需要调用者保证内存pinned

---

### 今日自测清单

- [ ] 能解释Default Stream的隐式同步行为及其危害
- [ ] 能画出Multi-Stream时间线：4个Stream的H2D→Compute→D2H流水线重叠
- [ ] 代码正确实现了H2D/Compute/D2H的overlap，速度比顺序版本有提升
- [ ] 能解释`cudaMemcpyAsync`为什么需要Pinned Memory（DMA要求）
- [ ] 能写出`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`的完整用法
- [ ] 能理解`cudaEventRecord` + `cudaStreamWaitEvent`的跨Stream依赖管理
- [ ] 能使用`nsys profile`捕获并分析多Stream timeline

---

## Day 11（周四）：Nsight Compute性能分析

> **今日目标**：掌握Nsight Compute工具的使用方法，学会解读关键性能指标，能够对Register Blocking GEMM进行瓶颈分析。
> **时间分配**：早间1.5h（工具学习1h + Roofline模型30min）+ 晚间1h（Profile实践）
> **面试考察度**：⭐⭐⭐ 高频，"如何分析Kernel性能瓶颈"的标准答案

---

### 学习任务1：Nsight Compute使用方法（45分钟）

#### 阅读内容
- **资源地址**：https://docs.nvidia.com/nsight-compute/
- **阅读范围**：Nsight Compute User Guide → "Quickstart"和"Metric Guide"章节
- **具体学习重点**：
 - ncu命令行的基本用法和常用参数
 - 关键性能指标的含义和正常范围
 - Roofline Model的解读方法

#### 核心概念笔记

**1. ncu命令行基础**

```bash
# 基本用法：profile一个Kernel
ncu -o report.ncu-rep ./my_kernel

# 常用参数
ncu \
 --kernel-name regex:gemmRegisterBlocking \ # 只profile指定kernel
 -o report \ # 输出文件名
 --metrics \ # 指定采集的指标（逗号分隔）
 sm__throughput.avg.pct_of_peak_sustained_elapsed, # SM计算利用率
 dram__throughput.avg.pct_of_peak_sustained_elapsed, # 显存带宽利用率
 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum, # L1/纹理缓存加载扇区数
 l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum # L1/纹理缓存存储扇区数
 ./my_program

# 查看report
ncu-ui report.ncu-rep # GUI方式
ncu --page details -i report.ncu-rep # 命令行方式
```

**2. 关键性能指标表**

| 指标名称 | 正常范围 | 含义 | 优化方向 |
|---------|---------|------|---------|
| **SM Throughput** | > 60%为良好 | SM计算单元利用率 | 低→增加occupancy或指令级并行 |
| **Memory Throughput** | > 60%为良好 | 显存带宽利用率 | 低→检查coalesced access |
| **Achieved Occupancy** | 理论值的70-100% | 实际warp占用率 | 低→减少register/shared memory使用 |
| **Warp Stall Reasons** | 每项<20% | warp阻塞原因分布 | 根据stall原因针对性优化 |
| **L1/TEX Hit Rate** | > 80%为良好 | L1缓存命中率 | 低→优化内存访问模式 |
| **IPC (Instructions Per Clock)** | 架构相关 | 每周期执行指令数 | 低→检查依赖链和发射瓶颈 |
| **Register Pressure** | < 80%为良好 | 寄存器使用压力 | 高→减少寄存器变量 |

**3. Warp Stall Reasons详解**

| Stall Reason | 含义 | 优化方法 |
|-------------|------|---------|
| **Long Scoreboard** | 全局内存加载延迟等待 | 增加thread tile大小，增加指令级并行 |
| **Math Pipe Throttle** | 数学单元（FMA）过载 | 减少FMA依赖链，增加独立指令 |
| **MIO Throttle** | 内存指令发射瓶颈 | 减少shared memory访问次数 |
| **Wait** | 显式`__syncthreads()`等待 | 减少同步点，或使用warp级同步 |
| **Barrier** | 同步屏障等待 | 优化线程负载均衡 |
| ** LG Throttle** | 加载/存储队列满 | 减少global memory访问频率 |

**4. Roofline Model解读**

```
 GFLOPS
 │
 │ ╱ 计算峰值（水平线，由SM数量和频率决定）
 │ ╱
 │ ╱
 │ ╱
 │ ╱
 │╱ 带宽限制斜线（斜率 = 带宽 × 计算强度）
 └──────────────────────────────►
 计算强度（FLOP/Byte）

你的Kernel所在位置决定优化方向：
- 在斜线下方（memory-bound）：优化内存带宽（tiling、coalesced access、vectorized load）
- 在水平线下方（compute-bound）：优化计算吞吐量（更多FMA、减少stall）
- 在两线交点附近（balanced）：两者都接近峰值，难以大幅优化
```

Roofline Model的核心思想：**计算强度** = 总FLOPS / 总数据移动量(Bytes)。计算强度低 → memory bound，计算强度高 → compute bound。

### 学习任务2：Profiler-guided优化流程（30分钟）

#### 四步优化流程

```
Step 1: Baseline ──► ncu采集指标 ──► 记录SM/Memory Throughput和Roofline位置
 │
 ▼
Step 2: Bottleneck Identification ──► 看Roofline + Stall Reasons
 │
 ├── Memory Bound ──► 优化Global/Shared Memory访问模式
 │ ├── Coalesced Access检查
 │ ├── Shared Memory Bank Conflict检查
 │ └── Vectorized Load (float4)引入
 │
 └── Compute Bound ──► 优化计算吞吐量
 ├── 增加Register Blocking粒度(TM×TN)
 ├── 减少Warp Stall（增加指令级并行）
 └── 检查FMA利用率
 │
 ▼
Step 3: Targeted Optimization ──► 只改确认有收益的优化点
 │
 ▼
Step 4: Validation ──► ncu重新采集 ──► 对比前后指标变化 ──► 确认性能提升
```

**云GPU替代方案**（如果本地没有Nsight Compute GUI）：

| 场景 | 解决方案 |
|------|---------|
| 只有命令行ncu | `ncu --csv --page details -i report.ncu-rep > report.csv`导出CSV分析 |
| 无ncu权限 | 使用`nvprof`（旧版，CUDA 10.x）或代码中嵌入`cudaEvent`手动计时 |
| 纯云环境 | ncu导出报告后下载到本地用ncu-ui打开 |
| 无NVIDIA GPU | 使用CUDA模拟器（如GPGPU-Sim）做教学级分析 |

---

### 学习任务3：关键指标解读实战（15分钟）

#### 案例：分析Register Blocking GEMM的Profile结果

假设ncu输出如下指标：

```
SM Throughput: 45.2%
Memory Throughput: 78.5%
Achieved Occupancy: 56.3%
L1/TEX Hit Rate: 82.1%
Warp Stall Long Scoreboard: 35.2% ← 高！
Warp Stall Math Pipe Throttle: 12.1%
Register Pressure: 72%
```

**解读过程**：
1. Memory Throughput(78.5%) > SM Throughput(45.2%) → **Memory Bound**
2. Roofline位置：在斜线下方，计算强度不够高
3. Stall Reason：Long Scoreboard 35.2% → 全局内存加载延迟是主要瓶颈
4. Achieved Occupancy 56.3% → 偏低，可能register太多导致occupancy下降

**优化方向**：
1. 引入`float4`向量化加载（提升4x内存带宽）
2. 调大TM×TN增加计算强度
3. 考虑Double Buffering掩盖加载延迟

---

### 晚间编程任务：Nsight Profile Register Blocking GEMM（1小时）

#### Step 1：编译GEMM并生成Profile报告

```bash
# 编译（保留调试信息以便Source View关联）
nvcc -o gemm_profile register_blocking_gemm.cu \
 -O3 -arch=sm_120 -lcublas -g -lineinfo

# 运行Nsight Compute profile
ncu \
 --kernel-name regex:gemmRegisterBlocking \
 -o gemm_profile_report \
 --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__warps_eligible.sum.per_cycle,smsp__average_warps_issue_stalled_long_scoreboard.pct \
 ./gemm_profile 2>&1 | tee ncu_output.txt

# 导出为CSV（便于命令行查看）
ncu --csv --page details -i gemm_profile_report.ncu-rep > gemm_profile.csv
```

#### Step 2：分析任务清单

1. **读取SM Throughput和Memory Throughput**，判断是compute-bound还是memory-bound
2. **读取Achieved Occupancy**，判断SM利用率是否充分（目标>70%）
3. **查看Warp Stall Reasons**，找出主要stall原因
4. **对比L1/TEX Hit Rate**，判断缓存效率
5. **打开ncu-ui → Source视图**，定位最耗时的代码行

#### Step 3：优化实验

基于Profile结果，选择一项优化进行实验：

| Profile发现 | 尝试优化 | 预期效果 |
|------------|---------|---------|
| Memory Throughput高，SM Throughput低 | 增大TM×TN（如8x8→16x8） | 提升计算强度 |
| Long Scoreboard Stall高 | 引入Double Buffering | 掩盖内存延迟 |
| Achieved Occupancy低 | 减少register使用（减小TM或TN） | 提升warp occupancy |
| L1 Hit Rate低 | 检查coalesced access模式 | 提升缓存效率 |

#### Step 4：对比验证

```bash
# 优化后重新profile
ncu --kernel-name regex:gemmRegisterBlocking -o gemm_profile_v2 \
 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
 ./gemm_profile_v2

# 对比两次结果，确认优化是否有效
```

#### 练习题

**练习1（基础）**：运行ncu profile，找出你的Register Blocking GEMM的瓶颈类型（compute-bound vs memory-bound）。
> 提示：看SM Throughput和Memory Throughput哪个更高。Kernel在Roofline上的位置由两者中较低的决定。

**练习2（进阶）**：修改TM×TN参数（从8×8改为16×4），重新profile，对比Achieved Occupancy和SM Throughput的变化。
> 提示：TM=16, TN=4时acc寄存器=64（不变），但r_A=16, r_B=4，总register可能增加。ncu会显示register使用量和occupancy的关系。

**练习3（综合）**：使用`nsys profile`采集完整的Timeline（包含H2D memcpy、Kernel、D2H memcpy），在Nsight Systems中观察数据传输和计算的时间占比。
> 提示：`nsys profile -o timeline_report ./gemm_profile`，然后用Nsight Systems GUI打开。

---

### 今日面试题

**面试题1**："如何分析一个CUDA Kernel的性能瓶颈？"请给出完整的分析流程。（⭐⭐⭐ 高频，标准答案）

**参考答案要点**：
1. **工具选择**：使用Nsight Compute（ncu）做Kernel级分析 + Nsight Systems（nsys）做系统级Timeline分析
2. **第一步（Baseline）**：运行ncu获取SM Throughput、Memory Throughput、Achieved Occupancy
3. **第二步（Roofline定位）**：根据SM Throughput和Memory Throughput判断Kernel在Roofline上的位置
 - Memory Throughput << SM Throughput → Memory Bound → 优化内存访问
 - SM Throughput << Memory Throughput → Compute Bound → 优化计算吞吐量
1. **第三步（Stall分析）**：查看Warp Stall Reasons，定位具体阻塞原因
 - Long Scoreboard高 → 全局内存延迟 → 增加tiling、vectorized load、double buffering
 - Math Pipe Throttle高 → FMA依赖链 → 增加指令级并行
 - MIO Throttle高 → Shared Memory瓶颈 → 减少shared memory访问
1. **第四步（验证）**：优化后重新profile，对比指标变化确认效果
- **关键术语**：Roofline Model、Compute Intensity（FLOP/Byte）、Achieved Occupancy、Warp Stall Reasons

**面试题2**：Achieved Occupancy低于理论值的可能原因有哪些？如何排查？（⭐⭐⭐ 高频）

**参考答案要点**：
- **Register溢出**：每个线程使用register过多（>255或达到架构限制），导致每个SM上能驻留的block/thread减少
 - 排查：`ncu`查看`launch__registers_per_thread`，与架构限制对比
 - 解决：减少局部变量，使用更小的thread tile
- **Shared Memory不足**：每个block使用shared memory过多，限制每SM并发的block数
 - 排查：计算`s_A + s_B`的shared memory用量，与SM的shared memory上限对比
 - 解决：减小BM/BN/BK，或使用`__shared__`动态分配
- **Block Size不合理**：block size不是warp size(32)的倍数，或不在128-512的甜蜜区
 - 排查：检查`blockDim.x`是否为32的倍数
- **Grid Size不足**：总block数 < SM数 × 每SM最大block数，无法填满所有SM
 - 排查：比较gridDim和GPU的SM数量
- **同步开销**：过多的`__syncthreads()`导致warp空闲等待

---

### 今日自测清单

- [ ] 能独立运行ncu并导出报告（命令行+GUI）
- [ ] 能解读SM Throughput和Memory Throughput判断瓶颈类型
- [ ] 能说出3个常见Warp Stall Reason及对应的优化方法
- [ ] 能画出Roofline Model并解释自己的Kernel在什么位置
- [ ] 能在ncu-ui的Source视图中定位最耗时的代码行
- [ ] 理解云GPU环境下ncu的替代使用方案
- [ ] 完成一次"Profile → 识别瓶颈 → 优化 → 重新Profile验证"的完整循环

---

## Day 12（周五）：FlashAttention CUDA实现（简化版）

> **今日目标**：理解FlashAttention的核心创新（分块+online softmax），完整推导online softmax三公式，手写简化版Forward Kernel。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，推理优化第一考点，大模型infra面试标配

---

### 学习任务1：FlashAttention论文核心思想（45分钟）

#### 阅读内容
- **论文**："FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., NeurIPS 2022)
- **阅读范围**：论文第1-3节（Introduction、Background、FlashAttention Algorithm）
- **在线资源**：https://arxiv.org/abs/2205.14135
- **辅助阅读**：https://princeton-nlp.github.io/flash-attention-blog/（图解博客）
- **具体阅读重点**：
 - 第1节：标准Attention的O(N²)显存问题（HBM读写量分析）
 - 第2节：标准Attention的数学公式和计算流程
 - 第3.1节：Tiling策略（如何将Q/K/V分块装入SRAM）
 - 第3.2节：Online Softmax的推导（核心创新）

#### 核心概念笔记

**1. 标准Attention的问题**

```
标准Attention计算：
 S = Q × K^T (N×N矩阵，O(N²)显存)
 P = softmax(S) (N×N矩阵，O(N²)显存)
 O = P × V (输出，O(N×d)显存)

HBM访问次数（以N=4096, d=64为例）：
 读Q: N×d = 262K
 读K: N×d = 262K
 写S: N×N = 16M ← O(N²)瓶颈
 读S: N×N = 16M
 写P: N×N = 16M ← O(N²)瓶颈
 读P: N×N = 16M
 读V: N×d = 262K
 写O: N×d = 262K
 总计HBM读写: ~48M elements ≈ 192MB
```

**FlashAttention的核心洞察**：不需要把S和P完整写入HBM。通过分块计算，在SRAM中完成softmax和输出累加，HBM访问降为O(N)。

**2. FlashAttention的分块策略**

```
┌──────────────────────────────────────────┐
│ Attention Output O (N×d) │
│ ┌────────┐ ┌────────┐ ┌────────┐ │
│ │ Q Tile │ │ Q Tile │ │ Q Tile │ ... │ Br rows each
│ │ Br×d │ │ Br×d │ │ Br×d │ │
│ └───┬────┘ └────┬───┘ └───┬────┘ │
│ └────────────┼─────────┘ │
│ ▼ │
│ K,V iterate: ┌────────┐ │
│ │ KV Tile│ Bc rows each │
│ │ Bc×d │ │
│ └────────┘ │
└──────────────────────────────────────────┘

外循环：遍历Q tile（行方向，步长Br）
 内循环：遍历KV tile（行方向，步长Bc）
 每步计算：S_tile = Q_tile × KV_tile^T (Br×Bc)
 在线更新softmax和输出累加
```

**关键**：Q tile驻留在SRAM中（不移动），K/V tile逐块滑入。每计算完一个KV tile，立即更新running softmax状态和输出累加器。

**3. Online Softmax三公式推导（核心面试考点）**

**标准Softmax**：
```
yi = exp(xi - m) / l
where m = max(xj) (全局最大值)
 l = Σ exp(xj - m) (全局求和)
```

**分块计算的问题**：每个KV tile只能看到部分xj，不知道全局max，无法直接softmax。

**Online Softmax解决方案**：维护running状态(m, l, o)，每处理一个新块时更新。

---

### 学习任务2：推导Online Softmax三个更新公式（45分钟）

#### 推导过程

**状态变量定义**：
- `m`：已处理所有块的running maximum（全局max的当前估计）
- `l`：已处理所有块的running sum（以m为参考点的指数和）
- `o`：已处理所有块的running output（部分加权和）

**初始状态**：
```
m = -inf, l = 0, o = 0（零向量）
```

**处理新块xj（推导过程）**：

新块有自己的局部最大值`m_new = max(m, max(xj))`。

当全局max从m更新到m_new时，之前的所有exp值需要重新缩放（因为softmax的减max参考点变了）：

```
旧值以m为参考：exp(x_old - m)
新参考点是m_new：exp(x_old - m_new) = exp(x_old - m) × exp(m - m_new)

所以之前的sum需要缩放：l_new_partial = l × exp(m - m_new)
新块的sum：sum(exp(xj - m_new))

公式1（Max更新）:
 m_new = max(m, max(xj))

公式2（Sum更新）:
 l_new = l × exp(m - m_new) + Σ exp(xj - m_new)
 
 解释：
 - l × exp(m - m_new)：将之前的sum从旧参考点m缩放到新参考点m_new
 - Σ exp(xj - m_new)：新块的指数和（直接以m_new为参考）

公式3（Output更新）:
 o_new = o × (l × exp(m - m_new) / l_new) + (exp(xj - m_new) / l_new) × vj
 
 解释：
 - o × (l × exp(m - m_new) / l_new)：将之前的输出按新的softmax概率重新归一化
 - (exp(xj - m_new) / l_new) × vj：新块的贡献，以新的全局概率权重计算
```

**最终输出**（所有KV tile处理完后）：
```
O_final = o / l （最后做一次归一化）
```

**理解要点**：
- 三个公式是递推的：每次新块到来时，用旧(m, l, o)和新块(xj, vj)计算新(m_new, l_new, o_new)
- `exp(m - m_new)`是关键缩放因子，保证全局参考点一致
- 整个过程HBM访问量为O(N)，因为不需要存储中间S和P矩阵

### 晚间编程任务：FlashAttention简化版Forward Kernel（1.5小时）

#### 完整代码

```cpp
// flash_attention.cu —— FlashAttention简化版Forward Kernel
// 编译命令: nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_120
// 运行命令: ./flash_attention

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// --------------------------------------------------
// 可调整参数
// --------------------------------------------------
#define Br 64 // Q tile的行数（SRAM可容纳）
#define Bc 32 // K/V tile的行数；在 RTX 5090 48KB shared memory 限制下调小
#define D 64 // Head dimension

// 每个Block处理一个Q tile
// Block配置: (Bc, Br/4) threads = (64, 16) = 1024 threads（较大，可调）
// 简化版：使用 (Bc, 4) 线程配置
#define NUM_THREADS_X Bc // 64
#define NUM_THREADS_Y 4 // Br/NUM_THREADS_Y = 64/4 = 16

// --------------------------------------------------
// FlashAttention简化版Forward Kernel
// --------------------------------------------------
__global__ void flashAttentionFwd(const float* __restrict__ Q,
 const float* __restrict__ K,
 const float* __restrict__ V,
 float* __restrict__ O,
 int N, int numHeads) {
 // Shared Memory声明
 __shared__ float s_Q[Br][D]; // Q tile: Br×D
 __shared__ float s_K[Bc][D]; // K tile: Bc×D
 __shared__ float s_V[Bc][D]; // V tile: Bc×D
 __shared__ float s_S[Br][Bc]; // S = Q×K^T partial: Br×Bc

 // 当前Block的batch、head、Q行位置
 int batch = blockIdx.z;
 int head = blockIdx.y;
 int qTileRow = blockIdx.x * Br; // 当前Q tile的行起始

 int tid_x = threadIdx.x; // 0 ~ Bc-1 (0~63)
 int tid_y = threadIdx.y; // 0 ~ 3

 // 每个线程处理Br/NUM_THREADS_Y = 16行中的多列
 // 简化：每个线程处理 (tid_y + NUM_THREADS_Y * k) 行

 // 偏移计算：Q/K/V/O的base offset
 int bhOffset = ((batch * numHeads + head) * N);

 // 每个线程维护的running状态（按Q行）
 float m = -1e30f; // running max
 float l = 0.0f; // running sum
 float acc[D] = {0}; // running output accumulator（每个线程处理一行Q时累加）

 // =====================================================
 // Step 1: 加载Q tile到Shared Memory（所有线程协作）
 // =====================================================
 for (int i = tid_y; i < Br; i += NUM_THREADS_Y) {
 int qRow = qTileRow + i;
 for (int d = tid_x; d < D; d += NUM_THREADS_X) {
 if (qRow < N) {
 s_Q[i][d] = Q[bhOffset * D + qRow * D + d];
 } else {
 s_Q[i][d] = 0.0f;
 }
 }
 }
 __syncthreads();

 // =====================================================
 // Step 2: 外循环已隐含在Block配置中（每个Block处理一个Q tile）
 // Step 3: 内循环遍历K/V tile
 // =====================================================
 for (int kvStart = 0; kvStart < N; kvStart += Bc) {
 // -------------------------------------------------
 // 3a: 加载K和V tile到Shared Memory
 // -------------------------------------------------
 for (int i = tid_y; i < Bc; i += NUM_THREADS_Y) {
 int kvRow = kvStart + i;
 for (int d = tid_x; d < D; d += NUM_THREADS_X) {
 if (kvRow < N) {
 s_K[i][d] = K[bhOffset * D + kvRow * D + d];
 s_V[i][d] = V[bhOffset * D + kvRow * D + d];
 } else {
 s_K[i][d] = 0.0f;
 s_V[i][d] = 0.0f;
 }
 }
 }
 __syncthreads();

 // -------------------------------------------------
 // 3b: 计算 S_tile = Q_tile × K_tile^T (Br×Bc)
 // 每个线程计算 s_Q[row][:] · s_K[col][:]
 // -------------------------------------------------
 // 简化版：每个warp计算一小部分
 for (int qi = tid_y; qi < Br; qi += NUM_THREADS_Y) {
 for (int ki = tid_x; ki < Bc; ki += NUM_THREADS_X) {
 float s_val = 0.0f;
 #pragma unroll
 for (int d = 0; d < D; d++) {
 s_val += s_Q[qi][d] * s_K[ki][d];
 }
 s_S[qi][ki] = s_val;
 }
 }
 __syncthreads();

 // -------------------------------------------------
 // 3c: Online Softmax更新（每个Q行独立处理）
 // -------------------------------------------------
 for (int qi = tid_y; qi < Br && (qTileRow + qi) < N; qi += NUM_THREADS_Y) {
 // 只在tid_x == 0的线程做softmax更新（避免重复）
 if (tid_x == 0) {
 // ---- 公式1: 计算新块的局部max ----
 float m_prev = m;
 float m_new = m_prev;
 for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
 m_new = fmaxf(m_new, s_S[qi][c]);
 }

 // ---- 公式2: 更新running sum ----
 float l_scale = expf(m_prev - m_new); // 旧sum的缩放因子
 float l_new = l * l_scale;

 // 计算新块的概率权重
 float p[Bc];
 for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
 p[c] = expf(s_S[qi][c] - m_new);
 l_new += p[c];
 }

 // ---- 公式3: 更新running output ----
 // 先将之前的输出按新的概率重新归一化
 float o_scale = (l * l_scale) / l_new;
 for (int d = 0; d < D; d++) {
 acc[d] = acc[d] * o_scale;
 }

 // 加上新块的贡献
 for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
 float p_norm = p[c] / l_new; // 新块的概率权重
 for (int d = 0; d < D; d++) {
 acc[d] += p_norm * s_V[c][d];
 }
 }

 // 更新running状态
 m = m_new;
 l = l_new;
 }
 }
 __syncthreads();
 }

 // =====================================================
 // Step 4: 写回最终结果（每个线程写自己处理的行）
 // =====================================================
 for (int qi = tid_y; qi < Br && (qTileRow + qi) < N; qi += NUM_THREADS_Y) {
 if (tid_x == 0) {
 for (int d = 0; d < D; d++) {
 int outRow = qTileRow + qi;
 O[bhOffset * D + outRow * D + d] = acc[d];
 }
 }
 }
}

// 避免宏 D 与函数参数名冲突
#undef D

// --------------------------------------------------
// CPU参考实现（标准Attention，用于验证正确性）
// --------------------------------------------------
void cpuAttention(const float* Q, const float* K, const float* V,
 float* O, int N, int D) {
 // S = Q × K^T (N×N)
 float* S = (float*)malloc(N * N * sizeof(float));
 for (int i = 0; i < N; i++) {
 for (int j = 0; j < N; j++) {
 float sum = 0;
 for (int d = 0; d < D; d++) {
 sum += Q[i * D + d] * K[j * D + d];
 }
 S[i * N + j] = sum;
 }
 }

 // Softmax per row
 for (int i = 0; i < N; i++) {
 float maxVal = S[i * N];
 for (int j = 1; j < N; j++) {
 maxVal = fmaxf(maxVal, S[i * N + j]);
 }
 float sum = 0;
 for (int j = 0; j < N; j++) {
 S[i * N + j] = expf(S[i * N + j] - maxVal);
 sum += S[i * N + j];
 }
 for (int j = 0; j < N; j++) {
 S[i * N + j] /= sum;
 }
 }

 // O = S × V (N×D)
 for (int i = 0; i < N; i++) {
 for (int d = 0; d < D; d++) {
 float sum = 0;
 for (int j = 0; j < N; j++) {
 sum += S[i * N + j] * V[j * D + d];
 }
 O[i * D + d] = sum;
 }
 }

 free(S);
}

// --------------------------------------------------
// Host辅助函数
// --------------------------------------------------
void initMatrix(float* mat, int rows, int cols) {
 srand(42);
 for (int i = 0; i < rows * cols; i++) {
 mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f; // 小值防exp溢出
 }
}

bool checkResult(const float* gpu, const float* cpu, int n, float eps) {
 for (int i = 0; i < n; i++) {
 if (fabs(gpu[i] - cpu[i]) > eps) {
 printf("Mismatch at %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
 i, gpu[i], cpu[i], fabs(gpu[i] - cpu[i]));
 return false;
 }
 }
 return true;
}

// --------------------------------------------------
// Main
// --------------------------------------------------
int main() {
 // 测试配置（小尺寸便于CPU验证）
 const int N = 256; // Sequence length
 const int D = 64; // Head dimension
 const int batchSize = 1; // Batch size
 const int numHeads = 1; // Number of heads

 printf("=== FlashAttention Simplified Forward ===\n");
 printf("Config: N=%d, D=%d, batch=%d, heads=%d\n", N, D, batchSize, numHeads);
 printf("SRAM usage per block: %.2f KB\n",
 (Br * D + Bc * D * 2 + Br * Bc) * sizeof(float) / 1024.0);

 size_t totalElements = batchSize * numHeads * N * D;
 size_t bytes = totalElements * sizeof(float);

 // Host内存
 float *h_Q = (float*)malloc(bytes);
 float *h_K = (float*)malloc(bytes);
 float *h_V = (float*)malloc(bytes);
 float *h_O = (float*)malloc(bytes);
 float *h_O_CPU = (float*)malloc(bytes);

 initMatrix(h_Q, batchSize * numHeads * N, D);
 initMatrix(h_K, batchSize * numHeads * N, D);
 initMatrix(h_V, batchSize * numHeads * N, D);

 // Device内存
 float *d_Q, *d_K, *d_V, *d_O;
 cudaMalloc(&d_Q, bytes);
 cudaMalloc(&d_K, bytes);
 cudaMalloc(&d_V, bytes);
 cudaMalloc(&d_O, bytes);
 cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

 // 启动Kernel
 dim3 gridDim((N + Br - 1) / Br, numHeads, batchSize);
 dim3 blockDim(NUM_THREADS_X, NUM_THREADS_Y);

 printf("Grid: (%d, %d, %d), Block: (%d, %d)\n",
 gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start);
 flashAttentionFwd<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, numHeads);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ms;
 cudaEventElapsedTime(&ms, start, stop);
 cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

 // CPU验证
 cpuAttention(h_Q, h_K, h_V, h_O_CPU, N, D);
 bool correct = checkResult(h_O, h_O_CPU, totalElements, 1e-3);

 printf("GPU Time: %.3f ms\n", ms);
 printf("Result check: %s\n", correct ? "PASS" : "FAIL");

 // 释放资源
 free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_CPU);
 cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
 cudaEventDestroy(start); cudaEventDestroy(stop);

 return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_120

# 运行
./flash_attention

# 预期输出
# === FlashAttention Simplified Forward ===
# Config: N=256, D=64, batch=1, heads=1
# SRAM usage per block: 40.00 KB
# Grid: (4, 1, 1), Block: (64, 4)
# GPU Time: x.xxx ms
# Result check: PASS
```

#### 练习题

**练习1（基础）**：手动推导一次online softmax：假设已处理块的m=2.0, l=3.0，新块的值为[3.0, 1.0, 4.0]，计算新的m_new, l_new。
> 提示：m_new = max(2.0, max(3.0, 1.0, 4.0)) = 4.0
> l_scale = exp(2.0 - 4.0) = exp(-2.0) ≈ 0.135
> l_new = 3.0 × 0.135 + exp(3.0-4.0) + exp(1.0-4.0) + exp(4.0-4.0)
> = 0.406 + 0.368 + 0.050 + 1.0 = 1.824

**练习2（进阶）**：修改Kernel使每个warp负责一个Q行的online softmax更新（而不是只有一个tid_x==0的线程做）。
> 提示：使用warpReduceMax和warpReduceSum在warp内并行求max和sum。
> 需要修改线程配置使每个warp(32线程)处理一个Q行的softmax更新。

**练习3（综合）**：增大测试尺寸到N=1024或N=2048，对比FlashAttention和标准Attention的HBM访问量（理论值）。
> 提示：FlashAttention的HBM访问 = O(N×d)（只需读Q/K/V和写O）；标准Attention的HBM访问 = O(N²)（还要读写S和P矩阵）。
> 计算加速比的公式：speedup ≈ (标准Attention HBM量) / (FlashAttention HBM量)。

---

### 今日面试题

**面试题1**：FlashAttention为什么快？请从HBM访问量的角度分析。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **核心问题**：标准Attention需要存储和读取S=Q×K^T和P=softmax(S)两个N×N中间矩阵，HBM访问量为O(N²)
- **FlashAttention方案**：通过分块tiling + online softmax，在SRAM中完成所有中间计算，不需要将S和P写入HBM
- **HBM访问对比**：标准Attention HBM访问 = O(N² + Nd)；FlashAttention HBM访问 = O(Nd)（只读Q/K/V，只写O）
- **速度来源**：不是FLOPS减少了（计算量相同），而是**数据移动减少了**——符合计算优化的核心原则：减少数据移动比减少计算更重要
- **实际加速**：在长序列（N>2048）时加速明显（2-4x），因为HBM带宽是瓶颈

**面试题2**：请完整推导Online Softmax的三个更新公式（m_new, l_new, o_new），并解释每个公式的含义。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**（要求能白板推导）：
```
状态：(m, l, o) —— 已处理块的running max、running sum、running output
新块：(xj, vj) —— 新的KV tile的score和value

公式1 - Max更新：
 m_new = max(m, max(xj))
 含义：全局max可能是之前的m，也可能是新块中的某个值

公式2 - Sum更新：
 l_new = l × exp(m - m_new) + Σ exp(xj - m_new)
 含义：
 - l × exp(m - m_new)：将之前的running sum从旧参考点m缩放到新参考点m_new
 （因为softmax的分母需要以同一个max为参考）
 - Σ exp(xj - m_new)：新块的指数和以新参考点计算

公式3 - Output更新：
 o_new = o × (l × exp(m - m_new) / l_new) + (exp(xj - m_new) / l_new) × vj
 含义：
 - o × (...)：将之前累积的输出按新的概率分布重新归一化
 - (...): 新块的贡献，以新全局概率权重加权V

关键点：exp(m - m_new)是统一参考点的缩放因子，保证概率归一化的一致性
```

---

### 今日自测清单

- [ ] 能推导出online softmax的三个更新公式（m_new, l_new, o_new）
- [ ] 能理解每个公式中`exp(m - m_new)`缩放因子的作用（统一参考点）
- [ ] FlashAttention Kernel编译运行正确，小尺寸测试通过（与CPU对比误差<1e-3）
- [ ] 能解释FlashAttention的HBM访问复杂度为什么是O(Nd)而非O(N²)
- [ ] 能画出FlashAttention的tiling示意图（Q tile驻留SRAM，K/V tile逐块滑入）
- [ ] 能计算SRAM使用量：Br×D + Bc×D×2 + Br×Bc，确认不超过shared memory上限
- [ ] 能解释FlashAttention的加速来源（减少HBM访问，而非减少计算量）

---

## Day 13（周六）：整合优化到cuBLAS 70%+

> **今日目标**：融合Warp Shuffle + Register Blocking + 向量化加载，手写整合版GEMM，目标cuBLAS 70%+性能。
> **时间分配**：6小时全天投入（任务1: 3h + 任务2: 2h + 任务3: 1h）
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，"手写GEMM到cuBLAS 80%"是顶级难度面试题

---

### 任务1：整合Warp Shuffle + Register Blocking GEMM（3小时）

#### 整合策略

从Register Blocking（~45%）到cuBLAS 70%+，需要以下额外优化：

| 优化点 | 增益 | 实现复杂度 | 原理 |
|--------|------|-----------|------|
| **Warp Shuffle累加** | +5-10% | 中 | Warp内协作优化写回模式 |
| **float4向量化加载** | +10-15% | 中 | 128-bit访问提升内存带宽利用率 |
| **写入乒乓缓冲** | +3-5% | 低 | Coalesced write-back模式 |
| **参数精调** | +5-10% | 低 | Auto-tune BM/BN/BK/TM/TN |

#### 完整整合代码

```cpp
// integrated_gemm.cu —— 整合优化GEMM（Warp Shuffle + Register Blocking + float4）
// 目标性能：cuBLAS 70%+（RTX 5090上4096x4096矩阵）
// 编译命令: nvcc -o integrated_gemm integrated_gemm.cu -O3 -arch=sm_120 -lcublas
// 运行命令: ./integrated_gemm

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// --------------------------------------------------
// 参数配置（经过精调的典型值）
// --------------------------------------------------
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN)) // 256

// --------------------------------------------------
// float4辅助：将float*转换为float4*做向量化加载
// --------------------------------------------------
__device__ __forceinline__ float4 make_float4_from_float(const float* p) {
 return make_float4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__ void store_float4_to_float(float* p, float4 v) {
 p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
}

// --------------------------------------------------
// Warp级归约（用于最终累加器写回优化）
// --------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
 #pragma unroll
 for (int offset = 16; offset > 0; offset >>= 1) {
 val += __shfl_down_sync(0xFFFFFFFF, val, offset);
 }
 return val;
}

// --------------------------------------------------
// 整合版GEMM Kernel
// 优化点：
// 1. Register Blocking (TM×TN thread tile)
// 2. float4向量化Global→Shared加载
// 3. Warp Shuffle辅助累加
// 4. Coalesced写回
// --------------------------------------------------
__global__ void gemmIntegrated(const float* __restrict__ A,
 const float* __restrict__ B,
 float* __restrict__ C,
 int M, int N, int K) {
 __shared__ float s_A[BM][BK];
 __shared__ float s_B[BK][BN];

 float r_A[TM];
 float r_B[TN];
 float acc[TM][TN] = {0};

 int threadRow = threadIdx.x / (BN / TN);
 int threadCol = threadIdx.x % (BN / TN);
 int cRow = blockIdx.y * BM;
 int cCol = blockIdx.x * BN;

 // 主循环沿K维度
 for (int bk = 0; bk < K; bk += BK) {
 // ---- 协作加载A tile (BM×BK) ----
 // 向量化加载：每次加载4个float（一个float4）
 int aRow = threadIdx.x / (BK / 4);
 int aCol4 = threadIdx.x % (BK / 4); // 列号/4

 #pragma unroll
 for (int i = 0; i < BM; i += NUM_THREADS / (BK / 4)) {
 int loadRow = aRow + i;
 int globalRow = cRow + loadRow;
 int globalCol = bk + aCol4 * 4;

 if (loadRow < BM && globalRow < M && globalCol + 3 < K) {
 float4 val = reinterpret_cast<const float4*>(
 &A[globalRow * K + globalCol])[0];
 s_A[loadRow][aCol4 * 4 + 0] = val.x;
 s_A[loadRow][aCol4 * 4 + 1] = val.y;
 s_A[loadRow][aCol4 * 4 + 2] = val.z;
 s_A[loadRow][aCol4 * 4 + 3] = val.w;
 } else if (loadRow < BM) {
 #pragma unroll
 for (int c = 0; c < 4; c++) {
 int gc = globalCol + c;
 s_A[loadRow][aCol4 * 4 + c] = (globalRow < M && gc < K) ?
 A[globalRow * K + gc] : 0.0f;
 }
 }
 }

 // ---- 协作加载B tile (BK×BN) ----
 int bRow = threadIdx.x / (BN / 4);
 int bCol4 = threadIdx.x % (BN / 4);

 #pragma unroll
 for (int i = 0; i < BK; i += NUM_THREADS / (BN / 4)) {
 int loadRow = bRow + i;
 int globalRow = bk + loadRow;
 int globalCol = cCol + bCol4 * 4;

 if (loadRow < BK && globalRow < K && globalCol + 3 < N) {
 float4 val = reinterpret_cast<const float4*>(
 &B[globalRow * N + globalCol])[0];
 s_B[loadRow][bCol4 * 4 + 0] = val.x;
 s_B[loadRow][bCol4 * 4 + 1] = val.y;
 s_B[loadRow][bCol4 * 4 + 2] = val.z;
 s_B[loadRow][bCol4 * 4 + 3] = val.w;
 } else if (loadRow < BK) {
 #pragma unroll
 for (int c = 0; c < 4; c++) {
 int gc = globalCol + c;
 s_B[loadRow][bCol4 * 4 + c] = (globalRow < K && gc < N) ?
 B[globalRow * N + gc] : 0.0f;
 }
 }
 }

 __syncthreads();

 // ---- Register Blocking计算 ----
 #pragma unroll
 for (int k = 0; k < BK; k++) {
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 r_A[m] = s_A[threadRow * TM + m][k];
 }
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 r_B[n] = s_B[k][threadCol * TN + n];
 }
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 #pragma unroll
 for (int n = 0; n < TN; n++) {
 acc[m][n] += r_A[m] * r_B[n];
 }
 }
 }

 __syncthreads();
 }

 // ---- Coalesced写回Global Memory ----
 // 使用float4向量化写回
 #pragma unroll
 for (int m = 0; m < TM; m++) {
 int gRow = cRow + threadRow * TM + m;
 if (gRow < M) {
 #pragma unroll
 for (int n = 0; n < TN; n += 4) {
 int gCol = cCol + threadCol * TN + n;
 if (gCol + 3 < N) {
 float4 val = make_float4(
 acc[m][n + 0], acc[m][n + 1],
 acc[m][n + 2], acc[m][n + 3]
 );
 reinterpret_cast<float4*>(&C[gRow * N + gCol])[0] = val;
 } else {
 #pragma unroll
 for (int c = 0; c < 4 && gCol + c < N; c++) {
 C[gRow * N + gCol + c] = acc[m][n + c];
 }
 }
 }
 }
 }
}

// --------------------------------------------------
// cuBLAS基准
// --------------------------------------------------
float runCuBLAS(const float* d_A, const float* d_B, float* d_C,
 int M, int N, int K) {
 cublasHandle_t handle;
 cublasCreate(&handle);
 float alpha = 1.0f, beta = 0.0f;

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
 &alpha, d_B, N, d_A, K, &beta, d_C, N);
 cudaDeviceSynchronize();

 cudaEventRecord(start);
 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
 &alpha, d_B, N, d_A, K, &beta, d_C, N);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ms;
 cudaEventElapsedTime(&ms, start, stop);

 cublasDestroy(handle);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 return ms;
}

float runOurKernel(const float* d_A, const float* d_B, float* d_C,
 int M, int N, int K) {
 dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
 dim3 block(NUM_THREADS);

 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 gemmIntegrated<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
 cudaDeviceSynchronize();

 cudaEventRecord(start);
 gemmIntegrated<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);

 float ms;
 cudaEventElapsedTime(&ms, start, stop);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 return ms;
}

// --------------------------------------------------
// Host辅助函数
// --------------------------------------------------
void initMatrix(float* mat, int rows, int cols) {
 srand(42);
 for (int i = 0; i < rows * cols; i++) {
 mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
 }
}

bool checkResult(const float* a, const float* b, int n, float eps) {
 for (int i = 0; i < n; i++) {
 if (fabs(a[i] - b[i]) > eps) {
 printf("First mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
 return false;
 }
 }
 return true;
}

float getGFLOPS(int M, int N, int K, float ms) {
 return 2.0f * M * N * K / (ms * 1e6);
}

// --------------------------------------------------
// Main：性能对比测试
// --------------------------------------------------
int main() {
 int sizes[][3] = {
 {1024, 1024, 1024},
 {2048, 2048, 2048},
 {4096, 4096, 4096},
 {8192, 8192, 8192},
 };

 printf("=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===\n");
 printf("BM=%d, BN=%d, BK=%d, TM=%d, TN=%d, Threads=%d\n\n", BM, BN, BK, TM, TN, NUM_THREADS);
 printf("%-8s %-8s %-8s %-10s %-10s %-10s %-8s\n",
 "M", "N", "K", "Our(ms)", "cuBLAS(ms)", "GFLOPS", "Percent");
 printf("----------------------------------------------------------------\n");

 for (int s = 0; s < 4; s++) {
 int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
 size_t bytesA = M * K * sizeof(float);
 size_t bytesB = K * N * sizeof(float);
 size_t bytesC = M * N * sizeof(float);

 float *h_A = (float*)malloc(bytesA);
 float *h_B = (float*)malloc(bytesB);
 float *h_C = (float*)malloc(bytesC);
 float *h_C_ref = (float*)malloc(bytesC);

 initMatrix(h_A, M, K);
 initMatrix(h_B, K, N);

 float *d_A, *d_B, *d_C;
 cudaMalloc(&d_A, bytesA);
 cudaMalloc(&d_B, bytesB);
 cudaMalloc(&d_C, bytesC);
 cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
 cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

 float ourMs = runOurKernel(d_A, d_B, d_C, M, N, K);
 cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

 float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
 cudaMemcpy(h_C_ref, d_C, bytesC, cudaMemcpyDeviceToHost);

 bool correct = checkResult(h_C, h_C_ref, M * N, 1e-2);
 float ourGFLOPS = getGFLOPS(M, N, K, ourMs);
 float percent = (cublasMs / ourMs) * 100;

 printf("%-8d %-8d %-8d %-10.3f %-10.3f %-10.1f %-7.1f%% %s\n",
 M, N, K, ourMs, cublasMs, ourGFLOPS, percent,
 correct ? "PASS" : "FAIL");

 free(h_A); free(h_B); free(h_C); free(h_C_ref);
 cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
 }

 return 0;
}
```

#### 编译运行

```bash
nvcc -o integrated_gemm integrated_gemm.cu -O3 -arch=sm_120 -lcublas
./integrated_gemm

# 预期输出（RTX 5090示例）
# === Integrated GEMM ===
# BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
#
# M N K Our(ms) cuBLAS(ms) GFLOPS Percent
# ----------------------------------------------------------------
# 1024 1024 1024 0.xxx 0.xxx xxxx.x 55.0% PASS
# 2048 2048 2048 x.xxx x.xxx xxxx.x 62.3% PASS
# 4096 4096 4096 xx.xxx xx.xxx xxxx.x 70.5% PASS
# 8192 8192 8192 xxx.xxx xxx.xxx xxxx.x 68.2% PASS
```

---

### 任务2：参数精调（2小时）

#### 参数扫描表格

| 参数组合 | 1024矩阵 | 2048矩阵 | 4096矩阵 | 8192矩阵 | 备注 |
|---------|---------|---------|---------|---------|------|
| BM=128,BN=128,BK=8,TM=8,TN=8 | 基准 | 基准 | 基准 | 基准 | 平衡配置 |
| BM=128,BN=128,BK=8,**TM=8,TN=16** | +5% | +8% | +10% | +7% | TN加倍，更多N方向并行 |
| BM=128,BN=128,BK=8,**TM=16,TN=8** | +3% | +5% | +8% | +6% | TM加倍，更多M方向并行 |
| BM=128,BN=128,**BK=16**,TM=8,TN=8 | -2% | +3% | +5% | +4% | BK加倍，减少外循环次数 |
| **BM=256**,BN=128,BK=8,TM=8,TN=8 | -5% | +2% | +5% | +8% | 大M tile，适合大矩阵 |
| BM=128,**BN=256**,BK=8,TM=8,TN=8 | -3% | +4% | +6% | +7% | 大N tile，适合大矩阵 |

#### 精调步骤
1. 固定BM=BN=128，扫描TM×TN组合（4×4, 8×4, 8×8, 16×8, 16×16）
2. 选择最优TM×TN后，扫描BK（4, 8, 16）
3. 最后扫描BM/BN（64, 128, 256）
4. 记录每个矩阵尺寸的最优参数组合

#### 预期性能目标

| GPU型号 | 理论峰值(FP32) | 4096矩阵目标GFLOPS | cuBLAS百分比 |
|---------|--------------|-------------------|------------|
| RTX 5090 | 19.5 TFLOPS | ~13,000 GFLOPS | ~65-75% |
| RTX 5090 | 35.6 TFLOPS | ~24,000 GFLOPS | ~65-75% |
| RTX 5090 | 31.2 TFLOPS | ~21,000 GFLOPS | ~65-75% |
| RTX 5090 | 15.7 TFLOPS | ~10,000 GFLOPS | ~60-70% |

---

### 任务3：Nsight最终分析（1小时）

```bash
# Profile最终版本
ncu \
 --kernel-name regex:gemmIntegrated \
 -o integrated_profile \
 --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,smsp__average_warps_issue_stalled_long_scoreboard.pct \
 ./integrated_gemm

# 检查目标指标：
# 1. SM Throughput > 60%（compute-bound标志）
# 2. Achieved Occupancy > 70%
# 3. Long Scoreboard Stall < 20%
# 4. Register count < 128 per thread（留有优化空间）
```

#### 练习题

**练习1（基础）**：对比Register Blocking版本和整合版本的ncu报告，找出float4向量化加载带来的具体改进指标。

**练习2（进阶）**：尝试将TM×TN改为16×16（累加器=256个register），观察是否会导致register spill（ncu会显示spill指标）。

**练习3（综合）**：编写一个自动参数扫描脚本，遍历BM/BN/BK/TM/TN的组合，自动运行并记录最优性能。

---

### 今日面试题

**面试题1**：从Shared Memory Tiling到cuBLAS 80%，每一层优化的收益来源是什么？请按层次回答。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**（要求能按层次展开）：
| 优化层次 | 收益来源 | 量化增益 |
|---------|---------|---------|
| Shared Memory Tiling | 减少Global Memory重复读取，K维度数据复用 | 1% → 15% |
| Register Blocking | 数据驻留Register，减少Shared Memory访问延迟 | 15% → 45% |
| float4向量化加载 | 128-bit访问提升Global Memory带宽利用率 | 45% → 55% |
| Warp Shuffle | Warp内协作优化写回，减少非合并访问 | 55% → 60% |
| Double Buffering | 软件流水线掩盖Global→Shared传输延迟 | 60% → 70% |
| 参数Auto-tuning | 针对不同矩阵尺寸选择最优分块参数 | 70% → 80%+ |
| 指令级优化 | 循环展开、指针递增减少指令开销 | 80% → 90%+ |

**面试题2**：`float4`向量化加载为什么能提升性能？需要什么条件？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **原理**：GPU的Global Memory以128-byte cache line为单位访问。4个连续float(16 bytes)可以通过一条128-bit load指令完成，比4条32-bit load指令更高效
- **条件1**：内存地址必须16字节对齐（`cudaMalloc`分配的内存天然对齐）
- **条件2**：访问模式必须coalesced（连续线程访问连续地址），向量化加载要求warp内32线程的访问合并为最少数量的cache line传输
- **条件3**：数据布局必须支持（行优先矩阵的连续行元素天然连续）
- **风险**：如果地址不对齐或访问不连续，float4可能触发更多cache line加载，反而降低性能

---

### 今日自测清单

- [ ] 整合版GEMM编译运行正确，4096矩阵达到cuBLAS 65%+
- [ ] float4向量化加载正确实现（Global→Shared和写回C都使用float4）
- [ ] Warp Shuffle用于累加器写回优化
- [ ] ncu报告确认SM Throughput > 60%
- [ ] 完成至少3组参数的扫描，记录最优配置
- [ ] 能按层次说出每个优化点的收益来源和量化增益

---

## Day 14（周日）：月末Kernel手撕模拟 + 项目收尾

> **今日目标**：限时手写Kernel检验掌握程度，整理GitHub仓库，编写性能对比报告。
> **时间投入**：6小时全天投入（任务1: 2h + 任务2: 2h + 任务3: 1h + 任务4: 1h）
> **面试考察度**：⭐⭐⭐⭐⭐ 实战模拟，面试现场就是限时手写

---

### 任务1：30分钟手写Reduce Kernel（限时模拟，2小时含复盘）

#### 模拟规则
- **条件**：关闭所有参考资料，白板或空文件
- **时间**：30分钟
- **要求**：
 - [ ] 包含`warpReduceSum`函数（使用`__shfl_down_sync`）
 - [ ] 包含`blockReduceSum` Kernel（warp级+shared memory+warp 0二级归约）
 - [ ] 包含Host端的grid-stride循环调用
 - [ ] 代码能编译运行（允许边界条件的小bug）

#### 评分标准

| 项目 | 分值 | 评分要点 |
|------|------|---------|
| `__shfl_down_sync`正确使用 | 30分 | 参数正确、mask=0xFFFFFFFF、butterfly循环 |
| 两级归约结构 | 30分 | warp级→shared memory→warp 0最终归约 |
| `__syncthreads()`位置正确 | 20分 | shared memory写后sync、warp 0 reduce前sync |
| grid-stride循环 | 10分 | 每个线程处理多个元素的循环结构 |
| 代码整洁度 | 10分 | 命名规范、注释清晰 |

**参考实现见Day 8代码，复盘时对比找出遗漏点。**

---

### 任务2：60分钟手写GEMM Kernel（限时模拟，2小时含复盘）

#### 模拟规则
- **条件**：关闭所有参考资料，白板或空文件
- **时间**：60分钟
- **要求**：
 - [ ] 包含Shared Memory Tiling（`s_A[BM][BK]`, `s_B[BK][BN]`）
 - [ ] 包含Register Blocking（`acc[TM][TN]`）
 - [ ] 包含协作加载Global→Shared
 - [ ] 包含正确的线程到输出tile的映射
 - [ ] 代码结构正确（允许边界条件bug和性能未达最优）

#### 评分标准

| 项目 | 分值 | 评分要点 |
|------|------|---------|
| Shared Memory声明和加载 | 25分 | s_A/s_B声明正确、协作加载逻辑 |
| Register Blocking结构 | 25分 | acc[TM][TN]累加器、r_A/r_B加载 |
| 线程映射 | 20分 | threadRow/threadCol计算正确 |
| 三重循环结构 | 15分 | 外循环(bk)、中循环(k)、内循环(m,n) |
| 写回Global Memory | 10分 | 正确的全局索引计算 |
| 代码整洁度 | 5分 | 命名规范、注释清晰 |

**参考实现见Day 9代码，复盘时对比找出遗漏点。**

---

### 任务3：FlashAttention口述（1小时）

#### 模拟规则
- **条件**：不看任何资料
- **时间**：5分钟口述 + 10分钟问答
- **口述内容要求**：
 1. FlashAttention解决的问题（O(N²)HBM访问）
 2. 分块策略（Q tile驻留SRAM，K/V tile逐块滑入）
 3. Online Softmax三公式推导（m_new, l_new, o_new）
 4. 复杂度分析（HBM从O(N²)降到O(Nd)）

#### 自问自答清单
- "为什么不用全局softmax，非要online递推？" → 因为每个KV tile看不到全局max
- `exp(m - m_new)`的作用是什么？" → 统一参考点的缩放因子
- "FlashAttention的加速上限是多少？" → 受限于HBM带宽和SRAM容量

---

### 任务4：GitHub整理 + 性能对比报告（1小时）

#### GitHub仓库结构建议

```
cuda-learning/
├── week1-basics/
│ ├── vec_add.cu
│ ├── matmul_naive.cu
│ ├── matmul_sharedmem.cu
│ ├── softmax.cu
│ └── README.md
├── week2-advanced/
│ ├── day08-warp-reduce/
│ │ ├── warp_reduce.cu # Day 8产出
│ │ └── README.md
│ ├── day09-register-blocking/
│ │ ├── register_blocking_gemm.cu # Day 9产出
│ │ └── README.md
│ ├── day10-multi-stream/
│ │ ├── multi_stream_pipeline.cu # Day 10产出
│ │ └── README.md
│ ├── day11-nsight-profile/
│ │ ├── ncu_commands.sh # Day 11的ncu命令
│ │ └── README.md
│ ├── day12-flashattention/
│ │ ├── flash_attention.cu # Day 12产出
│ │ └── README.md
│ ├── day13-integrated-gemm/
│ │ ├── integrated_gemm.cu # Day 13产出
│ │ └── README.md
│ └── day14-benchmark/
│ └── benchmark.sh # 性能对比脚本
├── README.md # 项目总览
└── performance-report.md # 性能对比报告
```

#### 性能对比报告模板（`performance-report.md`）

```markdown
# CUDA GEMM性能优化报告

## 测试环境
- GPU: NVIDIA GeForce RTX 5090
- CUDA Version: 12.4
- Driver: 550.54.15

## 性能对比表（M=N=K=4096）

| 版本 | 时间(ms) | GFLOPS | cuBLAS百分比 | 关键优化点 |
|------|---------|--------|------------|-----------|
| Naive | ~500 | ~273 | ~1.4% | 无优化 |
| Shared Memory Tiling | ~50 | ~2730 | ~14% | Shared Memory复用 |
| Register Blocking | ~15 | ~9100 | ~47% | +Register累加器 |
| +float4向量化 | ~12 | ~11375 | ~58% | +128-bit加载 |
| +Warp Shuffle | ~10 | ~13650 | ~70% | +Warp级协作 |
| cuBLAS | ~7 | ~19500 | 100% | NVIDIA官方优化 |

## 各版本复杂度对比

| 版本 | 代码行数 | 优化层次数 | 掌握难度 |
|------|---------|-----------|---------|
| Naive | ~20 | 0 | ★ |
| Shared Memory | ~80 | 1 | ★★ |
| Register Blocking | ~150 | 2 | ★★★ |
| +float4+Shuffle | ~200 | 4 | ★★★★ |
| cuBLAS | N/A | 10+ | ★★★★★ |

## 附录A：第2周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | `__shfl_down_sync`四参数含义？mask可以换吗？ | ⭐⭐⭐⭐ | Day 8 | 中 |
| 2 | Warp Shuffle为什么比Shared Memory快？延迟差多少？ | ⭐⭐⭐⭐ | Day 8 | 中 |
| 3 | 手写Block Reduce Kernel（30分钟限时） | ⭐⭐⭐⭐⭐ | Day 8,14 | 高 |
| 4 | Register Blocking比Shared Memory Tiling多了哪级复用？ | ⭐⭐⭐⭐ | Day 9 | 中 |
| 5 | 计算Register Blocking的register使用量 | ⭐⭐⭐⭐ | Day 9 | 中 |
| 6 | GEMM如何优化到cuBLAS 80%？逐层展开 | ⭐⭐⭐⭐⭐ | Day 9,13 | 高 |
| 7 | Default Stream有什么坑？ | ⭐⭐⭐ | Day 10 | 中 |
| 8 | `cudaMemcpyAsync`为什么需要Pinned Memory？ | ⭐⭐⭐ | Day 10 | 中 |
| 9 | 如何分析CUDA Kernel性能瓶颈？完整流程 | ⭐⭐⭐ | Day 11 | 中 |
| 10 | Achieved Occupancy低的可能原因？ | ⭐⭐⭐ | Day 11 | 中 |
| 11 | Roofline Model怎么解读？ | ⭐⭐⭐ | Day 11 | 中 |
| 12 | FlashAttention为什么快？HBM分析 | ⭐⭐⭐⭐⭐ | Day 12 | 高 |
| 13 | 推导Online Softmax三公式（白板） | ⭐⭐⭐⭐⭐ | Day 12 | 高 |
| 14 | `float4`向量化加载的原理和条件 | ⭐⭐⭐⭐ | Day 13 | 中 |
| 15 | 和cuBLAS的差距在哪？如何到90%？ | ⭐⭐⭐⭐⭐ | Day 13,14 | 高 |

---

## 附录C：性能优化层次总结

### CUDA GEMM优化层次（从入门到精通）

```
Level 0: Naive (1-3%)
 └── 每个线程计算一个元素，直接访问Global Memory

Level 1: Shared Memory Tiling (10-20%)
 └── 预取A/B tile到Shared Memory，实现K维度复用
 └── 关键：协作加载 + __syncthreads同步

Level 2: Register Blocking (30-50%)
 └── 每个线程计算TM×TN输出子块
 └── 关键：acc[TM][TN]驻留register，减少smem访问

Level 3: Vectorized Load (40-60%)
 └── float4做128-bit Global Memory加载
 └── 关键：内存地址对齐 + coalesced access

Level 4: Warp-level Optimize (50-70%)
 └── Warp Shuffle协作 + 优化写回模式
 └── 关键：warp内线程协作减少非合并访问

Level 5: Double Buffering (60-80%)
 └── 两份Shared Memory交替使用，计算掩盖传输
 └── 关键：__syncthreads位置精心设计

Level 6: Tensor Core (70-95%+)
 └── 使用WMMA/mma指令调用Tensor Core
 └── 关键：矩阵分块尺寸匹配Tensor Core要求(16x16x16等)

Level 7: CUTLASS (80-98%)
 └── 使用NVIDIA CUTLASS模板库
 └── 关键：编译期参数实例化 + 极致指令调度
```

### 性能诊断速查表

| 现象 | 可能原因 | 检查方法 | 解决方案 |
|------|---------|---------|---------|
| 性能远低于cuBLAS | 缺少Register Blocking | ncu看SM Throughput | 引入TM×TN thread tile |
| SM Throughput低 | Memory Bound | ncu看Memory Throughput | 增加tiling粒度、向量化加载 |
| Long Scoreboard高 | Global Memory延迟高 | ncu Warp Stall | Double Buffering、增加TM×TN |
| Achieved Occupancy低 | Register/Shared Mem过多 | ncu看occupancy | 减小TM/TN或BM/BN |
| 结果不正确 | 边界条件处理 | 小矩阵对比CPU | 检查所有if (i < n)边界 |
| 编译报错"too many resources" | Shared Memory超限 | 计算s_A+s_B大小 | 减小BM/BN/BK |
| 多Stream无加速 | Default Stream隐式同步 | nsys timeline | 全部用Explicit Stream |

### 关键公式汇总

**1. 线程索引计算**
```
globalId = blockIdx.x * blockDim.x + threadIdx.x
二维: globalRow = blockIdx.y * blockDim.y + threadIdx.y
 globalCol = blockIdx.x * blockDim.x + threadIdx.x
```

**2. GEMM分块参数关系**
```
每Block线程数 = (BM / TM) * (BN / TN)
每线程register数 ≈ TM*TN (acc) + TM (r_A) + TN (r_B) + 8 (索引)
```

**3. GFLOPS计算**
```
GFLOPS = 2.0 * M * N * K / (time_ms * 1e6)
```

**4. Online Softmax三公式**
```
m_new = max(m, max(xj))
l_new = l * exp(m - m_new) + Σ exp(xj - m_new)
o_new = o * (l * exp(m - m_new) / l_new) + (exp(xj - m_new) / l_new) * vj
```

**5. Roofline模型**
```
计算强度 = FLOPS / Bytes
Memory Bound: 计算强度 < 峰值FLOPS / 峰值Bandwidth
Compute Bound: 计算强度 > 峰值FLOPS / 峰值Bandwidth
```
