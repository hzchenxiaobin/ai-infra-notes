# Week 2：CUDA 进阶优化与性能分析

> 核心目标：掌握 Warp Shuffle、Register Blocking、CUDA Stream 异步执行、Nsight 性能分析和 FlashAttention CUDA 实现

| 项目 | 说明 |
|------|------|
| 前置要求 | 已完成 Week 1 学习，掌握向量加法、Naive GEMM、Shared Memory Tiling GEMM、Softmax Kernel |
| 建议时长 | 工作日每天 2.5h，周末每天 6h，周计 24.5h |
| 本周产出 | Warp Reduce Kernel、Register Blocking GEMM（cuBLAS 40%+）、Multi-Stream 重叠执行、Nsight 分析报告、FlashAttention 简化版 Forward Kernel、整合优化 GEMM（cuBLAS 70%+） |
| 周日里程碑 | 手写优化 GEMM 达到 cuBLAS 70%+ 性能，完成简化版 FlashAttention Forward Kernel |

---

## 🧭 本周学习地图

```
Day 1: Warp Shuffle 原语 → Warp Reduce Kernel（两级归约）
        ↓
Day 2: Register Blocking + 2D Tiling → GEMM cuBLAS 40%+
        ↓
Day 3: CUDA Streams 异步 → H2D/Compute/D2H 重叠流水线
        ↓
Day 4: Nsight Compute → Register Blocking GEMM 瓶颈分析
        ↓
Day 5: FlashAttention → Online Softmax 推导 + Forward Kernel
        ↓
Day 6: 整合 Warp Shuffle + Register Blocking → GEMM cuBLAS 70%+
        ↓
Day 7: 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告
```

---

## Day 1：Warp Shuffle 原语与 Warp/Block Reduce

### 🎯 目标

通过今天的学习，你将：

1. 理解 Warp Shuffle 的硬件原理和相比 Shared Memory 的延迟优势
2. 掌握 `__shfl_sync`、`__shfl_up_sync`、`__shfl_down_sync`、`__shfl_xor_sync` 四个原语
3. 理解 Butterfly 模式归约的 5 步通信过程
4. 手写完整的 Warp Reduce + Block Reduce 两级归约 Kernel
5. 理解为什么需要两级归约，以及第二级归约为什么也用 Warp Shuffle
6. 能对照昇腾 CANN 的 Vector Unit 理解 Warp 级通信的跨平台一致性

> 💡 **为什么重要**：Warp Shuffle 是手写 Reduce、Scan、GEMM 写回优化的标配技术，也是面试中「手写 Block Reduce」的核心考点。它比 Shared Memory 快一个数量级，是 GPU 内部最快的线程间通信方式。

---

### 学前导读：为什么需要 Warp 级通信

在 Week 1 中，我们学习了 Shared Memory 作为 block 内线程共享数据的手段。但在很多场景下，通信只发生在 **同一个 Warp 内的 32 个线程之间**，此时 Shared Memory 并不是最优选择。

#### Shared Memory 的局限

Shared Memory 虽然比 Global Memory 快得多（~20-30 cycles vs ~400-800 cycles），但它仍然有几个开销：

1. **需要同步**：写 Shared Memory 后必须调用 `__syncthreads()` 保证可见性
2. **需要两条指令**：先 store 到 Shared Memory，再 load 从 Shared Memory
3. **延迟仍有 20-30 cycles**：对于高频的 Warp 内通信，这个延迟不可忽略

#### Warp Shuffle 的优势

从 Kepler 架构（Compute Capability 3.0）起，NVIDIA 引入了 Warp Shuffle 原语，允许同一 Warp 内的线程**直接交换寄存器数据**，无需经过 Shared Memory：

| 通信方式 | 延迟（周期） | 是否需要 `__syncthreads()` | 指令数 | 适用场景 |
|---------|------------|-------------------------|--------|---------|
| Shared Memory 中转 | ~20-30 cycles | 是（需同步） | ≥ 2 条（store + load） | Block 级通信（跨 Warp） |
| **Warp Shuffle 直连** | **~1-2 cycles** | **否（硬件同步）** | **1 条** | **Warp 内通信（32 线程）** |
| Register 直连 | ~0 cycles | 否 | 0 | 同一线程内复用 |

**核心洞察**：Shuffle 的延迟比 Shared Memory 低一个数量级，因为它直接通过 Warp 内部的专用交换网络传输数据，无需经过 Shared Memory 的读写路径。

> 💡 **一句话总结**：Warp Shuffle 是 GPU 内部最快的线程间通信方式，比 Shared Memory 快 10-20 倍，但仅限于同一 Warp 内的 32 个线程。

---

### 理论学习

#### 1.1 Warp Shuffle 四大家族

CUDA 提供了四个 Warp Shuffle 原语，分别对应不同的通信模式：

| 原语名称 | 函数签名 | 作用描述 | 使用场景 |
|---------|---------|---------|---------|
| `__shfl_sync` | `T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)` | 从 `srcLane` 线程读取 `var` 值 | 广播：线程 0 的结果广播给 Warp 内所有线程 |
| `__shfl_up_sync` | `T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` | 从 `threadIdx-delta` 线程读取 | 前缀和：每个线程获取左侧 delta 距离线程的值 |
| `__shfl_down_sync` | `T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` | 从 `threadIdx+delta` 线程读取 | 归约：Warp 内折半累加 |
| `__shfl_xor_sync` | `T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)` | 从 `threadIdx ^ laneMask` 线程读取 | Butterfly 交换：归约、位反转排序 |

##### 四个原语的数据流向图

```
__shfl_sync(mask, val, srcLane=4):
  Lane:  0   1   2   3   4   5   6   7  ... 31
  数据:  ?   ?   ?   ?   V4  ?   ?   ?  ... ?
  结果:  V4  V4  V4  V4  V4  V4  V4  V4  ... V4    ← 所有 lane 都得到 lane 4 的值

__shfl_up_sync(mask, val, delta=2):
  Lane:  0   1   2   3   4   5   6   7  ... 31
  数据:  V0  V1  V2  V3  V4  V5  V6  V7  ... V31
  结果:  V0  V1  V0  V1  V2  V3  V4  V5  ... V29  ← 每个 lane 从上方 delta 处取值

__shfl_down_sync(mask, val, delta=2):
  Lane:  0   1   2   3   4   5   6   7  ... 31
  数据:  V0  V1  V2  V3  V4  V5  V6  V7  ... V31
  结果:  V2  V3  V4  V5  V6  V7  V8  V9  ...  ?   ← 每个 lane 从下方 delta 处取值

__shfl_xor_sync(mask, val, laneMask=2):
  Lane:  0   1   2   3   4   5   6   7  ... 31
  交换:  2   3   0   1   6   7   4   5  ... 29  ← lane i 与 lane (i^2) 交换
```

#### 1.2 参数详解

以 `__shfl_down_sync` 为例，它有四个参数：

```cpp
float val = __shfl_down_sync(0xFFFFFFFF, myVal, 16, 32);
//              │              │           │      │    │
//              │              │           │      │    └── width: 参与 shuffle 的宽度（默认 32）
//              │              │           └─────────────── delta: 向下偏移量
//              │              └─────────────────────────── var: 要传递的变量
//              └────────────────────────────────────────── mask: 线程掩码，0xFFFFFFFF=全部 32 线程
//              └────────────────────────────────────────── 返回值: 从目标线程读取的值
```

##### mask 参数详解

`mask` 是一个 32 位无符号整数，每一位对应 Warp 内的一个 lane（线程）：

- `0xFFFFFFFF`（全部 32 位为 1）：表示 Warp 内所有 32 个线程都参与
- `0x0000FFFF`：表示只有 lane 0-15 参与
- `0xFFFF0000`：表示只有 lane 16-31 参与

> ⚠️ **注意**：从 Volta 架构（CC 7.0）开始，必须使用 `_sync` 后缀版本（带显式 mask）。旧版 `__shfl_down`（无 mask）已被弃用，因为 Volta 引入了独立线程调度，需要显式指定哪些线程参与 shuffle。

##### width 参数详解

`width` 控制 shuffle 操作的分组宽度，必须是 2 的幂（2, 4, 8, 16, 32）：

```cpp
// width=32（默认）：整个 Warp 一起 shuffle
val = __shfl_down_sync(0xFFFFFFFF, val, 1, 32);  // lane 0 读 lane 1

// width=16：Warp 分成两组（lane 0-15, lane 16-31），各自独立 shuffle
val = __shfl_down_sync(0xFFFFFFFF, val, 1, 16);  // lane 0 读 lane 1，lane 16 读 lane 17
```

#### 1.3 Warp Reduce Butterfly 模式

Warp Reduce（求和）使用 `__shfl_down_sync` 实现折半累加，整个过程像蝴蝶展翅，因此称为 **Butterfly 模式**：

```
Step 1 (offset=16): lane0 <- lane0+lane16, lane1 <- lane1+lane17, ... lane15 <- lane15+lane31
Step 2 (offset=8):  lane0 <- lane0+lane8,  lane1 <- lane1+lane9,  ... lane7  <- lane7+lane15
Step 3 (offset=4):  lane0 <- lane0+lane4,  lane1 <- lane1+lane5,  ... lane3  <- lane3+lane7
Step 4 (offset=2):  lane0 <- lane0+lane2,  lane1 <- lane1+lane3
Step 5 (offset=1):  lane0 <- lane0+lane1
Result: lane 0 持有 Warp 内 32 个线程的累加和
```

5 步操作（`log₂32 = 5`）即可完成 32 个线程的归约，每步只需 1 条 shuffle 指令。

对应的代码：

```cuda
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

##### `__shfl_down_sync` vs `__shfl_xor_sync` 的区别

两者都能实现归约，但有一个关键区别：

| 原语 | 归约后哪些 lane 有结果 | 适用场景 |
|------|---------------------|---------|
| `__shfl_down_sync` | **只有 lane 0** 有最终结果 | 只需要一个结果的归约 |
| `__shfl_xor_sync` | **所有 lane** 都有最终结果 | 需要所有线程都知道归约结果的场景（如 broadcast） |

```cuda
// down 模式：只有 lane 0 有结果
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;  // 只有 lane 0 的 val 是完整的
}

// xor 模式：所有 lane 都有结果
__inline__ __device__ float warpReduceSumAll(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;  // 所有 lane 的 val 都是完整的
}
```

#### 1.4 两级归约：Warp Reduce + Block Reduce

##### 为什么需要两级归约？

单个 Warp 只有 32 个线程。但一个 Block 可能有 1024 个线程（32 个 Warp）。Warp 级归约只解决 32 线程的汇总，Block 级需要将 32 个 Warp 的结果再做一次汇总。

##### 两级归约的执行流程

```
Block (1024 threads = 32 warps)
│
├── Step 1: 每个线程从 Global Memory 读取并做 per-thread 累加
│           (使用 grid-stride loop 处理大数组)
│
├── Step 2: Warp 级归约（每个 Warp 的 32 个线程累加到 lane 0）
│           使用 __shfl_down_sync 的 butterfly 模式
│
├── Step 3: lane 0 将 Warp 的部分和写入 Shared Memory
│           warpSums[0], warpSums[1], ..., warpSums[31]
│           __syncthreads()  ← 等待所有 Warp 完成
│
└── Step 4: Warp 0 做最终归约
            lane 0~31 读取 warpSums[0]~warpSums[31]
            再执行一次 warpReduceSum
            lane 0 写入最终结果到 Global Memory
```

##### 为什么第二级归约也用 Warp Shuffle？

因为 Warp 0 有 32 个 lane，正好处理最多 32 个 Warp 的部分和。用 `__shfl_down_sync` 做第二级归约比用 Shared Memory 循环更快。

> 💡 **核心记忆点**：两级归约 = Warp Shuffle（快路径，处理 32 线程）+ Shared Memory 中转（慢路径，处理跨 Warp）+ Warp 0 二次 Shuffle（最终汇总）。

---

### 昇腾对照

| CUDA 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| `__shfl_down_sync` | `__all_reduce` / `__reduce_add`（Ascend C 内置） | CUDA 需要显式写 shuffle 循环；昇腾 Ascend C 提供 Warp 级归约内置函数，调用更简洁 |
| `0xFFFFFFFF` mask | 隐式全线程参与 | CUDA 需要显式指定 mask 控制哪些 lane 参与；昇腾的 Vector Unit 操作默认全线程参与 |
| Warp 内寄存器交换 | Vector Unit 内联通信 | 昇腾的 Vector Unit（VU）内部同样有高速数据交换通路，但抽象层级更高 |
| `__reduce_add_sync` (CUDA 11+) | `__reduce_add` (Ascend C) | 两者都是 Warp 级归约的封装 API，功能等效 |
| Warp 级 Reduce 延迟 ~1-2 cycle | Vector Unit 指令延迟 ~2-4 cycle | 量级相当，都是片上 fastest communication path |

**关键差异**：CUDA 的 Shuffle 原语更底层，需要开发者手动控制 butterfly 模式；昇腾的 Ascend C 提供了更高级的归约内置函数，但底层同样依赖类似的寄存器级数据交换。

---

### Coding 任务：手写 Warp Reduce Kernel

#### 任务 1：创建 warp_reduce.cu

创建文件 [kernels/warp_reduce.cu](kernels/warp_reduce.cu)：

```cuda
// warp_reduce.cu —— Warp 级 + Block 级两级归约完整实现
// 编译命令: nvcc -o warp_reduce warp_reduce.cu -O3 -arch=sm_80
// 运行命令: ./warp_reduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// --------------------------------------------------
// Warp 级归约：使用 __shfl_down_sync 折半累加
// --------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------
// Warp 级归约（XOR 模式）：使用 __shfl_xor_sync
// 用途：when you need reduction result in ALL lanes, not just lane 0
// --------------------------------------------------
__inline__ __device__ float warpReduceSumXor(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------
// Block 级归约：每个 warp 先归约，然后 warp 0 做最终归约
// --------------------------------------------------
__global__ void blockReduceSum(const float* input, float* output, int n) {
    __shared__ float warpSums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Step 1: 每个线程从 global memory 读取并做 per-thread 累加
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Step 2: Warp 级归约（每个 warp 的 32 个线程累加到 lane 0）
    sum = warpReduceSum(sum);

    // Step 3: lane 0 将 warp 的部分和写入 shared memory
    if (lane == 0) {
        warpSums[wid] = sum;
    }
    __syncthreads();

    // Step 4: Warp 0 做最终归约
    if (wid == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// --------------------------------------------------
// 多 block 版本：需要第二次 kernel 调用汇总
// --------------------------------------------------
float launchReduce(const float* d_input, float* d_temp, float* d_output,
                   int n, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = min(blocks, 1024);

    blockReduceSum<<<blocks, threadsPerBlock>>>(d_input, d_temp, n);
    cudaDeviceSynchronize();

    blockReduceSum<<<1, 256>>>(d_temp, d_output, blocks);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// --------------------------------------------------
// Host 辅助函数
// --------------------------------------------------
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
}

float cpuReduceSum(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return static_cast<float>(sum);
}

int main() {
    const int n = 1 << 22;  // 4,194,304 个元素
    printf("=== Warp Shuffle Block Reduce ===\n");
    printf("Array size: %d (%.2f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

    float* h_input = (float*)malloc(n * sizeof(float));
    initData(h_input, n);

    float *d_input, *d_temp, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_temp, 1024 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float gpuSum = launchReduce(d_input, d_temp, d_output, n, 256);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float cpuSum = cpuReduceSum(h_input, n);
    float diff = fabs(gpuSum - cpuSum);

    printf("GPU Sum: %.6f\n", gpuSum);
    printf("CPU Sum: %.6f\n", cpuSum);
    printf("Diff:    %.6f (%s)\n", diff, diff < 1e-3 ? "PASS" : "FAIL");
    printf("Time:    %.3f ms (%.2f GB/s bandwidth)\n",
           ms, n * sizeof(float) / (ms * 1e6));

    free(h_input);
    cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
```

#### 任务 2：编译与运行

```bash
# 编译（根据 GPU 架构选择 arch 参数）
# Ampere (A100, RTX 30xx): sm_80
# Turing (RTX 20xx, T4): sm_75
# Volta (V100): sm_70
nvcc -o warp_reduce kernels/warp_reduce.cu -O3 -arch=sm_80

# 运行
./warp_reduce
```

**预期输出**：

```
=== Warp Shuffle Block Reduce ===
Array size: 4194304 (16.00 MB)
GPU Sum: 20973.4xxxx
CPU Sum: 20973.4xxxx
Diff:    0.00xxxx (PASS)
Time:    0.xxx ms (xx.xx GB/s bandwidth)
```

#### 任务 3：使用 ncu 查看 Warp Shuffle 效率

```bash
ncu \
  --metrics \
    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread \
  ./warp_reduce
```

**观察重点**：
- Occupancy 应该较高（因为 kernel 使用的寄存器和 Shared Memory 都很少）
- 执行时间应该非常短（因为 Shuffle 延迟极低）

#### 为什么 grid-stride loop 能处理大数组？

代码中有这样一段循环：

```cuda
for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    sum += input[i];
}
```

这叫 **grid-stride loop**（网格步长循环）。当数组大小 `n` 远大于总线程数（`blockDim.x * gridDim.x`）时，每个线程需要处理多个元素。

```
假设 n = 4194304, threadsPerBlock = 256, blocks = 1024
总线程数 = 256 × 1024 = 262144
每个线程处理 n / 262144 ≈ 16 个元素

线程 0 处理: input[0], input[262144], input[524288], ...
线程 1 处理: input[1], input[262145], input[524289], ...
```

grid-stride loop 的好处：
1. **不限制数组大小**：无论 `n` 多大都能处理
2. **coalesced access**：相邻线程访问相邻地址（步长 = 1）
3. **自动负载均衡**：每个线程处理的元素数相近

---

### 扩展实验

#### 实验 1：`__shfl_down_sync` vs `__shfl_xor_sync`

修改 `warpReduceSum`，分别使用 `__shfl_down_sync` 和 `__shfl_xor_sync`，在归约后打印所有 lane 的值：

```cuda
// 验证：down 模式只有 lane 0 有结果，xor 模式所有 lane 都有结果
sum = warpReduceSum(sum);       // down 模式
if (lane < 4) printf("down: lane %d = %f\n", lane, sum);

sum = warpReduceSumXor(sum);    // xor 模式
if (lane < 4) printf("xor:  lane %d = %f\n", lane, sum);
```

**预期结果**：
- `down` 模式：只有 lane 0 有正确的累加和，其他 lane 的值是部分和
- `xor` 模式：所有 lane 都有相同的完整累加和

#### 实验 2：实现 Warp Reduce Max

将 `warpReduceSum` 改为求最大值的版本：

```cuda
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
```

用这个函数找出数组中的最大元素，与 CPU 结果对比。

#### 实验 3：Segmented Warp Reduce

使用 mask 参数实现分组归约：同一个 Warp 内分两组（lane 0-15 和 lane 16-31），各自独立归约。

```cuda
// mask=0x0000FFFF 表示只激活 lane 0-15
// mask=0xFFFF0000 表示只激活 lane 16-31
float val = __shfl_down_sync(0x0000FFFF, val, offset);  // lane 0-15 内部归约
float val = __shfl_down_sync(0xFFFF0000, val, offset);  // lane 16-31 内部归约
```

思考：mask 参数的作用是什么？在什么场景下需要部分 Warp 参与？

---

### 常见错误与调试

| 错误 | 原因 | 解决方法 |
|------|------|---------|
| 编译警告 `__shfl_down` deprecated | 使用了旧版无 `_sync` 的 API | 改用 `__shfl_down_sync(mask, ...)` |
| 归约结果不正确 | `__syncthreads()` 位置错误 | 确保在 Shared Memory 写入后、Warp 0 读取前同步 |
| 只有部分 Warp 的结果被汇总 | 第二级归约时 `numWarps` 计算错误 | `numWarps = (blockDim.x + warpSize - 1) / warpSize` |
| 大数组结果与 CPU 不一致 | 累加顺序不同导致浮点误差 | 使用 double 做 CPU 验证，允许 1e-3 误差 |
| ncu 报告 occupancy 很低 | block 内线程数太少 | 使用 256 或 512 线程 per block |

**调试技巧**：
- 先用小数组（如 1024 元素，1 个 block）测试正确性
- 在 `warpReduceSum` 后打印 lane 0 的值，确认 Warp 级归约正确
- 逐步增大数组大小，验证多 block 场景

---

### 验证 Checklist

- [ ] 能解释 `__shfl_down_sync` 四个参数的含义（mask, val, delta, width）
- [ ] 能画出 Warp 内 32 线程执行 shuffle butterfly 的 5 步通信图（offset=16→8→4→2→1）
- [ ] Warp Reduce 代码编译运行正确，GPU 结果与 CPU 对比误差 < 1e-3
- [ ] 能解释 `0xFFFFFFFF` 的含义：32 位掩码，激活 Warp 内所有 32 个 lane
- [ ] 能解释为什么两级归约需要 `__syncthreads()`（Warp 间同步点）
- [ ] 能说出 `__shfl_down_sync` 和 `__shfl_xor_sync` 的区别（down=单向偏移，xor=蝴蝶交换）
- [ ] 理解 Volta+ 架构必须使用 `_sync` 版本的原因（独立线程调度需要显式 mask）
- [ ] 能对照昇腾的 Vector Unit 解释 Warp 级通信的跨平台一致性

---

### 今日总结

Day 1 我们掌握了 Warp Shuffle 这一 GPU 内最快的线程间通信原语：

1. **Warp Shuffle 四大家族**：`__shfl_sync`（广播）、`__shfl_up_sync`（前缀和）、`__shfl_down_sync`（归约）、`__shfl_xor_sync`（Butterfly 交换）
2. **延迟优势**：Shuffle 延迟 ~1-2 cycles，比 Shared Memory（~20-30 cycles）快 10-20 倍
3. **Butterfly 归约**：5 步 `__shfl_down_sync`（offset=16→8→4→2→1）完成 32 线程归约
4. **两级归约**：Warp 级 Shuffle → Shared Memory 中转 → Warp 0 二次 Shuffle
5. **grid-stride loop**：处理大数组的标准模式，保证 coalesced access
6. **mask 参数**：控制哪些 lane 参与 shuffle，可实现 segmented reduce

掌握这些后，你就拥有了手写高性能 Reduce、Scan、Broadcast 的核心工具。

---

### 面试要点

1. **`__shfl_down_sync(0xFFFFFFFF, val, 16)` 的四个参数分别是什么含义？`0xFFFFFFFF` 可以换成其他值吗？**

   - 参数 1 `mask=0xFFFFFFFF`：32 位线程掩码，每一位对应一个 lane。`0xFFFFFFFF` 表示全部 32 个 lane 参与。可以换成其他值实现部分 Warp 操作（如 segmented reduce）
   - 参数 2 `val`：要传递的本线程变量值
   - 参数 3 `delta=16`：目标 lane 的偏移量，即读取 `(laneId + delta) % width` 线程的值
   - 参数 4（省略，默认 32）`width`：参与 shuffle 的宽度，默认 32（整个 Warp）
   - 注意：从 Volta 架构开始必须使用 `_sync` 后缀版本（显式 mask），旧版 `__shfl_down` 已被弃用

2. **为什么 Warp Shuffle 比 Shared Memory 更适合做 Warp 内归约？实际延迟差距有多大？**

   - **延迟差距**：Shuffle 延迟约 1-2 cycles，Shared Memory 访问延迟约 20-30 cycles，差距约 10-20 倍
   - **原因 1（硬件路径）**：Shuffle 通过 Warp 内部的专用交换网络直接从源寄存器读取数据，不经过 Shared Memory 的读写路径
   - **原因 2（无需同步）**：Shuffle 在 Warp 内部是隐式同步的（SIMT 执行模型保证 Warp 内线程同步），不需要 `__syncthreads()`；Shared Memory 方式需要 `__syncthreads()` 保证数据可见性
   - **原因 3（指令数少）**：Shuffle 是一条指令完成读+交换；Shared Memory 需要至少两条指令（store + load）
   - **局限**：Shuffle 只适用于 Warp 内（最多 32 线程），跨 Warp 通信仍需 Shared Memory

3. **为什么需要两级归约？谁来做第二级归约？**

   - 单个 Warp 只有 32 线程，一个 Block 可能有 1024 线程（32 个 Warp）。Warp 级归约只解决 32 线程的汇总
   - 第二级归约由 **Warp 0** 完成：lane 0~31 读取 Shared Memory 中 32 个 Warp 的部分和，再执行一次 `warpReduceSum`
   - 为什么用 Warp Shuffle 而不是 Shared Memory 循环？因为 Warp 0 有 32 个 lane，正好处理最多 32 个 Warp 的部分和，Shuffle 比 Shared Memory 循环更快

4. **`__shfl_down_sync` 和 `__shfl_xor_sync` 在归约场景下有什么区别？**

   - `__shfl_down_sync`：归约后**只有 lane 0** 有最终结果，适合只需要一个结果的场景
   - `__shfl_xor_sync`：归约后**所有 lane** 都有相同的最终结果，适合需要所有线程都知道归约结果的场景（如 broadcast 归约结果给所有线程）

5. **grid-stride loop 是什么？为什么需要它？**

   - grid-stride loop 让每个线程处理多个元素，步长为 `blockDim.x * gridDim.x`（总线程数）
   - 当数组大小远大于总线程数时，保证每个线程都能处理多个元素
   - 好处：不限制数组大小、保证 coalesced access、自动负载均衡

---

## Day 2：Register Blocking 与 2D Tiling

### 🎯 目标

通过今天的学习，你将：

1. 理解从 Shared Memory Tiling 到 Register Blocking 的进化路径
2. 掌握 Thread Tile 概念：每个线程负责计算输出矩阵的 TM×TN 子块
3. 理解三级数据复用层次：Global Memory → Shared Memory → Register
4. 掌握 Register 使用量的计算方法
5. 理解 Double Buffering（软件流水线）的原理
6. 实现 Register Blocking GEMM，性能达到 cuBLAS 40%+

> 💡 **为什么重要**：Register Blocking 是「如何优化 GEMM 到 cuBLAS 80%」这一顶级面试题的关键转折点，它把性能从 ~15% 提升到 ~45%，是从入门到进阶的分水岭。

---

### 学前导读：从 Shared Memory Tiling 到 Register Blocking

在 Week 1 中我们学习了 Shared Memory Tiling GEMM。它的核心是：把 A/B 的子矩阵预取到 Shared Memory，实现 K 维度的数据复用。但这还不够快，因为每个线程仍然只计算 C 的一个元素，对 Shared Memory 的访问非常频繁。

**Register Blocking 的核心思想**：让每个线程计算 C 的一个 **TM×TN 子块**，把累加器驻留在寄存器中，从而减少对 Shared Memory 的访问次数。

```
优化层次          数据驻留位置        复用对象           性能目标
─────────────────────────────────────────────────────────────
Naive GEMM        Global Memory      无                 ~1-3% peak
Shared Mem Tiling Shared Memory      A/B tile           ~15-25% peak
Register Blocking Register           A子行/B子列+累加器  ~40-60% peak
Warp-level        Register+Shuffle   Warp内协作          ~60-80% peak
软件流水线         全部 + 双缓冲       计算掩盖传输        ~80-95% peak
```

---

### 理论学习

#### 2.1 Register Blocking 数据流图

```
Global Memory (A[M][K], B[K][N])
    │
    ▼ 协作加载（所有线程参与）
Shared Memory (s_A[BM][BK], s_B[BK][BN])
    │
    ├──► Register (r_A[TM]) ──┐
    │                           ▼
    └──► Register (r_B[TN]) ──► FMA累加 (acc[TM][TN])
    │                              │
    ◄──────────────────────────────┘
         重复 BK 次（内层 k 循环）
```

每个线程的执行流程：
1. 从 Shared Memory 加载 TM 个 A 元素到 `r_A[TM]`
2. 从 Shared Memory 加载 TN 个 B 元素到 `r_B[TN]`
3. 做 TM×TN 次 FMA 累加到 `acc[TM][TN]`
4. 重复 BK 次（内层 k 维度循环）
5. 一个 BK 块处理完后，加载下一 BK 块，重复

#### 2.2 关键参数定义

| 参数 | 含义 | 典型值 | 计算公式 |
|------|------|--------|---------|
| BM | Block tile 的 M 维度 | 128 | 可调 |
| BN | Block tile 的 N 维度 | 128 | 可调 |
| BK | Block tile 的 K 维度 | 8 | 较小值减少 shared mem 占用 |
| TM | 每个线程负责的 M 方向输出数 | 8 | acc 寄存器 = TM×TN |
| TN | 每个线程负责的 N 方向输出数 | 8 | acc 寄存器 = TM×TN |
| 每 Block 线程数 | BM/TM × BN/TN | 256 | (128/8)×(128/8)=16×16=256 |

#### 2.3 Register 使用量计算

每个线程的 register 消耗：
- 累加器：`acc[TM][TN]` = TM × TN = 64 个 float
- A 加载寄存器：`r_A[TM]` = 8 个 float
- B 加载寄存器：`r_B[TN]` = 8 个 float
- 索引/临时变量：~8 个 float
- **总计**：~88 个 float ≈ 88 个 register

> ⚠️ 注意：现代 GPU 每个线程最多 255 个 register（如 A100）。如果 TM=TN=16，累加器就有 256 个 register，会溢出到 local memory 导致性能暴跌。

#### 2.4 线程到输出 tile 的二维映射

```
输出 tile (BM×BN = 128×128) 被划分为 (BM/TM)×(BN/TN) = 16×16 = 256 个 thread tile
每个 thread tile = TM×TN = 8×8

threadIdx.x 的范围: 0 ~ 255
threadRow = threadIdx.x / (BN / TN) = threadIdx.x / 16  → 范围 0~15
threadCol = threadIdx.x % (BN / TN) = threadIdx.x % 16   → 范围 0~15

线程(threadRow, threadCol) 负责输出的行范围:
  [blockIdx.y * BM + threadRow * TM,  blockIdx.y * BM + (threadRow+1) * TM)
负责输出的列范围:
  [blockIdx.x * BN + threadCol * TN,  blockIdx.x * BN + (threadCol+1) * TN)
```

#### 2.5 Double Buffering（软件流水线）

```
单缓冲： [Load Tile 0] ──► [Compute Tile 0] ──► [Load Tile 1] ──► [Compute Tile 1]
                          ▲ 空闲等待（Load 不能被 Compute 掩盖）

双缓冲： [Load Tile 0→Buf0] ──► [Compute Tile 0 同时 Load Tile 1→Buf1]
                              ──► [Compute Tile 1 同时 Load Tile 2→Buf0]
         ▲ Compute 和 Load 并行执行，掩盖传输延迟
```

实现方式：声明两份 shared memory buffer，奇偶 tile 交替使用。

---

### 昇腾对照

| CUDA 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| Register Blocking（Thread Tile） | Split-K / FRACTAL_NZ 分块 | 昇腾的 FRACTAL_NZ 布局本质上也是一种多级分块 |
| `acc[TM][TN]` 驻留寄存器 | L0 Buffer accumulator 驻留 | CUDA 用寄存器文件做累加器；昇腾用 L0 Buffer |
| Shared Memory | L1 Buffer (UB/Tiling Buffer) | 都用于存储从 HBM 预取的 tile 数据 |
| Double Buffering | Fixpipe 自动流水线 | 昇腾硬件自动完成，CUDA 需手动实现 |
| FMA 指令 | Cube Core 的 MAC 指令 | 昇腾 Cube Core 用矩阵乘累加（MAC）指令 |

---

### Coding 任务：Register Blocking GEMM

#### 任务 1：创建 register_blocking_gemm.cu

创建文件 [kernels/register_blocking_gemm.cu](kernels/register_blocking_gemm.cu)：

```cuda
// register_blocking_gemm.cu —— Register Blocking 矩阵乘法完整实现
// 编译命令: nvcc -o register_gemm register_blocking_gemm.cu -O3 -arch=sm_80 -lcublas
// 运行命令: ./register_gemm

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void gemmRegisterBlocking(const float* __restrict__ A,
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

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile
        int aRow = threadIdx.x / BK;
        int aCol = threadIdx.x % BK;
        #pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS / BK) {
            int loadRow = aRow + i;
            if (loadRow < BM && (cRow + loadRow) < M && (bk + aCol) < K) {
                s_A[loadRow][aCol] = A[(cRow + loadRow) * K + (bk + aCol)];
            } else if (loadRow < BM) {
                s_A[loadRow][aCol] = 0.0f;
            }
        }

        // 协作加载 B tile
        int bRow = threadIdx.x / BN;
        int bCol = threadIdx.x % BN;
        #pragma unroll
        for (int i = 0; i < BK; i += NUM_THREADS / BN) {
            int loadRow = bRow + i;
            if (loadRow < BK && (bk + loadRow) < K && (cCol + bCol) < N) {
                s_B[loadRow][bCol] = B[(bk + loadRow) * N + (cCol + bCol)];
            } else if (loadRow < BK) {
                s_B[loadRow][bCol] = 0.0f;
            }
        }

        __syncthreads();

        // Register Blocking 计算
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

    // 写回 Global Memory
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

float runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

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

void initMatrix(float* mat, int rows, int cols) {
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
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
    return 2.0f * M * N * K / (ms * 1e6);
}

int main() {
    int sizes[][3] = {{1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096}};

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
        float *h_C_ref = (float*)malloc(sizeC);
        initMatrix(h_A, M, K);
        initMatrix(h_B, K, N);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);
        cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blockDim(NUM_THREADS);

        gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ourMs;
        cudaEventElapsedTime(&ourMs, start, stop);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

        float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C_ref, d_C, sizeC, cudaMemcpyDeviceToHost);

        bool correct = checkResult(h_C, h_C_ref, M, N, 1e-2);
        float percent = (cublasMs / ourMs) * 100;

        printf("%-10d %-10d %-10d %-12.3f %-12.3f %-9.1f%% %s\n",
               M, N, K, ourMs, cublasMs, percent, correct ? "PASS" : "FAIL");

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}
```

#### 任务 2：编译运行

```bash
nvcc -o register_gemm kernels/register_blocking_gemm.cu -O3 -arch=sm_80 -lcublas
./register_gemm
```

**预期输出**：

```
=== Register Blocking GEMM ===
Parameters: BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256
M          N          K          Our(ms)      cuBLAS(ms)   Percent
------------------------------------------------------------
1024       1024       1024       0.xxx        0.xxx        35.2%  PASS
2048       2048       2048       x.xxx        x.xxx        42.1%  PASS
4096       4096       4096       xx.xxx       xx.xxx       45.8%  PASS
```

#### 任务 3：检查 Register 使用量

```bash
nvcc -Xptxas -v -o register_gemm kernels/register_blocking_gemm.cu -O3 -arch=sm_80 -lcublas
```

观察输出中的 `Used N registers`，确认没有 `spill stores/loads`。

---

### 扩展实验

#### 实验 1：调整 TM 和 TN

修改 TM 和 TN 的值（如 TM=4, TN=4 或 TM=16, TN=4），用 `nvcc -Xptxas -v` 观察 register 使用量变化和性能变化。

#### 实验 2：实现 Double Buffering

声明两份 shared memory buffer（`s_A[2][BM][BK]`），奇偶 tile 交替使用，用计算掩盖 global→shared memory 的传输延迟。

#### 实验 3：使用 Warp Shuffle 优化累加器

在 Register Blocking 基础上，使用 `__shfl_xor_sync` 实现 Warp 内线程的累加器交换，减少写回 global memory 时的非合并访问。

---

### 验证 Checklist

- [ ] 能解释 Register Blocking 相比纯 Shared Memory Tiling 多了一级数据复用（Global→Shared→Register）
- [ ] 能计算 register usage：TM×TN 个累加器 + TM + TN 个加载寄存器 + 索引变量
- [ ] 代码编译运行正确，性能达到 cuBLAS 40%+（4096 矩阵）
- [ ] 能画出数据流图：Global Memory → Shared Memory → Register → FMA 累加
- [ ] 能计算每 Block 线程数 = (BM/TM) × (BN/TN)
- [ ] 能解释 Double Buffering 的原理（用计算掩盖数据传输延迟）
- [ ] 能对照昇腾的 FRACTAL_NZ + Split-K 解释 register blocking 的数据分片哲学

---

### 今日总结

Day 2 我们掌握了 Register Blocking 这一 GEMM 优化的核心转折点：

1. **三级数据复用**：Global Memory → Shared Memory → Register，每多一级复用都大幅减少访存延迟
2. **Thread Tile**：每个线程计算 TM×TN 子块，累加器驻留寄存器，减少 Shared Memory 访问
3. **Register 计算**：TM=TN=8 时约 88 个 register，在 255 上限内
4. **协作加载**：所有线程协作把 A/B tile 从 Global 加载到 Shared Memory
5. **Double Buffering**：双缓冲掩盖传输延迟，是从 45% 到 70% 的关键优化

---

### 面试要点

1. **"如何把手写 GEMM 优化到 cuBLAS 80% 的性能？"请逐层展开优化策略。**

   按层次展开：
   - Naive（~1%）→ Shared Memory Tiling（~15%）→ Register Blocking（~40%）→ Warp-level（~60%）→ Vectorized Load（~70%）→ Double Buffering（~80%）→ Auto-tuning（~90%+）

2. **Register Blocking 中的 `acc[TM][TN]` 为什么要放在 register 而不是 shared memory？**

   - 寄存器访问延迟 ~0 cycle，Shared Memory 延迟 ~20-30 cycles
   - `acc[TM][TN]` 被访问 TM×TN×BK 次（内层循环），放在 register 极大减少延迟
   - 放 shared memory 会占用 BM×BN×4 = 64KB，超过 shared memory 上限

3. **Register 使用量如何计算？什么时候会 spill？**

   - `acc[TM][TN]` + `r_A[TM]` + `r_B[TN]` + 索引变量
   - TM=TN=8 时约 88 个 register；TM=TN=16 时累加器就有 256 个，会 spill 到 local memory

---

## Day 3：CUDA Streams 与异步执行

### 🎯 目标

通过今天的学习，你将：

1. 理解 CUDA Stream 异步执行模型
2. 掌握 Default Stream 的隐式同步行为及其"坑"
3. 掌握多 Stream 并行策略与 `cudaMemcpyAsync` 的使用
4. 理解 Pinned Memory 对异步传输的必要性
5. 实现多 Stream H2D/Compute/D2H 重叠流水线
6. 能使用 `cudaEvent` 管理跨 Stream 依赖

> 💡 **为什么重要**：多 Stream 异步执行是提升端到端吞吐的关键技术。理解 Default Stream 的隐式同步行为，能避免"看似创建了多 Stream 却没有任何并发"的性能陷阱。

---

### 学前导读：为什么需要异步执行

CPU 和 GPU 是独立的计算资源。如果所有操作都同步执行（CPU 提交后等 GPU 完成），CPU 大量时间在空等。异步执行让 CPU 提交任务后立即返回，GPU 在后台执行，二者并行工作。

更进一步，GPU 内部有独立的硬件引擎：**Copy Engine**（负责 H2D/D2H 传输）和 **Compute Engine**（负责 Kernel 执行）。如果安排得当，拷贝和计算可以同时进行，这就是 Multi-Stream 重叠流水线的核心。

---

### 理论学习

#### 3.1 Stream 的本质

Stream 是 GPU 上操作（Kernel 执行、内存拷贝）的队列。同一个 Stream 内的操作按 FIFO 顺序执行，不同 Stream 之间的操作可以并发（只要资源允许）。

```
Stream 1: [H2D拷贝1] → [Kernel1] → [D2H拷贝1]
Stream 2: [H2D拷贝2] → [Kernel2] → [D2H拷贝2]
           ↑ H2D拷贝2可以与Kernel1并发执行（copy engine和compute unit独立）
```

#### 3.2 Default Stream 的"坑"

| 特性 | Default Stream (Stream 0) | Explicit Stream |
|------|-------------------------|-----------------|
| 创建方式 | 隐式存在，无需创建 | `cudaStreamCreate(&stream)` |
| 同步行为 | **隐式同步所有其他 Stream** | 只同步本 Stream 的操作 |
| 与 Host 关系 | `cudaMemcpy` 阻塞 Host | `cudaMemcpyAsync` 非阻塞 Host |
| 适用场景 | 简单程序、调试 | 生产环境、性能优化 |

**Default Stream 的隐式同步规则（极易出错）**：
- 规则 1：Default Stream 上的操作会等待所有其他 Stream 的先前操作完成
- 规则 2：其他 Stream 上的操作会等待 Default Stream 的先前操作完成
- **后果**：即使创建了多 Stream，只要在 Default Stream 上做一次 `cudaMemcpy`，所有并发都被打断

**解决方案**：始终使用 `cudaStreamNonBlocking` 标志创建 Stream，或编译时加 `--default-stream per-thread`。

```cuda
// 创建不与 Default Stream 同步的 Stream（推荐做法）
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

#### 3.3 `cudaMemcpy` vs `cudaMemcpyAsync`

| 函数 | 同步性 | 是否可指定 Stream | 内存要求 | 使用场景 |
|------|--------|----------------|---------|---------|
| `cudaMemcpy` | **同步**（阻塞 Host 直到完成） | 否（Default Stream） | 任意内存 | 简单程序 |
| `cudaMemcpyAsync` | **异步**（立即返回） | 是 | 必须使用 **pinned 内存** | 多 Stream 并发 |

**Pinned Memory（页锁定内存）**：通过 `cudaMallocHost` 分配，不会被 OS 换出到磁盘，支持 DMA 直接传输。

> 为什么 `cudaMemcpyAsync` 需要 Pinned Memory？因为异步传输使用 DMA 引擎直接访问内存，如果内存被 OS 换出到磁盘，DMA 无法访问。普通 pageable 内存会被 CUDA 驱动先复制到临时 pinned buffer，导致异步退化为同步。

#### 3.4 多 Stream 重叠流水线

```
无 Stream（顺序）： [H2D拷贝] ──► [Kernel计算] ──► [D2H拷贝]
                   总计 = H2D + Compute + D2H

Multi-Stream（重叠）：
  Stream1: [H2D chunk1] ──► [Kernel chunk1] ──► [D2H chunk1]
  Stream2:        [H2D chunk2] ──► [Kernel chunk2] ──► [D2H chunk2]
  Stream3:               [H2D chunk3] ──► [Kernel chunk3] ──► [D2H chunk3]
  Stream4:                      [H2D chunk4] ──► [Kernel chunk4] ──► [D2H chunk4]
                   ↑ H2D与Kernel重叠，Kernel与D2H重叠
                   总计 ≈ max(H2D + D2H, Compute) + 流水线填充
```

#### 3.5 cudaEvent 跨 Stream 依赖

当 Stream 间存在数据依赖时，用 Event 实现精确同步：

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream A 中记录事件
cudaEventRecord(event, streamA);

// Stream B 等待该事件
cudaStreamWaitEvent(streamB, event, 0);
```

---

### 昇腾对照

| CUDA 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| `cudaStreamCreate` | `aclrtCreateStream` | 函数语义完全一致 |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` | 等待流中所有操作完成 |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync` | 异步内存拷贝，都需要 pinned memory |
| Default Stream 隐式同步 | 昇腾 Stream 默认行为 | 类似机制 |
| `cudaMallocHost`（Pinned Memory） | `aclrtMallocHost` | 页锁定内存，用于 DMA 传输 |
| Copy Engine + Compute Engine | 昇腾 DMA 引擎 + AI Core | 硬件架构类似，都支持拷贝与计算并发 |

---

### Coding 任务：Multi-Stream 重叠流水线

#### 任务 1：创建 multi_stream_pipeline.cu

创建文件 [kernels/multi_stream_pipeline.cu](kernels/multi_stream_pipeline.cu)：

```cuda
// multi_stream_pipeline.cu —— 多 Stream 重叠流水线完整实现
// 编译命令: nvcc -o multi_stream multi_stream_pipeline.cu -O3 -arch=sm_80
// 运行命令: ./multi_stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = A[i] + B[i];
        for (int j = 0; j < 100; j++) {
            sum = sum * 0.999f + 0.001f;
        }
        C[i] = sum;
    }
}

// 顺序版本（baseline）
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

// Multi-Stream 重叠版本
float multiStreamVersion(float* h_A, float* h_B, float* h_C,
                          float* d_A, float* d_B, float* d_C,
                          int totalSize, int chunkSize, int nStreams) {
    int numChunks = (totalSize + chunkSize - 1) / chunkSize;
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

int main() {
    const int totalSize = 1 << 24;  // 16,777,216 个元素
    const int chunkSize = 1 << 18;  // 262,144 个元素 per chunk
    const int nStreams = 4;

    printf("=== Multi-Stream Overlap Pipeline ===\n");
    printf("Total size: %d (%.2f MB)\n", totalSize,
           totalSize * sizeof(float) / (1024.0 * 1024.0));
    printf("Chunk size: %d (%.2f MB)\n", chunkSize,
           chunkSize * sizeof(float) / (1024.0 * 1024.0));
    printf("Num chunks: %d, Num streams: %d\n\n",
           (totalSize + chunkSize - 1) / chunkSize, nStreams);

    size_t totalBytes = totalSize * sizeof(float);
    float *h_A, *h_B, *h_C_seq, *h_C_multi;
    cudaMallocHost(&h_A, totalBytes);
    cudaMallocHost(&h_B, totalBytes);
    cudaMallocHost(&h_C_seq, totalBytes);
    cudaMallocHost(&h_C_multi, totalBytes);

    srand(42);
    for (int i = 0; i < totalSize; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, totalBytes);
    cudaMalloc(&d_B, totalBytes);
    cudaMalloc(&d_C, totalBytes);

    printf("Running sequential version...\n");
    float seqMs = sequentialVersion(h_A, h_B, h_C_seq, d_A, d_B, d_C, totalSize, chunkSize);
    printf("Sequential: %.3f ms\n\n", seqMs);

    printf("Running multi-stream version (nStreams=%d)...\n", nStreams);
    float multiMs = multiStreamVersion(h_A, h_B, h_C_multi, d_A, d_B, d_C,
                                        totalSize, chunkSize, nStreams);
    printf("Multi-Stream: %.3f ms\n\n", multiMs);

    bool correct = true;
    for (int i = 0; i < totalSize; i++) {
        if (fabs(h_C_seq[i] - h_C_multi[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    float speedup = seqMs / multiMs;
    printf("=== Performance Summary ===\n");
    printf("Sequential:   %.3f ms\n", seqMs);
    printf("Multi-Stream: %.3f ms\n", multiMs);
    printf("Speedup:      %.2fx\n", speedup);
    printf("Result check: %s\n", correct ? "PASS" : "FAIL");

    cudaFreeHost(h_A); cudaFreeHost(h_B);
    cudaFreeHost(h_C_seq); cudaFreeHost(h_C_multi);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

#### 任务 2：编译运行

```bash
nvcc -o multi_stream kernels/multi_stream_pipeline.cu -O3 -arch=sm_80
./multi_stream
```

**预期输出**：

```
=== Multi-Stream Overlap Pipeline ===
...
=== Performance Summary ===
Sequential:   xxx.xxx ms
Multi-Stream: xx.xxx ms
Speedup:      1.2x ~ 1.8x
Result check: PASS
```

#### 任务 3：使用 nsys 观察多 Stream 重叠

```bash
nsys profile -o multi_stream_timeline ./multi_stream
```

用 Nsight Systems GUI 打开 `.nsys-rep` 文件，在 Timeline 视图中观察不同 Stream 的操作条是否有重叠区域。

---

### 扩展实验

#### 实验 1：对比 NonBlocking 标志

修改代码使用 `cudaStreamCreate`（不带 NonBlocking 标志）代替 `cudaStreamCreateWithFlags`，观察性能差异。

#### 实验 2：实现 cudaEvent 跨 Stream 依赖

添加第三个处理 Stream（Stream C），它必须在 Stream A 和 Stream B 的 D2H 都完成后才能开始。使用 `cudaEventRecord` + `cudaStreamWaitEvent`。

#### 实验 3：调整 Stream 数量和 Chunk 大小

测试不同 nStreams（1, 2, 4, 8）和 chunkSize 下的加速比，找到最优配置。

---

### 验证 Checklist

- [ ] 能解释 Default Stream 的隐式同步行为及其危害
- [ ] 能画出 Multi-Stream 时间线：4 个 Stream 的 H2D→Compute→D2H 流水线重叠
- [ ] 代码正确实现了 H2D/Compute/D2H 的 overlap，速度比顺序版本有提升
- [ ] 能解释 `cudaMemcpyAsync` 为什么需要 Pinned Memory（DMA 要求）
- [ ] 能写出 `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)` 的完整用法
- [ ] 能理解 `cudaEventRecord` + `cudaStreamWaitEvent` 的跨 Stream 依赖管理
- [ ] 能对照昇腾 CANN 的 Stream API（aclrtCreateStream 等）写出对应代码
- [ ] 能使用 `nsys profile` 捕获并分析多 Stream timeline

---

### 今日总结

Day 3 我们掌握了 CUDA Stream 异步执行模型：

1. **Stream 是 GPU 操作的队列**：同 Stream 内 FIFO，跨 Stream 可并发
2. **Default Stream 的坑**：隐式同步所有 Explicit Stream，一处 `cudaMemcpy` 就打断全部并发
3. **Pinned Memory**：`cudaMemcpyAsync` 的必要条件，DMA 直接访问需要页锁定内存
4. **多 Stream 重叠**：利用 Copy Engine 和 Compute Engine 独立性，实现 H2D/Compute/D2H 流水线
5. **cudaEvent**：管理跨 Stream 依赖的精确同步工具

---

### 面试要点

1. **CUDA 的 Default Stream 有什么"坑"？在什么情况下会意外导致性能下降？**

   - 隐式同步规则：Default Stream 上的操作会等待所有其他 Stream 的先前操作完成，反之亦然
   - 陷阱场景：创建了多 Stream 做并发优化，但某处调用了 `cudaMemcpy`（默认走 Default Stream），导致所有 Stream 的并发被打断
   - 解决方案：全部使用 Explicit Stream + `cudaMemcpyAsync`，或 `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`

2. **`cudaMemcpyAsync` 相比 `cudaMemcpy` 需要什么额外条件？为什么必须使用 Pinned Memory？**

   - 必须使用 Pinned Memory（page-locked），因为异步传输使用 DMA 引擎直接访问内存
   - 如果内存被 OS 换出到磁盘，DMA 无法访问，驱动会先复制到临时 pinned buffer，导致异步退化为同步
   - 分配方式：用 `cudaMallocHost` 或 `cudaHostAlloc` 代替 `malloc`
