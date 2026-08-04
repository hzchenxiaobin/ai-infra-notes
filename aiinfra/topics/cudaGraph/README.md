# CUDA Graph 专题：原理、用法与面试考点

> **适用对象**：准备 AI Infra / 推理引擎方向岗位的求职者，以及做 LLM 推理性能优化的工程师
> **相关专题**：[CUDA 手撕题专题](../cuda/README.md)、[PyTorch 专题](../pytorch/README.md)、[vLLM 专题](../vllm/README.md)

---

## 一、为什么需要 CUDA Graph

### 1. Kernel Launch 开销问题

- 每次 kernel launch 在 CPU 侧有固定开销（约 **3~10 μs**，含参数准备、驱动调用），kernel 本身执行时间越短，launch 开销占比越高
- LLM decode 阶段 batch 小、每层 kernel 多（QKV proj、attention、RoPE、RMSNorm、MLP…… 一个 layer 几十次 launch），CPU 提交速度跟不上 GPU 执行速度，GPU 出现**气泡（idle gap）**，利用率上不去
- 这是典型的 **CPU-bound** 场景：瓶颈不在 GPU 算力，而在 CPU 提交任务的速度

### 2. CUDA Graph 的思路

把一连串 kernel launch（以及 memcpy、memset、event 等操作）**录制**成一张有向无环图（DAG），之后用**一次提交** `cudaGraphLaunch` 重放整图：

- CPU 侧提交开销从 "N 次 launch" 降为 "1 次 graph launch"，几乎为零
- GPU 可以提前拿到整图的依赖关系，相邻无依赖的 kernel 可以**并发调度**，减少 kernel 间的 gap
- 驱动可以对整图做优化（如预分配、减少同步检查）

一句话总结：**CUDA Graph 用空间（固化执行序列）换时间（消除 launch 开销），是 decode 阶段小 batch 推理的标配优化**。

## 二、核心概念与 API 流程

### 1. 两个阶段

| 阶段 | 说明 |
|------|------|
| **Capture（录制）** | 把后续要执行的操作录成 `cudaGraph_t`（不真正执行，或执行一次用于 warmup） |
| **Replay（重放）** | 实例化为 `cudaGraphExec_t` 后，通过 `cudaGraphLaunch` 反复执行 |

### 2. 两种建图方式

**方式一：Stream Capture（推荐，最常用）**

```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// 1. 开始录制（通常用一个非默认 stream）
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 2. 正常发起一系列 kernel / memcpy，这些调用不会真正执行，而是被录进图里
kernelA<<<grid, block, 0, stream>>>(...);
kernelB<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

// 3. 结束录制，得到图
cudaStreamEndCapture(stream, &graph);

// 4. 实例化（只做一次，开销较大）
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 5. 之后每次推理只重放
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);
```

**方式二：显式建图 API（Explicit Graph Creation）**

用 `cudaGraphAddKernelNode` / `cudaGraphAddMemcpyNode` / `cudaGraphAddDependencies` 手工搭建节点和边。灵活但繁琐，一般只在需要精细控制图结构（如手工构造并行分支）时使用。

### 3. Capture 模式

`cudaStreamBeginCapture` 的三种模式，决定录制期间其他线程的"非捕获"launch 是否报错：

- `cudaStreamCaptureModeGlobal`：最严格，全局任何非录制线程的 CUDA 调用都会使 capture 失败（默认）
- `cudaStreamCaptureModeThreadLocal`：只约束本线程
- `cudaStreamCaptureModeRelaxed`：最宽松

实践中 PyTorch 内部使用 thread local 模式，避免干扰其他线程。

## 三、使用限制（面试高频考点）

CUDA Graph 的本质是**把执行序列固化**，所以所有被固化的东西都不能变：

### 1. 输入地址必须固定

- 图中记录的是**指针值**，不是数据。重放时 kernel 读的还是录制时那些地址
- 因此输入必须先 copy 到一块**固定的静态 buffer**，再 launch graph；输出也从固定地址读
- PyTorch 中体现为：`static_input.copy_(new_input)` → `graph.replay()` → 读 `static_output`

### 2. Shape / grid / block 必须固定

- 图的拓扑和每个节点的 launch 参数都固化了，**batch size 变了就得重新 capture 一张图**
- 推理框架的做法：为若干档 batch size（如 1/2/4/8/16…256）各 capture 一张图，运行时按 padding 到最近的档位选图（vLLM 的 `cudagraph_capture_sizes` 就是这么做的）

### 3. 录制期间的禁止操作

- **不能有任何同步操作**：`cudaStreamSynchronize`、`cudaDeviceSynchronize`、同步版 `cudaMemcpy`（capture 期间根本没真正执行，同步没有意义）
- **不能用默认流（legacy default stream）** 做隐式同步
- **不能有 CPU-GPU 依赖的逻辑**：如 `.item()`、`.cpu()`、根据 GPU 数据做 if 分支 —— 动态控制流录不进去
- **不能申请/释放显存**（`cudaMalloc` 在 capture 中受限；PyTorch 用专门的 **graph memory pool** 解决：capture 期间分配走私有 pool，重放时复用同一块虚拟地址）
- 未 capture 进图的额外 stream 操作若与图有依赖，需通过 event（fork/join）显式录进图里

### 4. 哪些算子 / 场景不适合

- 含动态 shape、动态控制流的算子
- 含 `cudaMallocAsync` 之外显存操作的自定义算子
- CPU 回调、host function node 以外的 host 侧逻辑

## 四、PyTorch 中的用法

### 1. 手动 capture（`torch.cuda.CUDAGraph`）

```python
import torch

# 静态输入/输出 buffer（地址固定）
static_input = torch.zeros(batch, hidden, device="cuda")

# warmup：在 side stream 上跑几遍，触发 cuBLAS/cuDNN 的 autotune 和 workspace 分配
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):          # 内部处理 stream、graph pool、禁用同步
    static_output = model(static_input)

# 推理时：copy 输入 → replay → 读输出
static_input.copy_(real_input)
g.replay()
result = static_output
```

几个易错点：

- **必须先 warmup**：cuBLAS/cuDNN 首次运行会做 algo 选择和 workspace 分配，这些操作不能发生在 capture 里
- **capture 要用 side stream**：`torch.cuda.graph` 上下文管理器自动处理
- **graph pool 共享**：多张图可以共享一个 memory pool（`torch.cuda.graph(g, pool=other.pool())`），节省显存；vLLM 全档位 batch size 的图就是这么做的
- capture 区域外的 tensor 若在图内被写入，重放会改它的值（地址固化）

### 2. 自动 capture：`torch.compile(mode="reduce-overhead")`

- `mode="reduce-overhead"` 底层就是 CUDA Graph（Inductor 自动生成）
- 相关开关：`triton.cudagraphs=True`（默认开）
- 注意 dynamic shape 会自动跳过 cudagraph 或按 shape 分桶 capture

### 3. `make_graphed_callables`

- `torch.cuda.make_graphed_callables(module, sample_args)`：把 nn.Module 的 forward（可选 backward）整体图化，训练场景可用
- 限制：仅支持静态 shape、单 GPU、autograd 有额外约束，实际训练用较少，推理多用手动 capture 或 torch.compile

## 五、推理框架中的应用

### 1. vLLM

- decode 阶段默认启用 CUDA Graph（V1 引擎中 `CUDAGraphMode`），prefill 不走图（shape 太动态）
- **Full graph vs Piecewise graph**：
  - Full graph：整个模型一张图，要求所有算子都可图化
  - **Piecewise cudagraph**（V1 默认）：把模型按 attention 切分成若干段，attention 之外的纯计算段各自 capture 成图，attention 保持 eager —— 因为 attention 的 seq_len / batch 组合是动态的
- 按 `cudagraph_capture_sizes` 分档 capture，运行时 pad 到最近档位；多档图共享 memory pool
- 相关八股：为什么 prefill 不用 CUDA Graph？（shape 动态 + compute-bound，launch 开销占比小，不值得）

### 2. TensorRT-LLM / SGLang

- TensorRT-LLM：engine 构建时可启用 CUDA Graph，按 batch size 分档
- SGLang：`--enable-torch-compile` 或自带的 radix 前缀 + cuda graph runner，思路同 vLLM

### 3. 收益量级

- decode 小 batch 场景，CUDA Graph 通常带来 **10%~30%+** 的端到端提升；kernel 越小越多，收益越大
- 大 batch / prefill 场景收益趋近于 0（GPU 已 compute-bound）

## 六、常见面试问题（八股）

1. **CUDA Graph 原理是什么？为什么能加速？**
   录制 kernel 序列为 DAG 一次提交，消除 CPU launch 开销，减少 kernel 间 gap；驱动可见全局依赖，调度更优。本质是 CPU-bound 场景的优化。

2. **CUDA Graph 有什么限制 / 哪些操作不能 capture？**
   静态 shape、固定地址；capture 期间不能同步、不能 malloc、不能 host-device 数据依赖；动态控制流录不进去。

3. **输入数据变了怎么办？**
   copy 到静态 buffer 再 replay；shape 变了要重新 capture（推理框架按 batch size 分档 + padding）。

4. **为什么 capture 前要 warmup？**
   让 cuBLAS/cuDNN 完成 autotune 和 workspace 分配，避免这些"一次性操作"被录进图或导致 capture 失败。

5. **显存问题怎么解决？**
   PyTorch 的 graph memory pool：capture 期间分配走私有 pool，地址固化后反复复用；多图共享 pool。

6. **vLLM 里 CUDA Graph 怎么用？piecewise 是什么？为什么 prefill 不用？**
   见第五节。

7. **CUDA Graph vs CUDA Stream 并发，有什么区别？**
   Stream 并发是运行时调度，依赖程序员手工拆流 + event 同步，launch 开销仍在；Graph 是编译期固化依赖，一次提交，开销近零。Graph 内也可以有多条并行分支（capture 时用多 stream + event fork/join 录进图）。

8. **如何调试 CUDA Graph？**
   `cudaGraphDebugDotPrint` 导出 dot 图；nsys 看 graph launch 的 timeline；capture 失败时用 `cudaStreamGetCaptureInfo` 查状态；PyTorch 侧 `torch.cuda.CUDAGraph.enable_debug_mode()`。

9. **Graph 能否更新（不重新 capture）？**
   可以，`cudaGraphExecKernelNodeSetParams` 修改节点参数，或 `cudaGraphExecUpdate` 在拓扑小变时增量更新（v12+ 还有 conditional node / device graph launch 等高级特性）。

## 七、进阶主题（了解即可）

- **Conditional nodes**（CUDA 12.3+）：图内支持 if/while 控制流节点，可在 device 侧按条件走不同子图
- **Device graph launch**（CUDA 12.0+）：从 kernel 里 launch 图，device-side 调度
- **Graph node 类型**：kernel / memcpy / memset / child graph / event record & wait / host function / memset 等
- **Instantiate 开销**：首次 instantiate 较重（ms 级），推理框架通常在启动时预 capture 所有档位

---

## 参考资料

- [NVIDIA CUDA C Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [NVIDIA Blog: Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [PyTorch 文档：torch.cuda.CUDAGraph](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
- [PyTorch Blog: Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [vLLM 文档：CUDA Graph / CUDAGraphMode](https://docs.vllm.ai/)
- [torch.compile reduce-overhead 模式说明](https://pytorch.org/docs/stable/torch.compiler.html)
