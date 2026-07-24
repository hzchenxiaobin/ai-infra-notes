# AI Infra 面试题与参考答案（北美面经篇）

> **来源**：知乎帖子 [《AI infra 面试经验贴》](https://zhuanlan.zhihu.com/p/1970722821522061231)（作者"抠抠歪"，理工科 PhD，北美 HPC 背景，面了美国十几家公司，方向为 kernel 开发与分布式通信）
> **说明**：原帖用 `(x)` `(xx)` `(xxx)` 标注题目在面试中出现的频率（低/中/高），本文保留该标注，并为每道题附上参考答案
> **原帖结构**：① 基础知识 ② 手撕 GPU kernel ③ LeetCode ④ 系统设计

---

## 一、基础知识（xxx）

### 1. GPU 相关（xxx）

**Q：GPU 的硬件架构和软件编程架构分别是什么？**

- 硬件层级：GPU → GPC（Graphics Processing Cluster）→ SM（Streaming Multiprocessor）→ CUDA Core / Tensor Core；SM 内有 shared memory / L1 cache、register file、warp scheduler；各 SM 共享 L2 cache 与 HBM/GDDR 显存
- 软件层级：grid → thread block → warp（32 线程，SIMT 执行）→ thread；block 调度到 SM 上执行，block 内线程可用 shared memory 通信、`__syncthreads()` 同步
- 编程模型要点：kernel 配置 `<<<grid, block>>>`，host/device 异构编程，异步 stream

**Q：TMA、CUTLASS、CuTe DSL 分别是什么？（NVIDIA 新特性）**

- **TMA**（Tensor Memory Accelerator，Hopper 引入）：硬件异步批量拷贝单元，一条指令完成 global ↔ shared 的多维 tensor tile 搬运，配合 mbarrier 做异步完成通知，解放线程去做计算，是 warp specialization 流水线的基础
- **CUTLASS**：NVIDIA 的 CUDA C++ 模板库，把 GEMM 等算子抽象为 threadblock/warp/thread 多级 tile + collective/mainloop/epilogue，可复用高性能组件组装 kernel
- **CuTe DSL**：CUTLASS 3.x 的核心抽象——Layout（逻辑坐标到物理偏移的映射函数）与 Tensor 的代数系统，用 `composition`/`tiled_divide` 等操作统一描述数据布局与 tiling；CuTe DSL 也指其 Python 绑定（cutlass 4.x 的 `cutlass.cute`），可用 Python 写高性能 kernel

**Q：如何判别 compute-bound 还是 memory-bound？什么是 Roofline Model？**

- **Roofline Model**：横轴为算术强度（arithmetic intensity，$I = \text{FLOPs} / \text{Bytes}$），纵轴为性能（FLOP/s）。理论峰值由两段构成：$\min(\text{峰值算力},\ I \times \text{峰值带宽})$
- 判别：计算 kernel 的算术强度 $I$，与机器的拐点（ridge point，$\text{峰值算力}/\text{峰值带宽}$，H100 约 295 TFLOPS ÷ 3.35 TB/s ≈ 88 FLOP/Byte）比较：$I$ 大于拐点为 compute-bound，否则 memory-bound
- 工程做法：用 Nsight Compute 看 SM busy 与 DRAM throughput 哪个更接近打满；elementwise/softmax 类是 memory-bound，大 GEMM 是 compute-bound

**Q：Nsight Compute 和 Nsight Systems 的区别？**

- **Nsight Systems**：系统级时间线分析——CPU/GPU 活动、kernel 启动开销、memcpy、stream 间依赖、通信（NCCL）与计算是否重叠
- **Nsight Compute**：单 kernel 级别剖析——occupancy、memory workload、bank conflict、stall reason（warp 为什么停等）、Roofline 图
- 典型流程：先用 Systems 定位哪个 kernel/间隙是瓶颈，再用 Compute 深挖该 kernel

**Q：什么是 Brent's Theorem？**

- 并行计算理论：若一个问题总工作量 $T_1$（单线程时间）、关键路径长 $T_\infty$（无限处理器的最短时间），则用 $p$ 个处理器的执行时间满足 $T_p \le T_1 / p + T_\infty$
- 意义：并行化不能突破 $T_\infty$；设计并行算法（如 scan、reduction）时要同时优化 work 和 span

**Q：grid/block size 怎么选？**

- block 大小取 warp 的整数倍，常用 128/256/512；太小（如 32）会浪费 SM 的 block slot，太大则 register/shared memory 受限导致 occupancy 低
- grid 大小 ≥ SM 数量 × 每 SM 可驻留 block 数，保证 load balance；对 elementwise 类常取 `ceil(N / (blockDim * items_per_thread))`
- 最终依据 occupancy 与实际 profiling 调优，而非理论值

**Q：什么是 coalesced memory access？**

- 同一 warp 的 32 个线程访问**连续、对齐**的 global memory 地址，硬件合并成尽量少的 memory transaction（如一个 128B cache line 一次取回）
- 反例：stride 访问（每线程隔 $k$ 个元素）会把带宽浪费 $k$ 倍；SOA 比 AOS 更容易合并

**Q：Shared memory bank conflict 的原理与优化？**

- shared memory 分 32 个 bank，相邻 4B 字落在相邻 bank；一个 warp 内**多个线程访问同一 bank 的不同地址**时访问被串行化（$n$-way conflict 慢 $n$ 倍）；访问同一地址是 broadcast，不冲突
- 优化：**padding**（如 `tile[32][33]` 多一列错开 bank）；**swizzling**（用 XOR 重排物理列号，如 `col ^ (row % 8)`，CUTLASS/CuTe 的标准做法）；改变访问模式

**Q：向量化读取（vectorized load）怎么做？**

- 用 `float4`/`int4`（128 bit）一次加载 16B，减少指令数、提升单线程带宽利用率
- 要求指针 16B 对齐且长度是 4 的倍数；这就是为什么对 $2^n$ 取模/求商的位运算（`x >> 2`、`x & 3`）在 kernel 里很常用
- 进阶：`__ldg`/`ld.global.nc` 只读缓存路径、cp.async / TMA 异步搬运

**Q：如何减少 register spill？**

- spill 是寄存器不够用，编译器把变量放到 local memory（实际在显存上），代价高
- 手段：减小每线程 tile 尺寸；用 `__launch_bounds__` 限制线程数引导分配；`#pragma unroll` 适度（过度展开反而 spill）；重计算代替缓存中间值；把大数组挪到 shared memory；`maxrregcount` 编译选项

**Q：什么是 warp specialization？**

- 让不同 warp 承担不同角色（producer/consumer）：producer warp 专职用 TMA/cp.async 搬运数据，consumer warp 专职计算，通过 mbarrier/pipeline 同步
- Hopper 上 FlashAttention-3、CUTLASS mainloop 的核心技术，把访存与计算彻底解耦重叠

### 2. ML 相关（xxx）

**Q：CNN 卷积层的时间/空间复杂度？**

- 输入 $H \times W \times C_{in}$，卷积核 $K \times K$，输出 $C_{out}$ 通道：
- 时间（FLOPs）：$O(H' W' K^2 C_{in} C_{out})$
- 空间：参数量 $K^2 C_{in} C_{out}$，激活显存 $O(H' W' C_{out})$

**Q：ResNet 解决了什么问题？**

- 深层网络的退化问题（深度增加精度反而下降，非过拟合）；残差连接 $y = F(x) + x$ 让梯度有高速通路，缓解梯度消失，使百层以上网络可训练

**Q：ViT 的基本结构？**

- 图像切成 $16 \times 16$ patch → 线性投影为 token → 加 position embedding → 标准 Transformer encoder（MHSA + FFN）→ CLS token 分类；attention 复杂度随 patch 数 $O(N^2)$ 增长

**Q：U-Net 中如何 upscale？**

- 两种方式：① 转置卷积（transposed convolution，可学习上采样）；② 插值上采样（bilinear/nearest）+ 普通卷积（更常用，避免棋盘伪影）；配合与 encoder 对应层的 skip connection 融合细节

**Q：RNN 与 Transformer 的对比？**

- RNN 顺序计算无法并行、长程依赖梯度消失；Transformer 用 self-attention 任意两 token 直接交互、可并行训练，但 attention $O(N^2)$ 复杂度

**Q：Self-Attention 的计算流程？（xxx）**

- $Q = XW_Q,\ K = XW_K,\ V = XW_V$；$\text{Attn} = \text{softmax}(QK^T / \sqrt{d_k}) V$
- 除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度饱和；复杂度 $O(N^2 d)$，显存 $O(N^2)$（naive 实现）

**Q：FlashAttention 三个版本的核心思想？（xxx）**

- **v1**：online softmax + tiling——把 $QK^TV$ 分块算，不物化 $N \times N$ 矩阵，显存从 $O(N^2)$ 降到 $O(N)$；核心是边扫边维护 running max $m$ 和归一化因子 $l$，用 rescale 修正之前的部分和
- **v2**：减少非 matmul FLOPs；调整循环顺序（Q 在外层）以按序列维并行、减少 shared memory 读写；更好的 warp 划分
- **v3**（Hopper）：用 TMA + warp specialization 做 pingpong 调度，GEMM 与 softmax 在 warpgroup 内/间重叠，支持 FP8 分块量化，算力利用率打到 H100 的 75%+

**Q：手撕 forward/backward、一层 MLP（numpy/pytorch 级）？**

```python
# y = x @ W1 + b1; h = relu(y); out = h @ W2 + b2
def forward(x, W1, b1, W2, b2):
    y = x @ W1 + b1
    h = np.maximum(y, 0)
    out = h @ W2 + b2
    cache = (x, y, h)
    return out, cache

def backward(dout, cache, W1, W2):
    x, y, h = cache
    dW2 = h.T @ dout; db2 = dout.sum(0)
    dh = dout @ W2.T
    dy = dh * (y > 0)          # relu 导数
    dW1 = x.T @ dy;  db1 = dy.sum(0)
    dx = dy @ W1.T
    return dx, dW1, db1, dW2, db2
```

> 练习推荐：[Deep-ML](https://www.deep-ml.com/problems)、Karpathy 的 micrograd 视频

### 3. 分布式计算相关（xx）

**Q：Strong Scaling 与 Weak Scaling？**

- **Strong scaling**：问题规模固定，增加处理器看加速比（Amdahl 定律限制：串行部分 $s$ 决定上限 $1/s$）
- **Weak scaling**：每处理器分到的规模固定，整体规模随处理器数增长（Gustafson 定律）；训练大模型属于 weak scaling 场景

**Q：常用的 collective communication 及原理？**

- Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AllToAll
- **Ring AllReduce** = ReduceScatter + AllGather 两阶段，每卡只收发 $2(N-1)/N$ 倍数据，带宽最优、与卡数无关；NCCL 默认实现（小消息用 tree/doubling 降低延迟）

**Q：PP / MP（TP）/ FSDP 的区别？**

- **PP（流水线并行）**：按层切分到不同卡，micro-batch 流水（GPipe / 1F1B）减少气泡
- **TP（张量并行）**：单层内切权重，层内需要 AllReduce，通信频繁，一般限于 NVLink 域内
- **FSDP**（ZeRO-3 的 PyTorch 实现）：参数/梯度/优化器状态全切分；前向反向前 AllGather 参数，反向后 ReduceScatter 梯度；显存最省，通信可与前一层计算重叠

**Q：Megatron 的 MLP 为什么"先行切再列切"？**

- 以 PyTorch 权重 `W[out, in]` 视角：第一层 GEMM（h→4h）**按行切**（切输出维，即 Megatron 术语的 ColumnParallel），每卡独立算出自己的 $4h/n$ 个输出并就地过 GeLU，**无需通信**；第二层 GEMM（4h→h）**按列切**（切输入维，RowParallel），每卡算部分和，最后**一次 AllReduce** 合并
- 这样整个 MLP 前向只引入一次 AllReduce，是通信最少的设计；Attention 同理（QKV 切头→行切，输出投影列切）

**Q：RDMA 是什么？MPI + GPU 怎么配合？**

- RDMA：网卡直接读写远端内存，绕过 CPU/内核（GPUDirect RDMA 让 NIC 直接访问 GPU 显存），是跨机高带宽低延迟通信的基础
- MPI + GPU：CUDA-aware MPI 可直接收发 device buffer；NCCL 则针对 GPU 集合通信做了拓扑感知优化
- PGAS/SHMEM（NVSHMEM）：分区全局地址空间，一个 rank 可直接 put/get 另一 rank 的显存，适合细粒度通信（如 MoE all-to-all），是当前热点

### 4. Infrastructure 相关（xx）

**Q：CI/CD pipeline 出 fault 怎么办？如何 rollback？**

- fault：定位失败 stage（测试/构建/部署）→ 保留日志与产物 → 修复后重跑；pipeline 设计要保证可重入、幂等
- rollback：制品版本化（不可变镜像 tag/digest），回滚 = 把上个稳定版本的制品重新部署；蓝绿/金丝雀发布让回滚只是切流量；数据库迁移要考虑前后兼容（expand-and-contract）

**Q：Docker / K8s 基本概念？如何上传和管理 K8s 用的镜像？**

- Docker：镜像（只读分层模板）/容器（运行实例）；Dockerfile 构建 → `docker tag` → `docker push` 到 registry（ECR/GCR/DockerHub）
- K8s：Pod（最小调度单位）/Deployment（副本与滚动更新）/Service（服务发现）；镜像由 Pod spec 中的 `image` 字段指定，kubelet 从 registry 拉取（`imagePullPolicy`、私有仓库用 imagePullSecret）

**Q：Python 环境/包管理？**

- venv/conda 隔离环境；pip + requirements.txt（或 uv/poetry 锁版本）；`pip freeze` 固定依赖；多版本 CUDA 库注意 wheel 与驱动匹配

**Q：const pointer 与 pointer to const？**

```cpp
const int* p;  // pointer to const：指向的内容不能通过 p 改，p 本身可以改
int* const p = &x;  // const pointer：p 不能改指向，内容可以改
const int* const p = &x;  // 都不能改
// 读法：从右往左读
```

**Q：虚函数与继承的底层？**

- 含虚函数的类有 vptr 指向 vtable，调用按对象动态类型查表派发；有运行时开销（间接跳转 + 阻止内联）；虚析构保证 `delete base_ptr` 正确；多重继承有 this 指针调整

**Q：动态链接 vs 静态链接？**

- 静态：编译时把库代码拷进可执行文件，部署简单、体积大、升级要重新链接
- 动态：运行时加载 `.so`/`.dll`，节省内存可热升级，但有依赖地狱（版本冲突）、符号解析开销（PLT/GOT）；CMake 中 `add_library(x STATIC/SHARED)` 控制

**Q：CMake 基础？**

- `cmake_minimum_required` / `project` / `add_executable` / `add_library` / `target_link_libraries` / `target_include_directories` / `find_package`；现代 CMake 用 target 为中心而非全局变量

**Q：C++ 模板元编程（TMP）要点？**

- 编译期计算：模板特化/偏特化、SFINAE（`std::enable_if`）、`constexpr`/consteval、C++20 concepts 约束；`if constexpr` 编译期分支；variadic template 与 parameter pack 展开
- CUTLASS 大量用 TMP 把 tile 形状、数据类型作为模板参数在编译期展开——读源码前必须掌握

**Q：Strategy 模式与 Abstract Factory 模式？（考过）**

- **Strategy**：把算法族封装为可互换的对象，运行时注入替换（如不同 attention backend 实现同一接口）
- **Abstract Factory**：提供创建一族相关对象的接口，不指定具体类（如按平台创建 CUDA/ROCm 全套算子）
- 参考：[refactoring.guru](https://refactoring.guru/design-patterns)

### 5. 硬件 / 操作系统 / 网络（x）

**Q：Write-through 与 write-back cache？**

- write-through：写同时更新 cache 和内存，一致性好、写慢；write-back：只写 cache，脏行淘汰时才写回内存，快但一致性复杂（多核需要 MESI 协议）

**Q：Memory fence 是什么？**

- 内存屏障，禁止编译器/CPU 把屏障前后的内存访问重排，保证可见性顺序；C++ 中 `std::atomic_thread_fence`，配合 memory order（relaxed/acquire/release/seq_cst）使用

**Q：Heap 与 stack 的区别？**

- stack：编译期大小确定、自动管理、分配快（移动栈指针）、容量小（MB 级）；heap：动态分配（malloc/new）、手动/RAII 管理、容量大、有碎片和分配开销

**Q：Lock-free programming 基本概念？**

- 用 CAS（`compare_exchange`）等原子操作而非互斥锁做同步，避免阻塞与优先级反转；难点：ABA 问题、内存序选择、难调试；无锁 ≠ 无等待（wait-free 才保证每步有进展）

**Q：内核态/用户态与 zero copy？**

- 普通 `read/write` 网络发文件要 4 次拷贝 + 4 次上下文切换（磁盘→内核→用户→socket 内核→网卡）
- **zero copy**：`sendfile`/`splice` 让数据只在内核空间流转（配合 DMA scatter-gather 可做到真正零拷贝）；Kafka、Nginx 大量使用

**Q：五层网络模型与三次握手？**

- 五层：物理 → 链路 → 网络（IP）→ 传输（TCP/UDP）→ 应用（HTTP）
- 三次握手：SYN → SYN+ACK → ACK；两次不够——无法防止历史失效的连接请求突然到达导致服务端资源浪费，且双方都要确认对方的收发能力

---

## 二、手撕 GPU Kernel（xxx）

> 原帖：CUDA 是加分项，Triton/CuTe 也可以写；推荐在 [LeetGPU](https://leetgpu.com/challenges) 刷题（本仓库 `leetgpu/` 目录即配套题解），kernel 优化合集可看 DefTruth 的合订本

### 1. Reduction（xxx）

**Q：写 warp 级和 block 级 reduction？**

```cuda
// warp 级：shuffle down，免 shared memory 免同步
__inline__ __device__ float warpReduce(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
// block 级：每 warp 归约 → shared → 第一个 warp 再归约
__inline__ __device__ float blockReduce(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduce(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane] : 0.f;
    if (wid == 0) v = warpReduce(v);
    return v;  // 仅 thread 0 有效
}
```

**Q：Mark Harris 的经典优化点有哪些？**

- 每线程先串行累加多个元素（减少 block 数与全局往返）
- sequential addressing（`s >>= 1` 步进）代替交错寻址，避免 shared memory bank conflict
- 最后 32 个元素用 warp unroll（`volatile`/shuffle）省去 `__syncthreads()`
- 循环完全展开（模板参数化 block size）

**Q：softmax / LayerNorm / BatchNorm 怎么从 reduction 延伸？**

- softmax：两遍（max reduce → 指数和 reduce）或 online 一遍；LayerNorm：每行做 mean/var 归约（可用 Welford 单遍算法避免两次遍历）；BatchNorm：按通道归约，训练用 batch 统计、推理用 running 统计
- follow-up：**FlashAttention 里的 softmax**——online softmax：扫到第 $j$ 块时维护 running max $m_j$ 与归一化和 $l_j$，更新公式 $m_{new} = \max(m, \tilde m)$，历史部分和乘 $e^{m - m_{new}}$ rescale 后累加，无需存整行分数

### 2. Elementwise（xxx）

**Q：写 sigmoid kernel，以及如何处理 warp divergence？**

sigmoid：$y = 1 / (1 + e^{-x})$，每线程一个（或 vec4 四个）元素，memory-bound。

原帖考点——分支：

```cuda
if (c < N) v += a;
else       v += b;
```

同 warp 内线程走不同分支会串行执行两个路径（divergence）。消除方法——用算术 mask（编译器也会自动做 if-conversion/predication）：

```cuda
int check = c < N;
v += check * a + (1 - check) * b;
// 等价：v += fminf/fmaxf、三元表达式在简单情况下也会被编译成 select
```

### 3. SGEMM（xx）

**Q：手写 tiled SGEMM 的骨架？**

```cuda
// 骨架：block tile BMxBN 放进 shared，每线程算 TMxTN 微块
__global__ void sgemm(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    __shared__ float As[BM][BK + PAD];   // PAD 防 bank conflict
    __shared__ float Bs[BK][BN + PAD];
    float acc[TM][TN] = {0};
    for (int k0 = 0; k0 < K; k0 += BK) {
        // 1) 协作搬运 A/B tile 到 shared（coalesced + 向量化）
        load_tile(As, A, ...); load_tile(Bs, B, ...);
        __syncthreads();
        // 2) 每线程从 shared 读片段做 TMxTN 外积累加（register blocking）
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) { /* outer product acc += */ }
        __syncthreads();
    }
    write_back(C, acc, ...);  // 合并写回
}
```

要点：shared memory 复用把算术强度拉高到 compute-bound；向量化加载；双缓冲（cp.async/TMA）重叠搬运与计算。

**Q：follow-up——什么是 Split-K？**

- 当 $M, N$ 小但 $K$ 很大时，$M \times N$ 维并行度不足；把 $K$ 维切成若干段，每段由一个 block 算部分和，最后用 atomicAdd 或第二个 reduction kernel（semaphore/workspace）合并
- 牺牲额外写开销换并行度，cuBLAS/CUTLASS 对小 M/N 大 K 的 shape 自动启用；同类技术还有 Stream-K（按工作量均分 tile，解决尾波量化问题）

### 4. Scan（xx）

**Q：手写 parallel prefix sum（scan）？**

- **Hillis-Steele**：work-efficient 差但 span 短（$O(n\log n)$ work，$\log n$ 步），block 内常用
- **Blelloch**：up-sweep（归约）+ down-sweep（分发），$O(n)$ work 的 exclusive scan，见 [GPU Gems 3 第 39 章](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- 大数组：block 内 scan → block 和写入数组 → 对 block 和再 scan → 回加（Chained Scan / decoupled look-back）

**Q：Stream Compaction 怎么做？（真实面试题）**

输入数组 + 条件谓词，把满足条件的元素紧凑保留到输出前部：

1. 对每个元素算 flag（满足为 1）
2. 对 flag 做 **exclusive scan**，得到每个元素在输出中的位置
3. scatter：满足条件的元素写到 `out[pos[i]]`

复杂度 $O(n)$，是稀疏数据处理（如稀疏 attention、粒子模拟中删除无效粒子）的标准原语。

**Q：Radix Sort 为什么用 scan？（口述题）**

- LSD radix sort：按每个 digit（如 4 bit）分桶；每轮对 bucket 标志做 scan 计算各元素的稳定目标位置；$\lceil 32/4 \rceil = 8$ 轮完成 32 位整数排序，$O(n)$ 且稳定

### 5. Transpose（x）

**Q：矩阵转置 kernel 的考点？**

- naive 版本读合并、写跨步（或反之），带宽打对折
- 正解：block 内 $32 \times 32$ tile 经 **shared memory 中转**——global 合并读 → shared → 合并写回 global；shared 数组声明为 `[32][33]` padding 消除 bank conflict
- 该技巧与 SGEMM 的 tile 搬运同源，所以面试常直接考 SGEMM

---

## 三、LeetCode（xx）

> 原帖：比传统岗少很多，但有公司因第一轮 LeetCode 的 follow-up 没写出挂人；难度约周赛 1200–1700，重数据结构

**Q：常考算法类型？**

- 二分、双指针、贪心
- **拓扑排序**（考过两次）：① 反向传播——计算图的逆拓扑序执行 backward；② 简化编译器——给指令依赖关系排执行顺序
  - 解法：Kahn 算法（入度为 0 入队，BFS 逐层剥离）或 DFS 后序逆序

**Q：位运算常考什么？（low-level / kernel 组尤其多）**

```cpp
// 判奇偶
x & 1
// 对 2^n（如 4，vec4 向量化访问的粒度）求商/取余
div = x >> 2;      // x / 4
mod = x & (4 - 1); // x % 4
// 其他高频：x & (x-1) 消最低位 1；x & (-x) 取最低位 1；异或交换/找只出现一次的数
```

> 推荐[灵神的位运算帖](https://leetcode.cn/discuss/post/CaOJ45/)

---

## 四、系统设计（x）

**Q：设计 load balancer（round robin）？**

```python
class RoundRobinLB:
    def __init__(self, backends):
        self.backends, self.idx = backends, 0
    def next(self):
        srv = self.backends[self.idx % len(self.backends)]
        self.idx += 1            # 并发下用原子自增 / 每线程局部计数
        return srv
```

follow-up：加权轮询（平滑 WRR，如 Nginx）、健康检查摘除故障节点、一致性哈希（会话保持/缓存亲和）。

**Q：设计简易推荐系统？**

两阶段：**召回**（协同过滤/双塔向量 ANN 检索，从亿级取千级候选）→ **排序**（特征 + CTR 预估模型如 DeepFM，精排几十条）；进阶提冷启动、去重与多样性重排、在线 serving 的延迟预算。

**Q：设计异步 double buffering 读写？**

- 两个缓冲区 ping-pong：计算消费 buffer A 时，后台异步填充 buffer B，下一轮交换角色，把访存/IO 延迟藏进计算
- GPU 上的对应物：cp.async / TMA 多级流水（CUTLASS pipeline），生产者（拷贝 warp）写 buffer、用 mbarrier 通知，消费者（计算 warp）等 barrier 读取
- 关键：同步原语（barrier/event）保证写完成才读、读完成才覆写

---

## 原帖的其他要点

- 作者背景：PhD + 北美 HPC 三年，ML 经验少，边面边学，最终收获 2 个 offer
- 考察频率排序：**手撕 GPU kernel > 基础八股 > LeetCode > 系统设计**
- 练习资源：[LeetGPU](https://leetgpu.com/challenges)、[Deep-ML](https://www.deep-ml.com/problems)、DefTruth 的 CUDA kernel 优化合订本、[refactoring.guru](https://refactoring.guru/design-patterns)、[GPU Gems 3 第 39 章](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

> ⚠️ 注意：CSDN 转载版（[blog.csdn.net](https://blog.csdn.net/2401_84204413/article/details/154695552)）在原帖尾部拼接了一段"大模型学习资料"广告，非原帖内容，本文已剔除。
