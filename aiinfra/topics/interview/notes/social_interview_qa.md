# AI Infra 社招面试实录整理与参考答案（面试官视角）

> **来源**：知乎帖子 [《记AI-infra/大模型推理社招面试一兄弟的全过程(v3.0)》](https://zhuanlan.zhihu.com/p/1920946738270810330)（作者"不归牛顿管的熊猫"，公众号"AI不止算法"，面试官视角记录）
> **获取说明**：知乎有反爬保护，正文无法直接抓取。本文内容通过搜索引擎快照逐段还原，**已确认的题目**标注 ✅，从面试官点评/上下文推断的考察范围标注 🔶（推断部分答案按该领域标准答法给出，供备考参考）

---

## 一、帖子背景与面试流程

- **场景**：社招，候选人来自 AI 芯片公司，工作内容是 CPU/GPU 算子库开发（CPU 用 oneDNN 类库，GPU 有自研对标 CUTLASS 的模板算子库，以 GEMM 为中心的算子如 FlashAttention 基于它构建）
- **流程**（面试官自述）：
  1. 自我介绍（约 5 分钟，候选人详细介绍简历上的负责内容、成果、技术栈）
  2. 项目深挖（从自我介绍中挑点追问）
  3. 大模型推理 / 量化 / CUDA 的基础与进阶问答
  4. 手撕代码：一道 **C++ 类实现题**（团队对 C++ 能力有明确要求，没按套路出 CUDA kernel 或 LeetCode）
- **面试官点评**：候选人整体不错——推理/量化/CUDA 基础和进阶问题都能答上来，代码题写得七七八八，对简历上的项目也能讲清楚

> 💡 **可借鉴的面试结构认知**：社招 AI Infra 岗 = 项目深挖 + 领域八股（推理/量化/CUDA）+ 一道代码题；C++ 功底是很多团队的硬指标。

## 二、已确认的面试题与参考答案

### Q1 ✅ 项目题：你们 CPU 和 GPU 算子库分别用什么？

- 候选人答案（帖子实录）：CPU 主要用 xxxdnn（oneDNN 类）；GPU 以前用 xxxdnn，后来有自研对标 CUTLASS 的软件产品，以 GEMM 为中心的算子（如 FlashAttention）一般用它
- **考点**：是否了解所在技术栈的生态位，而不是只会调用

### Q2 ✅ 对 CUTLASS 这类算子模板库有没有个人理解？（支持范围、技术实现、customize）

候选人答案（帖子实录）：和 CUTLASS 一样是一个 C++ 模板库，用模板泛化了算子的很多特性——GPU 平台、数据类型、layout 等，通过模板尽可能复用以 GEMM 为中心的算子的 common 部分。

**参考扩展答案**（面试官想听到的深度）：

- **技术实现**：CUTLASS 用 C++ 模板把 GEMM 分解为多层 tile（threadblock → warp → thread/instruction），每层由 Collective（mainloop）+ Epilogue 组合；3.x 以 CuTe 为核心，用 `Layout`（形状/步长的代数）统一描述张量坐标到内存偏移的映射，tiling 就是 layout 的 `composition`/`tiled_divide`
- **支持范围**：数据类型（FP64→FP16/BF16→FP8→INT8/INT4→FP4）、架构（sm_70 到 sm_120，含 TMA/warp specialization 的 Hopper/Blackwell 特性）、算子族（GEMM、Conv、Group GEMM、FMHA 示例）
- **customize**：组合不同的 `TileShape`、`ClusterShape`、mainloop policy（如 TMA 多级流水）、epilogue 融合（bias/激活/量化 scale）；用模板特化而非虚函数，所有分支在编译期展开，零运行时开销

### Q3 ✅（某 CUDA 技术）有没有写过？那你口头描述一下逻辑

帖子实录片段："候选人：有的。我：那你口头描述一下逻辑。"——具体主题在快照中缺失（🔶），从上下文看属于 CUDA/算子类手写经验的口头考察。

**备考建议**：这类"口述 kernel 逻辑"最常见于 **GEMM tiling** 和 **FlashAttention online softmax**，按下面要点组织口述：

- **GEMM**：分 block tile → 协作搬运 A/B 到 shared memory（合并访存 + 向量化）→ 每线程 register blocking 算微块 → `__syncthreads` 保证可见性 → 双缓冲/cp.async 重叠搬运与计算
- **FlashAttention**：外层循环 K/V 的 tile，内层对每块算 $S = QK^T$，用 running max $m$ 和归一化因子 $l$ 做 online softmax，历史累加值乘 $e^{m_{old}-m_{new}}$ rescale，全程不物化 $N \times N$ 矩阵

### Q4 ✅ 动态量化和静态量化有听说过吗？区别是？

**参考答案**：

- 两者指**激活值**的量化策略（权重都是离线量化好的，没有争议）：
- **静态量化**：推理前用校准数据集离线统计激活分布，固定 scale/zero-point；运行时直接整数 GEMM 后乘 scale，无额外开销；缺点是需要代表性校准数据，遇到分布外输入误差大
- **动态量化**：运行时按当前输入（per-tensor 或 per-token）实时计算 scale 再量化；无需校准数据、精度更好（自适应输入分布）；缺点是每个 GEMM 前要多算一次 reduce（求 absmax），有运行时开销，部分硬件/算子路径不支持
- LLM 推理中常见组合：**权重静态量化 + 激活动态 per-token 量化**（W8A8 动态方案，vLLM/SGLang 默认路径）；TensorRT 的 INT8 多为静态量化（需要 calibration）

### Q5 ✅ 手撕代码：实现一个简单的 C++ 类

帖子实录："最后再做个题吧，帮忙实现一个简单的 C++ 类，要求有这个功能，那个功能。"——具体要求在快照中缺失（🔶）。面试官明确说**没出 CUDA kernel 和 LeetCode，考的是 C++ 功底**。

**备考建议**：社招高频 C++ 类实现题就那几道，按优先级准备：

- **手写 `shared_ptr`**（最高频）：控制块（引用计数 + 弱计数）、拷贝/移动语义、线程安全边界（计数原子操作，但指向对象不保证）
- **手写线程池**：任务队列 + 条件变量 + `std::function` 类型擦除，支持 `enqueue` 返回 `future`
- **单例**：Meyers singleton（局部静态变量，C++11 起线程安全）
- 每道题都要能讲清：拷贝构造/赋值为什么禁用或自定义、析构时机、异常安全

```cpp
// 最简 shared_ptr 骨架（面试手写版）
template <typename T>
class SharedPtr {
    T* ptr_;
    std::atomic<long>* count_;
public:
    explicit SharedPtr(T* p = nullptr)
        : ptr_(p), count_(new std::atomic<long>(1)) {}
    ~SharedPtr() { release(); }
    SharedPtr(const SharedPtr& o) : ptr_(o.ptr_), count_(o.count_) {
        count_->fetch_add(1, std::memory_order_relaxed);
    }
    SharedPtr& operator=(const SharedPtr& o) {
        if (this != &o) { release(); ptr_ = o.ptr_; count_ = o.count_;
                          count_->fetch_add(1, std::memory_order_relaxed); }
        return *this;
    }
    SharedPtr(SharedPtr&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr; o.count_ = nullptr;
    }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    long use_count() const { return count_ ? count_->load() : 0; }
private:
    void release() {
        if (count_ && count_->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete ptr_; delete count_;
        }
    }
};
```

## 三、考察范围（🔶 从面试官点评推断）

帖子点评确认覆盖了这三大块的基础 + 进阶问题，快照未能还原具体题目，按该岗位标准高频题列出：

### 大模型推理

- KV Cache 为什么需要、大小怎么算（$2 \times L \times N_{layer} \times N_{kv\_head} \times d_{head} \times$ 精度字节）
- PagedAttention 解决什么问题（显存碎片 + 按需分配，类比 OS 虚拟内存分页）
- Continuous Batching 相对静态 batching 的收益（请求级进出，消除尾部等待）
- Prefill/Decode 两阶段的瓶颈差异（compute-bound vs memory-bound，为什么 PD 分离）
- 常见推理框架（vLLM/SGLang/TensorRT-LLM）的核心优化点

### 量化

- 常见 LLM 量化方法及区别：**GPTQ**（逐层二阶误差补偿的权重量化）、**AWQ**（保护 salient 权重通道的激活感知量化）、**SmoothQuant**（把激活 outlier 难度迁移到权重，W8A8）、**FP8**（E4M3/E5M2，Hopper/Blackwell 硬件加速）
- 为什么 INT8/INT4 后精度仍能保持（权重分布集中 + 逐通道/per-group scale + 少量 outlier 通道特殊处理）
- per-tensor / per-channel / per-group 量化的精度-开销权衡
- KV Cache 量化与权重量化的区别（KV 是动态生成的激活，误差会随序列累积）

### CUDA

- GPU 硬件/编程模型：SM、warp、occupancy、Roofline 判别 compute/memory-bound
- shared memory bank conflict 及消除（padding/swizzling）
- Tensor Core 与 mma/wgmma 指令，为什么要 16B 对齐
- CUDA Graph 解决什么（小 kernel 启动开销，decode 阶段必备）
- FlashAttention 各版本优化点（tiling/online softmax → 并行度与非 matmul FLOPs → Hopper warp specialization + FP8）

## 四、从这篇帖子提炼的面试建议

1. **简历上的项目必须能扛住连环追问**：面试官从自我介绍里挑点深挖（算子库 → CUTLASS 理解），答不出"你自己的理解"会被认为只是调库
2. **社招重 C++ 功底**：很多推理引擎团队 host 侧全是 C++，手撕 `shared_ptr`/线程池比手撕 kernel 更常见
3. **推理/量化/CUDA 三块八股要成体系**：基础题（KV Cache、量化分类）到进阶题（PagedAttention、SmoothQuant 数学原理、CUDA Graph）都会被覆盖
4. **面试官也在观察表达**：候选人"非常详细地介绍负责内容、成果、对团队的贡献"被正面记录——自我介绍就要结构化讲清 背景→行动→量化结果
