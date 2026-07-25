---
name: leetgpu-solution
description: 用于在 aiinfra/topics/cuda/ 下编写 LeetGPU (https://leetgpu.com) CUDA 挑战题解。规定了目录组织、题解文档结构（6 段式）、手绘 sketch 风 SVG 插图规范、Kernel 代码要求、网站构建集成与面试考点衔接。触发于"写 leetgpu 题解"、"补全 CUDA 挑战题解"、"加一道 leetgpu 题解"等请求。
---

# 写 LeetGPU 题解 Skill

本工程(`ai-infra-notes`)的 LeetGPU 题解是对 [https://leetgpu.com](https://leetgpu.com) 在线 CUDA 挑战平台的题解归档。本 skill 描述如何产出符合仓库惯例的题解。

## 1. 题目来源与 slug 规则

题目由**使用者指定**（平台 URL、题目名或编号均可），本 skill 不做自动选题。题目元数据可参考本地仓库 `/mnt/workspace/code/github/leetgpu-challenges`（目录结构 `challenges/<difficulty>/<number>_<name>/`），每道题 `challenge.py` 中的 `name` 字段即平台展示名，也用于生成 URL slug。

**slug 推导规则**：`<slug>` = 平台 URL slug，由 `challenge.py` 的 `name` 经「小写化 → 空格转 `-` → 去括号/斜杠」得到（如 `"General Matrix Multiplication (GEMM)"` → `general-matrix-multiplication-gemm`，`"1D Convolution"` → `1d-convolution`）。题解文件名固定为 `<slug>.md`（位于 `aiinfra/topics/cuda/`），slug 即唯一标识。

### 1.1 同类练习题推荐映射

每篇题解末尾的「## 同类练习题」章节从下表取材，**为每道题推荐 4 道考查相同 CUDA 概念的练习题**，附直达链接与一句话关联说明。本表是推荐映射的**唯一权威来源**——新增/修改题解时，直接从下表对应行复制 4 条推荐到题解的「同类练习题」表格中，避免各题解推荐不一致。

> 📌 **使用规则**：
> 1. 题解的「同类练习题」章节内容**必须与下表完全一致**（题目编号、关联说明、主线总结），不得自行增删。
> 2. 表中尚无对应行的题目，参照同领域已有行的格式补一行，保持「4 条推荐 + 选材主线」结构。
> 3. 下表的「领域」分组（A–L）按概念组织，便于按概念查找。
> 4. 题目编号对应 `leetgpu-challenges` 仓库目录前缀 `<number>_`；链接格式固定为 `https://leetgpu.com/challenges/<slug>`。

#### A. 基础并行模式（Element-wise / Memory-bound）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `vector-addition` | #21 ReLU（同为逐元素 kernel，多了分支判断，练习 coalesced 读写）· #31 Matrix Copy（纯拷贝，专注带宽优化与 float4 向量化）· #68 Sigmoid（数学函数逐元素，练习 fused kernel 思想）· #63 Interleave（写索引映射练习，coalesced 写回） | memory-bound 逐元素 kernel，练习 grid-stride loop 与合并访存 |
| `relu` | #23 Leaky ReLU（带负斜率分支，对比无分支优化）· #52 SiLU（融合 sigmoid+mul，练习 fused kernel）· #68 Sigmoid（纯数学函数逐元素，练习 exp 实现）· #65 GeGLU（GELU 激活，更复杂的逐元素融合） | 逐元素激活函数 family，练习分支/无分支 kernel 与合并访存 |
| `matrix-addition` | #31 Matrix Copy（纯矩阵拷贝，专注 coalesced 带宽优化）· #1 Vector Addition（1D 向量加法，grid-stride 基础）· #8 Matrix Addition（同题，可对比不同 tile 写法）· #62 Value Clipping（逐元素 clamp，练习 2D 索引） | 2D grid 映射 + 合并访存，练习矩阵级 elementwise kernel |
| `matrix-copy` | #1 Vector Addition（grid-stride + coalesced 基础）· #8 Matrix Addition（2D coalesced）· #3 Matrix Transpose（非连续访存对比）· #63 Interleave（写索引映射练习） | 纯带宽优化 + coalesced 拷贝，练习 memory-bound kernel 的极限优化 |
| `scalar-multiply` | #1 Vector Addition（grid-stride + coalesced 基础）· #21 ReLU（逐元素 + 分支）· #8 Matrix Addition（2D 逐元素）· #62 Value Clipping（逐元素 clamp） | 标量 × 向量逐元素，练习最简 elementwise kernel |
| `reverse-array` | #63 Interleave（写索引映射 + coalesced）· #1 Vector Addition（1D grid-stride 基础）· #31 Matrix Copy（coalesced 带宽优化）· #62 Value Clipping（逐元素 + 索引） | 1D 并行 in-place swap + coalesced，练习数据重排类 kernel |
| `vector-reversal` | #19 Reverse Array（同类型基础题）· #63 Interleave（索引重排练习）· #62 Value Clipping（逐元素索引）· #31 Matrix Copy（coalesced 带宽优化） | 1D 向量反转 + coalesced，练习 in-place swap 与索引映射 |
| `element-reversal` | #19 Reverse Array（同类型基础题）· #63 Interleave（索引重排练习）· #62 Value Clipping（逐元素索引）· #31 Matrix Copy（coalesced 带宽优化） | 逐元素反转 + 索引映射，练习 elementwise 重排 |
| `leaky-relu` | #21 ReLU（最简激活函数对比，无负斜率）· #52 SiLU（融合激活函数，练习 __expf）· #68 Sigmoid（数学函数逐元素，练习 exp 实现）· #65 GeGLU（GELU 门控变体，更复杂激活） | 逐元素激活函数 family，练习分支/无分支 kernel 与合并访存 |
| `sigmoid` | #21 ReLU（最简激活函数对比）· #52 SiLU（融合 sigmoid+mul，练习 fused kernel）· #23 Leaky ReLU（分支激活对比）· #54 SwiGLU（SwiGLU 使用 sigmoid 组件） | 逐元素数学函数，练习 __expf 快速数学与合并访存 |
| `color-inversion` | #1 Vector Addition（grid-stride + coalesced 基础）· #21 ReLU（逐元素 kernel，分支开销）· #66 RGB to Grayscale（多通道加权求和，类似逐元素）· #8 Matrix Addition（2D grid 逐元素） | 逐元素图像变换，练习多通道 coalesced 访存与索引映射 |
| `interleave` | #1 Vector Addition（grid-stride + coalesced 基础）· #31 Matrix Copy（纯拷贝带宽优化）· #19 Reverse Array（1D 并行 in-place swap）· #62 Value Clipping（逐元素 clamp） | 写索引映射练习，coalesced 写回与数据重排 |

#### B. 卷积与池化（Convolution & Pooling）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `1d-convolution` | #10 2D Convolution（halo 扩展到二维）· #11 3D Convolution（体数据 halo）· #90 Causal Depthwise Conv1d（因果卷积变体）· #28 Gaussian Blur（可分离卷积） | 1D shared memory halo，练习卷积边界填充与 tile 加载 |
| `2d-convolution` | #9 1D Convolution（halo 基础入门）· #11 3D Convolution（体数据 halo 扩展）· #28 Gaussian Blur（可分离卷积，行列分离优化）· #42 2D Max Pooling（滑窗 reduction，类似 tiling 模式） | shared memory halo + 常数内存，练习卷积类 kernel 的边界处理与 tiling |
| `causal-depthwise-conv1d` | #9 1D Convolution（1D 卷积基础，halo 填充入门）· #10 2D Convolution（2D shared memory halo + 常数内存）· #11 3D Convolution（3D 体数据 halo 扩展）· #28 Gaussian Blur（可分离卷积，行列分离优化） | 因果卷积 + depthwise 分组，练习卷积边界处理与通道独立并行 |
| `2d-max-pooling` | #10 2D Convolution（2D shared memory halo + tiling）· #9 1D Convolution（1D 卷积，halo 基础）· #28 Gaussian Blur（可分离卷积，滑窗模式）· #90 Causal Depthwise Conv1d（因果卷积变体） | 滑窗 reduction，练习 2D 索引映射与 padding 边界处理 |

#### C. 归约与扫描（Reduction & Scan）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `reduction` | #17 Dot Product（元素乘 + 全局归约，归约的直接应用）· #43 Count Array Element（计数归约 + atomic，对比归约与 atomic）· #27 MSE（平方差归约，归约在损失函数中的应用）· #51 Max Subarray Sum（scan + 归约的综合练习） | 树形归约 + warp shuffle，练习并行归约这一核心模板 |
| `prefix-sum` | #70 Segmented Prefix Sum（分段 scan，段边界处理进阶）· #72 Stream Compaction（predicate + scan 得到输出位置）· #47 Subarray Sum（prefix sum 直接应用求子和）· #82 Linear Recurrence（线性递推，scan 的数学扩展） | warp scan + 三阶段分块 scan，练习并行前缀扫描这一核心模板 |
| `segmented-prefix-sum` | #16 Prefix Sum（分段 scan 的基础）· #72 Stream Compaction（scan 的另一应用）· #82 Linear Recurrence（scan 的数学扩展）· #94 SSM Selective Scan（分段 scan 的前沿应用） | 分段 scan + 段边界处理，练习 prefix sum 的高阶变体 |
| `stream-compaction` | #16 Prefix Sum（stream compaction 的基础）· #70 Segmented Prefix Sum（分段 scan 进阶）· #43 Count Array Element（predicate 计数）· #87 Speculative Decoding Verification（compaction 的推理应用） | predicate + scan 得到输出位置，练习 scan 的筛选应用 |
| `dot-product` | #4 Reduction（树形归约，dot product 的基础组件）· #58 FP16 Dot Product（半精度归约）· #27 MSE（平方差归约的变体）· #17 Dot Product（同题，可对比不同归约写法） | 元素乘 + block 归约，练习融合 kernel 与归约 |
| `max-subarray-sum` | #16 Prefix Sum（本题的核心基础）· #47 Subarray Sum（prefix sum 直接应用）· #48 2D Subarray Sum（扩展到二维）· #72 Stream Compaction（scan 的另一应用） | prefix sum + Kadane scan + 归约，练习 scan 的综合应用 |
| `histogramming` | #43 Count Array Element（计数归约，atomic vs reduction 对比）· #44 Count 2D Array Element（2D 计数，扩展到多维 atomic）· #29 Top K Selection（bitonic 排序 + 堆归约，相关并行模式）· #36 Radix Sort（Radix Sort，histogram + scan 综合） | shared memory 直方图 + atomic 冲突，练习计数类并行模式 |
| `count-array-element` | #4 Reduction（树形归约，count 的归约基础组件）· #44 Count 2D Array Element（2D 计数，扩展到多维 atomic）· #13 Histogramming（shared memory 直方图，atomic + reduction 综合应用）· #27 Mean Squared Error（平方差归约，归约在损失函数中的应用） | predicate 归约 + atomic 计数，练习 count 类 kernel 的归约与 atomic 权衡 |
| `argmax` | #4 Reduction（树形归约，argmax 的基础组件）· #29 Top K Selection（排序归约进阶）· #5 Softmax（先求 max 再归一化）· #17 Dot Product（block 归约练习） | 归约变体（求最大值索引），练习比较归约与 warp shuffle |
| `top-k-selection` | #60 Top-p Sampling（排序 + 累积概率 + 采样）· #15 Sorting（通用并行排序）· #36 Radix Sort（按位 histogram + scan 排序）· #71 Parallel Merge（归并排序网络） | bitonic 排序 + 堆归约，练习并行排序与选择 |
| `subarray-sum` | #16 Prefix Sum（prefix sum 直接应用求子和）· #4 Reduction（树形归约基础组件）· #48 2D Subarray Sum（扩展到二维前缀和）· #51 Max Subarray Sum（scan + 归约综合练习） | prefix sum 直接应用，练习范围归约与 block reduce |
| `mean-squared-error` | #4 Reduction（树形归约，MSE 的基础组件）· #17 Dot Product（block 归约，类似模式）· #25 Categorical Cross Entropy Loss（归约 + log，损失函数变体）· #58 FP16 Dot Product（半精度归约，低精度变体） | 归约在损失函数中的应用，练习 fused kernel + block reduce |
| `fp16-dot-product` | #4 Reduction（树形归约基础组件）· #17 Dot Product（FP32 版 dot product 对比）· #57 FP16 Batched Matrix Multiplication（FP16 + Tensor Core，半精度 GEMM）· #27 Mean Squared Error（归约在损失函数中的应用） | 半精度归约，练习 __half 类型转换与 FP32 累加精度保证 |

#### D. 矩阵乘法与 GEMM（GEMM & Matmul）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `matrix-multiplication` | #22 GEMM（完整 GEMM，register blocking + 双缓冲进阶）· #30 Batched Matrix Multiplication（batched GEMM，多组矩阵并行）· #37 Matrix Power（重复 matmul，练习 tiling 复用）· #32 INT8 Quantized MatMul（INT8 量化 GEMM，低精度计算） | tiled matmul + register tiling，练习 GEMM 这一 compute-bound 核心模板 |
| `gemm` | #2 Matrix Multiplication（naive tiled matmul，对比基础写法）· #30 Batched Matrix Multiplication（batched GEMM，多矩阵并行调度）· #32 INT8 Quantized MatMul（INT8 量化 GEMM，低精度 + scale）· #57 FP16 Batched MatMul（FP16 + Tensor Core，半精度 GEMM） | GEMM tiling / register blocking / 双缓冲，练习 compute-bound kernel 优化全链路 |
| `batched-matrix-multiplication` | #22 GEMM（完整 GEMM，register blocking 基础）· #57 FP16 Batched MatMul（半精度 + Tensor Core）· #32 INT8 Quantized MatMul（低精度 batch）· #37 Matrix Power（重复 matmul 调度） | batched GEMM + 多组矩阵并行调度，练习 batch 维度的 kernel 设计 |
| `matrix-transpose` | #31 Matrix Copy（纯拷贝带宽优化，对比转置的访存模式）· #10 2D Convolution（2D shared memory halo + tiling）· #2 Matrix Multiplication（tiled matmul，同样用 shared mem 分块）· #63 Interleave（写索引重排，coalesced 练习） | shared memory tiling + bank conflict padding，练习矩阵数据重排类 kernel |
| `int8-quantized-matmul` | #22 GEMM（GEMM tiling 基础）· #30 Batched Matrix Multiplication（batched GEMM）· #81 INT4 Weight-Only Quantized MatMul（4-bit 量化进阶）· #64 Weight Dequantization（反量化基础操作） | INT8 量化 GEMM，练习低精度计算与 requantize 流程 |
| `sparse-matrix-vector-multiplication` | #17 Dot Product（warp shuffle 归约，SpMV 行内归约的基础组件）· #75 Sparse Matrix-Dense Matrix Multiplication（稀疏 GEMM，SpMV 的矩阵版进阶）· #22 General Matrix Multiplication (GEMM)（稠密 GEMM tiling，对比稀疏 vs 稠密访存模式）· #4 Reduction（树形归约，SpMV 行内归约的基础组件） | CSR 稀疏格式 + warp shuffle 行内归约，练习不规则访存与稀疏矩阵乘模板 |

#### E. 注意力机制（Attention）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `softmax-attention` | #12 Multi-Head Attention（FlashAttention 思想）· #53 Causal Self-Attention（因果掩码，下三角掩码）· #17 Dot Product（attention 的基础组件）· #5 Softmax（attention 的基础组件） | fused softmax+matmul + 数值稳定，练习 attention score 计算全流程 |
| `attention` | #6 Softmax Attention（本题的基础版本）· #12 Multi-Head Attention（head 并行进阶）· #53 Causal Self-Attention（因果掩码）· #59 Sliding Window Self-Attention（局部窗口） | attention score + softmax + weighted sum，练习 fused attention 全流程 |
| `causal-self-attention` | #59 Sliding Window（另一种局部 attention 窗口）· #80 GQA（KV head 复用的 attention 变体）· #12 Multi-Head Attention（head 并行）· #6 Softmax Attention（无 mask 基础版） | 因果掩码 + fused attention，练习 mask 对 attention 的影响 |
| `multi-head-attention` | #6 Softmax Attention（单 head 基础版）· #80 GQA（KV head 共享变体）· #53 Causal Self-Attention（因果掩码）· #74 GPT-2 Block（attention 的综合应用） | FlashAttention 思想 + head 并行，练习融合 attention 的高阶优化 |
| `sliding-window-self-attention` | #53 Causal Self-Attention（因果掩码变体）· #80 GQA（KV head 共享变体）· #6 Softmax Attention（无窗口基础版）· #92 Decaying Causal Attention（衰减因子变体） | 局部注意力窗口 + fused attention，练习窗口 mask 对 attention 的影响 |
| `grouped-query-attention` | #12 Multi-Head Attention（MHA 基础版）· #53 Causal Self-Attention（mask 变体）· #96 INT8 KV-Cache Attention（量化 + KV cache）· #59 Sliding Window（另一种 attention 变体） | KV head 共享 + attention，练习 GQA 的分组调度 |
| `int8-kv-cache-attention` | #80 GQA（KV head 复用的 attention 基础）· #64 Weight Dequantization（反量化基础）· #53 Causal Self-Attention（attention 基础）· #32 INT8 Quantized MatMul（INT8 计算基础） | 量化 KV cache + attention，练习低精度推理与 attention 的结合 |
| `attn-w-linear-bias` | #6 Softmax Attention（fused softmax+matmul 基础版）· #53 Causal Self-Attention（因果掩码变体）· #12 Multi-Head Attention（head 并行进阶）· #59 Sliding Window Self-Attention（滑窗注意力变体） | 线性偏置注意力，练习 attention + positional bias 的融合 |
| `decaying-causal-attention` | #53 Causal Self-Attention（因果掩码基础版）· #59 Sliding Window Self-Attention（滑窗注意力变体）· #6 Softmax Attention（无掩码基础版）· #80 Grouped Query Attention (GQA)（KV head 共享变体） | 衰减因子 + 因果掩码，练习 attention mask 变体与增量衰减计算 |

#### F. 归一化与嵌入（Normalization & Embedding）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `softmax` | #50 RMS Normalization（RMS Norm，归约 + 归一化变体）· #6 Softmax Attention（fused softmax+matmul，数值稳定进阶）· #4 Reduction（树形归约，softmax 的基础组件）· #40 Batch Normalization（Batch Norm，mean/var 归约归一化） | 三遍 kernel + 数值稳定，练习归约与归一化的融合 |
| `rms-normalization` | #40 Batch Normalization（mean/var 归约归一化）· #105 Group Normalization（分组归约）· #5 Softmax（max+sum 归约 + 归一化）· #50 RMS Normalization（同题对比不同实现） | 归约 + 归一化（root mean square），练习 norm 类 kernel |
| `batch-normalization` | #50 RMS Normalization（归约 + 归一化变体）· #105 Group Normalization（分组归约）· #4 Reduction（mean/var 归约的基础组件）· #5 Softmax（max + sum 归约归一化） | mean/var 归约 + 归一化，练习统计归约类 norm kernel |
| `group-normalization` | #40 Batch Normalization（mean/var 归约归一化，跨 batch 维度）· #50 RMS Normalization（RMS Norm，归约 + 归一化变体）· #5 Softmax（max+sum 归约 + 归一化）· #4 Reduction（树形归约，norm 的基础组件） | 分组归约归一化，练习两遍 scan + shared memory reduction |
| `token-embedding-layer` | #61 RoPE（位置嵌入的另一种实现）· #41 Simple Inference（embedding 的推理应用）· #64 Weight Dequantization（查表式反量化）· #106 Token Embedding Layer（同题，可对比不同实现） | gather / lookup table，练习嵌入查表类 kernel |
| `rope-embedding` | #106 Token Embedding（嵌入查表基础）· #54 SwiGLU（融合 elementwise 进阶）· #52 SiLU（fused elementwise）· #50 RMS Normalization（归约 + elementwise） | 复数旋转 + elementwise，练习位置编码的并行实现 |

#### G. Transformer 组件与推理优化（Transformer Blocks & Inference）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `gpt-2-transformer-block` | #12 Multi-Head Attention（block 的核心组件）· #50 RMS Norm（归一化组件）· #54 SwiGLU（激活/MLP 组件）· #85 LoRA Linear（低秩线性层变体） | LN + Attention + MLP 综合模块，练习多 kernel 流水线与模块融合 |
| `swiglu` | #52 SiLU（SwiGLU 的激活组件）· #21 ReLU（最简激活对比）· #65 GeGLU（GELU 门控变体）· #84 SwiGLU MLP Block（SwiGLU 的完整 MLP 应用） | 融合激活 + 门控乘法，练习 fused MLP 组件 kernel |
| `swiglu-mlp-block` | #54 SwiGLU（SwiGLU 激活组件，本 block 的核心 elementwise）· #22 GEMM（GEMM tiling，3 个 matmul 的基础组件）· #74 GPT-2 Transformer Block（更大的 transformer block 综合）· #52 SiLU（SiLU 激活，SwiGLU 的子组件） | 融合 MLP block，SwiGLU 的完整应用 |
| `silu` | #21 ReLU（最简激活函数对比）· #68 Sigmoid（silu 的组件）· #54 SwiGLU（融合激活 + 门控进阶）· #23 Leaky ReLU（分支激活对比） | 融合 sigmoid + mul 逐元素，练习 fused activation kernel |
| `lora-linear` | #41 Simple Inference（基础推理管线）· #64 Weight Dequantization（低精度推理基础）· #84 SwiGLU MLP Block（融合 MLP 模块）· #2 Matrix Multiplication（低秩 matmul 基础） | 低秩适配 + 融合低秩矩阵，练习推理优化中的低秩计算 |
| `simple-inference` | #85 LoRA Linear（低秩适配的推理变体）· #106 Token Embedding（推理管线组件）· #74 GPT-2 Block（完整推理模块）· #2 Matrix Multiplication（Linear 的 CUDA 实现） | PyTorch Linear 前向封装，练习推理管线的最简形态 |
| `speculative-decoding-verification` | #29 Top-K Selection（排序归约基础）· #60 Top-p Sampling（排序 + 采样）· #72 Stream Compaction（scan + predicate）· #16 Prefix Sum（验证的 scan 基础） | draft token 验证 + scan，练习推理优化中的并行验证 |
| `adder-transformer` | #74 GPT-2 Transformer Block（完整 transformer block 综合应用）· #12 Multi-Head Attention（标准 MHA，对比加法注意力）· #6 Softmax Attention（softmax attention 基础版）· #85 LoRA Linear（低秩线性层变体） | 加法注意力替代 softmax，练习多 kernel 推理流水线 |
| `moe-topk-gating` | #29 Top K Selection（bitonic 排序 + 堆归约基础）· #60 Top-p Sampling（排序 + 累积概率 + 采样）· #5 Softmax（softmax，top-k 后的归一化）· #84 SwiGLU MLP Block（MoE 中的 MLP 组件） | top-k 选择 + softmax，练习排序归约与 MoE 路由 |

#### H. 量化与低精度（Quantization）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `weight-dequantization` | #32 INT8 Quantized MatMul（量化计算的应用）· #81 INT4 Weight-Only（4-bit 打包反量化）· #96 INT8 KV-Cache（量化 attention 应用）· #85 LoRA Linear（低秩 + 量化推理） | 量化反量化到 fp16/fp32，练习低精度推理的基础操作 |

#### J. 高级算法与数学（Advanced Algorithms & Math）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `nearest-neighbor` | #22 General Matrix Multiplication (GEMM)（GEMM tiling，nearest neighbor 的分块复用同构）· #20 K-Means Clustering（K-Means 距离矩阵，pairwise distance 的迭代应用）· #4 Reduction（树形归约，argmin 更新的归约基础组件）· #33 Ordinary Least Squares（线性代数 + 归约，距离/矩阵计算的另一变体） | pairwise distance + shared memory tiling 数据复用，练习 compute-bound kernel 的算术强度提升 |
| `2d-jacobi-stencil` | #10 2D Convolution（2D shared memory halo + tiling，stencil 的加权变体）· #9 1D Convolution（1D shared memory halo，stencil 的一维基础）· #42 2D Max Pooling（滑窗 reduction，类似的 tiling + 边界处理模式）· #11 3D Convolution（3D shared memory halo，stencil 扩展到体数据） | stencil 计算 + shared memory halo 边界复用，练习网格类 kernel 的邻居冗余读消除 |

> 💡 **选材原则**：每道题的 4 条推荐遵循「**1 道同类型基础题 + 1 道进阶变体 + 1 道综合应用 + 1 道跨领域延伸**」的结构，确保从基础到进阶的渐进练习路径。推荐题优先选已有题解的题（可回看自己的题解），其次选相同概念的相关题。

## 2. 目录组织

所有题解**扁平存放**在 `aiinfra/topics/cuda/` 下，按题目 slug 命名，一篇题解一个文件：

```
aiinfra/topics/
├── images/                              # topics 级共享 SVG/PNG 插图（含 cuda_* 前缀图）
│   ├── cuda_softmax_three_pass.svg
│   └── ...
└── cuda/
    ├── README.md                        # CUDA 手撕题专题（面经总结）
    ├── SKILL.md                         # 本文件
    ├── vector-addition.md               # #1 Vector Addition
    ├── relu.md                          # #21 ReLU
    ├── prefix-sum.md                    # #16 Prefix Sum
    └── ...                              # 每道题一个 <slug>.md
```

**规则**：

1. **题解根目录**：`aiinfra/topics/cuda/`，不要写到其他位置。
2. **扁平组织**：所有题解直接放在 `aiinfra/topics/cuda/` 下，不按 `weekN/dayM/` 分层。
3. **题解文件名**：`<slug>.md`，其中 `<slug>` 是 LeetGPU 平台的题目 URL slug（如 `vector-addition`、`prefix-sum`），slug 即唯一标识。
4. **图片目录**：`aiinfra/topics/images/`，topics 下各专题共享。题解插图建议加 `cuda_<slug>_` 前缀避免冲突（如 `cuda_prefix_sum_overview.svg`），在题解中用 `../images/xxx.svg` 相对路径引用。

> 📌 **构建脚本**：`build/topics.py` 已支持本目录的扁平题解——扫描 `<slug>.md`（排除 `README.md` / `SKILL.md` / `dayN.md`），生成 `public/cuda/<slug>.html` 并在专题概览页与侧边栏自动加入口。

## 3. 题解文档结构

每篇题解 `.md` 遵循固定 **6 段结构**（参考下文模板与本目录已有题解）：

```markdown
# LeetGPU <题目名> 题解

## 1. 题目概述
- **标题 / 题号**：<题目名>
- **链接**：https://leetgpu.com/challenges/<slug>
- **难度**：简单 / 中等 / 困难
- **标签**：CUDA、<概念标签1>、<概念标签2>

（题意描述 + 输入输出 + 约束条件）

## 2. CPU 基线 / 朴素 GPU 方法
（CPU 串行实现 + 朴素 GPU 实现，说明瓶颈）

## 3. GPU 设计
### 3.1 并行化策略
### 3.2 存储层次使用（global / shared / register）
### 3.3 关键技巧（warp shuffle / tiling / coalesced 等）

## 4. Kernel 实现
（完整可编译 CUDA 代码：#include、__global__ kernel、main()、
  cudaMalloc/Memcpy、验证逻辑、cudaFree）

## 5. 性能分析与优化
（ncu profiling 命令 + 关键指标 + 优化方向）

## 6. 复杂度分析
（时间复杂度、空间复杂度、算术强度、瓶颈类型 memory/compute-bound）
```

**写作规范**：

- **中文为主**，概念加粗，善用 `> 💡` / `> ⚠️` blockquote。
- 代码块标注语言：` ```cuda` / ` ```cpp` / ` ```bash` / ` ```text`。
- **Kernel 代码必须完整可编译**：包含 `#include`、`__global__` kernel、`main()`、host 端 `cudaMalloc`/`cudaMemcpy`、验证逻辑、`cudaFree`。
- 代码块首行带注释：`// <filename>.cu —— <说明>` + `// 编译命令: nvcc ...`。
- 图片引用用相对路径：`![<中文alt>](../images/<filename>.svg)`（题解位于 `aiinfra/topics/cuda/` 下，`../images/` 解析到共享的 `aiinfra/topics/images/`）。
- 每篇题解引用 **2-4 张 SVG/PNG 插图**，并配 `### 4.2 代码详解` 子节（详见 §5）。

### 数学公式

- 行内公式用 `$...$`，块级公式用 `$$...$$`
- **禁止**用反引号 `` `...` `` 包裹数学公式，否则会被渲染为等宽代码，KaTeX 不会识别
- 公式内函数/运算符使用 LaTeX 命令：`\exp`、`\log`、`\sum`、`\max`、`\frac`、`\sqrt`，避免直接写 `exp`、`log`、`Σ`、`√`


## 4. 图片风格：手绘 sketch 风（Excalidraw-like）

**所有插图统一为手绘 sketch 风**，与 LeetCode 题解保持一致。具体要求：

- **禁止 ASCII 图片**：所有示意图、流程图、架构图一律用 SVG，不要在 Markdown 中嵌入 ASCII 字符画（如用 `+---+`、`|   |` 拼成的表格或流程图）

### 4.1 视觉特征

| 维度 | 要求 |
|------|------|
| **线条** | 手绘不均匀、略带抖动，避免完美直线或圆润矢量边 |
| **笔触** | 粗糙、类似马克笔/铅笔描边，线宽可略有变化 |
| **配色** | 极简，一般不超过 3-4 种柔和颜色（蓝 `#e8f0fe`/`#446688`、绿 `#e6f4ea`/`#4a7a3a`、橙 `#fff8e1`/`#d6a040`、红 `#fce4ec`/`#b85450`），背景白色或米白 `#fafafa` |
| **形状** | 简单几何块——矩形、网格、箭头、圆角框，不画复杂 3D 或写实元素 |
| **标签** | 手写感字体：英文用 `Comic Sans MS` / `Bradley Hand`，CJK 用 `Kaiti SC` / 楷体 |
| **整体感觉** | 轻松白板涂鸦，标注随意、轻微错位也无妨，优先可读性和直观性 |

### 4.2 SVG 实现技法（仓库统一用法）

用 SVG 滤镜 `feTurbulence` + `feDisplacementMap` 给所有图形叠加轻微抖动，实现手绘效果。**每张 SVG 顶部固定引入以下** `<defs>`：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 360"
     font-family="'Comic Sans MS', 'Segoe UI', 'Kaiti SC', 楷体, cursive">
  <defs>
    <filter id="rough2">
      <feTurbulence type="fractalNoise" baseFrequency="0.025" numOctaves="2" seed="7"/>
      <feDisplacementMap in="SourceGraphic" scale="1.5"/>
    </filter>
  </defs>

  <rect width="520" height="360" fill="#fafafa"/>

  <!-- 所有矩形/路径/文本都加 filter="url(#rough2)" -->
  <rect x="80" y="48" width="50" height="30" fill="#e8f0fe"
        stroke="#446688" stroke-width="1.5" rx="4" filter="url(#rough2)"/>
  ...
</svg>
```

**要点**：

- `font-family` 必须包含 `'Comic Sans MS'` 和 `'Kaiti SC', 楷体`，保证中英文都有手写感。
- 每个图形元素（`rect`/`path`/`text`/`circle`）都加 `filter="url(#rough2)"`。
- 也可直接用文本编辑器手写 SVG（推荐）；`aiinfra/topics/images/` 只保留 SVG/PNG 成品，不含生成脚本。

### 4.3 常见图类型

| 图类型 | 用途 | 示例 |
|--------|------|------|
| **概念图** | 直观展示并行策略（grid-block-thread 映射、tiling 分块、reduction 树） | `reduction_overview.svg` |
| **存储层次图** | global → shared → register 的数据流 | `matmul_tiled.svg` |
| **性能对比图** | naive vs optimized 的带宽/延迟对比 | `reduction_grid_stride.svg` |

### 4.4 图片命名

全小写 + 下划线，语义化，建议加题目 slug 前缀避免冲突：

- `reduction_overview.svg`、`reduction_block_internal.svg`（reduction 题）
- `matmul_naive.svg`、`matmul_tiled.svg`（matrix multiplication 题）
- `top-k-selection_bitonic.png`、`top-k-selection_heap_reduce.png`（top-k-selection 题）

## 5. 代码详解与 SVG 图解规范

每篇题解除了 6 段基础结构外，还应包含 **代码详解** 子节和足够的 **SVG 图解**。这是题解质量的核心区分点——有图有详解的题解能让读者从"看懂代码"升级到"理解为什么这样写"。

### 5.1 代码详解子节

在 **§4 Kernel 实现** 的代码块之后（通常在 `### 4.1 LeetGPU 提交版本` 和 `## 5. 性能分析` 之间），添加 `### 4.2 代码详解` 子节。

#### 结构模板

```markdown
### 4.2 代码详解

<1-2 句话概括 kernel 的核心策略>

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标计算** | `i = blockIdx.x * blockDim.x + threadIdx.x` | thread 到全局索引的映射 |
| **加载/计算** | `... = input[...]` | 核心访存或计算逻辑 |
| **同步** | `__syncthreads()` | 屏障的作用与缺失后果 |
| **写回** | `output[...] = ...` | 结果写回 |

**关键索引关系**：
- `<变量>` = `<公式>` — <含义>
- ...

> 💡 **关键洞察**：<一句话点出 kernel 设计的本质洞察>
```

#### 详解内容要求

| 维度 | 要求 |
|------|------|
| **逐行覆盖** | kernel 中每一段关键代码都要在表格或列表中有对应解释 |
| **索引计算** | 必须解释 `threadIdx` → `blockIdx` → 全局坐标的映射链 |
| **同步语义** | 每次 `__syncthreads()` 要说明"等什么"和"不等会怎样" |
| **Worked Example** | 对复杂 kernel（convolution、attention、scan 等）给出具体数值的逐步演算 |
| **变量表** | 对多变量 kernel 给出"变量-含义-初始值"对照表 |
| **关键洞察** | 用 `> 💡` blockquote 点出 kernel 设计的核心洞察（1-2 句） |

#### 不同复杂度的详解深度

| Kernel 类型 | 详解深度 | 示例文件 |
|-------------|----------|----------|
| **简单 element-wise**（vector-add、relu、scalar-multiply） | 3-5 行表格 + 索引公式 | `vector-addition.md` |
| **Shared memory tiling**（transpose、matmul、convolution） | 完整索引映射表 + worked example + bank conflict 分析 | `matrix-transpose.md` |
| **归约类**（reduction、softmax、dot-product） | warp shuffle 步骤分解 + block reduce 两阶段流程图 | `reduction.md` |
| **融合 kernel**（flash attention、online softmax） | 三公式逐步数值演算 + k 循环数据流图 + `__syncthreads` 作用表 | `softmax-attention.md` |
| **多 kernel 流水线**（GPT-2 block、stream compaction） | kernel 链调用顺序表 + 每 kernel 一句话 + HBM IO 表 | `gpt-2-transformer-block.md` |

### 5.2 SVG 图解规范

#### 每篇题解的 SVG 数量要求

| 题解类型 | SVG 数量 | 说明 |
|----------|----------|------|
| 简单题（element-wise） | 1-2 张 | 至少 1 张概念图（数据流或索引映射） |
| 中等题（shared memory / reduction） | 2-3 张 | 概念图 + 存储层次图或索引详解图 |
| 复杂题（attention / scan / 多 kernel） | 3-5 张 | 概念图 + 数据流图 + 逐步演算图 + 性能对比图 |

> ⚠️ **最低要求**：每篇含 CUDA kernel 的题解**至少 1 张 SVG**。PyTorch 题解（无 CUDA kernel）无需 SVG。

#### SVG 内容类型

| 类型 | 用途 | 何时使用 | 命名模式 |
|------|------|----------|----------|
| **概念总览图** | 展示 kernel 整体数据流和并行策略 | 每篇必选 | `cuda_<slug>_overview.svg` |
| **索引计算图** | 逐步展示坐标映射（thread→shared→global） | tiling / halo 类必选 | `cuda_<slug>_index_calculation.svg` |
| **逐步演算图** | 具体数值的 step-by-step 推演 | attention / online softmax 类必选 | `cuda_<slug>_worked.svg` |
| **数据流图** | 多阶段 kernel 的 pipeline 流程 | 多 kernel 或三遍扫描类必选 | `cuda_<slug>_dataflow.svg` |
| **性能对比图** | naive vs optimized 的指标对比 | 有明确优化对比时可选 | `cuda_<slug>_roofline.svg` |
| **block 映射图** | grid/block 到数据的映射关系 | 多 head / batched 类可选 | `cuda_<slug>_block_mapping.svg` |

#### SVG 引用路径

题解位于 `aiinfra/topics/cuda/`，SVG 位于 `aiinfra/topics/images/`，因此引用路径为：

```markdown
![<中文描述>](../images/<filename>.svg)
```

#### SVG 创建要点

1. **手绘 sketch 风**：使用 `feTurbulence` + `feDisplacementMap` 滤镜（详见 §4.2）
2. **配色一致**：蓝（输入）、绿（输出）、橙（shared/中间）、红（关键操作/警告）
3. **中文标注**：标题、图例、公式说明用中文；变量名和代码用 monospace 英文
4. **具体数值**：worked example 类 SVG 必须用具体数字（如 `N=3, d=2, scale=0.707`），不能只有抽象符号
5. **viewBox**：使用 `viewBox="0 0 W H"` 而非固定 width/height，保证响应式缩放

### 5.3 slug 唯一性

扁平组织下同一 slug 只对应一个文件（`aiinfra/topics/cuda/<slug>.md`），不存在跨目录的副本或 stub。新增题解前先 `ls aiinfra/topics/cuda/<slug>.md` 确认未撞名；若已存在，直接在该文件上修改完善。

### 5.4 完成度检查清单

为题解补充 SVG + 代码详解后，用以下清单自检：

- [ ] 至少 1 张 SVG 引用（`![...](../images/...svg)`）
- [ ] SVG 文件存在于 `aiinfra/topics/images/`
- [ ] `### 4.2 代码详解` 子节存在（或等效的详解标题）
- [ ] 详解覆盖 kernel 的关键代码段（索引计算、访存模式、同步屏障）
- [ ] 复杂 kernel 有 worked example（具体数值逐步推演）
- [ ] `> 💡 关键洞察` blockquote 存在
- [ ] `## 同类练习题` 章节存在，且内容与 §1.1 推荐映射完全一致（4 条推荐 + 选材主线）
- [ ] `python3 build.py` 构建成功
- [ ] 生成的 HTML 中 SVG 路径可正常访问

## 6. 网站构建集成

题解写完后由 `build/topics.py` 读取并生成网页：

- **扫描** `aiinfra/topics/cuda/` 下所有 `<slug>.md` 题解文件（排除 `README.md`、`SKILL.md`、`dayN.md`）。
- 解析一级标题 `# LeetGPU <题目名> 题解` 作为侧边栏与卡片标题（取 `<题目名>`）。
- 图片路径 `../images/xxx.svg` 在题解页被重写为 `./images/xxx.svg`（输出目录扁平化）。
- 生成 `public/cuda/index.html`（概览页，自动追加「📝 LeetGPU 题解」卡片区）与 `public/cuda/<slug>.html`（各题解页）。
- `aiinfra/topics/images/` 中 `cuda_*` 前缀的 SVG 自动复制到 `public/cuda/images/` 部署。

**验证命令**：

```bash
python3 build.py                     # 组合构建全站
```

**自检清单**：

- [ ] 题解位于 `aiinfra/topics/cuda/<slug>.md`（扁平存放，slug 唯一）
- [ ] 一级标题 `# LeetGPU <题目名> 题解`
- [ ] 含 6 段结构（题目概述/CPU基线/GPU设计/Kernel实现/性能分析/复杂度分析）
- [ ] Kernel 代码完整可编译（含 main、cudaMalloc、验证、cudaFree）
- [ ] 含 2-4 张 SVG/PNG 插图，引用格式 `![中文alt](../images/xxx.svg)`
- [ ] 含 `### 4.2 代码详解` 子节（逐行解释 + 索引表 + 关键洞察）
- [ ] SVG 为手绘 sketch 风（含 `feTurbulence` 抖动滤镜 + Comic Sans/Kaiti SC 字体）
- [ ] 复杂 kernel 有 worked example（具体数值逐步推演）
- [ ] 含 `## 同类练习题` 章节，内容与 §1.1 推荐映射一致
- [ ] 含 ncu profiling 命令与关键指标
- [ ] `python3 build.py` 构建成功并生成对应题解页
- [ ] `git push origin` 推送题解（commit + push 到远程）

## 7. 面试导向：题解与面试的衔接

LeetGPU 题解不仅是刷题归档，也是 **AI Infra 面试「手撕 CUDA kernel」环节**的直接准备材料。面经整理（`aiinfra/topics/interview/`）显示：大模型相关岗位**手撕 CUDA 与 LeetCode 的比例约 4:1**，现场写 GEMM、FlashAttention、Reduction 等 kernel 并追问优化方向是高频环节。写题解时应让读者"做完题就能答面试"。

### 7.1 高频手撕 kernel 与 LeetGPU 题目映射

| 面试高频手撕题 | 对应 LeetGPU 题目 | 面试高频追问 |
|----------------|-------------------|--------------|
| 手写 GEMM + 优化 | `matrix-multiplication` / `gemm` | shared memory tiling 为什么有效、register blocking、双缓冲、bank conflict、tensor core |
| FlashAttention / fused softmax | `softmax-attention` / `multi-head-attention` | online softmax 数值稳定、FA v1/v2/v3 改进、KV cache 影响 |
| Reduction / Softmax | `reduction` / `softmax` | warp shuffle、`__syncthreads` 语义、block 两级归约、warp divergence |
| Matrix Transpose | `matrix-transpose` | bank conflict 成因与 padding 消除、coalesced 读写权衡 |
| Convolution | `1d-convolution` / `2d-convolution` | shared memory halo、常数内存、边界处理 |
| 量化 kernel | `int8-quantized-matmul` / `weight-dequantization` | per-tensor / per-channel / per-group 粒度、scale 计算、SmoothQuant / AWQ / GPTQ |
| Scan / Prefix Sum | `prefix-sum` | warp scan、三阶段分块 scan、inclusive/exclusive |

### 7.2 写作时的面试导向要求

题解各部分应主动覆盖面试高频追问点：

| 面试追问点 | 在题解中的落点 |
|------------|----------------|
| 为什么这样优化（shared mem / tiling / kernel 融合） | §3 GPU 设计 + `> 💡 关键洞察` |
| `__syncthreads` 什么时候需要、缺失会怎样 | §4.2 代码详解的同步语义说明 |
| bank conflict / warp divergence 的成因与规避 | §4.2 代码详解 + §5 性能分析 |
| 怎么定位性能瓶颈（ncu 指标、roofline） | §5 性能分析 |
| 还能怎么进一步优化 | §5 性能分析的优化方向 |

### 7.3 可选的「面试考点」小节

对面试高频题（GEMM、attention、reduction、量化等），可在题解末尾（`## 同类练习题` 之前）加一节，把题解浓缩为面试应答材料：

```markdown
## 面试考点

- **手撕要求**：<面试官可能要求现场默写的 kernel 核心结构，1-2 句>
- **高频追问**：<3-5 个追问点，各附一句话答案>
- **进阶延伸**：<Hopper TMA / tensor core / FA v2 等进阶话题>
```

> 📚 **面经参考**：`aiinfra/topics/interview/`（面试形式、CUDA/推理/训练/量化高频考点分类、逐题参考答案）。
