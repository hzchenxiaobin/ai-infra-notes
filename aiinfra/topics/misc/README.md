# 杂七杂八：AI Infra 碎片知识整理

> 收纳那些不够单独开一个专题、但值得记录的碎片知识——硬件特性、概念辨析、面试快问快答、踩坑备忘等。
> 随时往里塞，不追求体系化，追求"下次遇到能快速查到"。

---

## 📋 分类索引

| 分类 | 内容 |
|------|------|
| [硬件与架构](#硬件与架构) | TMA、Warp Specialization、Blackwell/Hopper 新特性等 |
| [深入教程](#深入教程) | 单点深挖的长文教程 |
| [概念辨析](#概念辨析) | 易混淆术语对比（SIMT vs SIMD、cp.async vs TMA 等） |
| [面试快问快答](#面试快问快答) | 一两句话能答完的小题 |
| [踩坑备忘](#踩坑备忘) | 开发中踩过的小坑 + 解法 |

---

## 深入教程

| 教程 | 内容 |
|------|------|
| [TMA：Hopper 的张量内存加速器](tma.md) | 传统访存痛点、TMA descriptor 机制、`cp.async.bulk.tensor` 指令、mbarrier `expect_tx` 追踪、边界/swizzle 自动处理、TMA vs cp.async 对比、数据通路旁路 L1 |
| [Warp Specialization：Hopper 的生产者-消费者并行范式](warp_specialization.md) | 传统同步瓶颈、TMA+WGMMA+mbarrier 三件套、pipeline 设计、与 Triton `num_stages` 对比、CUTLASS 源码结构 |

---

## 硬件与架构

### TMA（Tensor Memory Accelerator）

Hopper（sm_90）引入的**硬件单元**，用于在 GMEM 和 SMEM 之间搬运多维张量块（tile），替代传统的 `cp.async` / `ld.global`。详见独立教程：[TMA 深入教程](tma.md)

### Warp Specialization（warp 专化）

Hopper 上推崇的编程范式，与 TMA 配合使用。详见独立教程：[Warp Specialization 深入教程](warp_specialization.md)

> 💡 **面试一句话**：Warp specialization 是 producer/consumer 分工范式——TMA warp 管搬数据，WGMMA warp 管算，用 mbarrier 异步通知，让访存和计算真正重叠。

### TMA 与 Warp Specialization 的关系

| 维度 | TMA | Warp Specialization |
|------|-----|---------------------|
| 本质 | 硬件单元 | 编程范式 |
| 解决什么 | 地址计算 + 异步搬运 | 访存/计算重叠 |
| 互相依赖 | 可独立用（但配 WS 才发挥威力） | 依赖 TMA 的异步性才有意义 |
| Triton 支持 | 部分（3.0+） | 不完全（software pipeline 代替） |

> ⚠️ 两者配合是 Hopper 上 GEMM/FA 达到 peak 性能的关键，也正是"Triton 滞后 1-2 架构周期"的典型场景。

---

## 概念辨析

> 此节收录易混淆术语的对比，每条尽量一句话说清差异。

（待补充）

---

## 面试快问快答

> 一两句话能答完的小题，适合面试前快速过一遍。

（待补充）

---

## 踩坑备忘

> 开发中踩过的小坑 + 解法，避免重复踩。

（待补充）

---

## 如何往这里加内容

1. 学到/遇到一个知识点，不够开专题但值得记 → 直接往对应分类下加
2. 新分类在索引表和正文中各加一行
3. 每条尽量带一句 `> 💡 **面试一句话**：<...>` 方便速记
4. 如果某条膨胀超过 ~50 行，考虑拆出去开独立教程，这里留链接
