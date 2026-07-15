# Triton 专题

> **适用对象**：已完成 week1 CUDA 基础教程，理解 thread/block/grid、Global Memory、Shared Memory 概念
> **本周目标**：掌握 Triton 的 block 级编程模型，能用 Triton 写出常见 GPU kernel（vector add、GEMM、softmax 等），理解 Triton 与手写 CUDA、CUTLASS 的适用场景
> **时间投入**：工作日每天 2.5h，周末每天 5h

---

## 为什么学 Triton

Triton 是 OpenAI 开源的 Python DSL，让你用接近 Python 的语法写出接近手写 CUDA 性能的 GPU kernel：

| 维度 | 手写 CUDA | CUTLASS | Triton |
|------|-----------|---------|--------|
| 性能 | 100%（基准） | 90-98% cuBLAS | 80-90% 手写 CUDA |
| 开发效率 | 数天~数周 | 数小时~数天 | 数分钟~数小时 |
| 学习曲线 | 陡峭 | 陡峭 | 平缓 |
| 灵活性 | 完全自定义 | 模板参数化 | block 级自定义 |
| 适用场景 | 库开发、极致性能 | GEMM/卷积算子库 | 快速原型、自定义算子 |

> 💡 **一句话总结**：Triton 暴露 block 级编程，隐藏 thread 级细节——你用 Python 写算法逻辑，编译器自动处理 Shared Memory、线程同步、向量化。

---

## 本周学习计划

| 天数 | 主题 | 核心产出 |
|------|------|----------|
| Day 1 | Triton 总览与环境搭建 | 跑通 `vector_add` kernel |
| Day 2 | Block 级编程与内存访问 | 实现 `matrix_transpose` |
| Day 3 | Softmax 与归约操作 | 实现 `online_softmax` |
| Day 4 | GEMM 基础 | 实现 Shared Memory Tiling GEMM |
| Day 5 | FlashAttention 与注意力算子 | 实现简化版 FlashAttention |
| Day 6 | Triton 与 CUDA/CUTLASS 对比 | 性能对比报告 |
| Day 7 | 进阶专题与总结 | 面试题复盘 |

---

## 前置准备

- Python >= 3.8
- PyTorch >= 2.0（自带 Triton）
- GPU Compute Capability >= 7.0

```bash
# 验证 Triton 安装
python3 -c "import triton; print(triton.__version__)"

# 验证 PyTorch 可用
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 目录结构

```
aiinfra/topics/triton/
├── README.md          # 本文件（专题概览）
├── day1.md            # 每日学习文档
├── kernels/           # 可运行代码示例
│   └── vector_add.py
├── notes/             # 源码与原理笔记
└── benchmark/         # 性能对比
```
