# LeetGPU Profiling — ncu 性能分析

本目录汇总了 [LeetGPU 题解](../../LeetGPU/) 中所有带有 ncu profiling 代码的可执行程序，方便一键编译和性能分析。

## 目录结构

```
profiling/LeetGPU/
├── Makefile                          # 统一编译 + 各题 ncu profiling target
├── README.md                         # 本文件
├── vector-add.cu                     # Vector Add（memory-bound，block size 对比）
├── relu.cu                           # ReLU（element-wise，occupancy 调优）
├── matrix-transpose.cu               # Matrix Transpose（coalescing + bank conflict）
├── reduction.cu                      # Reduction（warp shuffle + 两级归约）
├── histogram.cu                      # Histogram（atomic + shared memory 对比）
├── matrix-addition.cu                # Matrix Addition（float4 向量化）
├── matrix-multiplication.cu          # Matrix Multiplication（shared memory tiling GEMM）
├── softmax.cu                        # Softmax（三遍扫描，memory-bound）
└── argmax.cu                         # Argmax（带状态追踪的归约）
```

## 快速开始

```bash
# 编译所有程序
make all

# 运行所有程序（正确性验证）
make profile-all

# 单独编译运行某个程序
make vector_add && ./vector_add
```

## 各题 ncu profiling 命令

每道题有对应的 `profile-<slug>` target：

```bash
make profile-vector-add          # Vector Add: DRAM throughput + occupancy
make profile-relu                # ReLU: occupancy + registers + DRAM throughput
make profile-matrix-transpose    # Matrix Transpose: bank conflicts + DRAM throughput
make profile-reduction           # Reduction: bank conflicts + occupancy
make profile-histogram           # Histogram: DRAM + bank conflicts + occupancy
make profile-matrix-addition     # Matrix Addition: DRAM + occupancy + registers
make profile-matmul              # GEMM: SM throughput + DRAM + occupancy
make profile-softmax             # Softmax: DRAM + SM + occupancy + stall
make profile-argmax              # Argmax: occupancy + DRAM + registers
```

## 各题 ncu 观察要点

| 题目 | 瓶颈类型 | 关键指标 | 预期值 | 对应题解 |
|------|---------|---------|--------|---------|
| Vector Add | memory-bound | `dram__throughput` | 高（>60%） | [Vector Add 题解](../../LeetGPU/leetgpu-vector-add-solution.md) |
| ReLU | memory-bound | `dram__throughput`, `registers_per_thread` | DRAM 高，寄存器少 | [ReLU 题解](../../LeetGPU/leetgpu-relu-solution.md) |
| Matrix Transpose | memory-bound | `bank_conflicts`, `dram__throughput` | bank conflict 应为 0（padding 后） | [Transpose 题解](../../LeetGPU/leetgpu-matrix-transpose-solution.md) |
| Reduction | memory-bound | `bank_conflicts`, `occupancy` | occupancy 高，冲突低 | [Reduction 题解](../../LeetGPU/leetgpu-reduction-solution.md) |
| Histogram | memory-bound | `dram__throughput`, `bank_conflicts` | shared mem 版 DRAM 更低 | [Histogram 题解](../../LeetGPU/leetgpu-histogram-solution.md) |
| Matrix Addition | memory-bound | `dram__throughput`, `occupancy` | float4 版 DRAM 更高 | [Matrix Add 题解](../../LeetGPU/leetgpu-matrix-addition-solution.md) |
| Matrix Multiplication | compute-bound | `sm__throughput`, `dram__throughput` | SM >> DRAM | [Matmul 题解](../../LeetGPU/leetgpu-matrix-multiplication-solution.md) |
| Softmax | memory-bound | `dram__throughput`, `stall` | DRAM 高，Long Scoreboard 高 | [Softmax 题解](../../LeetGPU/leetgpu-softmax-solution.md) |
| Argmax | memory-bound | `occupancy`, `dram__throughput` | DRAM 高 | [Argmax 题解](../../LeetGPU/leetgpu-argmax-solution.md) |

## 瓶颈判定方法

```
dram__throughput 高 + sm__throughput 低  → memory-bound
dram__throughput 低 + sm__throughput 高  → compute-bound
两者都低                                  → latency-bound
```

> 💡 本目录中 8/9 道题是 memory-bound（AI < Ridge Point），只有 Matrix Multiplication 在大矩阵时是 compute-bound。这反映了 CUDA 入门题目的典型特征：访存是主要瓶颈。
