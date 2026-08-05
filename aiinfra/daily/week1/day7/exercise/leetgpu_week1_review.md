# Week 1 LeetGPU 复盘记录

> 用法：每道题提交通过后，把耗时和 ncu 关键指标填进下表；综合练习（Day 7）记录在最后。
> 题目与考点对应关系见 [Day 7 README 的本周回顾表](../README.md)。

## 每日题目记录

| Day | 题目 | 耗时 | 关键 ncu 指标 | 备注 |
|-----|------|------|---------------|------|
| Day 1 | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | | | 1D 索引、边界检查 |
| Day 2 | [ReLU](https://leetgpu.com/challenges/relu) | | achieved occupancy | block size 对 occupancy 的影响 |
| Day 3 | [Matrix Addition](https://leetgpu.com/challenges/matrix-addition) | | `gpu__time_duration.sum` | block 形状调参（16×16 / 32×8 / 32×16） |
| Day 4 | [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) | | `dram__throughput` | naive vs tiling+padding |
| Day 5 | [Reduction](https://leetgpu.com/challenges/reduction) | | bank conflict 计数 | warp shuffle + smem 中转 |
| Day 6 | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | | `dram__throughput` / `sm__throughput` | naive 版 + Roofline 定位 |

## 综合练习（Day 7）

| 题目 | 耗时 | GFLOPS / 带宽利用率 | 瓶颈类型 | 优化尝试 |
|------|------|---------------------|---------|---------|
| Matrix Transpose | | | memory-bound | tiling / padding / block size |
| Matrix Multiplication | | | 理论 compute-bound，naive 实测为准 | naive + ncu 定位（tiling 属 Week 2 预习） |

## 心得记录

- 本周踩过的坑：
- 对 memory-bound / compute-bound 判断的直觉：
