# Day 3 Profiling 任务：设备属性与 Occupancy Calculator

## 今日目标
学会读取 GPU 硬件属性，并用官方工具验证理论 occupancy。

## 需要运行的工具

### 任务 1：运行 deviceQuery

```bash
/usr/local/cuda/extras/demo_suite/deviceQuery
```

如果找不到，尝试：

```bash
find /usr/local/cuda -name deviceQuery
```

**记录关键字段**：

| 字段 | 数值 | 含义 |
|------|------|------|
| Device name | | GPU 型号 |
| CUDA Capability | | 计算能力 |
| Multiprocessors | | SM 数量 |
| CUDA Cores per SM | | 每个 SM 的 CUDA core 数 |
| Max threads per SM | | |
| Max threads per block | | |
| Shared memory per block | | |
| Registers per block | | |
| Warp size | | |
| Memory Clock Rate | | |
| Memory Bus Width | | |
| Peak Memory Bandwidth | | |

### 任务 2：运行 CUDA Occupancy Calculator

打开 Excel 版本的 CUDA Occupancy Calculator（通常位于 CUDA Samples 目录），或在线版本：

- 输入你的 GPU 计算能力
- 输入 Day 2 kernel 的 block 大小、寄存器用量、共享内存用量
- 记录理论 occupancy

### 任务 3：编译并运行 occupancyCalculator sample

```bash
cd /usr/local/cuda/samples/1_Utilities/occupancyCalculator
make
./occupancyCalculator
```

## 数据记录

| 项目 | 数值 |
|------|------|
| GPU 型号 | |
| SM 数量 | |
| 每个 SM 最大 warp 数 | |
| 显存带宽（理论） | |
| Day 2 kernel 理论 occupancy | |
| Day 2 kernel 实际 occupancy | |

## 思考题

1. 如何根据 `memoryClockRate` 和 `memoryBusWidth` 计算理论显存带宽？
2. 理论 occupancy 和实际 occupancy 为什么会有差异？
3. 你的 GPU 峰值算力是多少？如何计算？
