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

## 显存带宽计算说明

GDDR（Graphics Double Data Rate）显存采用**双倍数据速率**技术：数据在时钟信号的**上升沿**和**下降沿**都会被传输，因此实际数据传输速率是 deviceQuery 报告的 `Memory Clock Rate` 的 2 倍。

### 计算公式

```text
理论显存带宽 = MemoryClockRate × 2 × MemoryBusWidth / 8
```

其中：

- `MemoryClockRate`：deviceQuery 报告的显存时钟频率（单位：MHz）
- `×2`：GDDR 双倍数据速率，每个时钟周期传输 2 次数据
- `MemoryBusWidth`：显存位宽（单位：bit）
- `/8`：将 bit 转换为 byte

### 示例（RTX 5090）

```text
Memory Clock Rate = 14001 MHz
Memory Bus Width  = 512-bit

带宽 = 14001 × 2 × 512 / 8 / 1000
     = 1792.13 GB/s
```

> 注意：deviceQuery 中的 `Memory Clock Rate` 是**基准时钟频率**，不是实际数据传输速率。计算带宽时必须先乘以 2，否则结果只有真实带宽的一半。

### 简单类比

把显存时钟想象成公交调度：

- **SDR（单倍数据速率）**：每"滴"一声发一趟车
- **GDDR（双倍数据速率）**：每"滴"一声发两趟车（上升沿一趟、下降沿一趟）

同样的"心跳频率"下，GDDR 能搬运的数据量是 SDR 的 2 倍。

## 思考题

1. 如何根据 `memoryClockRate` 和 `memoryBusWidth` 计算理论显存带宽？
2. 理论 occupancy 和实际 occupancy 为什么会有差异？
3. 你的 GPU 峰值算力是多少？如何计算？
