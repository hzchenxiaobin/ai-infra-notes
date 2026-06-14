# Day 1 Profiling 任务：hello_gpu

## 今日目标
理解 kernel launch 的基本开销和 thread/block 调度方式。

## 需要编译的代码

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/hello_gpu kernels/hello_gpu.cu
```

## Profiling 任务

### 任务 1：使用 nsys 查看 kernel launch 时间线

```bash
nsys profile -o profiles/day1_hello_gpu_timeline ./kernels/hello_gpu
```

**观察重点**：
- `cudaLaunchKernel` 的 CPU 端耗时
- kernel 在 GPU 上的实际执行时长
- 多个 block 是否并行执行

### 任务 2：使用 ncu 查看基础指标

```bash
ncu \
  --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  ./kernels/hello_gpu
```

**观察重点**：
- 活跃 warp 比例
- kernel 实际占用的 cycle 数

## 数据记录

| 指标 | 数值 | 备注 |
|------|------|------|
| CPU launch 耗时 | | |
| GPU 执行耗时 | | |
| 活跃 warp 比例 | | |
| 输出线程总数 | | |

## 思考题

1. 改变 grid/block 配置后，kernel 执行时间是否变化？为什么？
2. `printf` 在 kernel 中对性能有什么影响？
