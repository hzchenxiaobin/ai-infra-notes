# Week1 Day4：Nsight Compute (ncu) 性能分析任务 —— 矩阵转置

> **来源**：`week1/day4`（Memory Hierarchy 深入）
> - 源码：`week1/day4/kernels/transpose.cu` → 复制为本目录 `transpose.cu`
> - Profiling 任务：`week1/day4/notes/day4_transpose.md`
> - 教程正文：`week1/day4/README.md`（任务 3、扩展实验 3）
>
> 本目录把 Day 4 中需要用 Nsight Compute / Nsight Systems 分析矩阵转置性能
> 的部分抽出来，方便在有 GPU 的机器上统一执行。
> 目录结构为 `ncu/<week>/<day>/`，便于后续其他日期的 ncu 任务归档。

## 背景分析对象

`transpose.cu` 包含两个 kernel：

| kernel | 读 | 写 | 说明 |
|--------|----|----|------|
| `transpose_naive` | coalesced（按行读） | **stride access**（按列写） | 写操作成为瓶颈 |
| `transpose_optimized` | coalesced read → shared memory tile | coalesced write | 用 `TILE_DIM+1` padding 消除 bank conflict |

优化原理：先按行从 global memory 读入 shared memory（coalesced read），
`__syncthreads` 后交换 block 坐标，再按行从 shared memory 写回 global memory
（coalesced write）。转置发生在 shared memory 内部（按行写、按列读），
而 global memory 两侧都是连续访问。

## 步骤 1：编译

```bash
cd ncu
nvcc -o transpose transpose.cu
./transpose
# 预期输出：Transpose correctness: PASS
```

## 步骤 2：使用 ncu 对比 memory throughput

> 注：`transpose.cu` 包含两个 kernel，ncu 会分别采集。

```bash
ncu \
  --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    sm__cycles_elapsed.avg \
  ./transpose
```

观察重点：
- `transpose_naive` 与 `transpose_optimized` 的耗时差异（`sm__cycles_elapsed.avg`）
- DRAM 带宽占比（`dram__throughput.avg.pct_of_peak_sustained_elapsed`）
- global memory read/write 字节数

## 步骤 3：分析 L1/L2 cache 与显存读写量

```bash
ncu --metrics \
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum \
  ./transpose
```

对比 naive 和 optimized 版本的实际显存读写量。

## 步骤 4：使用 nsys 查看完整时间线

```bash
mkdir -p profiles
nsys profile -o profiles/day4_transpose_timeline ./transpose
```

## 数据记录

| 版本 | 执行时间 (cycles) | DRAM Throughput % | L1 Read Bytes | L1 Write Bytes | DRAM Read Bytes | DRAM Write Bytes |
|------|------------------|-------------------|---------------|----------------|-----------------|------------------|
| naive | | | | | | |
| optimized | | | | | | |

## 思考题

1. Naive 版本的瓶颈是什么？读还是写？
2. Shared memory 优化版如何做到 coalesced write？
3. 为什么 shared memory tile 要加 padding？（见 Day 5 的 bank conflict 分析）

### 思考题 2 详细解析：Shared memory 优化版如何做到 coalesced write？

以 `TILE_DIM=4`、矩阵 `8×8` 为例，Block(1,0) 的完整工作流程：

- **左侧 Input**：Block(1,0) 负责读取原矩阵中列范围 `[4,7]`、行范围 `[0,3]` 的 tile；
- **中间 Shared Memory**：先把数据按 `tile[threadIdx.y][threadIdx.x]` 写入 tile，
  再按 `tile[threadIdx.x][threadIdx.y]` 读出，在 tile 内部完成转置；
- **右侧 Output**：同一个 block 把数据写到输出矩阵中行范围 `[4,7]`、列范围 `[0,3]`
  的 tile，这正是原 tile 的转置位置。

#### 数据划分与线程映射

CUDA 把矩阵划分成 `TILE_DIM × TILE_DIM` 的 tile，每个 block 负责一个 tile。

- 原矩阵中，block `(blockIdx.x, blockIdx.y)` 对应的 tile 左上角为：
  - `x_base = blockIdx.x * TILE_DIM`
  - `y_base = blockIdx.y * TILE_DIM`
- block 内线程 `(threadIdx.x, threadIdx.y)` 负责该 tile 内的元素：
  - `x = x_base + threadIdx.x`
  - `y = y_base + threadIdx.y`

以 Block(1,0) 为例：它读取 Input 中 `A[0..3][4..7]`，经过 shared memory 中转后，
写入 Output 的 `O[4..7][0..3]`。

#### 第一阶段：coalesced read

```cuda
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
tile[threadIdx.y][threadIdx.x] = in[y * width + x];
```

一个 warp 内 `threadIdx.y` 相同，`threadIdx.x` 连续变化：

- `y` 固定 → `y * width` 是常量；
- `x = blockIdx.x * TILE_DIM + threadIdx.x` 连续递增；
- 地址 `in[y * width + x]` 连续 → **coalesced read** ✅。

**为什么用 `tile[threadIdx.y][threadIdx.x]` 存储？** 因为这样每个线程把读到的元素
放到 shared memory 中与输入矩阵相同行/列位置的单元里，保持 tile 内部的行主序布局。

#### 第二阶段：coalesced write

```cuda
x = blockIdx.y * TILE_DIM + threadIdx.x;
y = blockIdx.x * TILE_DIM + threadIdx.y;
out[y * height + x] = tile[threadIdx.x][threadIdx.y];
```

关键点：**交换了 `blockIdx.x` 和 `blockIdx.y`，但 `threadIdx.x` 仍然对应输出地址
的连续维度**。

输出矩阵 `out` 是行优先的 `height × width`。一个 warp 内：

- `y = blockIdx.x * TILE_DIM + threadIdx.y` 固定；
- `x = blockIdx.y * TILE_DIM + threadIdx.x` 连续递增；
- 地址 `out[y * height + x]` 连续 → **coalesced write** ✅。

**为什么用 `tile[threadIdx.x][threadIdx.y]` 读出？** 因为输出阶段一个 warp 内
`threadIdx.y` 固定、`threadIdx.x` 连续变化。如果仍然按 `tile[threadIdx.y][threadIdx.x]`
读出，所有线程会读到 tile 的同一行，写出的地址反而不连续。交换索引后，相邻线程从
tile 的不同行取数据，但写回 global memory 时地址连续。

#### 转置发生在哪里？

在 shared memory 内部：

```cuda
tile[threadIdx.y][threadIdx.x] = ...   // 按行写入 tile
... = tile[threadIdx.x][threadIdx.y]; // 按列读出 tile
```

shared memory 的随机访问延迟低，所以这里的非连续访问不是瓶颈。global memory 两侧
则都被改造成了连续访问。

#### 地址正确性验证

跟踪线程 `(a, b)`：

- 第一阶段写入：`tile[b][a] = in[(by*TILE_DIM + b) * width + (bx*TILE_DIM + a)]`
- 第二阶段读取：`tile[a][b]`，它来自线程 `(b, a)` 第一阶段写入，对应原矩阵元素
  `in[(by*TILE_DIM + a) * width + (bx*TILE_DIM + b)]`
- 第二阶段写出：`out[(bx*TILE_DIM + b) * height + (by*TILE_DIM + a)] = ...`

最终满足转置定义 `output[j][i] = input[i][j]`。
