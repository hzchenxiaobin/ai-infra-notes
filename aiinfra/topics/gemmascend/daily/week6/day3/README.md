# 第 6 周 · 第 3 天（总第 38 天）：运行或推演 Grouped GEMM

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 45 分钟，源码导读约 25 分钟，实践步骤约 70 分钟，自测题约 15 分钟。

前两天我们把 Grouped GEMM 的两种形态读完了：第 36 天的 M-Grouped（组在行上，B 逐组一份）和第 37 天的 K-Grouped（组在收缩维上，D 逐组一份）。今天是**实践日**：不引入新概念，而是把"读懂"变成"会算"。计划给出三个任务：有环境就运行 `mgrouped` / `kgrouped` 两组测试；核心练习是**跟踪一个 group 从 layout 到输出区间的映射**；没有环境就**手工构造两个不等长 group 并推演调度过程**。通过标准是一条 MoE 全链路问题：**给定一个 MoE 输入，能说明 token、group、expert 权重和输出的对应关系**。本文所有手算例子都给出可核对的中间量，做完后可直接与源码行为对账。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 运行（或静态描述）`python tests/test_bf16_gemm.py mgrouped --skip-prof` 与 `kgrouped --skip-prof` 的完整流程：CLI 选择子如何映射到 `GemmType`（`tests/test_bf16_gemm.py:39-42`）、用例数量为什么分别是 25 和 288（`tests/generators.py:579-603, :616-649`）、`run_gemm_test` 对分组类型跳过了哪些 Normal 才有的分支（`tests/common.py:154, :186, :210`）；
2. 掌握**"group → 输出区间"追踪配方**：给定 `grouped_layout`、对齐度 256 和一组 Block 配置，独立算出任意组 g 的有效区间、padding 区间、起始 block 编号与 block 数量（M 维），或 K 循环参数 `current_shape_k / k_idx_base / num_k_blocks / actual_k`（K 维）；
3. 对两个不等长 group 的手算例（M 维 `[300, 700]`、K 维 `[500, 1000]`）分别推演某个 AI Core 的 `get_next_block` 调用序列：组游标何时推进、`in_group_idx` 如何换算、M 维的 `m_block_idx` 平移量 `last_psum_m / BLOCK_M` 从哪来（`scheduler.hpp:144-177`）；
4. 解释测试体系如何"抓住"分组错误：M-Grouped 的输入零 padding + 全行比较（`generators.py:338-340, :357`）、K-Grouped 的 NaN 预填 + 空组清零 + fp32 累加逐位对照（`common.py:190, :202-206`；`epilogue.hpp:108-141`），以及 perf 口径差异（`common.py:223-224`）；
5. 回答通过标准：给定一个 MoE 输入（token 序列、路由结果、expert 权重、`grouped_layout`），说清 token、group、expert 权重和输出在前向（M-Grouped）与反向 wgrad（K-Grouped）两个方向上的对应关系，并能画出对应图。

## 二、预备知识

- **第 17 天**：Normal 调度——每个核通过 `block_idx = (++current_iter) × kNumCores + core_id` 领任务，今天的推演全靠这条公式。
- **第 20 天**：`gm_ptr` 的 `index()` / `group()` / stride——今天用它核算组偏移落点（B 的 `group()`、A/B 的 K 平移、D 的 slab）。
- **第 22 天**：K 维两层循环与 `actual_k` 截断——K-Grouped 的"非对齐组尾"复用同一机制。
- **第 29、30 天**：`generators.py` 与 `common.py` 的测试流程——今天重点看它们的分组分支与验证差异。
- **第 31 天**：运行测试的经验（`--skip-prof`、失败层级判断）——今天照搬运行方法。
- **第 36 天**：M-Grouped 的 psum 行布局、分组游标、"只有 B 拿组偏移"、BLOCK_M 钉死 256。
- **第 37 天**：K-Grouped 的 psum-over-K 布局、共享 M×N 网格、`expected_k` 提示、空组清零。

> 推荐先花 5 分钟把第 37 天末尾的"三种 GEMM 类型对比表"默画一遍；今天所有推演都以它为底图。另注：第五节 A 组任务需要 Ascend 环境（Ascend 950 系列 + CANN 9.20，见仓库 `README.md` 的 Requirements）；无环境时完成 B 组静态任务即可达到当天目标。

## 三、详细讲解

### 3.1 今天的任务：从"读懂"到"会算"

读源码时容易产生"我懂了"的错觉，分组调度尤其如此——`grouped_layout`、组游标、block 平移这些量的组合行为，只有亲手算过一遍才会真正内化。今天的方法论是三条追踪线，全部围绕计划的核心句"跟踪一个 group 从 layout 到输出区间的映射"展开：

1. **布局线**：`grouped_layout`（长度 G 的前缀和数组）→ 每组的有效区间与 padding 区间；
2. **调度线**：某组的区间 → 它占用的 block 编号范围 → 某个 core 何时领到它、组游标何时推进；
3. **寻址线**：组号 → B 的 `group()` 偏移 / A·B 的 K 平移 / D 的 slab 偏移 → 输出区间。

有环境时再加上第四条**验证线**：跑测试，把每条输出行还原成上述三个量，确认与手算一致。下面 3.2 节先给出测试入口全景，3.3、3.4 节分别完成 M 维和 K 维的完整手算，3.5 节解释测试如何验证，3.6 节回答 MoE 通过标准。

### 3.2 测试入口全景：从命令行到用例

两条命令的入口在 `tests/test_bf16_gemm.py`：

```python
# tests/test_bf16_gemm.py:25-26, :39-42
parser.add_argument('select', nargs='?', default='all',
                    choices=['all', 'unaligned-k', 'unaligned-mn', 'strided', 'normal', 'mgrouped', 'kgrouped'])
parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true', ...)
...
elif sel == 'mgrouped':
    test_bf16_gemm(GemmType.MGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
elif sel == 'kgrouped':
    test_bf16_gemm(GemmType.KGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
```

`test_bf16_gemm` 只是给 `run_gemm_test` 套上 bf16×bf16、`threshold=1e-5` 的外壳（`:6-18`）。`run_gemm_test`（`tests/common.py:131-236`）的流程对两种分组类型是一致的，差异都来自 `gemm_type`：

```text
disable_cpu_fallback()                       # 禁止算子静默回落 CPU（common.py:142）
tests = enumerate_gemm_tests(gemm_type, ...) # 枚举用例（:149）
┌─ 第一遍：JIT 预编译 ──────────────────────────────────────────────┐
│ for test: 用 device='meta' 生成描述符（不分配 NPU 存储，:157）       │
│   只展开分组允许的转置：M-Grouped 只有 'nt'，K-Grouped 只有 'tn'    │
│   （generators.py:290-298）                                         │
│   派生 kernel：bf16 同型 + 输出 bf16 → 随机舍入变体（:169-171）；    │
│   非 K-Grouped → 追加 fp32 输出变体（:172-174）；                    │
│   K-Grouped 非 accumulate → 追加 fp32 累加变体（:165-168）          │
│ par_compile(...)  # dry-run 模式多线程编译（:175; par_compile.py:16-34）│
└────────────────────────────────────────────────────────────────────┘
┌─ 第二遍：逐用例真跑 ──────────────────────────────────────────────┐
│ 无 bias / 无 alpha（分组不支持，:154, :186）                         │
│ K-Grouped 非 accumulate → D 用 NaN 预填（:190）                      │
│ gemm_dg(...) → torch.npu.synchronize() → assert_equal(d, d_ref)    │
│   （:199-201；cmp_spans 见 3.5 节）                                 │
│ K-Grouped → 追加"直接输出 == fp32 累加后 cast"逐位对照（:202-206）   │
│ aclnn 交叉验证：分组直接跳过（:210; gemm_aclnn :87-91）              │
│ bf16 → 随机舍入与 fp32 参考核对（:215-221）                          │
│ perf（未 --skip-prof 时）：K-Grouped 的 k 换成 sum(ks_cpu)（:223-228）│
└────────────────────────────────────────────────────────────────────┘
```

**用例数量**（`Num tests:` 打印，`common.py:150`）：

| 测试 | 组合维度 | 数量 |
|---|---|---|
| `mgrouped` | 1 个最小用例 `(G=128, expected_m=1, N=4, K=4)`（`generators.py:595`）+ 3 种 `(G, expected_m)` × 4 种 `(N, K)` × 2 种 `major_b`（`:597-603`） | **25** |
| `kgrouped` | 8 种 `(G, M, N, expected_k)` 形状 × 3 种变体（normal / empty / single）× 4 种 `(accumulate, cd_dtype)` × 3 档 `ks-cpu` 提示（`:616-649`） | **288** |

每个 `GemmTestDesc` 还会派生 2～3 个 kernel 变体（流程图第一遍），所以实际编译数大于用例数。M-Grouped 用例的每组行数是 `max(int(uniform(0.7, 1.3) × expected_m), 0)`（`generators.py:217`）——**天然不等长**，正是今天"两个不等长 group"练习的自动化版本；K-Grouped 的 `empty` 变体强制约 25% 的组为空且首组必空、末组必非空，`single` 变体只留中间一组非空（`:606-613`）。

### 3.3 练习一（M-Grouped）：跟踪一个 group 从 layout 到输出区间

#### 追踪配方（四步）

给定 `grouped_layout = psum[0..G-1]`（int32，累积行数）、对齐度 `kAlignment = 256`（`csrc/jit/jit.hpp:59`，launch 时传入 kernel，`csrc/jit_kernels/bf16_gemm.hpp:65`）和 `BLOCK_M`（M-Grouped 恒等于 256，见下）：

1. **组边界**：组 g 的有效行区间为 `[start_g, end_g)`，其中 `start_0 = 0`、`start_g = align(psum[g-1], 256)`、`end_g = psum[g]`；`[end_g, align(end_g, 256))` 是组间 padding（`scheduler.hpp:17-23` 注释）。
2. **组内 block 数**：`num_m_blocks_g = ceil((end_g - start_g) / BLOCK_M)`。因为 `start_g` 是 256 的倍数且 `BLOCK_M = 256 = kAlignment`（`scheduler.hpp:146` 的 `static_assert(kAlignment == BLOCK_M)`），组内 m-block 编号加上 `start_g / BLOCK_M` 就是无缝的全局 m-block 编号。
3. **block 编号区间**：组 g 占用的全局 block 编号为 `[cum_g, cum_g + num_m_blocks_g × num_n_blocks)`，其中 `cum_g` 是之前各组"组内 m-block 数 × num_n_blocks"的累积（`scheduler.hpp:148-155`）。
4. **输出区间**：D 与 A 同排列（都是 `[total_m, N]` 平坦缓冲），组 g 的输出行就是 `[start_g, end_g)`；padding 行的输出是 0（输入零 padding + 整块计算，3.5 节）。

#### 手算例：两个不等长 group

设 `G = 2`、两组实际行数 `[300, 700]`（`expected_m = 512` 只是生成器的期望值，不影响手工推导）。按 `generate_psum_layout`（`generators.py:212-220`）：

```text
组 0：layout[0] = 0+300 = 300；   下一组起点 = align(300, 256) = 512
组 1：layout[1] = 512+700 = 1216；  total_m = align(1216, 256) = 1280

grouped_layout = [300, 1216]   # int32, 长度 2
A: [1280, K]（K-major）  B: [2, N, K]  D: [1280, N]
```

全局 M 空间的划分（`··` 段为组间 padding）：

```text
M 轴（total_m = 1280）
  0────300   512────────────────1216   1280
  ├─ 组 0 ─┤··│───── 组 1 ─────────┤··│
  │ 300 行 │212│       700 行       │64│
  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
  组 0 输出行区间 [0, 300)        组 1 输出行区间 [512, 1216)
```

设 `N = 7168`、`BLOCK_N = 256` → `num_n_blocks = 28`；`BLOCK_M = 256`（钉死：`heuristics.hpp:104-106` 过滤 + `scheduler.hpp:146` 断言）。代入配方：

| 组 | 有效行 | padding 行 | 组内 m-block 数 | 全局 m-block | block 编号区间 |
|---|---|---|---|---|---|
| 0 | `[0, 300)` | `[300, 512)` | `ceil(300/256) = 2` | `{0, 1}` | `[0, 2×28) = [0, 56)` |
| 1 | `[512, 1216)` | `[1216, 1280)` | `ceil(704/256) = 3` | `{2, 3, 4}` | `[56, 56+84) = [56, 140)` |

注意组 0 的 m-block 1 覆盖行 `[256, 512)`——**前 44 行有效、后 212 行是 padding**，整块参与计算（`get_actual_m` 对 M-Grouped 恒返回 `BLOCK_M`，`scheduler.hpp:129-136`），正确性由输入零 padding 保证。

#### 推演 core 0 的 `get_next_block` 序列

设 `kNumCores = 32`（`common.hpp:51`），core 0 依次领到 `block_idx = 0, 32, 64, 96, 128, 160, …`（`scheduler.hpp:142`）。对照 m-grouped 分支（`:144-162`）逐次推演：

| 调用 | block_idx | 组 | 组游标动作 | in_group_idx | 组内 (m, n) | 全局 m_block | 行区间 | B 取哪组 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 构造时已初始化（`current_psum_m=300, num_m_blocks=2`，`:99-102`） | 0 | (0, 0) | 0 | `[0, 256)` | `B[0]` |
| 2 | 32 | 0 | 无 | 32 | (1, 4) | 1 | `[256, 512)` | `B[0]` |
| 3 | 64 | 1 | 推进：`last_psum_m=align(300,256)=512`，`current_psum_m=1216`，`current_m_block_cumsum=2`，`num_m_blocks=ceil(704/256)=3` | 8 | (0, 8) | 0+512/256=**2** | `[512, 768)` | `B[1]` |
| 4 | 96 | 1 | 无 | 40 | (1, 12) | 3 | `[768, 1024)` | `B[1]` |
| 5 | 128 | 1 | 无 | 72 | (2, 16) | 4 | `[1024, 1280)` | `B[1]` |
| 6 | 160 | — | 推进：`++current_group_idx == 2 == num_groups` → 返回 false | — | — | — | 结束 | — |

两个关键换算，请合上表格自己重推一遍：

- **`in_group_idx = block_idx − current_m_block_cumsum × num_n_blocks`**（`:159`）：调用 3 中 `64 − 2×28 = 8`；
- **`m_block_idx += last_psum_m / BLOCK_M`**（`:161`）：调用 3 中组内 m-block 0 平移到全局 2（`512/256`），对应行 `[512, 768)`——这就是"组 1 从 layout 到输出行区间"的调度落点。

组内坐标走 `get_swizzled_block`（`:202-248`）；本例两组的 `local_num_m_blocks`（2 和 3）都小于 `kGroupM = 4`，命中快速路径的行-major 映射（`m = idx / num_n_blocks, n = idx % num_n_blocks`，`:209-211`）。当某组 m-block 数 ≥ 4 时才会进入 swizzle 路径——手算时先检查这一条可以省很多事。

**B 的组偏移**：Kernel 拿到 block 后用 `b_group_idx = scheduler.get_group_idx()`（`bf16_gemm.hpp:100`）取 `b = gm_b.group(b_group_idx, shape_n, shape_k)`（`:103`），字节偏移 = `g × N × K × sizeof(bf16)`（`common.hpp:167-172`）。A 与 D 完全不加组偏移——它们本来就是平坦的。

### 3.4 练习二（K-Grouped）：跟踪一个 group 从 layout 到输出区间

#### 追踪配方（四步）

给定 `grouped_layout`（K 的前缀和）、`kAlignment = 256`、`BLOCK_K`、`M/N/BLOCK_M/BLOCK_N`：

1. **组边界**：组 g 的有效 K 区间 `[start_g, end_g)`，`start_g = align(psum[g-1], 256)`（组 0 为 0）、`end_g = psum[g]`；`total_k = align(psum[G-1], 256)`（`scheduler.hpp:25-31` 注释）。
2. **block 区间**：所有组**共享**同一个 M×N 网格，`num_blocks = ceil(M/BLOCK_M) × ceil(N/BLOCK_N)` 不随组变；每个**非空**组恰好消耗 `num_blocks` 个编号（`scheduler.hpp:163-177`）。
3. **K 循环参数**：组 g 的 `current_shape_k = end_g − start_g`、`k_idx_base = start_g`、`num_k_blocks = ceil(current_shape_k / BLOCK_K)`；第 i 块的 `k_idx = k_idx_base + i × BLOCK_K`、`actual_k = min(current_shape_k − i × BLOCK_K, BLOCK_K)`（`bf16_gemm.hpp:95-97, :112-114`）——padding 的 K 行**不参与**计算。
4. **输出区间**：D 是 `[G, M, N]`，组 g 对应整个 slab `D[g]`，字节偏移 = `g × M × N × sizeof(dtype)`（`bf16_gemm.hpp:186-188` 的 `gm_d.group(d_group_idx, shape_m, shape_n)`）；空组不消耗 block，非累加时由 AIV 的 `aiv_zero_empty_groups` 清零（`:175-176`；`epilogue.hpp:108-141`）。

#### 手算例：两个不等长 group

设 `G = 2`、两组实际 K `[500, 1000]`（`ks = [500, 1000]`）：

```text
组 0：layout[0] = 500；    下一组起点 = align(500, 256) = 512
组 1：layout[1] = 512+1000 = 1512；  total_k = align(1512, 256) = 1536

grouped_layout = [500, 1512]
A: [1536, M]（MN-major）  B: [1536, N]（MN-major）  D: [2, M, N]
```

```text
K 轴（total_k = 1536）
  0────500  512──────────────────1512  1536
  ├ 组 0 ─┤·│───── 组 1 ────────────┤··│
  │ K=500 │12│        K=1000        │24│
  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
  A/B 只取 [0,500) 与 [512,1512)；输出是 D[0]、D[1] 两份完整 slab
```

设 `M = 384`、`N = 4096`、`BLOCK_M = 128`、`BLOCK_N = 256`（K-Grouped 的 BLOCK_M 不受 256 钉死约束）→ `num_m_blocks = 3`、`num_n_blocks = 16`、`num_blocks = 48`。代入配方：

| 组 | 有效 K | padding K | block 编号区间 | current_shape_k | k_idx_base | sf_idx_base |
|---|---|---|---|---|---|---|
| 0 | `[0, 500)` | `[500, 512)` | `[0, 48)` | 500 | 0 | 0 |
| 1 | `[512, 1512)` | `[1512, 1536)` | `[48, 96)` | 1000 | 512 | 512/64 = 8 |

两组的 K 循环参数（`BLOCK_K = 256`）：

| 组 | num_k_blocks | (k_idx, actual_k) 序列 | 覆盖的 A/B 行 |
|---|---|---|---|
| 0 | `ceil(500/256) = 2` | (0, 256), (256, **244**) | `[0, 500)`，不触碰 `[500, 512)` |
| 1 | `ceil(1000/256) = 4` | (512, 256), (768, 256), (1024, 256), (1280, **232**) | `[512, 1512)`，不触碰 `[1512, 1536)` |

#### 推演 core 0 的 `get_next_block` 序列

总 block 数 = 2 × 48 = 96，core 0 依次领 `0, 32, 64, 96, …`。对照 k-grouped 分支（`scheduler.hpp:163-177`）：

| 调用 | block_idx | 组 | 组游标动作 | in_group_idx | (m, n) | 输出落点 | K 循环 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 构造时已初始化（`current_psum_k=500, current_shape_k=500`，`:103-107`） | 0 | (0, 0) | `D[0]` 行 `[0,128)` 列 `[0,256)` | 2 块，覆盖 `[0,500)` |
| 2 | 32 | 0 | 无 | 32 | (2, 0) | `D[0]` 行 `[256,384)` 列 `[0,256)` | 同上 |
| 3 | 64 | 1 | 推进：`last_psum_k=align(500,256)=512`，`current_psum_k=1512`，`current_k_block_cumsum=48`，`current_sf_k_cumsum=8`，`current_shape_k=1000` | 16 | (1, 0) | `D[1]` 行 `[128,256)` 列 `[0,256)` | 4 块，覆盖 `[512,1512)` |
| 4 | 96 | — | 推进：`++current_group_idx == 2 == num_groups` → 返回 false | — | — | 结束 | — |

三个关键换算：

- **`in_group_idx = block_idx − current_k_block_cumsum`**（`:178`）：调用 3 中 `64 − 48 = 16`；
- **M/N 坐标不平移**：`get_swizzled_block(in_group_idx, num_m_blocks, …)` 用的是全网格（`:179`）——M/N 坐标在所有组里含义一致；
- **组差异只体现在 K 与基址**：`eff_shape_k = scheduler.current_shape_k` 决定 K 循环长度（`bf16_gemm.hpp:95`），`k_idx = k_idx_base + k_blk_idx × BLOCK_K` 给**平坦的 A/B** 加 K 平移（`:113, :117-118`），`gm_d.group(g, …)` 给 D 加 slab 偏移（`:188`）。

本例 `num_m_blocks = 3 < kGroupM = 4`，组内坐标仍走行-major 快速路径，手算友好。对比 3.3 节：**M 维追踪的落点是"D 的行区间"，K 维追踪的落点是"D 的第 g 份 slab"**——这是两种分组里"输出区间"一词的不同含义。

### 3.5 测试如何知道算对了：验证手段与分组语义的对应

理解验证手段才能在测试失败时定位层级（第 31 天方法论在分组场景的延续）：

| 验证手段 | 源码位置 | 抓住的分组性质 |
|---|---|---|
| A 的 padding 行清零（数据与参考同时清） | `generators.py:338-340`（`zero_operand_slices_` `:194-205`，span 来自 `iter_psum_sep` `:230-235`） | M-Grouped 组尾 m-block **整块计算**，padding 行也产出结果——必须保证是 0 |
| SF 的 padding 行填**随机垃圾** | `generators.py:342-346`（K 维对应 `:384-389`） | 0 数据 × 任意 scale = 0，所以 SF 不需要清零；kernel 误读 SF 也只影响 padding 行 |
| `cmp_spans = None` → 全 `total_m` 行比较 | `generators.py:357`；`common.py:201`；`utils.py:46-58` | D 的 padding 行必须是精确 0（参考侧对应行也是 0），与第一条形成闭环 |
| K-Grouped 的 D 用 **NaN 预填** | `common.py:190` | 每个输出元素都必须被写：空组 slab 由 `aiv_zero_empty_groups` 清零（`epilogue.hpp:108-141`，调用点 `bf16_gemm.hpp:175-176`），任何漏写都会留下 NaN |
| `empty` / `single` 变体 | `generators.py:606-613` | 空组不消耗 block 但要产出 0；只有一组非空时其余组照样正确 |
| fp32 累加逐位对照 | `common.py:202-206`；`utils.py:61-67` | 直接 bf16 输出必须与"fp32 累加再 cast"逐位一致（`torch.equal` 级别） |
| `ks-cpu` 三档提示（list / none / empty） | `generators.py:634`；`gemm.hpp:35-43` | 提示只进 `expected_k_from_ks` 供启发式选配置，换档后结果必须完全一致 |
| aclnn 跳过 | `common.py:87-91, :210` | 分组算子无 aclnn 对照，参考值只有 `d_ref`（fp32 逐组矩阵乘） |
| 随机舍入对照 | `common.py:215-221` | bf16 输出的随机舍入路径在分组下也逐位正确 |

**perf 口径**：K-Grouped 在打印前把 `k` 替换为 `sum(test.ks_cpu)`（**真实**总 K，不含 padding，`common.py:223-224`）；M-Grouped 没有对应替换，`m` 是 generate 返回的 `total_m`（**含** padding 行，`generators.py:359`）。同一个"每用例一行 perf"（`utils.py:106-114`，去掉 `--skip-prof` 后由 `bench_msprof` 测量，`common.py:228`），两种分组的 FLOPS 分母口径不同——比较性能数据时要记住这一点。仓库 `README.md` 的 M-Grouped 性能表（`:134-147`）就是用这套口径测的（4/8 组 × 8192/4096 行每组等 MoE 形状）。

### 3.6 MoE 对应关系（通过标准的答案骨架）

计划要求："给定一个 MoE 输入，能说明 token、group、expert 权重和输出的对应关系。"用一个 mini 例子把两个方向都走一遍。

**前向（M-Grouped）**。设 6 个 token、2 个 expert，路由结果 `t0→e1, t1→e0, t2→e0, t3→e1, t4→e0, t5→e1`。Dispatch 按 expert 排序后：

```text
   A [total_m, K]（K-major，行 = token）         B [G, N, K]
  ┌───────────────────────┐                   ┌──────────────┐
  │ 组 0 (e0): t1  t2  t4  │                   │ B[0]=W_{e0}  │ expert 0 权重
  ├───────────────────────┤                   ├──────────────┤
  │ 组 1 (e1): t0  t3  t5  │                   │ B[1]=W_{e1}  │ expert 1 权重
  └───────────────────────┘                   └──────────────┘
  grouped_layout = [3, 6]（示意值；真实调用中组起点须对齐 256，
                    组 0 末尾会插入 253 行零 padding，total_m = align(6,256) = 256）

  D [total_m, N]：行 0..2 = t1,t2,t4 过 e0 的结果；行 3..5 = t0,t3,t5 过 e1 的结果
  行 i 的公式：D[i,:] = A[i,:] @ B[group(i)]^T，group(i) = 行 i 所在的组
  Combine：按路由表把 D 的行取回各 token 的原始位置（kernel 之外的上层职责）
```

**反向 wgrad（K-Grouped）**。同一个 dispatch 结果下，expert e 的权重梯度是 `dW_e = dY_e^T @ Act_e`，收缩维恰好是"路由到 e 的 token"。把 dY 与 Act 都按 expert 分段拼接成平坦缓冲：

```text
  A [total_k, M]（行 = token 的 dY，按 expert 分段）  B [total_k, N]（行 = 同批 token 的 Act）
       组 0 (e0): K 段 = t1, t2, t4                      同一段 K
       组 1 (e1): K 段 = t0, t3, t5                      同一段 K
  grouped_layout = [3, 6]（同样省略对齐 padding）

  D [G, M, N]：D[0] = A_{[0,3)}^T @ B_{[0,3)} = dW_{e0}；D[1] = dW_{e1}
  每份 slab 直接按 expert 索引，无需逆排列
```

对应关系总表：

| MoE 概念 | M-Grouped（前向） | K-Grouped（反向 wgrad） |
|---|---|---|
| token | A 的一行（dispatch 后顺序） | A、B 的一行（K 维上按 expert 分段） |
| expert | 组 g = 行区间 `[start_g, end_g)` | 组 g = K 区间 `[start_g, end_g)` |
| expert 权重 | 输入 `B[g]` `[N, K]` | 输出 `D[g]` `[M, N]`（权重梯度） |
| 输出 | D 的行区间 `[start_g, end_g)`，顺序同 A | D 的第 g 份完整 slab |
| 恢复原序 | Combine 逆排列（上层完成） | 直接按 expert 索引 |

测试用例的形状印证这一场景：M-Grouped 枚举 `(G, expected_m)` 到 `(8, 4096)`、N/K 取 `(7168, 3072)` 等（`generators.py:597-598`，即 README 性能表的 MoE expert 形状）；K-Grouped 的 `(4, 4096, 7168, 8192)` 等配置注释为 EP64/EP32/EP16（`:620-626`）——EP 度数决定每 rank 的 expert 数（= 组数 G）。

## 四、源码导读

今天以"核对推演"为目的读码：先自己算，再到源码里找对应的行。建议按表中顺序读：

| # | 文件 | 关键函数/对象 | 核对什么 |
|---|---|---|---|
| 1 | `tests/test_bf16_gemm.py` | CLI `:25-26`、mgrouped/kgrouped `:39-42` | 命令如何映射到 `GemmType` |
| 2 | `tests/generators.py` | `generate_psum_layout` `:212-220`、`iter_psum_slice` `:223-227`、`iter_psum_sep` `:230-235`、`zero_operand_slices_` `:194-205` | 手算的 layout 与 3.3/3.4 的区间一致 |
| 3 | `tests/generators.py` | `_generate_m_grouped_contiguous` `:326-359`（零 padding `:338-340`、SF 垃圾 `:342-346`、参考输出 `:348-353`、cmp_spans `:357`）、`_generate_k_grouped_contiguous` `:361-399` | 参考输出按"有效区间"逐组构造 |
| 4 | `tests/generators.py` | `_enumerate_m_grouped_contiguous` `:579-603`、变体 `:606-613`、`_enumerate_k_grouped_contiguous` `:616-649` | 25 / 288 的组合来源 |
| 5 | `tests/common.py` | `run_gemm_test` `:131-236`（meta 预编译 `:155-175`、NaN 预填 `:190`、断言 `:199-206`、perf 口径 `:223-228`）、`gemm_dg` 分组分支 `:67-82`、`gemm_aclnn` 跳过 `:87-91` | 第二遍执行流与 3.2 节流程图一致 |
| 6 | `deep_gemm/include/deep_gemm/scheduler.hpp` | 类注释 `:17-35`、构造 `:87-108`、`get_next_block` m 分支 `:144-162` / k 分支 `:163-177`、`get_swizzled_block` 快速路径 `:209-211` | 推演表中的每个换算 |
| 7 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | scheduler 初始化 `:73-74`、`eff_shape_k/k_idx_base` `:95-97`、`b_group_idx` 与 `b.group` `:100-103`、K 循环 `:112-118`、D 组偏移 `:157, :186-188`、空组清零 `:175-176` | 组号的三个消费点（B / A·B 平移 / D） |
| 8 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `aiv_zero_empty_groups` `:108-141` | 空组 slab 清零的 tile 划分 |
| 9 | `deep_gemm/include/deep_gemm/common.hpp` | `gm_ptr::index` `:156-165`、`group` `:167-172`、`MX_SF_DIVISOR` `:65`、`kNumAICores` `:51` | 地址计算与 32 核假设 |
| 10 | `csrc/apis/gemm.hpp` | `expected_k_from_ks` `:35-43`、M/K-Grouped case `:88-96`、GemmDesc 填充 `:164-196`（grouped_layout `:184`、expected_m/k `:186-187`） | Host 侧进了哪些分组信息 |
| 11 | `csrc/jit_kernels/heuristics.hpp` | m-grouped 过滤 `:104-106`、m_range 钉死 `:610-616`、k-grouped 估算 `:597-599` | BLOCK_M 钉死 256（M 维）与自由（K 维） |
| 12 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78`（kAlignment `:65`、launch 传参 `:73-77`） | grouped_layout 指针与 256 如何进 kernel |

**逐段导读一：`Scheduler` 两个分组分支**（行号见 `scheduler.hpp`）。m-grouped（`:144-162`）：`while` 循环用 `(current_m_block_cumsum + num_m_blocks) × num_n_blocks` 判断 block_idx 是否落入当前组，否则推进组并更新四个游标（`last_psum_m`、`current_psum_m`、`current_m_block_cumsum`、`num_m_blocks`）；随后 `in_group_idx` 减去累积量、组内 swizzle、`m_block_idx += last_psum_m / BLOCK_M` 平移。k-grouped（`:163-177`）：`current_shape_k > 0 and block_idx < current_k_block_cumsum + num_blocks` 双重守卫——空组因 `current_shape_k == 0` 直接被跳过且不累加 `current_k_block_cumsum`（`:174-175` 注释说明）；推进时多更新一个 `current_sf_k_cumsum = last_psum_k / 64`（`MX_SF_DIVISOR`）。两处都在 `++current_group_idx == num_groups` 时返回 false。核心不变式见 `:33-35` 注释：每个核的 block_idx 单调递增，所以 cumsum 游标只需前进、不用回退。

**逐段导读二：`_generate_m_grouped_contiguous` 的参考输出**（`generators.py:348-353`）。`d` 先整体置零，再对每个组的有效 span（`iter_psum_slice` 产出 `[0,300)`、`[512,1216)` 这样的切片）填入 `a_repr[ms, :] @ b_repr[i].T`——参考输出本身就把"组 → 输出行区间"写成了代码，是 3.3 节配方的可执行版本。K 维对应 `:391-397`：`d[i] = c[i] + a_repr[ks].T @ b_repr[ks]`。

**逐段导读三：`aiv_zero_empty_groups`**（`epilogue.hpp:108-141`）。遍历 `grouped_layout`，用 `k_begin == k_end` 判空（`k_begin = aligned(k_end, kAlignment)` 维护对齐起点）；发现空组后所有 AIV 协作，把一个全零 UB tile 按 `MAD_M × MAD_N` 网格铺满 `d.group(group_idx, shape_m, shape_n)`。这解释了 NaN 预填为何能抓漏写。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通 M-Grouped 测试**：

   ```bash
   python tests/test_bf16_gemm.py mgrouped --skip-prof
   ```

   确认 `Num tests:  25`。随机挑 3 行 ` > Perf`/`OK` 输出，还原它的 `(G, expected_m, N, K, major_b)` 与 `total_m`（m 列的值），再回答：该用例随机生成的每组行数大约是多少（`uniform(0.7, 1.3) × expected_m`，`generators.py:217`）？

2. **跑通 K-Grouped 测试**：

   ```bash
   python tests/test_bf16_gemm.py kgrouped --skip-prof
   ```

   确认 `Num tests:  288`。挑 3 行输出，说出它的变体（normal / empty / single）与 `ks-cpu` 提示档位，并指出该用例重点考察什么（empty → 空组清零；`ks-cpu=none` → 提示缺省；`acc=1` → fp32 累加路径）。

3. **观察配置差异**：

   ```bash
   export DG_PRINT_CONFIGS=1
   python tests/test_bf16_gemm.py mgrouped --skip-prof
   python tests/test_bf16_gemm.py kgrouped --skip-prof
   ```

   核对：M-Grouped 打印的 `GemmConfig` 中 `block_m` 恒为 256（`heuristics.hpp:104-106, :610-616`），K-Grouped 的 `block_m` 可以是其他值；`GemmDesc` 打印带 `num_groups` 与 `expected_m`/`expected_k`（`gemm_config.hpp:152-155`）。再对比同一形状下 `ks-cpu=list/none/empty` 三档的 `GemmConfig` 是否一致，体会"提示只影响配置选择"。

4. **M-Grouped 输出区间实验**（3.3 节手算例的可执行版）：

   ```python
   import torch, torch_npu
   import deep_gemm

   N, K = 512, 1024
   sizes = [300, 700]                              # 两个不等长 group
   layout, psum = [], 0
   for m in sizes:
       layout.append(psum + m)
       psum = (psum + m + 255) // 256 * 256
   total_m = psum                                  # 1280
   g_layout = torch.tensor(layout, dtype=torch.int32, device='npu')
   print(layout, total_m)                          # [300, 1216] 1280

   a = torch.randn(total_m, K, dtype=torch.bfloat16, device='npu')   # K-major
   a[300:512] = 0                                  # 组 0 padding 清零（API 契约）
   a[1216:1280] = 0                                # 组 1 padding 清零
   b = torch.randn(2, N, K, dtype=torch.bfloat16, device='npu')      # B[g] = expert g 权重
   d = torch.full((total_m, N), float('nan'), dtype=torch.bfloat16, device='npu')

   deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, g_layout)
   torch.npu.synchronize()

   # (1) padding 行必须被写成精确 0，而不是残留 NaN
   print(torch.all(d[300:512] == 0), torch.all(d[1216:1280] == 0))   # True True
   # (2) 有效行等于对应 expert 的乘积（累加顺序差异 → 用余弦 diff，阈值 1e-5，
   #     与 tests/utils.py:36-43 的 calc_diff 同式）
   def diff(x, y):
       x, y = x.float(), y.float()
       return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()
   ref0 = (a[0:300].float() @ b[0].float().T).to(torch.bfloat16)
   ref1 = (a[512:1216].float() @ b[1].float().T).to(torch.bfloat16)
   print(diff(d[0:300], ref0), diff(d[512:1216], ref1))               # 都 < 1e-5
   ```

   把 `sizes` 改成 `[300, 0, 700]` 重跑（layout 变为 `[300, 512, 1216]`）：空组占 0 行、`total_m` 不变，验证"空组在 M 维自动消失、无需清零"（与下面 K 维空组对照）。

5. **K-Grouped 输出区间实验**（3.4 节手算例的可执行版）：

   ```python
   M, N = 512, 256
   ks = [500, 1000]                                # 两个不等长 group
   layout, psum = [], 0
   for k in ks:
       layout.append(psum + k)
       psum = (psum + k + 255) // 256 * 256
   total_k = psum                                  # 1536
   g_layout = torch.tensor(layout, dtype=torch.int32, device='npu')   # [500, 1512]

   a = torch.randn(total_k, M, dtype=torch.bfloat16, device='npu')    # [K, M]，MN-major
   b = torch.randn(total_k, N, dtype=torch.bfloat16, device='npu')
   a[500:512] = 0; a[1512:1536] = 0                # 组间 padding 清零
   b[500:512] = 0; b[1512:1536] = 0
   d = torch.full((2, M, N), float('nan'), dtype=torch.bfloat16, device='npu')

   deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, g_layout)
   torch.npu.synchronize()

   ref0 = (a[0:500].float().T @ b[0:500].float()).to(torch.bfloat16)
   ref1 = (a[512:1512].float().T @ b[512:1512].float()).to(torch.bfloat16)
   print(diff(d[0], ref0), diff(d[1], ref1))       # 都 < 1e-5

   # 空组变体：组 0 为空，D[0] 应被清零而非残留 NaN
   ks2, layout2, psum2 = [0, 1000], [], 0
   for k in ks2:
       layout2.append(psum2 + k)
       psum2 = (psum2 + k + 255) // 256 * 256
   g2 = torch.tensor(layout2, dtype=torch.int32, device='npu')        # [0, 1000]
   d2 = torch.full((2, M, N), float('nan'), dtype=torch.bfloat16, device='npu')
   deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d2, ks2, g2)
   torch.npu.synchronize()
   print(torch.all(d2[0] == 0))                    # True（aiv_zero_empty_groups 的效果）
   ```

6. **（可选）性能观察**：去掉 `--skip-prof` 重跑 `mgrouped`，把 2～3 个用例的 `m, n, k, TFLOPS, GB/s` 记入表格，与 `README.md:134-147` 的 M-Grouped 表对照量级（注意自己用例的 `m` 是含 padding 的 `total_m`）。

### B 组：无 Ascend 环境（静态任务）

1. **psum 手算**：设 `G = 3`、各组实际行数 `[100, 0, 700]`、对齐 256。写出 `grouped_layout`（应为 `[100, 256, 956]`）、`total_m`（1024）、每组的有效/padding 行区间、每组的 `last_psum_m / current_psum_m / num_m_blocks`（组 2 应为 256/956/3），并指出哪个组是空组、它占几个 block（0 个）。对照 `generate_psum_layout`（`generators.py:212-220`）与 `iter_psum_slice`/`iter_psum_sep`（`:223-235`）核对。
2. **M-Grouped 调度推演**：用 3.3 节的例子（`grouped_layout=[300,1216]`、`N=7168`、`BLOCK_N=256`、`kNumCores=32`），不看正文表格，独立推演 **core 5** 的 `get_next_block` 序列（block_idx = 5, 37, 69, 101, 133，共 5 块；第 3 次调用时应推进到组 1 且 `in_group_idx=13`，最终全局 m_block 依次为 0, 1, 2, 3, 4）。再回答：core 20 领几块（4 块：20, 52, 84, 116）？为什么不同 core 的块数可以不同（`block_idx = iter×32+core_id` 超出 140 即结束）？
3. **K-Grouped 推演**：用 3.4 节的例子（`grouped_layout=[500,1512]`、`M=384, N=4096, BLOCK_M=128, BLOCK_N=256`），独立推演 **core 13** 的序列（block_idx = 13, 45, 77；前两块属组 0，第三块属组 1，`in_group_idx` 分别为 13, 45, 29，落点都是各自 slab 的 n_block 13 列区）。再推演空组变体 `grouped_layout=[0,1000]` 下 **core 0** 的调用序列（只有两次成功调用且都落在组 1：第一次调用时构造初始化的组 0 因 `current_shape_k == 0` 立即被跳过，`current_k_block_cumsum` 保持 0；第三次调用 `block_idx=64` 已越出 48 → 返回 false；`D[0]` 不被 AIC/AIV 碰，由 `aiv_zero_empty_groups` 清零）。
4. **地址计算**：bf16、`G=2`。沿用 3.3 例（`N=7168, K=1024`）：(a) M-Grouped 中组 1 的 B 偏移 `group(1, 7168, 1024)` = `1×7168×1024×2 = 14,680,064` 字节。沿用 3.4 例（`M=384, N=4096, total_k=1536`）：(b) K-Grouped 中组 1 的 A 平移 = `k_idx_base × stride_outer × sizeof = 512×384×2 = 393,216` 字节（MN-major，`stride_outer = M`）；(c) D 的 slab 偏移 = `1×384×4096×2 = 3,145,728` 字节；(d) fp8 情形下 SF 的组偏移 = `sf_idx_base × shape_m × sizeof(int16) = 8×384×2 = 6,144` 字节（`MX_SF_DIVISOR=64`，`common.hpp:65`）。
5. **mini-MoE 手工对应**（通过标准演练）：按 3.6 的 6-token 例子，写出 dispatch 后 A 的行序（t1, t2, t4, t0, t3, t5）、每个 token 属于哪个组/哪个 expert、`B[0]/B[1]` 分别是谁的权重、D 每行是谁的输出；再写出 wgrad 方向下 `D[0]/D[1]` 各是哪个 expert 的权重梯度。合上教程对同伴（或纸面）复述一遍。
6. **代码对照**：通读 `scheduler.hpp:144-177` 与 `bf16_gemm.hpp:95-118, :157, :175-190`，在纸上标出分组信息的流转：`grouped_layout`（GM 上的 int32 数组，经 `csrc/jit_kernels/bf16_gemm.hpp:77` 传入）→ Scheduler 游标 → `get_group_idx()/current_shape_k/k_idx_base` → B 的 `group()` / A·B 的 `index()` 平移 / D 的 `group()` / `aiv_zero_empty_groups`。数一数组号分别"转手"几次，与 Normal GEMM 对比。

## 六、当天产出

一份**"group → 输出区间"追踪记录**，至少包含：

1. 两个手算例的布局推导（`grouped_layout`、`total_m/total_k`、每组有效/padding 区间、block 编号区间、K 循环参数）；
2. 一张 core 视角的 `get_next_block` 推演表（M 维与 K 维各一张，含组游标推进时刻与 `in_group_idx` 换算）；
3. MoE 对应关系说明（3.6 节表格的自己版本）——这是**通过标准**的书面答案；
4. 有环境的同学附上：`mgrouped`/`kgrouped` 的运行记录（`Num tests` 与 2～3 行输出还原）、A 组第 4/5 步实验的验证结果，以及 `DG_PRINT_CONFIGS=1` 观察到的配置差异。

写完后把画不出/算不出的格子标出来，回到第四节对应站点重读。

## 七、自测题

1. `mgrouped` 与 `kgrouped` 两条命令分别枚举多少用例？数量由哪些维度组合而来？
   > 25 = 1 个最小用例 + 3 种 (G, expected_m) × 4 种 (N, K) × 2 种 major_b（`generators.py:579-603`）；288 = 8 形状 × 3 变体 × 4 输出选项 × 3 档 ks 提示（`:616-649`）。每个用例还会派生 2～3 个 kernel 变体（`common.py:165-174`）。
2. 给定 `grouped_layout = [300, 1216]`（M 维、BLOCK_M=256），组 1 的有效行区间、padding 区间、起始全局 m_block 与 block 数量各是多少？
   > 有效 `[512, 1216)`、padding `[1216, 1280)`、起始全局 m_block = `512/256 = 2`、组内 3 个 m-block；block 数量 = 3 × num_n_blocks。
3. "跟踪一个 group 从 layout 到输出区间"在 M 维与 K 维分别追踪什么落点？
   > M 维：D 的行区间 `[align(psum[g-1],256), psum[g])`（D 与 A 同排列，平移量为 `last_psum_m/BLOCK_M`）；K 维：D 的第 g 份完整 slab `D[g]`（`gm_d.group(g, M, N)`，偏移 `g×M×N×sizeof`）。
4. 某个 core 何时推进组游标？推进时 M 维更新哪四个量、K 维更新哪五个量？
   > 当 `block_idx` 越出当前组的 block 范围：M 维是 `(cum_m + num_m_blocks)×num_n_blocks`，K 维是 `current_shape_k > 0 and block_idx < cum_k + num_blocks`。M 维更新 `current_group_idx/last_psum_m/current_psum_m/current_m_block_cumsum/num_m_blocks`（组号+四个游标）；K 维更新组号 + `last_psum_k/current_psum_k/current_k_block_cumsum/current_sf_k_cumsum/current_shape_k`（`scheduler.hpp:150-155, :169-176`）。
5. M-Grouped 组尾的 padding 行为什么可以整块计算而不出错？测试如何配合？
   > `get_actual_m` 对 M-Grouped 恒返回 BLOCK_M（`scheduler.hpp:129-136`），整块计算含 padding 行；输入侧由 `zero_operand_slices_` 把 padding 行清零（数据与参考同时清，`generators.py:338-340`），0 × 权重 = 0，输出 padding 行自然为 0；比较时 `cmp_spans=None` 对全 `total_m` 行断言（`:357`）。SF 的 padding 行填垃圾也无妨（0 数据 × 任意 scale = 0，`:342-346`）。
6. K-Grouped 空组的输出由谁写？测试靠什么手段发现漏写？
   > 非累加时由 AIV 的 `aiv_zero_empty_groups` 把整个 `[M, N]` slab 铺零（`epilogue.hpp:108-141`，调用点 `bf16_gemm.hpp:175-176`）；测试把 D 预填 NaN（`common.py:190`），任何未被写的元素会保留 NaN 而被 `assert_equal` 抓住。累加模式下空组保持 `C[g]`（调用方先 `d.copy_(c)`），无需清零。
7. 两种分组的 perf 行口径有何不同？
   > K-Grouped 在打印前把 k 替换为 `sum(test.ks_cpu)`（真实总 K，`common.py:223-224`）；M-Grouped 的 m 是含 padding 的 `total_m`（`generators.py:359` 直接返回 padded 尺寸）。FLOPS 分母一个不含 padding、一个含 padding，横向比较时要注意。
8. （通过标准）给定 MoE 输入：token 序列、top-k 路由、各 expert 权重与 `grouped_layout`，说明前向和反向 wgrad 中 token、group、expert 权重与输出的对应关系。
   > 前向（M-Grouped）：dispatch 把 token 按 expert 排序拼接成 A 的行，组 g 的行区间 `[start_g, end_g)` 全部属于 expert g，`B[g]` 是它的权重，输出行区间与 A 行区间一致，Combine 按路由表逆排列。反向 wgrad（K-Grouped）：dY 与 Act 按 expert 分段拼接成 A/B 的 K 维，组 g 的 K 区间对应路由到 expert g 的 token，输出 `D[g]` 是该 expert 的权重梯度 `[M, N]`，直接按 expert 索引。参见 3.6 节图与表。
9. `ks-cpu` 提示的三个档位改变什么、不改变什么？为什么换档后测试结果必须一致？
   > 只改变 Host 侧 `expected_k_from_ks` 算出的 `GemmDesc.expected_k`（均值、下限 1，`gemm.hpp:35-43`），进而影响启发式选的 `GemmConfig`；设备侧唯一权威是 `grouped_layout`，kernel 不读 ks 提示，所以三档的正确性结果必须逐位一致（`common.py` 对三档都做同样的 `assert_equal`）。
10. 为什么 M-Grouped 的 BLOCK_M 被钉死 256，而 K-Grouped 自由选择？
    > M 维的组边界必须与 BLOCK_M 网格重合才能用 `last_psum_m / BLOCK_M` 把组内 m-block 平移到全局（要求整除，`scheduler.hpp:146` 的 `static_assert(kAlignment == BLOCK_M)`）；启发式相应地只保留 `block_m == 256` 的候选（`heuristics.hpp:104-106`，搜索范围钉在 `:610-616`）。K 维的组边界在收缩维上，与 M/N block 网格正交，组尾由 `actual_k` 截断兜底，因此 BLOCK_M 可自由搜索。

---

明天（第 39 天）进入 MQA Logits：阅读 `tests/test_attention.py`、`csrc/apis/attention.hpp` 与 `deep_gemm/include/deep_gemm/mqa_logits.hpp`，理解 Q、KV、weights 的形状与 logits 的计算流程，并对比它与普通 GEMM 的调度及 Epilogue 差异。



