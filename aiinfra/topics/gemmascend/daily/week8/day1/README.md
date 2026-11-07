# 第 8 周 · 第 1 天（总第 50 天）：建立性能分析基础

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 45 分钟，自测题约 15 分钟。

前七周我们都在回答"**算得对不对、代码怎么组织**"：调度、搬运、MAD、流水线、Epilogue、Grouped GEMM、MQA、MegaMoE。最后一周换一个问题——"**算得快不快、为什么快/不快**"。今天先建立性能分析的公共语言：性能测量的对象是什么（kernel 的设备执行时间，而不是墙钟时间）、三个核心指标怎么算（TFLOPS、有效带宽、算术强度）、如何判别受限类型（计算受限 / 访存受限 / 流水线受限），以及两个最容易污染测量结果的因素（冷/热缓存、JIT 编译耗时）。所有内容都锚定在本仓库自己的测量工具链上：`deep_gemm/testing/bench.py` 的 `bench_msprof` + `KernelProfile`，`tests/utils.py` 的 `count_bytes` + `print_perf`，以及 `README.md` 的 Performance 数据表。当天产出是**一份性能指标公式与解释笔记**（参考框架见第六节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 `bench_msprof`（`deep_gemm/testing/bench.py:284-400`）的完整测量流程：预热 → 冷缓存准备 → 采集窗口 → PMU 记录解析 → 跨迭代平均成 `KernelProfile`，并解释为什么每一步都不让 JIT 编译耗时和启动开销混进结果；
2. 手写三个指标的公式并用 `KernelProfile.tflops()` / `gbps()`（`bench.py:51-55`）验证：TFLOPS = FLOPs / 时间、有效带宽 = 逻辑搬运字节数 / 时间、算术强度 AI = FLOPs / 字节数，其中 FLOPs = 2·M·N·K（`tests/common.py:233`）、字节数由 `count_bytes`（`tests/utils.py:95-103`）按"每个张量只计一次"统计；
3. 用 Roofline 方法判别一个 shape 是计算受限还是访存受限：算出 AI，与"机器平衡点 = 峰值 TFLOPS / 峰值带宽"比较；并结合 `KernelProfile` 的流水线利用率字段（`aic_mad`、`aic_mte2`、`aic_fixpipe`、`aiv_vec` 等，`bench.py:32-45`）定位具体是哪条流水线饱和；
4. 解释冷缓存与热缓存的区别，以及 `bench_msprof` 如何用"2 倍 L2 大小的缓冲区清零"强制冷缓存（`_make_l2_flush_buffer`，`bench.py:263-268`；每次迭代前 `flush_buffer.zero_()`，`bench.py:344-346`），为什么 README 的性能口径注明 "with cold L2"（`README.md:102`）；
5. 区分 JIT 编译耗时与执行耗时：每个新配置触发一次模板实例化编译并进缓存（`DG_JIT_CACHE_DIR`，默认 `$HOME/.dj`，`README.md:90-91`），测试用 meta 张量 + `par_compile` 在测量前完成全部编译（`tests/common.py:152-176`），测量本身只取设备侧 PMU 时间戳；
6. 逐列读懂 README Performance 部分的四张表（Dense GEMM、FP8 推理、M-Grouped、MQA/MegaMoE/HC Prenorm，`README.md:100-185`），包括 Hardware limit、Utilization、Memory bandwidth、Communication bandwidth 各自的含义与来源；
7. 独立完成"shape → 指标 → 受限类型"的手工推演，并用 README 中已发表的数据验证自己的计算（例如 HC Prenorm 的 8192×24×28672 行）。

## 二、预备知识

- **第 1 天**：GEMM 计算量——`D = A × B`（A 是 M×K，B 是 K×N）共需 `2 × M × N × K` 次浮点运算（每对 (i,j) 做 K 次乘法和 K 次加法）。今天的 FLOPs 口径与此完全一致。
- **第 5 天**：Ascend 存储层次与数据流 `GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM`，以及 MTE1（L1→L0）、MTE2（GM→L1/UB）等搬运流水线的概念。今天 `KernelProfile` 的各字段正是按这些流水线统计利用率。
- **第 12 天**：JIT 配置、编译与启动——每个 `GemmConfig` 生成一份模板实例化代码，`jit->compile()` 编译并缓存、`jit->launch()` 启动。这是理解"编译耗时 vs 执行耗时"的基础。
- **第 22～25 天**：K 维两层循环、多级流水与同步、MAD 计算、AIC/AIV 分工与 Epilogue。性能分析要回答的"流水线受限"最终会落回这些代码结构上。
- **第 30～31 天**：测试数据生成与正确性验证流程——`calc_diff`（余弦相似度差异，`tests/utils.py:36-43`）、ACLNN 交叉验证、`--skip-prof` 的含义（`tests/test_bf16_gemm.py:26`："Skip performance measurements and just test correctness"）。今天的性能测量正是接在正确性验证之后的最后一步。
- **环境说明**：第五节 A 组任务需要 Ascend 环境（CANN 9.20 + torch_npu）；无环境时完成 B 组静态任务（公式手算、README 数据复核、测量流程标注、受限类型分类）即可达到当天目标。

## 三、详细讲解

### 3.1 性能测量的对象：为什么不能用墙钟时间

对 `time.time()` 前后一减得到"耗时"，在 GPU/NPU 上几乎是必然错的，原因至少有四个：

| 干扰源 | 说明 | `bench_msprof` 的对策 |
|---|---|---|
| JIT 编译 | 新配置首次调用要生成源码、调 Bisheng 编译、加载 `.o`，秒级耗时 | 测试先用 meta 张量并行预编译（`tests/common.py:156-175`），且预热循环在采集窗口之外 |
| Host 启动开销 | Python → C++ → `jit->launch()` 的主机侧耗时、异步队列 | 计时来自设备侧 PMU 记录的 `start_ns/end_ns`（`bench.py:177-179`），与 Host 无关 |
| 首次运行开销 | kernel 首次加载、指令 cache 冷启动 | `num_warmups=10` 次预热 + `torch.npu.synchronize()`（`bench.py:330-332`） |
| 窗口内其他 kernel | L2 flush、`npu_sleep` 等辅助 kernel 也在窗口里 | 用 `kernel_names` 按名字筛选（`bench.py:271-281`），docstring 明确提醒这一点（`:304-308`） |

因此测量对象被严格定义为：**单个 Device Kernel 一次执行的设备侧时长**。`bench_msprof` 的采集窗口里执行 `num_tests=30`（默认）次被测函数，PMU 为窗口内每个 kernel 任务留下一条记录（含 `start_ns`/`end_ns` 与各流水线计数器，`_PmuRecord`，`bench.py:128-191`），解析后按 kernel 名分组、跨迭代取平均（`_make_profile`，`bench.py:206-227`），得到一个 `KernelProfile`。测试里的调用形如：

```python
prof = bench_msprof(lambda: gemm_dg(test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab),
                    kernel_names='gemm_impl')        # tests/common.py:228
```

`'gemm_impl'` 是子串匹配——JIT 生成的 kernel 名是 `bf16_gemm_impl<...>`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:27`）这类长名，要求子串在窗口内**唯一**命中（`bench.py:274-279`），命中多个会直接报错。

三种后端（`bench.py:294`，默认 `'persistent'`）只是"怎么拿到 PMU 数据"的差别：`'slow'` 解析 msprof 导出的 `kernel_details.csv`（`_parse_kernel_details_csv`，`bench.py:449-508`，CSV 列名如 `Duration(us)`、`aic_mac_ratio` 等）；`'fast'` 解析一次性原始数据（`_parse_ffts_profile`，`bench.py:403-446`）；`'persistent'` 复用一个进程常驻的 PMU 会话（`_Session`，`bench.py:627-876`，用 `npu_sleep` 的两个模板实例 `sleep_implILj1E`/`sleep_implILj2E` 作窗口起止标记，`bench.py:512-513`）。今天只需要理解"无论哪个后端，最终都汇成一个 `KernelProfile`"。

### 3.2 三个核心指标：TFLOPS、有效带宽、算术强度

**指标一：TFLOPS（每秒万亿次浮点运算）。**

```text
FLOPs = 2 × M × N × K                （乘法和加法各计一次）
TFLOPS = FLOPs / 执行时间(秒)
```

仓库里的实现（`KernelProfile.tflops`，`bench.py:51-52`）：

```python
def tflops(self, flops: float) -> float:
    return flops / self.dur_ns / 1000     # dur_ns 是纳秒
```

单位推导：FLOPs / ns = 10⁹ FLOP/s = GFLOP/s，再除 1000 得 TFLOP/s。测试侧的调用把 `2.0 * m * n * k` 传进去（`tests/common.py:233`；HC Prenorm 同样传 `2.0 * m * n * k`，`tests/test_hyperconnection.py:94`）。

**指标二：有效带宽（GB/s）。**

```text
bytes = 输入 + 输出张量的总字节数（每个张量只计一次）
GB/s = bytes / 执行时间(秒)
```

实现（`KernelProfile.gbps`，`bench.py:54-55`）就是 `bytes_moved / self.dur_ns`（B/ns 恰好等于 GB/s）。字节数由 `count_bytes` 统计（`tests/utils.py:95-103`）：

```python
def count_bytes(*tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)
```

普通 GEMM 传 `(a, b, c, d)`（`tests/common.py:234`），HC Prenorm 传 `(a, b, d, sqr_sum)`（`tests/test_hyperconnection.py:95`）。**注意这是一个"逻辑搬运量"模型**：它假设每个张量从 GM 到 kernel 只走一遍，不统计 Kernel 内部 Tile 复用导致的重复 GM 读（那些通常被 L2 吸收）。因此算出来的 GB/s 是"有效带宽"——单位时间内完成的**有用**数据搬运量，真实 DRAM 流量可能更高，但有效带宽不会超过硬件带宽上限。

**指标三：算术强度 AI（Arithmetic Intensity，FLOP/B）。**

```text
AI = FLOPs / bytes = (2 × M × N × K) / (A、B、C、D 字节数之和)
```

AI 回答的是"每搬 1 字节能榨出多少次运算"，它**只由 shape 和 dtype 决定，与快慢无关**，是连接另外两个指标的桥梁：`TFLOPS = AI × GB/s`。

**一个完全可复核的算例（HC Prenorm，README.md:184 的 8192 行）。** 输入输出见 `tests/test_hyperconnection.py:65-68`：`a` 是 bf16 `[m, k]`，`b` 是 fp32 `[n, k]`，`d` 是 fp32 `[m, n]`，`sqr_sum` 是 fp32 `[m]`。取 `m=8192, n=24, k=28672`：

```text
FLOPs = 2 × 8192 × 24 × 28672                   = 11,274,289,152 ≈ 11.27 GFLOP
bytes = 8192×28672×2 (a, bf16) = 469,762,048
      +   24×28672×4 (b, fp32) =   2,752,512
      + 8192×   24×4 (d, fp32) =     786,432
      + 8192×      4 (sqr_sum) =      32,768
                                    -----------
                                    473,333,760 B ≈ 473.3 MB
AI    = 11,274,289,152 / 473,333,760            ≈ 23.8 FLOP/B
```

README 报告该 shape 耗时 136.7 us、82 TFLOPS、3463 GB/s。用公式复核：

```text
TFLOPS = 11,274,289,152 / 136.7e-6 s / 1e12 ≈ 82.5   ✓（README：82）
GB/s   =   473,333,760 / 136.7e-6 s / 1e9  ≈ 3462   ✓（README：3463）
```

两个数都对上了——这说明 README Performance 表的口径就是本节的公式 + `count_bytes` 的字节模型 + `bench_msprof` 的设备侧计时。**手工验算是今天最重要的技能**：给定 shape 和 dtype，你应该能不碰机器就算出任何一行的 TFLOPS/GB/s/AI。

### 3.3 用 Roofline 判别受限类型

Roofline 模型给每个 shape 一个理论上限：

```text
时间下界 = max( FLOPs / 峰值TFLOPS ,  bytes / 峰值带宽 )
             └─ 计算屋顶 ─┘            └─ 带宽屋顶 ─┘

机器平衡点 = 峰值TFLOPS / 峰值带宽   （单位 FLOP/B）
AI > 平衡点 → 计算受限；AI < 平衡点 → 访存受限
```

```text
性能上限
(TFLOPS)
   ▲            ┌─────────────── 计算屋顶：峰值 TFLOPS
   │           /
   │          /
   │         /  带宽屋顶：上限 = 峰值带宽 × AI（斜线）
   │        /
   │       /
   └──────┴──────────────────────────► 算术强度 AI (FLOP/B)
        平衡点
```

本仓库能查到的两个"屋顶"参数：

- **峰值 TFLOPS**（README.md:108-113 的 Hardware limit 列，Ascend 950DT）：FP4×FP4 = 1730，FP8×FP8 / FP8×FP4 = 865，BF16×BF16 = 432。表里的 Utilization 列就是 `实测 TFLOPS / 硬件上限`：1701/1730 = 98.3%、861/865 = 99.5%、431/432 = 99.8%。
- **峰值带宽**：仓库未直接给出 HBM 峰值，但 README.md:177 说 HC Prenorm "nearly saturates the HBM write bandwidth"，对应实测 3463 GB/s（README.md:184）——可以取 ≈3.5 TB/s 作为估算基准（下文平衡点均为估算值，用于建立数量级感觉）。

由此得到各 dtype 的平衡点估算：BF16 ≈ 432/3.5 ≈ **123 FLOP/B**，FP8 ≈ 865/3.5 ≈ **247 FLOP/B**，FP4 ≈ 1730/3.5 ≈ **494 FLOP/B**。用两个已算过的 shape 检验：

| Shape | AI (FLOP/B) | 对比平衡点 | 结论 | README 佐证 |
|---|---|---|---|---|
| Dense BF16 4096×7168×16384 | ≈ 2249 | ≫ 123 | 计算受限 | 431/432 TFLOPS = 99.8%（README.md:113） |
| HC Prenorm 8192×24×28672 | ≈ 23.8 | ≪ 123 | 访存受限 | 3463 GB/s，"nearly saturates HBM"（README.md:184, :177） |

Dense BF16 的字节明细（A+B+D 均 bf16，无 bias）：`4096×16384×2 + 16384×7168×2 + 4096×7168×2 = 427,819,008 B`，AI = 962,072,674,304 / 427,819,008 ≈ 2249。N=7168、K=16384 这类"胖"shape 天然 AI 高——K 越大每字节被复用得越狠；而 HC Prenorm 的 N=24 极小，AI 被压到个位数/十位数，注定跑带宽。

**第三类：流水线受限。** 计算与访存之外，kernel 还可能卡在某条**特定流水线**上。README 给了现成例子——MQA Logits "The kernel is FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization"（README.md:151）：FIX pipe（L0C → 输出方向的 fixpipe 流水，`bench.py:39` 注释 `aic_fixpipe: L0C -> out`）被打满 99%，而算力只有几百 TFLOPS、远未到 865。判别依据不是 AI，而是 `KernelProfile` 的流水线利用率字段（见 3.4）。更广义地，若**没有任何一条流水线饱和**、指标离两个屋顶都远，则是调度/同步/并行度受限（例如 M 很小时 Tile 数不够，AI Core 吃不饱）。

**判别流程总结**（今天的核心方法论）：

```text
① 算 AI = 2MNK / bytes，与该 dtype 的平衡点比较 → 计算受限 or 访存受限（方向判断）
② 看流水线利用率：mad% ≈ 100%？mte2% ≈ 100%？fixpipe%/vec% ≈ 100%？ → 定位具体瓶颈
③ 都不满 → 检查并行度（Tile 数 vs AI Core 数）、同步间隙、尾部效应
```

### 3.4 从 KernelProfile 的流水线利用率定位瓶颈

`KernelProfile`（`bench.py:27-71`）除了 `dur_ns`，还携带两组利用率字段，全部是"该流水线活跃 cycles / 自身 total_cycles"的 [0,1] 比值（docstring，`bench.py:29-30`）：

| 字段 | 流水线 | 语义（`bench.py:32-45` 注释） | 饱和时说明 |
|---|---|---|---|
| `aic_mad` | Cube MAD | 矩阵乘加本体 | 计算受限（理想 GEMM） |
| `aic_mte2` | MTE2 | GM → L1 | 装载流水打满 → 访存受限 |
| `aic_mte1` | MTE1 | L1 → L0 | 片上二级搬运紧张 |
| `aic_fixpipe` | fixpipe | L0C → out | 结果回写/后处理受限（MQA 即此类） |
| `aic_scalar` | Scalar | 标量单元 | 地址/循环控制开销大 |
| `aiv_vec` | Vector | AIV 向量计算 | Epilogue/反量化向量算子受限 |
| `aiv_mte2` / `aiv_mte3` | MTE2/MTE3 | GM ↔ UB | AIV 侧访存 |

这些数字的来源是 PMU 硬件计数器：`_PmuRecord`（`bench.py:128-191`）为每个 kernel 任务记录 `vec/mad/scalar/mte1/mte2/mte3/fixpipe` 计数与 `total_cycles`，`to_pipe_sample()` 把计数除以总 cycles 得比率（`bench.py:181-191`）；`slow` 后端的 CSV 里对应列 `aic_mac_ratio`、`aic_mte1_ratio` 等（`bench.py:452-457`）。AIC 与 AIV 记录用 `is_aiv` 标志区分（`bench.py:164-165`），混合 kernel（`__mix__`，如 `bf16_gemm_impl`）两类都会有。

测试输出的 `print_perf`（`tests/utils.py:106-115`）把最关键的两个直接打出来：

```text
 > Perf (m=  4096, n=  7168, k= 16384, layout=NT, fp32, acc=0, ...): 2229.900 us |  431 TFLOPS |  192 GB/s | mad= 99.x% | mte2= 12.x%
```

（上行格式按 `print_perf` 的格式串构造，数值以实际运行为准。）一行就能读出结论：mad 接近满、mte2 低 → 计算受限，与 99.8% 利用率互为印证。对照第 23～25 天读过的流水线代码：`mad%` 对应 L0A/L0B → L0C 的计算推进，`mte2%` 对应 GM → L1 的多阶段搬运，`fixpipe%` 对应 AIC store 阶段——**性能数字最终要能落回代码结构**，这是第 52 天单变量实验的基础。

### 3.5 冷缓存与热缓存

L2 是 HBM（GM）与 AI Core 之间的设备级缓存。同一个 GEMM 连续跑 30 次，A/B 矩阵很可能常驻 L2——第 2 次以后的迭代读的是 L2 而非 HBM，测出的"带宽"反映的是 L2 带宽，**显著高于真实部署时的 HBM 行为**。对工作集小、访存受限的 kernel，热缓存测量会严重高估性能。

`bench_msprof` 默认强制**冷缓存**（`flush_l2: bool = True`，`bench.py:289`）：

```python
def _make_l2_flush_buffer():
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    num_bytes = 2 * properties.L2_cache_size          # 2 倍 L2 容量
    return torch.empty(num_bytes // 4, dtype=torch.int, device='npu')   # bench.py:263-268

def profile_fn():
    for _ in range(num_tests):
        if flush_buffer is not None:
            flush_buffer.zero_()                      # 每次迭代前把 L2 刷干净  bench.py:344-346
        ...
        fn()
```

原理：对一块 2 倍于 L2 容量的缓冲区做写清零，写分配会把其他数据全部挤出 L2，下一次 `fn()` 的输入只能从 HBM 重新搬。README 的性能口径明确注明这一点："Measured on Ascend 950DT (CANN 9.20) using `bench_msprof` with cold L2"（README.md:102）。

两点补充：

1. flush 这个动作本身也是一个 kernel，混在采集窗口里——这正是 `kernel_names` 筛选存在的原因之一（docstring："this includes the helper kernels launched inside the window (L2 flush, npu_sleep, ...)"，`bench.py:305-307`）。
2. 需要测热缓存行为时传 `flush_l2=False`，仓库里有现成例子：paged MQA 的 metadata 构建 kernel 用 `flush_l2=False`（`tests/test_attention.py:413`）——它的输入（`context_lens`、`block_table`）本来就小且刚被写过，刷 L2 反而不符合真实时序。

### 3.6 JIT 编译耗时与执行耗时

第 12 天已经看到：JIT 为每种配置生成一份模板实例化代码并编译缓存。性能测量的铁律是**编译一次、不计入；执行多次、只计设备时间**。本仓库三层保障：

1. **测量前集中预编译**：`run_gemm_test` 先用 meta 张量（`device='meta'`，不分配 NPU 存储，`tests/common.py:156-157`）把所有 case 的 kernel 生成出来，在 `dry_run`（`set_dry_run(True)`，只编译不启动，`deep_gemm/testing/par_compile.py:6-13`）下用线程池并行编译（`par_compile`，`par_compile.py:16-35`），然后才进入逐 case 的正确性验证与性能测量（`tests/common.py:152-236`）。
2. **预热迭代**：`bench_msprof` 在采集窗口外先跑 `num_warmups=10` 次 `fn()` 并 `torch.npu.synchronize()`（`bench.py:330-332`），排空首次加载/启动的额外开销；连辅助 kernel `npu_sleep` 都提前预热（`bench.py:329`）。
3. **设备侧计时**：时长取自 PMU 记录的 `start_ns/end_ns`（`bench.py:177-179`），天然不含 Host 侧编译、launch 的毫秒级噪声。

编译结果的缓存由 `DG_JIT_CACHE_DIR` 控制（默认 `$HOME/.dj`，支持 `:` 分隔多路径，`README.md:90-91`）：同一配置第二次运行直接命中缓存，不再编译。相关的观测开关（今天了解、明天使用）：`DG_JIT_PRINT_LOAD_TIME` 打印 kernel 加载耗时（`README.md:93-94`）、`DG_PRINT_CONFIGS` 打印每个 shape 选中的配置（`README.md:89`）。

反过来，这组区别也定义了**两类"慢"**：首跑慢（编译 + 加载，秒级，可通过缓存摊薄）与稳态慢（设备执行，微秒级，只能靠优化 kernel 改善）。做性能对比时必须用稳态值，否则比较的是编译器而不是 kernel。

### 3.7 README Performance 数据表导读

`README.md:100-185` 是本仓库性能结论的权威口径，逐表梳理：

| 表 | 位置 | 关键列 | 要点 |
|---|---|---|---|
| Dense GEMM | `:106-113` | Latency / Computation / **Hardware limit** / Utilization | 四种 dtype 在 4096×7168×16384 上达 98.3%～99.8% 硬件上限；Hardware limit 给出峰值 TFLOPS（1730/865/865/432），是 Roofline 计算屋顶的直接来源 |
| FP8 推理 | `:115-132` | Latency / TFLOPS / **Memory bandwidth** | 同一 (N, K) 下对比 M=128 与 M=4096 两行：M=128 行 TFLOPS 低、带宽高（如 549 TFLOPS / 2456 GB/s，`:123`），M=4096 行 TFLOPS 高、带宽低（861 / 186，`:130`）——用表内两列相除得 AI：M=128 ≈ 223 FLOP/B（低于 FP8 平衡点估算 247，偏访存侧），M=4096 ≈ 4629 FLOP/B（计算受限，861/865 = 99.5%） |
| M-Grouped | `:134-147` | 同上 | MoE expert 场景（`#Groups` × `M per group`），M 很大 → 计算受限（850+ TFLOPS） |
| MQA Logits | `:149-158` | 同上 | 明确标注 FIX-pipe bound、FIX pipe 99%（`:151`）——流水线受限的官方案例；Decode 行（256 Q tokens）带宽高达 2006 GB/s，Prefill 行则 TFLOPS 更高 |
| MegaMoE | `:160-173` | 多一列 **Communication bandwidth** | EP8、top-k=6、1 shared expert 口径（`:162`）；通信带宽由 `barrier_comm_profiling=True` 的 `bench_msprof` 测得（`tests/test_mega_moe.py:204`，通信前用 `npu_sleep(int(2e7))` 保持设备忙碌 20ms，`bench.py:347-349`） |
| HC Prenorm | `:175-185` | Latency / TFLOPS / Memory bandwidth | N=24 的瘦 shape，带宽逼近 HBM（"nearly saturates the HBM write bandwidth"，`:177`）；M 从 13 到 8192，带宽从 519 升到 3463 GB/s——M 越小并行度越不足，连带宽屋顶都摸不到 |

读表习惯：先看 dtype 定计算屋顶，再算/查 AI 定方向，最后用带宽列或 TFLOPS 列离屋顶的距离判断"还有多少优化空间"。例如 FP8 推理 M=128 行：549/865 ≈ 63% 算力、2456/3500 ≈ 70% 带宽，两头都不满——这类 shape 的瓶颈在并行度/调度（M=128 只能切出很少的 Tile），正是第 17 天调度器知识在性能侧的投影。

## 四、源码导读

| # | 文件 | 关键符号 | 职责 |
|---|---|---|---|
| 1 | `deep_gemm/testing/bench.py` | `KernelProfile` `:27-71`、`bench_msprof` `:284-400`、`_make_l2_flush_buffer` `:263-268`、`_select_profiles` `:271-281`、`_PmuRecord` `:128-191`、`_make_profile` `:206-227` | 测量工具链主体 |
| 2 | `tests/utils.py` | `count_bytes` `:95-103`、`print_perf` `:106-115`、`calc_diff` `:36-43` | 字节统计与 perf 行打印 |
| 3 | `tests/common.py` | `run_gemm_test` 的预编译段 `:152-176` 与 perf 段 `:223-236` | FLOPs/字节口径的权威来源 |
| 4 | `tests/test_hyperconnection.py` | `CASE_SLICES` `:17-27`、perf 调用 `:86-97`、张量构造 `:64-68` | 可手工复核 README HC Prenorm 表的入口 |
| 5 | `tests/test_bf16_gemm.py` | CLI `:25-26`（`--skip-prof`） | 正确性/性能两种运行模式 |
| 6 | `deep_gemm/testing/par_compile.py` | `dry_run` `:6-13`、`par_compile` `:16-35` | 测量前的并行预编译（dry-run 只编译不启动） |
| 7 | `README.md` | Performance `:100-185`、环境变量 `:85-98` | 性能口径与硬件上限 |

**`deep_gemm/testing/bench.py` 逐段导读**（全文 965 行，后半是 PMU 管道底层，今天了解即可）：

- **`:27-71`（`KernelProfile`）**：先读 docstring——"All pipe fields are ratios in [0, 1] relative to their own total_cycles"。字段注释本身就是一张流水线语义表（`mad` Cube MAD、`mte1` L1→L0、`mte2` GM→L1、`fixpipe` L0C→out……）。`dur_us`（`:47-49`）、`tflops`（`:51-52`）、`gbps`（`:54-55`）三个方法就是 3.2 节公式的实现；`__repr__`（`:57-71`）把非零流水线拼成 `KernelProfile(12.3us, mad=98.1%, ...)` 形式，读日志时按此解析。
- **`:284-317`（`bench_msprof` 签名与 docstring）**：默认 `num_warmups=10, num_tests=30, flush_l2=True, backend='persistent'`。docstring 值得逐行读：`kernel_names` 的三种匹配/返回方式（`:298-301`）、`return_all_kernels` 会包含辅助 kernel 的警告（`:304-308`）、三个后端的差别（`:310-311`）。
- **`:318-351`（测量主流程）**：参数校验 → `npu_sleep(0)` 预热辅助 kernel（`:329`）→ 预热循环 + 同步（`:330-332`）→ 构造 L2 flush 缓冲（`:334`）→ `profile_fn` 的 30 次迭代，每次先 `flush_buffer.zero_()` 再 `fn()`（`:342-351`）。这一段是 3.1/3.5 节流程图的直接出处。
- **`:263-268`（`_make_l2_flush_buffer`）**：`2 * properties.L2_cache_size` 的 int 张量——冷缓存机制的实现，仅 5 行。
- **`:271-281`（`_select_profiles`）**：子串匹配；str 参数要求唯一命中否则报错并列出窗口内全部 kernel 名——排障时这个报错信息就是"窗口里到底跑了哪些 kernel"的清单。
- **`:128-191`（`_PmuRecord`）**：PMU 原始记录的 ctypes 结构（128 字节/条），字段 `vec/mad/scalar/mte1/mte2/mte3/fixpipe` + `total_cycles` + `start_ns/end_ns`；`dur_ns = end_ns - start_ns`（`:177-179`）、`to_pipe_sample()` 把计数除以总 cycles（`:181-191`）。设备侧计时的最终来源。
- **`:206-227`（`_make_profile`）**：把多次采样的时长与各流水线比率**取平均**合成一个 `KernelProfile`——所以输出指标已是 30 次迭代的均值，单次毛刺被抹平。
- **`:360-393`（`slow/fast` 后端）**：`torch_npu.profiler` 配 `AiCMetrics.PipeUtilization`（`:379-380`）；`fast` 解析原始 `ffts_profile` 切片（`:403-446`），`slow` 解析 msprof 导出的 `kernel_details.csv`（`:449-508`，docstring 列出了全部 CSV 列名，可作为字段对照表）。
- **`:505-965`（persistent 后端与 ctypes 绑定）**：`MsProfiler`/`ProfImpl` 直接绑 `libmsprofiler.so`/`libprofimpl.so`，`_Session` 维护常驻 PMU 会话并用 `sleep_implILj1E`/`sleep_implILj2E` 标记窗口（`:512-513`，`_Session.profile` `:863-876`）。今天只需知道它在做什么、为什么存在（避免每次测量重启 profiler 的开销），不必逐行读。
- **`:923-929` + `:965`**：`close_persistent_profiler` 与 `atexit` 钩子——常驻会话随进程退出关闭。

**`tests/utils.py` 逐段导读**：

- **`:95-103`（`count_bytes`）**：`numel × element_size` 求和，跳过 None、摊平 tuple——"每个张量只计一次"的逻辑搬运模型。注意 `tests/generators.py:755-757` 有一个同名的 `count_bytes`，实现相同，供 einsum 测试使用。
- **`:106-115`（`print_perf`）**：perf 行的格式串。重点看 `prof.tflops(flops)`、`prof.gbps(bytes_moved)`、`prof.aic_mad`、`prof.aic_mte2` 四个值如何嵌入——一行输出同时给出时长、两个屋顶指标和两条关键流水线利用率。`prof is None`（`--skip-prof` 或该 case 不测性能）时只打 `OK`（`:113-114`）。
- **`:36-58`（`calc_diff` / `assert_equal`）**：正确性与性能的分界线——性能测量永远发生在正确性断言**之后**（`tests/common.py:199-236`），先保证算得对，再谈跑得快。

**`tests/common.py` 关键段导读**：

- **`:152-176`**：meta 张量生成 + `par_compile`——"Compile from metadata without allocating NPU storage"（`:156`）。这是 3.6 节"编译不入账"的代码证据；也解释了为什么测性能前测试要花一段时间"Compiling"（tqdm 进度条）。
- **`:223-236`**：perf 段。`k = sum(test.ks_cpu)`（K-Grouped 取总 K，`:224`）；门槛 `if not skip_prof and test.profile and k:`（`:225-226`）；`bench_msprof(..., kernel_names='gemm_impl')`（`:228`）；`print_perf(m, n, k, tags, 2.0 * m * n * k, count_bytes(a, b, c, d), prof)`（`:230-236`）——FLOPs 与字节口径的权威出处。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **先正确性后性能（复现两种模式）**：

   ```bash
   python tests/test_bf16_gemm.py normal --skip-prof    # 只测正确性，perf 行打印 OK
   python tests/test_bf16_gemm.py normal                # 含性能测量
   ```

   对比同一 case 在两种模式下的输出行，确认 `--skip-prof` 只裁掉 perf 部分（`tests/test_bf16_gemm.py:26` 的帮助文本）；同时观察运行开始前的 `Compiling` 进度条——那是 `par_compile` 在预编译（`tests/common.py:175`），体会"编译不入账"。

2. **用公式复核 perf 行**：从 `normal` 模式的输出里挑一行（如 `layout=NT, fp32, acc=0` 的某个 shape），记录 `m/n/k`、`dur_us`、TFLOPS、GB/s；自己用 `2·m·n·k` 和 `count_bytes` 口径（bf16 下 A+B+D，含 bias 时加 C）手算两个指标，与打印值核对到舍入误差内。

3. **复现 README HC Prenorm 行**：

   ```bash
   python tests/test_hyperconnection.py peak
   ```

   对照 `README.md:179-185`（peak 切片正是 `CASE_SLICES['peak']`，`tests/test_hyperconnection.py:23-27`）：M=8192 行应接近 136.7 us / 82 TFLOPS / 3463 GB/s；记录 mad 与 mte2 的百分比，验证"访存受限 → mte2 高、mad 低"。再手算该行的 AI（≈23.8 FLOP/B，见 3.2 节）。

4. **冷/热缓存对比（小实验）**：写一个十行脚本直接调用公开 API（`bench_msprof` 从 `deep_gemm.testing` 导入，`deep_gemm/testing/__init__.py:1`）：

   ```python
   import torch, deep_gemm
   from deep_gemm.testing import bench_msprof

   m, n, k = 4096, 4096, 7168          # 工作集大于 L2 才容易看出差别，可再试更小的 shape
   a = torch.randn(m, k, dtype=torch.bfloat16, device='npu')
   b = torch.randn(n, k, dtype=torch.bfloat16, device='npu')
   d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')
   deep_gemm.bf16_gemm_nt(a, b, d)    # 先触发 JIT 编译并预热
   torch.npu.synchronize()
   flops, moved = 2.0 * m * n * k, (a.numel() + b.numel() + d.numel()) * 2
   for flush in (True, False):
       prof = bench_msprof(lambda: deep_gemm.bf16_gemm_nt(a, b, d),
                           kernel_names='gemm_impl', flush_l2=flush)
       print(f'flush_l2={flush}: {prof.dur_us:.1f} us, '
             f'{prof.tflops(flops):.0f} TFLOPS, {prof.gbps(moved):.0f} GB/s')
   ```

   预期：冷缓存的 GB/s 更低、耗时更长；对计算受限的大 shape 差别应很小（为什么？——MTE2 本来就不是瓶颈，L2 命中与否改变的是 mte2 的等待时间，而 mad 才是长杆）。把两组数记进表格。

5. **整理基线表**：把第 2～4 步的记录整理成一张表（shape、dtype、dur、TFLOPS、GB/s、mad%、mte2%、AI、受限类型判断），存为 `week8_day1_baseline.md`——第 51 天的基线记录以此为起点扩充。

### B 组：无 Ascend 环境（静态任务）

1. **指标手算**（核心任务）：对下表三个 shape 完成计算（dtype 与字节构成按第三列）：

   | Shape | dtype | 字节构成 | 需算出 |
   |---|---|---|---|
   | 4096×7168×16384 | bf16 A/B/D，无 bias | `count_bytes(a,b,c=None,d)` 口径 | FLOPs、bytes、AI |
   | 8192×24×28672 | a bf16、b fp32、d/sqr_sum fp32 | `count_bytes(a,b,d,sqr_sum)` 口径 | FLOPs、bytes、AI |
   | 128×7168×16384 | —（不手算字节） | — | 仅 FLOPs |

   参考答案：第一行 FLOPs = 962,072,674,304、bytes = 427,819,008、AI ≈ 2249；第二行见 3.2 节（11.27 GFLOP / 473.3 MB / 23.8）；第三行 FLOPs = 30,064,771,072，用 README.md:123 的 54.8 us 算出 ≈ 548 TFLOPS（README 报 549）。

2. **复核 README 数据**：用 3.2 节方法验证：(a) Dense BF16 行 431 TFLOPS 与 99.8% 利用率（README.md:113）；(b) HC Prenorm 的 M=13 行——FLOPs = 2×13×24×28672 = 17,891,328，bytes = 13×28672×2 + 24×28672×4 + 13×24×4 + 13×4 = 3,499,284，按 6.7 us 算出 ≈ 2.7 TFLOPS / ≈ 522 GB/s（README 报 3 / 519，差异来自延迟列的舍入）；(c) Dense 表三行利用率 1701/1730、861/865、431/432。

3. **测量流程标注**：合上教程，在纸上画出 `bench_msprof` 从被调用到返回 `KernelProfile` 的完整时序（至少包含：npu_sleep 预热、10 次预热 + 同步、flush 缓冲构造、30 次"flush → fn()"迭代、PMU 记录 → 按名筛选 → 平均），并在每一步旁标注"它防住了哪个测量陷阱"（JIT 编译 / 首启开销 / 辅助 kernel 混入 / 热缓存）。再对照 3.1 节与 `bench.py:318-351` 修正。

4. **受限类型分类练习**：对下列 README 数据行各给出受限类型判断与一句话理由（先自己判，再对照 3.3/3.7 节）：

   | 数据行 | 出处 |
   |---|---|
   | Dense FP8×FP8 4096×7168×16384（861 TFLOPS，99.5%） | README.md:112 |
   | FP8 推理 128×7168×16384（549 TFLOPS / 2456 GB/s） | README.md:123 |
   | FP8 推理 4096×7168×16384（861 TFLOPS / 186 GB/s） | README.md:130 |
   | MQA Prefill FP8（FIX pipe 99%） | README.md:151, :155 |
   | HC Prenorm 8192 行（82 TFLOPS / 3463 GB/s） | README.md:184 |

5. **写笔记**（当天产出）：按第六节的框架独立写一份性能指标公式与解释笔记。

## 六、当天产出

**性能指标公式与解释笔记**——建议一页纸 + 一张附录表，至少覆盖：

1. **三个公式**及其单位推导：TFLOPS = FLOPs/时间（`bench.py:51-52`）、有效带宽 = 逻辑字节数/时间（`bench.py:54-55`）、AI = FLOPs/字节；FLOPs = 2·M·N·K 与 `count_bytes` 的"每张量只计一次"口径（`tests/common.py:233-234`）。
2. **一个手算算例**（推荐 HC Prenorm 8192 行或 13 行），展示从 shape/dtype 到三个指标的完整计算并与 README 对数。
3. **受限类型判别流程**：AI 对平衡点 → 流水线利用率 → 并行度；附 `KernelProfile` 流水线字段速查表（mad/mte1/mte2/fixpipe/vec 各自的语义与"饱和意味着什么"）。
4. **测量方法学要点**：设备侧 PMU 计时、预热、`kernel_names` 筛选、冷缓存（2×L2 清零）、JIT 预编译不入账——每条注明对应的源码位置。
5. **附录**：A 组同学贴上自己的基线表（shape/dur/TFLOPS/GB/s/mad%/mte2%/AI/判断）；B 组同学贴上第五节 B 组第 4 题的分类答案表。

写完后自查：不看教程能否从任意一个 README 性能行反推出它的 AI 和受限类型？能否说清 `print_perf` 一行里五个数字各自的来源？

## 七、自测题

1. `bench_msprof` 测的是什么的耗时？为什么不用 Host 侧的墙钟时间？
   > 单个 Device Kernel 的设备侧执行时长：取自 PMU 记录的 `start_ns/end_ns`（`bench.py:177-179`），跨 30 次迭代平均（`_make_profile`，`bench.py:206-227`）。墙钟时间会混入 JIT 编译（秒级）、Host 启动开销、Python 开销和窗口内的辅助 kernel；`bench_msprof` 分别用预编译 + 预热（`:329-332`）、设备侧计时和 `kernel_names` 筛选（`:271-281`）把它们排除。
2. 写出 TFLOPS、有效带宽、算术强度的公式，并说明 `KernelProfile.tflops()` 里为什么是除以 `dur_ns` 再除以 1000。
   > TFLOPS = 2MNK/时间；有效带宽 = 逻辑搬运字节数/时间；AI = 2MNK/字节数。FLOPs/ns = 10⁹ FLOP/s = GFLOP/s，故再除 1000 得 TFLOP/s（`bench.py:51-52`）；字节/ns 恰为 GB/s，故 `gbps` 只除一次 `dur_ns`（`:54-55`）。
3. `count_bytes` 统计的字节数和真实 DRAM 流量是一回事吗？
   > 不是。它把每个张量只计一次（`tests/utils.py:95-103`），是"逻辑搬运量"模型，不含 Kernel 内 Tile 复用导致的重复 GM 读（通常被 L2 吸收）。由此算出的 GB/s 是有效带宽——不会超过硬件上限，但可能低于真实 DRAM 流量对应的带宽。
4. 一个 BF16 GEMM 的 AI 是 2000 FLOP/B，另一个是 20 FLOP/B，分别倾向什么受限？HC Prenorm 属于哪一类？
   > BF16 平衡点约 432 TFLOPS ÷ ≈3.5 TB/s ≈ 123 FLOP/B：2000 ≫ 123 → 计算受限（mad 饱和）；20 ≪ 123 → 访存受限（mte2/带宽饱和）。HC Prenorm N=24，AI ≈ 23.8（8192 行），访存受限，README 称其 "nearly saturates the HBM write bandwidth"（`README.md:177`）。
5. `KernelProfile` 里 `aic_mad`、`aic_mte2`、`aic_fixpipe` 分别对应哪段数据通路？哪个 README 场景是 fixpipe 受限？
   > `aic_mad`：Cube MAD 矩阵乘加本体；`aic_mte2`：GM→L1 装载；`aic_fixpipe`：L0C→out 结果回写（`bench.py:32-45` 注释）。MQA Logits 是 FIX-pipe bound，FIX pipe 利用率 99%（`README.md:151`）。
6. `bench_msprof` 如何强制冷缓存？为什么需要？什么时候应该关掉？
   > 分配一块 2 倍 L2 容量的 int 缓冲（`_make_l2_flush_buffer`，`bench.py:263-268`），每次迭代前 `zero_()` 把其他数据挤出 L2（`:344-346`）。不刷则连续迭代读的是 L2，带宽被高估、不符合真实部署。测热缓存行为或输入本来就热的时候关掉，如 paged MQA metadata kernel 用 `flush_l2=False`（`tests/test_attention.py:413`）。
7. 测试是怎么保证 JIT 编译不进入性能数字的？
   > 三层：测量前用 meta 张量 + `par_compile`（dry-run，只编译不启动）集中预编译全部 case（`tests/common.py:152-176`；`par_compile.py:6-13`）；`bench_msprof` 先跑 10 次预热再同步（`bench.py:330-332`）；计时取设备侧 PMU 时间戳而非 Host 时钟。编译结果缓存于 `DG_JIT_CACHE_DIR`（默认 `$HOME/.dj`，`README.md:90-91`）。
8. README Dense 表的 Utilization 列怎么算？FP8 推理表里 M=128 与 M=4096 两行为什么指标形态相反？
   > Utilization = 实测 TFLOPS / Hardware limit（如 431/432 = 99.8%，`README.md:108-113`）。M=128 行 AI ≈ 223 FLOP/B（表中两列相除 549/2456），低于 FP8 平衡点估算 ≈247，偏访存侧且带宽列大；M=4096 行 AI ≈ 4629，计算受限（861/865 = 99.5%）且带宽列小。M 小 → Tile 数少 → AI Core 并行度不足，两头都不满。

---

明天（第 51 天）把这些公式用起来：设置 `DG_PRINT_CONFIGS=1`、`DG_JIT_PRINT_COMPILER_COMMAND=1`，运行一个较小的 GEMM 测试，记录每个 shape 实际选中的 JIT 配置、耗时、TFLOPS 与带宽，形成一组**基线性能记录**——为第 52 天的单变量实验准备对照数据。
