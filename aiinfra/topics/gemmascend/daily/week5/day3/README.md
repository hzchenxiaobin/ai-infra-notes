# 第 5 周 · 第 3 天（总第 31 天）：运行并分析 BF16 测试

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 50 分钟，源码导读约 35 分钟，实践步骤约 60 分钟，自测题约 15 分钟。

前两天我们读懂了测试数据的出生地（`tests/generators.py` 的 desc 枚举与 `generate()`）和判定方式（`tests/common.py` 的两阶段编排、五类断言）。今天让这条链**真的跑起来**，并回答学习计划提出的三个问题：**`normal` 和 `unaligned-k` 两个场景到底各测什么边界？**（54 个与 20 个 desc 的构成，以及它们与内核 `actual_m/n/k`、`eff_mad_*`、零填充、原子累加等边界代码的一一对应）**测试失败时如何判断是环境、参考结果、布局还是 Kernel 问题？**（`dg:` / `aclnn:` 双断言的交叉判别 + 一张四层分诊表）**没有 Ascend 硬件时怎么学？**（选取 `unaligned-k` 的 `(1024, 2048, 17)` 用例，从 host 检查一路静态推演到写回 GM——其中 `block_k=128`、`MAD_K=64` 等配置是可以从启发式源码**确定性推出**的，只有 `BLOCK_M/N` 需要假设）。当天产出是**测试记录或静态推演笔记**（模板见第六节，参考答案见 3.6 节）。它也是本周后半段的地基：第 32～34 天的低精度路径将以同样的方法运行和分析 `test_fp8_gemm.py` 等脚本。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出两条命令的路由——`python tests/test_bf16_gemm.py normal --skip-prof` 进入 `test_bf16_gemm(GemmType.Normal, skip_prof=True)`（`tests/test_bf16_gemm.py:37-38`），`unaligned-k` 进入 `test_bf16_gemm(GemmType.Normal, unaligned_k=True, …)`（`:31-32`）——并推导两个场景的 desc 构成：`normal` = `_enumerate_normal`（`tests/generators.py:544-576`）的 27 个 fwd + 27 个 bwd = **54** 个，`unaligned-k` = `_enumerate_normal_unaligned_k`（`:519-541`）的 5 个 shape × 4 种 major = **20** 个，全部 `accumulate=True`、fp32 输出；
2. 背出**边界截断的完整清单**及行号——K 维三级：`num_k_blocks = ceil_div(eff_shape_k, BLOCK_K)`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:96`）、`actual_k = min(eff_shape_k − k_blk_idx·BLOCK_K, BLOCK_K)`（`:114`）、`eff_mad_k = min(actual_k − k_mad_idx, MAD_K)`（`:137`）；MN 维三级：`get_actual_m/n`（`deep_gemm/include/deep_gemm/scheduler.hpp:129-139`）、`eff_mad_m/eff_mad_n`（`bf16_gemm.hpp:125-126`，注意 N 侧还要对齐到 `get_frac_k<ub_dtype>()`）、drain 的 `num_rows/num_cols`（`deep_gemm/include/deep_gemm/epilogue.hpp:65-66`）——并解释 `as_mad_aligned`（`ascend/tile_ptr.hpp:52-55`）把 L0 维度向上对齐（bf16 均为 16）后，**搬运引擎 ND2NZ 的零填充**为什么保证多算的部分不改变结果；
3. 把 5 个 `unaligned-k` shape（k=4/17/15/234/4095）和 `normal` 的 4 类 desc（m=1 的 fwd、MN-major 的 dgrad、MN/MN 累加的 wgrad、bf16 非累加输出）逐一映射到触发的边界代码路径（3.2 节映射表）；
4. 独立完成 `(1024, 2048, 17)` 双 K-major 用例的**完整静态推演**：host 侧 shape/major 推断（`csrc/utils/layout.hpp:44-47`）→ 配置选择（可推出 `block_k=128`：`k % 256 ≠ 0` 使宽块被跳过，`csrc/jit_kernels/heuristics.hpp:648-651`；假设 `BLOCK_M=BLOCK_N=128` 后，`MAD 128×128×64` 由 `make_gemm_candidate` 的搜索确定性给出，`:124-147`）→ 调度（8×16 网格、4 个 wave、核 0 分到的 4 个块）→ 搬运（GM→L1 只搬 17 个 K 列）→ MAD（`eff_mad_k=17`、L0 K 维对齐到 32、`is_first=is_last=true`）→ 写回（drain 128×128、累加走原子加）；
5. 解释 `d` 的预填策略（`tests/common.py:190`：Normal 填 `randn` 垃圾、k-grouped 非累加填 NaN）为什么能让"漏写任何一个输出元素"的边界 bug 立刻显形——残留的垃圾值会让 `assert_equal` 的 diff 爆炸；
6. 按**环境 / 参考结果 / 布局 / Kernel** 四层对失败分诊：说出每层的典型信号（`ModuleNotFoundError` 与编译失败→环境；`dg:` 与 `aclnn:` 同时失败→优先怀疑参考；`DJ_PANIC "Ambiguous major axis"` 等→布局；仅 `dg:` 失败且 diff 超 threshold→Kernel），并用"仅特定 shape / 特定 major / 特定 trans_ab 失败"进一步定位到具体边界代码；
7. 会用观测工具核对假设：`DG_PRINT_CONFIGS=1` 让 `select_gemm_config` 打印每个 shape 的 `GemmDesc(...) GemmConfig(...)` 一行（`heuristics.hpp:749-753`，格式见 `csrc/jit_kernels/gemm_config.hpp:145-186`）；`ASCEND_LAUNCH_BLOCKING=1` 把异步 launch 错误变成同步失败（仓库 README 的 Environment Variables 节）；
8. 说出两个"可推导的配置事实"：m=1 的 desc 其 `BLOCK_M` 被钉死为 16（`m_limit = align(1, 16) = 16`，`heuristics.hpp:608-609`）；`can_direct_store`（`:521-533`）要求 `m % align_m == 0 && n % align_n == 0` 且 `!acc || cd_dtype == fp32`——所以 `unaligned-k` 的 acc+fp32 组合在形状对齐时**允许**直写，而 `normal` 的 bf16 累加用例不允许。

**当天产出：测试记录或静态推演笔记**（模板见第六节，参考答案见 3.6 节）。

## 二、预备知识

- **第 29 天**：`GemmTestDesc` 的字段、五个场景枚举器、`generate()` 的确定性与六元组输出——今天要数清每个场景产出多少 desc、每个 desc 长什么样。
- **第 30 天**：`run_gemm_test` 的两阶段编排、五类断言（dg / ACLNN / 直写 vs fp32 累加 / 随机舍入 / 性能）、`d` 的预填充、`assert_equal` 的阈值模式——今天是它们的"运行侧"消费。
- **第 17 天**：`Scheduler` 的持久块调度与 `get_next_block` 的 `block_idx = (++current_iter) * kNumCores + core_id`、GROUP_M swizzle 与 snake。
- **第 19 天**：`copy_gm_to_l1` / `copy_l1_to_l0a/b` 的 ND2NZ 与 transpose 路径、`actual_m/n/k` 的作用。
- **第 22 天**：`BLOCK_K` 与 `MAD_K` 两层 K 循环、`k_blk_idx` / `k_mad_idx` / `eff_mad_k`、`is_first` / `is_last`。
- **第 24～26 天**：L0C 累加与 `ENABLE_UPDATE/KEEP`、AIC store 与 AIV drain、累加的原子加（`epilogue.hpp:89-97`）、`direct_store` 与空组写零。
- **第 12 天**：JIT 编译与缓存——今天用 `DG_PRINT_CONFIGS` 观察配置选择的落点。
- **工具基础**：bash 环境变量、Python 异常栈的阅读；`torch.npu.synchronize()` 的含义（异步 launch 与同步点）。

另注：本文的"运行"部分需要 Ascend 环境（`./develop.sh` 构建后 `import deep_gemm` 才能找到 `_C` 扩展，`deep_gemm/__init__.py:25`）；无环境时完成第五节 A 组的静态推演即可达到当天目标。性能输出的字段解读展开是第 50～51 天的内容，今天只到"能记录"为止。

## 三、详细讲解

### 3.1 全景：两条命令各测什么

先看选择器。`tests/test_bf16_gemm.py` 的 CLI 只做路由（`:21-50`）：

```text
python tests/test_bf16_gemm.py normal --skip-prof
  └→ test_bf16_gemm(GemmType.Normal, skip_prof=True)          (:37-38)
     └→ run_gemm_test(bf16, bf16, Normal, …, threshold=1e-5)  (:13-18, tests/common.py:131)

python tests/test_bf16_gemm.py unaligned-k --skip-prof
  └→ test_bf16_gemm(GemmType.Normal, unaligned_k=True, …)     (:31-32)
     └→ run_gemm_test(bf16, bf16, Normal, unaligned_k=True, …)
```

`--skip-prof` 只跳过 `bench_msprof` 性能测量（`common.py:226-228`），正确性断言与 JIT 编译照常执行。两个场景的 desc 构成：

**`normal`（54 个 desc，`generators.py:544-576`）。** bf16 输入时 `nk_list = 7 个 bf16 输出的 (n,k) + 2 个 fp32 输出的 (n,k)` 共 9 项（`:559-564`）：

| 类别 | 生成代码 | 数量 | 形状与属性 | 主要覆盖点 |
|---|---|---|---|---|
| fwd | `:566-570` | 3 个 m × 9 = 27 | `(m, n, k)`，m ∈ {1, 128, 4096}；K/K major；非累加；前 7 列 bf16 输出、后 2 列 fp32 输出 | m=1 的极小 M 边界；n=129280 的超大 N；bf16/fp32 两种输出路径；非累加 bf16 输出触发随机舍入验证 |
| dgrad | `:574` | 9 | `(4096, k_pair, n_pair)`，K/MN major，非累加 bf16 | **B 为 MN-major**：GM→L1 的 nd2nz 行列互换（`ascend/copy.hpp:125-130`）与 L1→L0 的 transpose 搬运（`:139-144`） |
| wgrad（acc） | `:575` | 9 | `(n_pair, 4096, k_pair)`，MN/MN major，**累加**，fp32 输出 | 双 MN-major transpose + **累加路径**：`d.copy_(c)` 预填后内核原子加（`epilogue.hpp:91-96`） |
| wgrad（bf16） | `:576` | 9 | 同上但非累加 bf16 | MN-major + 非累加 + bf16 输出的组合；随机舍入验证 |

**`unaligned-k`（20 个 desc，`generators.py:519-541`）。** 5 个 shape × 4 种 (a_major, b_major) 组合，全部 `accumulate=True`（`:539`）、fp32 输出（`:536`）：

| shape (m, n, k) | K 尾特征（设 BLOCK_K=128） |
|---|---|
| (64, 64, 4) | k=4 < 16：连第一个 NZ 分形（16 列）都填不满 |
| (1024, 2048, 17) | k=17：1 个满分形 + 1 个只有 1 个有效列的分形 |
| (8192, 4096, 15) | k=15 < 16：单个"欠满分形" |
| (128, 128, 234) | 234 = 128 + 106：跨 2 个 k-block，尾块 actual_k=106，MAD 尾 eff_mad_k=42 |
| (2048, 8192, 4095) | 4095 = 31×128 + 127：32 个 k-block，尾块 actual_k=127（只差 1 列满块！） |

运行时的可见输出（有环境时）依次是：`Testing bf16 x bf16 gemm: Normal [unaligned-k]` 标题（`common.py:146-147`）、`Num tests:  54` / ` 20`（`:150`）、`Compiling` 进度条（`par_compile` 的 tqdm，`deep_gemm/testing/par_compile.py:25`）、然后每个 desc 一行 ` > Perf (m=…, n=…, k=…, layout=…, fp32, acc=…, …): OK`（`tests/utils.py:106-115`；`--skip-prof` 时以 `OK` 结尾）。`layout` 标签由 `major_opt` 给出（`generators.py:750-752`）：双 K-major → NT，(K, MN) → NN，(MN, K) → TT，(MN, MN) → TN。

### 3.2 核心映射表：测试场景 → 内核边界代码

这是今天的核心。BF16 内核的边界处理散落在四层，用一张表把它们和测试场景对上（行号均可在第四节的导读站点中核对）：

| 测试场景 / desc | 触发的边界处理 | 源码位置 |
|---|---|---|
| fwd m=1（如 `(1, 2112, 7168)`） | `actual_m = min(1, BLOCK_M) = 1`；L0A/L0C 按 16 行对齐（1 行有效 + 15 行零填充）；drain 只写 1 行；host 侧 `BLOCK_M` 被钉死为 16 | `scheduler.hpp:134`；`tile_ptr.hpp:52-55`；`epilogue.hpp:65, :81-82`；`heuristics.hpp:608-609` |
| fwd n=129280 / m=4096 | 多 wave 持久调度；GROUP_M=4 swizzle + N-first snake；N/M 尾块 `actual_n/m` | `scheduler.hpp:141-142, :202-247`；`:137-139` |
| dgrad（B 为 MN-major） | GM→L1 时行列互换：nd2nz 的行数 = actual_k、列数 = actual_mn；L1→L0 走 transpose 变体 | `copy.hpp:125-130`；`:139-144`（l0a）/ `:153-158`（l0b） |
| wgrad acc（MN/MN + 累加） | 累加语义：测试端 `d.copy_(c)` 后原地传入（`common.py:55-58`）；内核 AIV 侧原子加；L0C 内 `is_last ? ENABLE_UPDATE : ENABLE_KEEP` | `epilogue.hpp:91-96`；`bf16_gemm.hpp:149-151` |
| bf16 非累加输出（fwd/wgrad bf16/dgrad） | 随机舍入验证 + L0C→UB 的 F322BF16 量化模式 | `common.py:215-221`；`copy.hpp:231-264, :267-285` |
| unaligned-k 全部 | K 三级截断：`num_k_blocks` / `actual_k` / `eff_mad_k`；L0 K 维 `as_mad_aligned` 对齐到 16 的倍数；ND2NZ 零填充补齐 | `bf16_gemm.hpp:96, :114, :137`；`tile_ptr.hpp:52-55` |
| unaligned-k（MN-major 变体） | K 尾落在 GM→L1 搬运的**行**维（rows = actual_k），且走 transpose 路径 | `copy.hpp:127-129, :139-144` |
| unaligned-k（acc + fp32） | `can_direct_store` 允许"acc 且 fp32 输出"直写；直写时 `copy_l0c_to_gm` 用 `eff_mad_m/n` 截断写回，且 AIC 侧原子加 | `heuristics.hpp:521-533`（`:528` 的 acc 条件）；`bf16_gemm.hpp:82, :156-159` |
| （对照）k-grouped 空组 | `d` 预填 NaN + `aiv_zero_empty_groups` 写零 | `common.py:190`；`epilogue.hpp:108-141` |

三个容易忽略的事实：

1. **bf16 没有 `fill_tail_block`**。显式的尾部零填充函数 `fill_tail_block` / `fill_mn_tail_block`（`copy.hpp:333-368`）带 `static_assert(sizeof(T) == 1)`，只被 fp8/fp4 内核调用（如 `fp8_gemm.hpp:169-170`）。BF16 内核的 K 尾完全依赖三层 `min` 截断 + `as_mad_aligned` 向上取整 + **搬运引擎 ND2NZ 的隐式零填充**——补零列乘任何值都是 0，不改变累加结果，这是"多算无害"的设计。
2. **`eff_mad_n` 多一步对齐**（`bf16_gemm.hpp:126`）：`aligned(min(actual_n − n_mad_idx, MAD_N), get_frac_k<ub_dtype_t>())`，bf16 输出对齐 16、fp32 输出对齐 8——这是 L0C→UB 行跨度必须 32 字节对齐的要求（`epilogue.hpp:29` 的 static_assert）。注意**写回 GM 时用的是未对齐的 `num_cols = min(actual_n − n_mad, MAD_N)`**（`epilogue.hpp:66, :81-82`），所以对齐只影响片上计算范围，不会越界写 D。
3. **测试的预填让漏写显形**（`common.py:190`）：`d = torch.randn_like(d_ref)`（Normal 场景）。如果内核因边界处理错误漏写任何一个输出区域（例如 drain 的 `num_rows` 算错、尾块 n-block 被跳过），残留的随机垃圾会让 `calc_diff` 远超 1e-5——边界 bug 不需要专门构造就能被抓到。

### 3.3 完整静态推演：`(1024, 2048, 17)` 双 K-major（无环境主任务）

选取 `unaligned-k` 的 `(1024, 2048, 17)`、双 K-major、`accumulate=True`、fp32 输出的 desc，从 host 一路推到写回。**可确定推出的结论**与**需要假设的部分**会明确标注。

**第 1 步：host 检查与 desc（可推出）。** 生成器产出 `a: (1024, 17)` K-major（stride `(17, 1)`）、`b: (2048, 17)` K-major、`c = randn(1024, 2048, fp32)`、`d_ref = (a_repr @ b_repr.T + c).to(fp32)`（`generators.py:320-322`，alpha=None 时取 1.0）。调用 `deep_gemm.bf16_gemm_nt(a, b, d, c=d)`（累加约定：`common.py:55-58` 的 `d.copy_(c); extra_args['c'] = d`）。host 侧 `get_major_type_ab` 从 stride 推断 `major_a = major_b = Major::K`（`csrc/utils/layout.hpp:44-47`：`stride(-1) == 1` → K）；指针判同 `c->data_ptr() == d.data_ptr()` 成立 → `acc = true`（`csrc/apis/gemm.hpp:106-110, :180`）；形状由 `"mk"/"nk"/"mn"` 三个标注交叉解出 m=1024, n=2048, k=17（`layout.hpp:140-219`）。

**第 2 步：配置选择（部分可推出）。** bf16 → `elem_bits=16`、`k_alignment = 32·8/16 = 16`、m/n 对齐均为 16、`min_block_k = 256·8/16 = 128`、`max_block_k = 256`、`block_cap = 512`（`heuristics.hpp:604, :620-621`；硬件参数 `:52-71`）。

- **可推出**：`k=17`，`k % max_block_k = 17 % 256 ≠ 0` → 宽块被跳过（`:648-651`）→ **`block_k = 128`**；后续 K 细化循环也因 `17 % 16 ≠ 0` 不进入（`:672-673`）。
- **可推出**：m=1024 时 `m_range = {16, 512}`、n=2048 时 `n_range = {16, 512}`——`BLOCK_M/BLOCK_N` 由代价模型在范围内搜索（`:626-657`），静态无法确定。**演算时假设 `BLOCK_M = BLOCK_N = 128`**（有环境时用 `DG_PRINT_CONFIGS=1` 核对，见 3.5）。
- **可推出（在假设之上）**：给定 BLOCK 128×128×128，`make_gemm_candidate` 的 MAD 搜索是确定性的（`:124-147`）：split `(1,1)` 首先命中（`128 % 16 == 0`，`128·128·4B = 64KB ≤ L0C/2`）；`l0a_k = l0b_k = 64KB·8 / (2·128·16) = 128`，`max(k_alignment, block_k/num_l0_stages) = max(16, 64) = 64` → `mad_k = min(128, 128, 64) = 64`，且 `64 % 16 == 0`、`128 % 64 == 0`、不小于 `min(128, 128)/2 = 64` → **`MAD = 128×128×64`**，`num_l0_stages = 2`（`:111-112` 固定）。
- **可推出（在假设之上）**：`can_direct_store`（`:521-533`）逐条核对——Normal ✓、Identity ✓、`!acc || cd==fp32` ✓（acc 但 fp32 输出）、`m % 1 == 0` ✓（K-major A 的 align_m=1）、`n % 16 == 0` ✓（2048）、D 指针与行跨度 32 字节对齐 ✓（`stride_d·4B = 8192B`）；但 wave 条件 `ceil(1024/128)·ceil(2048/128) = 128 ≤ 32` 不成立，且 `m ≤ block_m`（1024 ≤ 128）也不成立 → **直写不可用**，走普通 AIC store + AIV drain 路径。

**第 3 步：调度（可推出）。** `num_m_blocks = 8`、`num_n_blocks = 16`、`num_blocks = 128`；32 核 → 4 个 wave。`kGroupM = 4`（`scheduler.hpp:47`）→ `tail_m_start = 8 − 8%4 = 8`，尾区为空，全部走 swizzle 主路径。`kGroupN = 32/4 = 8`、`n_groups = 16/8 = 2`、每超行 `4×16 = 64` 块。核 `c` 第 `t` 次迭代拿到 `block_idx = t·32 + c`（`:141-142`）。以**核 0** 为例（`:202-247` 的 swizzle + `:232` 的 XorN + `:235` 的 snake）：

```text
block_idx=0   → superrow 0, grp 0, local 0  → m_local=0, n_local=0^0=0 → (m0, n0)
block_idx=32  → superrow 0, grp 1, local 0  → (m0, n8)      [ng = 1]
block_idx=64  → superrow 1, grp 0, local 0  → ng = 2-1-0 = 1 → (m4, n8)   [snake 反转]
block_idx=96  → superrow 1, grp 1, local 0  → ng = 0        → (m4, n0)
```

每个块的 `m_idx = m_block·128`、`n_idx = n_block·128`；`actual_m = min(1024 − m_block·128, 128) = 128`、`actual_n = 128`——**M/N 全部整块，唯一的边界在 K**（这正是该用例的设计意图）。

**第 4 步：K 循环与搬运（可推出）。** Normal 类型 `eff_shape_k = shape_k = 17`（`bf16_gemm.hpp:95`）→ `num_k_blocks = ceil_div(17, 128) = 1`（`:96`）→ `actual_k = min(17 − 0, 128) = 17`（`:114`）。GM→L1（`:117-118` → `copy.hpp:120-124` K-major 分支）：`asc_copy_gm2l1_nd2nz(dst, src, stride=17·2B, …, n=128 行, d=17 列)`——引擎从 GM 每行读 17 个 bf16，写入 L1 的 NZ 分形：第 1 个分形 16 列全有效，第 2 个分形只有 1 个有效列、其余 15 列由引擎零填充。A、B 两个操作数同理。

**第 5 步：MAD（可推出）。** `m_mad/n_mad` 双循环各只跑 1 次（`actual_m = actual_n = 128`）；`eff_mad_m = 128`、`eff_mad_n = aligned(128, 8) = 128`（fp32 输出 → `get_frac_k<float>() = 8`）。K 侧：`k_mad_idx = 0 < actual_k = 17`，`eff_mad_k = min(17, 64) = 17`（`:137`）；`l0a_tile = l0a[0].as_mad_aligned(128, 17)` → **L0 的 K 维 = aligned(17, 16) = 32**（`tile_ptr.hpp:52-55`），即 2 个 C0 分形；`l0b_tile` 同理。`is_first = (k_blk_idx==0 && k_mad_idx==0) = true`，`is_last = (k_blk_idx==num_k_blocks−1 && 0+64 ≥ 17) = true`（`:134-135`）→ 单次 `mad(..., 128, 128, 17, ENABLE_UPDATE, clear_C=true)`（`:149-151` → `copy.hpp:371-381`）。补零的 15 列乘出 0，结果与真实 17 列点积一致。

**第 6 步：写回（可推出）。** 最后一个（也是唯一的）k-block 走 `epilogue.aic_store(l0c_tile, s_epi, 128, 128)`（`:161` → `epilogue.hpp:42-47`，L0C→UB 在 AIC 的 FIX 通路完成，fp32 输出为 NoQuant）；AIV 侧 `aiv_drain` 的 `num_rows = min(128, 128) = 128`、`num_cols = 128`（`epilogue.hpp:65-66`），`copy_ub_to_gm(d.index(m_idx, n_idx), …, 128, 128, stride=2048·4B)`（`:81-82`）。因为 `kWithAccumulation = true`，`aiv_initialize` 已设置 `asc_set_atomic_add_float()`（`:91-96`）——写回是**原子加**，叠加在测试端预填的 `c` 上，最终 `d = c + A@Bᵀ`，与 `d_ref` 语义一致。

把 3～6 步对到测试断言：`FF` 组合跑的就是上面这条路；`TF`（alpha）在 drain 前多一步 `ScaleByAlpha`；`FT`（bias）走 `AddBias`；`TT` 期望 host 拒绝。随后 ACLNN 交叉验证用 `aclnnInplaceAddmm` 独立算一遍（`csrc/apis/aclnn.hpp`，bf16 无额外跳过条件）。该 desc 是 `accumulate=True` → **没有**随机舍入验证（`common.py:215` 三条件之一不满足）。

### 3.4 另两类典型边界：m=1 与 MN-major

**m=1（fwd 用例）**：`m_limit = align(1, 16) = 16` 把 `m_range` 压成 `{16, 16}`，**`BLOCK_M = 16` 是唯一候选**（`heuristics.hpp:608-609`）——这是纯静态可推出的事实。内核里 `actual_m = 1`，L0A 的一个分形里 1 行有效 + 15 行零填充（`as_mad_aligned(1, …) → 16`），drain 只写 1 行到 GM。它同时是 `can_direct_store` 的典型受益者：`m % 1 == 0`、`n % 16 == 0`、`m ≤ block_m`（1 ≤ 16），形状条件全过（`:521-533`），是否最终启用还看 `refine_gemm_pipeline` 的代价比较（`:535-574`）。

**MN-major（dgrad / wgrad 用例）**：B 为 MN-major 时，GM→L1 的 nd2nz 参数行列互换——`n(行) = actual_k, d(列) = actual_mn`（`copy.hpp:127-129`），K 尾截断落在搬运的**行**维；L1→L0 走 `asc_copy_l12l0a/b_transpose`（`:139-144, :153-158`）。`unaligned-k` 的 4 种 major 组合正是把这组路径与 K 尾叠加：`NN`（K,MN）只 B 侧换路、`TT`（MN,K）只 A 侧换路、`TN`（MN,MN）双侧换路——20 个 desc 里同一 shape 的 4 种 layout 输出行，就是同一组边界在四条搬运路径上的重复验证。

### 3.5 失败分诊：环境 / 参考结果 / 布局 / Kernel

学习计划要求"如果失败，先判断是环境、参考结果、布局还是 Kernel 问题"。判据来自第 30 天的断言结构：`dg:` 与 `aclnn:` 是两个**互相独立**的实现，各自与 `d_ref` 比较（`common.py:201, :213`）。由此得到分诊表：

| 层级 | 典型信号 | 依据 / 源码位置 | 下一步 |
|---|---|---|---|
| **环境** | `ModuleNotFoundError: No module named 'deep_gemm._C'`（`import deep_gemm` 即失败，`__init__.py:25`）；JIT 编译子进程报错（找不到 `bisheng`/`ld.lld`，检查 `ASCEND_HOME_PATH`）；kernel launch 超时（`DG_JIT_LAUNCH_TIMEOUT`，默认 300s）；NPU OOM | 异常类型不是 `AssertionError`，栈顶在 import/编译/launch，而非 `assert_equal` | `./develop.sh` 重构建；核对 CANN 9.20 与 torch_npu；缩小用例 |
| **参考结果** | `dg:` 与 `aclnn:` **同时**失败且 diff 都很大（≈1 量级）——两个独立实现不可能以同样方式错 | 两者共享的只有输入数据与 `d_ref`（`common.py:201/:213` 都对着 `generators.py:320-322` 的公式） | 检查生成端是否被改动（shape/major/seed）；单独重算一个手工小例子验证 `d_ref` |
| **布局** | `DJ_PANIC: Ambiguous major axis …`（`csrc/utils/layout.hpp:38-39`）；`Conflicting shape dimension`（`:196-204`）；`K/MN-major operand requires unit stride`（`gemm_config.hpp:27-33`）；或仅特定 trans_ab / major 组合的 `dg:` 失败而其余通过 | dg 的 4 种 trans_ab 经抵消转置后 desc 相同（第 29 天 3.3）——**只挂一种 trans_ab 说明错在绑定规范化层而非内核**；major 相关的 panic 在 host 断言处直接抛出 | 检查传入张量的 shape/stride 与 API 后缀是否匹配 |
| **Kernel** | 仅 `dg:` 失败（`aclnn:` 通过）且 diff 超 1e-5；或 `alpha + bias must be rejected`（`common.py:197`，host 检查失效）；`stochastic rounding:` / `direct output:` 失败 | dg 内核或其 epilogue 与独立参考出现分歧 | 按失败场景细分（下表） |

Kernel 层的**场景 → 边界**细分（这是"将测试场景与源码中的边界处理对应起来"的失败版应用）：

| 失败范围 | 指向的边界代码 |
|---|---|
| 仅 `unaligned-k` 失败，且只在 k=4/15/17（k < 16 或 k % 16 ≠ 0） | K 尾：`actual_k`/`eff_mad_k`（`bf16_gemm.hpp:114, :137`）、`as_mad_aligned`（`tile_ptr.hpp:52-55`）、ND2NZ 零填充 |
| 仅 `unaligned-k` 的 k=234/4095（跨多块）失败 | 跨 k-block：`num_k_blocks`/`k_blk_idx`（`:96, :112`）、`is_last` 的判定（`:135`） |
| 仅 `unaligned-mn` 场景失败 | M/N 尾：`get_actual_m/n`（`scheduler.hpp:129-139`）、`eff_mad_m/n`（`bf16_gemm.hpp:125-126`）、drain 的 `num_rows/num_cols`（`epilogue.hpp:65-66`） |
| 仅 MN-major 的 desc 失败 | transpose 搬运：`copy.hpp:125-130, :139-158` |
| 仅累加用例（wgrad acc / unaligned-k）失败 | 原子加路径：`epilogue.hpp:91-96`、`bf16_gemm.hpp:82`；或 host 端 `d.copy_(c)` 约定 |
| 仅 m=1 或极小 shape 失败 | 分形内零填充行、`heuristics.hpp:608-609` 的小 m 配置 |
| 输出大面积保持预填的 randn 值 | 漏写：drain 循环边界或直写 `eff_mad_*`（`bf16_gemm.hpp:156-159`）——`common.py:190` 的预填正是探测器 |

**观测工具**：`DG_PRINT_CONFIGS=1`（或 `DG_JIT_DEBUG=1`）让 `select_gemm_config` 对每个新 shape 打印一行 `GemmDesc(m=…, n=…, k=…, …) GemmConfig(block=MxNxK, mad=MxNxK, l1_stages=…, …)`（`heuristics.hpp:749-753`；两个 formatter 在 `gemm_config.hpp:145-186`；`static printed` 集合去重，同一 desc+config 只打一次）。它用来核对 3.3 中假设的 `BLOCK_M/N`。`ASCEND_LAUNCH_BLOCKING=1`（README）让每次 launch 后同步，把异步报错拉回现场。这两个开关的更多用法是第 51 天的主题。

### 3.6 参考答案：静态推演笔记模板（当天产出）

```text
用例：unaligned-k (1024, 2048, 17)，双 K-major，acc=True，fp32 输出
├─ host：major_a/b = K（stride(-1)==1）；acc（指针判同）；m=1024, n=2048, k=17
├─ 配置：block_k=128（17%256≠0，heuristics.hpp:648-651）
│        BLOCK_M/BLOCK_N：假设 128/128（待 DG_PRINT_CONFIGS 核对）
│        MAD=128×128×64（由 make_gemm_candidate:124-147 确定性推出）；l0_stages=2
│        direct_store=false（128 blocks > 32 cores 且 m>block_m，:531-532）
├─ 调度：8×16=128 块，4 waves；核0：(m0,n0)→(m0,n8)→(m4,n8)→(m4,n0)（snake）
│        actual_m=actual_n=128（M/N 无尾）
├─ K 循环：num_k_blocks=1；actual_k=17；GM→L1 nd2nz 128行×17列，第2分形 1 有效+15 零
├─ MAD：eff_mad_k=17；L0 K 维 aligned(17,16)=32；is_first=is_last=true → 单次
│        mad(128,128,17,ENABLE_UPDATE,clear_C=true)
├─ 写回：aic_store(L0C→UB, NoQuant) → aiv_drain 写 128×128，原子加（acc）
└─ 断言：FF/FT/TF 三次 dg + TT 拒绝 + aclnn(Addmm)；无 SR（accumulate=True）
```

## 四、源码导读

建议按站点顺序阅读，总计约 35 分钟。所有行号以仓库当前版本为准。

### 站点 1：CLI 选择器（`tests/test_bf16_gemm.py` 全文，50 行）

`:6-18` 的 `test_bf16_gemm` 参数与 `threshold=1e-5`；`:25` 的 choices；`:31-47` 的路由。确认 `normal` 与 `unaligned-k` 都落在 `GemmType.Normal`，差别只在 `unaligned_k` 布尔位。

### 站点 2：两个场景枚举器（`tests/generators.py`）

`_enumerate_normal_unaligned_k`（`:519-541`）：shape 列表 `:520-525`、fp32 输出 `:536`、`accumulate=True` `:539`、tag `:540`；fp4 的偶数 K 调整 `:529-533`（bf16 不触发）。`_enumerate_normal`（`:544-576`）：`fp32_output_nk`/`bf16_output_nk` `:559-560`、`nk_list` 的 bf16 扩展 `:562-564`、fwd `:566-570`、bwd 三连 `:572-576`。亲手数一遍 27+27=54。

### 站点 3：内核边界三处 min（`deep_gemm/include/deep_gemm/bf16_gemm.hpp`）

AIC 分支 `:81-172`：`:95-96` eff_shape_k 与 num_k_blocks、`:105-106` actual_m/n、`:114` actual_k、`:117-118` 带 actual 的 GM→L1、`:123-127` MAD 双循环与 eff_mad_m/n、`:133-139` k_mad 循环与 eff_mad_k、`:134-135` is_first/is_last、`:138-139` as_mad_aligned、`:149-151` mad 调用、`:155-163` 末块直写或 aic_store。AIV 分支 `:173-192`：`:175-176` 空组写零（k-grouped）、`:180-191` drain 循环。

### 站点 4：调度器（`deep_gemm/include/deep_gemm/scheduler.hpp`）

`:141-142` 持久块索引公式；`:190-195` Normal 分支；`:202-248` `get_swizzled_block`——fast path `:209-211`、行主序尾区 `:212-216`、swizzle 主路径 `:217-247`（XorN `:232`、snake `:235`）；`:129-139` `get_actual_m/get_actual_n`（注意 m-grouped 恒返回 BLOCK_M 的 TODO `:130-132`）。

### 站点 5：搬运与对齐（`ascend/copy.hpp` + `ascend/tile_ptr.hpp`）

`copy_gm_to_l1`（`copy.hpp:101-131`）：K-major 分支 `:120-124`（nd2nz 的 n/d 参数）、MN-major 分支 `:125-130`。`copy_l1_to_l0a/l0b`（`:133-159`）：transpose 变体 `:139-144/:153-158`。`fill_tail_block`/`fill_mn_tail_block`（`:333-368`）：`static_assert(sizeof(T) == 1)`——**bf16 不走这里**，零填充由引擎隐式完成。`tile_ptr.hpp:52-55` 的 `as_mad_aligned`：MN 维对齐 `FRAC_MN=16`、K 维对齐 `get_frac_k<T>()`（bf16=16，`common.hpp:73-78`）。

### 站点 6：Epilogue 边界（`deep_gemm/include/deep_gemm/epilogue.hpp` + `epilogue/operators.hpp`）

`aic_store`（`epilogue.hpp:42-47`，AIC 侧 L0C→UB）；`aiv_drain`（`:49-87`）：`num_rows/num_cols` 的 min `:65-66`、`copy_ub_to_gm` 的截断写 `:81-82`；`aiv_initialize` 的原子加设置 `:91-96`；`aiv_zero_empty_groups` `:108-141`。`operators.hpp:93-122`：基类 `Operator::process_mad_block`（`:114`）是空操作、`Identity`（`:120`）直接继承——直通路径下 UB 数据不经变换，只有 `ScaleByAlpha`/`AddBias` 等才注入向量计算。

### 站点 7：配置选择与直写判定（`csrc/jit_kernels/heuristics.hpp`）

`select_gemm_config`（`:592-755`）：`block_cap` `:604`、min/max block_k `:620-621`、搜索主循环 `:626-657`、宽块条件 `:648-651`、`DJ_HOST_ASSERT(config.block_m, …)` `:658`。`make_gemm_candidate`（`:92-204`）：MAD 搜索 `:124-147`、L1/L0 stage 与容量检查 `:166-174`。`can_direct_store`（`:521-533`）逐条读五个条件。`DG_PRINT_CONFIGS` 打印 `:749-753`。

### 站点 8：host 布局检查与早退（`csrc/utils/layout.hpp`）

`major_check` 的歧义 panic `:37-42`、`get_major_type_ab` `:44-47`、`gemm_early_return` `:54-86`（m/n 为 0 直接返回；k 为 0 时把 `d` 置为 `c` 或清零——空 GEMM 也有确定语义）；`get_shape_by_spec` 的冲突 panic `:196-204`。这些是"布局层"失败信号的出处。

## 五、实践步骤

### A. 无 Ascend 环境：静态阅读与手工演算（主任务，约 60 分钟）

**A-1 normal 场景拆解（必做）。** 对照 `generators.py:544-576` 推导 54 个 desc 的构成表：fwd 27 个（3 个 m × 9 个 (n,k)）与 bwd 27 个（9 × dgrad/wgrad-acc/wgrad-bf16 三连）；为每类标注 (a_major, b_major)、accumulate、输出 dtype、会触发哪些断言（dg 三组合 + TT 拒绝 + aclnn + SR 与否——SR 条件见 `common.py:215`）。最后回答：为什么 fwd 的后 2 个 (n,k) 没有 SR 验证？

**A-2 (1024, 2048, 17) 完整推演（必做，当天产出的核心）。** 按 3.3 的六步走一遍，写出一页推演笔记（模板见 3.6）。至少要独立算出：`block_k`、（假设 BLOCK 128×128 后的）MAD 三维、`num_k_blocks`、`actual_k`、`eff_mad_k`、L0 的 K 维对齐值、核 0 在 4 个 wave 里分到的 4 个 (m_block, n_block)、drain 的 `num_rows/num_cols`、直写可用性判定。写完对照 3.3/3.6 订正，并把"可推出"与"假设"两类结论用不同记号标出。

**A-3 边界映射表（必做）。** 为下表每行填上"触发的边界代码（文件:行号）"和"该边界若写错，失败会出现在哪个断言/场景"（参考 3.2）：

| desc | 边界代码 | 写错时的失败表现 |
|---|---|---|
| unaligned-k (64, 64, 4) |  |  |
| unaligned-k (2048, 8192, 4095) |  |  |
| normal fwd m=1 |  |  |
| normal dgrad（B MN-major） |  |  |
| normal wgrad acc |  |  |

**A-4 失败分诊练习（必做）。** 判断以下六个案例分别属于环境 / 参考结果 / 布局 / Kernel（或组合）哪一层，给出依据与下一步动作（答案见第七节末尾）：

1. 运行 `python tests/test_bf16_gemm.py normal --skip-prof`，`import deep_gemm` 处抛 `ModuleNotFoundError: No module named 'deep_gemm._C'`；
2. 某次改动后，`dg:` 断言全部通过，但每个 desc 的 `aclnn:` 断言都失败（diff 约 2e-2）；
3. 新增了一个测试场景，该场景下 `dg:` 与 `aclnn:` 同时失败，diff 均约 1.0，其余场景全部通过；
4. 只有 `unaligned-k` 失败，且失败的 desc 全部是 k=17 与 k=15（diff 约 5e-3），k=234/4095 通过，`normal` 全通过；
5. `(T, T)` 组合抛出 `AssertionError: alpha + bias must be rejected`；
6. 抛出 `DJ_PANIC: Ambiguous major axis, the tensor has both last two dimensions contiguous, strides=[...]`。

**A-5 判定题自验（必做）。** 判断真假并给出行号依据（答案见第七节末尾）：

1. `unaligned-k` 场景的 20 个 desc 全部是 `accumulate=True` 且 fp32 输出，因此都没有随机舍入验证。（  ）
2. bf16 内核的 K 尾需要显式调用 `fill_tail_block` 把 L1 尾部清零。（  ）
3. 对 `(1024, 2048, 17)`，无论代价模型选什么 `BLOCK_M/BLOCK_N`，`block_k` 一定是 128。（  ）
4. `eff_mad_n` 的对齐值会直接决定写回 GM 的列数。（  ）
5. m=1 的 desc 其 `BLOCK_M` 可以被静态推出为 16。（  ）
6. `dg:` 断言失败但 `aclnn:` 通过，最可能的结论是参考结果 `d_ref` 算错了。（  ）

### B. 有 Ascend 环境：运行与观察（可选，约 40 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
python tests/test_bf16_gemm.py normal --skip-prof
python tests/test_bf16_gemm.py unaligned-k --skip-prof
```

1. 核对输出骨架：`Testing bf16 x bf16 gemm: Normal [unaligned-k]`、`Num tests:  54` / ` 20`、`Compiling` 进度条（`par_compile.py:25`）是否出现；数一数 ` > Perf (…): OK` 行数是否等于 desc 数；
2. 把 `unaligned-k` 的 20 行输出与 5 个 shape × 4 种 layout 对上（`layout=NT/NN/TT/TN`、`fp32`、`acc=1`、`unaligned-k`），确认 `k=` 列的 4/17/15/234/4095；
3. `DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py unaligned-k --skip-prof` 重跑，找到 `(1024, 2048, 17)`、`major_a=K, major_b=K` 的那行 `GemmConfig`，记录真实的 `block=…x…x…, mad=…x…x…`，与 A-2 的假设对照：`block_k=128` 是否吻合？`BLOCK_M/BLOCK_N` 差多少？`direct_store` 是 0 还是 1？
4. （可选）临时把 `tests/test_bf16_gemm.py:17` 的 `threshold` 改为 `1e-30` 重跑 `unaligned-k`，观察 `AssertionError: dg: … (rows …, diff=…)` 的定位信息格式（desc + trans + diff 值），然后**还原**；
5. （可选）`ASCEND_LAUNCH_BLOCKING=1` 重跑一次小场景，体会同步 launch 对报错定位的影响（会明显变慢，知道即可）。

## 六、当天产出

**测试记录或静态推演笔记**。把 A-1～A-5（或 B-1～B-3）的结果整理成一页笔记，至少包含：

1. **场景构成表**：normal 54 与 unaligned-k 20 的 desc 构成（A-1），标注每类的 major/acc/输出与触发的断言集；
2. **静态推演记录**：A-2 的六步推演（可直接用 3.6 模板），明确区分"可推出"与"假设"结论；有环境者附上 `DG_PRINT_CONFIGS` 的真实配置并标注与假设的差异；
3. **边界映射表**：A-3 填写完的表格；
4. **分诊记录**：A-4 六个案例的判定与依据；A-5 判定题的自评；
5. **运行记录**（有环境者）：两条命令的输出摘要（Num tests、Perf 行数、任一行的完整字段）、B-4 的失败信息样例；
6. **遗留问题清单**（给第 32/50/51 天）：至少记两条，例如"fp8/fp4 测试的断言集与 bf16 有何差异（明天）""mad%/mte2% 字段含义与瓶颈判断（第 50 天）""代价模型如何在这类 shape 上选 BLOCK_M/N（第 52 天单变量实验的素材）"。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. `normal` 场景的 54 个 desc 由哪两大类构成？每一小类（fwd / dgrad / wgrad-acc / wgrad-bf16）的 major 组合、累加性和输出 dtype 分别是什么？哪几类会触发随机舍入验证？
2. `unaligned-k` 的五个 shape 分别把 K 尾的哪种形态压到极限？(2048, 8192, 4095) 在 BLOCK_K=128 下有多少个 k-block、尾块 `actual_k` 是多少？
3. 写出 K 维和 MN 维的边界截断公式（各三级）及其源码行号。`eff_mad_k=17` 时 L0 里实际的 K 维是多少？多出来的列为什么不影响正确性？
4. `d` 预填 `randn`（`common.py:190`）在测试体系中扮演什么角色？举一个能被它抓到、但"输出数值算错"类验证抓不到的内核 bug 例子。
5. `dg:` 失败、`aclnn:` 通过说明什么？反过来呢？两者同时失败时为什么优先怀疑参考结果而不是内核？
6. 对 `(1024, 2048, 17)`，`block_k` 为什么能静态推出而 `BLOCK_M` 不能？假设 `BLOCK_M=BLOCK_N=128` 后，MAD 三维是如何从 `make_gemm_candidate` 确定性推出的？
7. `can_direct_store` 有哪些条件？为什么 `unaligned-k` 的 acc+fp32 用例形状对齐时允许直写，而 `normal` 的 bf16 累加用例（如 wgrad-acc 换成 bf16 输出）不允许？
8. `DG_PRINT_CONFIGS=1` 打印的内容是什么、在哪段代码触发？用它如何核对今天静态推演中的假设部分？

### 参考答案

1. fwd 27（m ∈ {1,128,4096} × 9 个 (n,k)，K/K、非累加，7 个 bf16 输出 + 2 个 fp32 输出）+ bwd 27（9 × 三连：dgrad K/MN 非累加 bf16；wgrad-acc MN/MN 累加 fp32；wgrad-bf16 MN/MN 非累加 bf16），`generators.py:566-576`。触发 SR 的是"同 dtype + 非累加 + bf16 输出"（`common.py:215`）：fwd 的 bf16 输出 7×3=21 个、dgrad 9 个、wgrad-bf16 9 个。
2. k=4（不足一个分形）、17（1+1 列）、15（欠满分形）、234（跨块尾 106）、4095（尾块只差 1 列满块）；4095 = 31×128 + 127 → 32 个 k-block，尾块 `actual_k = 127`（`bf16_gemm.hpp:96, :114`）。
3. K：`num_k_blocks = ceil_div(eff_shape_k, BLOCK_K)`（:96）、`actual_k = min(eff_shape_k − k_blk_idx·BLOCK_K, BLOCK_K)`（:114）、`eff_mad_k = min(actual_k − k_mad_idx, MAD_K)`（:137）。MN：`get_actual_m/n = min(shape − idx·BLOCK, BLOCK)`（`scheduler.hpp:129-139`）、`eff_mad_m/n`（`bf16_gemm.hpp:125-126`）、drain `num_rows/num_cols = min(actual − offset, MAD)`（`epilogue.hpp:65-66`）。`aligned(17, 16) = 32`（`tile_ptr.hpp:52-55`）；补零列乘任何值为 0，不改变累加结果（ND2NZ 隐式零填充；bf16 无 `fill_tail_block`，`copy.hpp:333-356` 仅 8 位类型）。
4. 它是"写完整性"探测器：任何漏写的输出元素会残留随机垃圾，使 `calc_diff` 爆炸。例：drain 的 `num_cols` 若误用对齐后的 `eff_mad_n`（32）而非 `min(actual_n, MAD_N)`（17）——数值正确的部分照常写入，但若反向（把有效列算少、跳过尾 n-block），该区域保持 randn，只有预填能抓住。
5. dg 失败 + aclnn 通过：dg 内核（或其绑定链路）与两个独立参考出现分歧 → Kernel 层（或进一步用场景细分边界）。反向：ACLNN 侧问题（环境/版本/绑定），dg 已被 `d_ref` 验证。同时失败：两个独立实现共享的只有输入与 `d_ref`，两个实现"以同样方式错"的概率远低于参考算错 → 优先查参考结果（`common.py:201/:213` 都对 `d_ref` 断言）。
6. `block_k` 由规则钉死：候选只有 min/max_block_k 两档（128/256），`k % 256 ≠ 0` 排除宽块（`heuristics.hpp:620-621, :648-651`），后续 K 细化又被 `17 % 16 ≠ 0` 拦住（`:672-673`）；`BLOCK_M/N` 在 {16, 512} 内由代价模型逐档评分（`:626-657`），静态无法复现。MAD：split `(1,1)` 首先通过整除与 L0C 检查，`mad_k = min(l0a_k=128, l0b_k=128, max(16, 128/2)=64) = 64`，满足整除与下限约束 → 128×128×64（`:124-147`）。
7. 五个条件（`heuristics.hpp:526-532`）：Normal + Identity epilogue；fp4 时双 K-major；`!acc || cd_dtype == fp32`；`m % align_m == 0 && n % align_n == 0`（K-major 时 align_m=1、align_n=16）；D 指针与行跨度 32 字节对齐 + wave 条件（块数 ≤ 核数，或 m ≤ block_m 且 mad_n 满足 L2 sector 条件）。bf16 累加 + bf16 输出违反第三条（acc 且非 fp32），因为直写把 L0C 结果直接落 GM，累加只能靠原子加——bf16 的原子加语义/代价与 fp32 不同，该组合被直接排除。
8. 每个 shape 一行 `GemmDesc(m=…, major_a=…, …) GemmConfig(block=…x…x…, mad=…x…x…, l1_stages=…, direct_store=…, …)`，在 `select_gemm_config` 末尾由 `DG_JIT_DEBUG` 或 `DG_PRINT_CONFIGS` 触发、`static printed` 去重（`heuristics.hpp:749-753`；formatter 在 `gemm_config.hpp:145-186`）。用它对照 A-2：`block_k` 与 `mad` 应与推演一致，`block_m/block_n/direct_store` 即为待核对的假设项。

**A-4 分诊答案**：1. **环境**（未执行 `./develop.sh`，`deep_gemm/__init__.py:25` 的 `from . import _C` 失败）。2. **参考/环境（ACLNN 侧）**——dg 已对 `d_ref` 验证通过，`aclnn:` 是独立第二参考，全量失败指向 ACLNN 环境或其绑定，而非内核。3. **参考结果**——两个独立实现同时大幅偏离，共享项是 `d_ref` 与输入生成；查新场景的 `generate()` 是否算错。4. **Kernel（K 尾）**——只在 k<16 或 k%16≠0 的 shape 失败，指向 `eff_mad_k`/`as_mad_aligned`/零填充路径（`bf16_gemm.hpp:137-139`、`tile_ptr.hpp:52-55`）；跨块（234/4095）通过说明 `num_k_blocks`/`actual_k` 逻辑正常。5. **Kernel/host API 层**——`gemm.hpp:113-116` 的互斥检查没有生效（`common.py:191-197` 期望它抛 `RuntimeError`）。6. **布局**——输入张量最后两维 stride 都是 1，major 无法推断（`csrc/utils/layout.hpp:38-39`），属调用方布局错误。

**A-5 判定答案**：1. **真**（`generators.py:539` 的 `accumulate=True` + `:536` 的 `torch.float`；SR 条件 `common.py:215` 要求非累加）。2. **假**（`copy.hpp:335` 的 `static_assert(sizeof(T) == 1)`，bf16 依赖 ND2NZ 隐式零填充）。3. **真**（候选仅 128/256 两档且 17%256≠0，`heuristics.hpp:620-621, :648-651`）。4. **假**（写回用未对齐的 `min(actual_n − n_mad, MAD_N)`，`epilogue.hpp:66, :81-82`；对齐只约束片上计算范围）。5. **真**（`align(1,16)=16` 把 m_range 压成 {16,16}，`heuristics.hpp:608-609`）。6. **假**（`d_ref` 若算错，`aclnn:` 也应对它失败；单侧失败指向失败的那一侧实现——这里是 dg 内核）。

---

**明日预告（第 32 天）**：学习 FP8/FP4 Kernel 路径。今天的方法原样迁移：阅读 `tests/test_fp8_gemm.py` 与 `tests/test_fp4_gemm.py` 的场景构成，追踪 `csrc/jit_kernels/fp8_gemm.hpp` 与 `deep_gemm/include/deep_gemm/fp8_gemm.hpp`、`fp8_dequant_gemm.hpp`，对比 FP8×FP8、FP4×FP4、FP8×FP4 三条执行路径，找到反量化与高精度累加的位置——届时今天整理的"场景 → 边界代码"映射表会再添三列。
