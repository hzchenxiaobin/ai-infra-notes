# 第 1 周 · 第 6 天：复盘与基础练习

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 6 天（复盘日）教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 40 分钟，综合练习约 70 分钟，通过标准自评约 20 分钟。

## 一、今日定位与学习目标

今天是第 1 周的复盘日，**不引入新内容**。学习计划给今天的任务是：

1. 用 Python 或纸笔实现一个小矩阵分块计算；
2. 不看笔记解释 Tiling、stride、MAD 和 Scale；
3. 补齐前五天的疑问，暂不深究不理解的硬件指令细节。

**通过标准**：能解释一个输出 Tile 需要哪些输入数据，以及数据为什么要分层搬运。

复盘的方法论提示：前五天是"往骨架上挂细节"，今天是"把骨架抽出来检查"。判断掌握程度的唯一标准是**闭卷复述**——能不看材料讲清楚的概念才算掌握，讲不清楚的正好借今天补上。第 2 周开始就要进入真实仓库的调用链（JIT、Host、Kernel），本周这条"GEMM 问题 → 布局 → 切分 → 数值 → 硬件"的主线如果不牢，后面读源码会处处卡顿。

## 二、知识梳理

### 2.1 一条主线：前五天各回答了什么问题

| 天 | 回答的问题 | 核心结论 |
| :---: | :--- | :--- |
| 1 | 算什么、有多重？ | `D = A × B`（A `[M,K]` × B `[K,N]` → D `[M,N]`）；FLOPs = `2MNK`；大 GEMM 算术强度高达数千 FLOP/B，**性能关键是数据复用**；输入输出放不进片上存储 |
| 2 | 数据在哪、怎么找？ | 内存是一维的，**布局 = (i, j) → 线性偏移的映射规则**；stride 说明元素散布在哪，形状只说明有多少；N/T 后缀是"逻辑形状与数学形状的关系"，major 是"物理上哪根轴连续"，两者独立；视图转置零拷贝，只有视图表达不了的布局才真搬数据 |
| 3 | 怎么切、怎么省？ | `BLOCK_M/N/K` 把大 GEMM 切成 Tile；**K 是累加维必须串行，M/N 方向 Tile 彼此独立可并行**；A 行块复用因子 = `num_n_blocks`，B 行块 = `num_m_blocks`；Tile 越大复用越好，但受 L0C/L1 容量硬约束 |
| 4 | 数怎么表示？ | **指数位决定动态范围、尾数位决定相对精度**；Scaling Factor（UE8M0，2 的幂）按组撑起小类型的范围，默认粒度 `(1, 1, 32)`；低精度乘积在 FP32 中精确，所以"低精度乘 + FP32 累加"成立 |
| 5 | 硬件怎么执行？ | GM/L1/L0A/L0B/L0C/UB 六层存储（512/64/64/256/256 KiB 每核）；AIC 管搬运与乘加、AIV 管写回加工；MTE2/MTE1/M/FIX/V/MTE3 六条流水线靠事件握手；**多级 stage 让搬运与计算重叠** |

### 2.2 贯穿全周的主线索：一个输出 Tile 的一生

前五天的概念不是五个孤岛，它们共同描述同一件事——**一个输出 Tile 从被认领到写回 GM 的完整过程**。这就是今天通过标准要求你能讲出来的故事：

```text
① 认领                ② 搬入                    ③ 进 L0                 ④ 计算                     ⑤ 写出
Scheduler 把    →     GM → L1（MTE2，       →     L1 → L0A/L0B      →     mad：L0A × L0B      →     L0C → UB（FIX）
M×N 网格中的一个           多级 stage 轮转）          （MTE1）                 累加进 L0C（FP32）           → AIV 加工（V）
Tile 发给某个核                                                                                          → 写回 GM（MTE3）
```

每一步对应的前几天概念与源码锚点（文件均在 `deep_gemm/include/deep_gemm/` 下）：

| 步骤 | 用到的概念 | 源码锚点（`bf16_gemm.hpp`，BF16 kernel） |
| :---: | :--- | :--- |
| ① 认领 | 第 3 天：`ceil(M/BLOCK_M) × ceil(N/BLOCK_N)` 个 Tile；边界截断 `actual_m/actual_n` | `Scheduler` 构造（`scheduler.hpp` 第 91～93 行的 `ceil_div`）；`get_next_block`（`bf16_gemm.hpp:89`）；`get_actual_m/n`（`scheduler.hpp` 第 129～139 行） |
| ② 搬入 | 第 1 天：放不下所以要分块；第 3 天：K 切成 `ceil(K/BLOCK_K)` 步；第 5 天：MTE2 + 多级 stage | K 循环（`bf16_gemm.hpp:112`）；`copy_gm_to_l1`（第 117～118 行）；`actual_k`（第 114 行） |
| ③ 进 L0 | 第 5 天：MAD 只认 L0A/L0B 这两个专用缓冲 | `copy_l1_to_l0a/l0b`（`bf16_gemm.hpp` 第 142～145 行） |
| ④ 计算 | 第 4 天：FP32 累加器；第 5 天：M 流水线、`is_first`/`ENABLE_KEEP` | `l0c_ptr<float> l0c`（`bf16_gemm.hpp:64`）；`mad(..., ENABLE_KEEP, is_first)`（第 149～151 行） |
| ⑤ 写出 | 第 5 天：FIX → AIV/V 加工 → MTE3；`kDirectStore` 旁路 | `copy_l0c_to_gm`（第 156～159 行，直通）或 `epilogue.aic_store`（第 161 行） |

一个 Tile 的完整输入需求（第 3 天的核心结论，也是通过标准的第一半）：

> **输出 Tile `(i, j)` 需要一个 A 行块（`A[i·BM : (i+1)·BM, 全部 K]`）加一个 B 行块（`B[j·BN : (j+1)·BN, 全部 K]`），K 维按 `BLOCK_K` 分步搬入、逐步累加进同一块 L0C，走完 K 才得到最终结果。**

### 2.3 四个核心概念的标准表述（闭卷对照用）

学习计划要求今天"不看笔记解释 Tiling、stride、MAD 和 Scale"。下面是参考答案，**先闭卷写自己的版本（练习 2），再回来对照**。每个概念给出定义、动机和源码锚点。

**Tiling（分块）**：把一个放不进片上存储的大 GEMM，切成输出侧 `ceil(M/BM) × ceil(N/BN)` 个 Tile、收缩维 `ceil(K/BK)` 步的计算策略。动机有两个：容量（真实 shape 的输入是片上缓冲的数百倍）和复用（A 行块被 N 方向所有 Tile 共享、B 行块被 M 方向所有 Tile 共享，Tile 越大每字节数据支撑的计算越多）。M/N 方向的 Tile 互相独立、可任意分配给 32 个核；K 是累加维，部分和必须落在同一块 L0C 上，只能串行。
源码锚点：`GemmConfig` 的 `block_m/block_n/block_k`（`csrc/jit_kernels/gemm_config.hpp` 第 98～100 行，注释 "Execution configuration: how to compute (tiling, pipeline, launch)"）；`Scheduler` 构造函数的 `ceil_div`（`scheduler.hpp` 第 91～93 行）；kernel 的 K 循环（`bf16_gemm.hpp:112`）；容量断言（第 66～69 行的四个 `static_assert`）。

**stride（步长）**：沿张量某个维度走一步，地址跳过多少个**元素**。它和形状是两个独立信息——形状说"有多少元素"，stride 说"这些元素散布在哪里"。Row-major `[M, K]` 的 stride 是 `(K, 1)`，column-major 是 `(1, M)`；切片、行间空洞都靠 stride 表达。kernel 寻址只需要"内层轴连续 + 一个外层 stride"两个信息，major 描述内层连续轴是 K 还是 MN。
源码锚点：`get_outer_stride()`（`csrc/jit_kernels/gemm_config.hpp` 第 23～34 行，注释写明 K-major/MN-major 各取 `stride(-2)`/`stride(-1)`）；`gm_ptr::index()`（`common.hpp` 第 156～165 行，按 major 走两个分支的地址公式）；`Major` 枚举（`common.hpp:82`）。

**MAD（Matrix Multiply-Accumulate，矩阵乘加）**：cube 计算单元执行的基本运算——从 L0A 取 A 片段、L0B 取 B 片段，矩阵乘积**累加**进 L0C。一个输出 Tile 的全部 K 步都累加进同一块 L0C：首个片段清零累加器（`is_first` → `clear_C`），后续片段继续累加（`ENABLE_KEEP`）。L0C 恒为 FP32，与操作数 dtype 无关；FP8/FP4 操作数走 `asc_mmad_mx`，把两侧 UE8M0 缩放融合进乘加。
源码锚点：`mad()`（`deep_gemm/include/deep_gemm/ascend/copy.hpp` 第 370～381 行，按操作数字节数分派 `asc_mmad`/`asc_mmad_mx`）；调用点（`bf16_gemm.hpp` 第 149～151 行）；L0C 声明（第 64 行 `l0c_ptr<float>`）。

**Scale（Scaling Factor，缩放因子）**：低精度类型的动态范围太小（E4M3 上限 448、E2M1 只有 16 个码点），所以把张量按粒度分组，每组除以一个缩放因子再量化，用时乘回来：`x ≈ round(x/sf) × sf`。仓库的 sf 是 UE8M0（纯 8 位指数，恒为 2 的幂），取"不小于 `amax/448`（E4M3）或 `amax/6`（E2M1）的最小 2 的幂"——向上取整保证量化永不饱和。默认粒度 `(1, 1, 32)`：每行、K 方向每 32 个元素一个 sf，相邻两个沿 K 打包进一个 int16，存储为 MN-major。SF 只能撑范围、撑不了分辨率，所以低精度乘法仍要 FP32 累加。
源码锚点：参考量化实现 `quantize_with_major()`（`tests/generators.py` 第 61～95 行，`sf_div = 448.0 / 6.0` 在第 80 行）；`ceil_to_ue8m0()`（第 27～31 行）；`gran_k == 32` 断言（`csrc/apis/layout.hpp:55`）；计算侧融合（`ascend/copy.hpp` 第 374～376 行的 `asc_mmad_mx` 分支）。

### 2.4 前五天要点速查表

| 概念 | 一句话 | 源码落点 | 详见 |
| :--- | :--- | :--- | :---: |
| M/N/K | K 是被消耗的收缩维，M/N 是输出保留维；K=0 直接清零返回 | `get_gemm_desc`（`csrc/apis/gemm.hpp:60`）、`gemm_early_return` | 第 1 天 |
| 2MNK / 算术强度 | 计算量与数据量之比决定算力受限还是带宽受限 | README 性能表 | 第 1 天 |
| N/T 后缀 | 传入张量相对数学形状是否转置；四种后缀算同一个 D | `bf16_gemm<false, true>` 等四个别名（`csrc/apis/gemm.hpp` 第 450～453 行） | 第 2 天 |
| Major::K / MN | 物理上哪根轴在内存里连续 | `common.hpp:82` | 第 2 天 |
| 视图转置 | 只改 shape/stride 元数据，零拷贝 | `get_transposed`（`csrc/utils/layout.hpp` 第 88～99 行） | 第 2 天 |
| 输出 Tile 数 | `ceil(M/BM) × ceil(N/BN)`；K 步数 `ceil(K/BK)` | `scheduler.hpp` 第 91～93 行 | 第 3 天 |
| 复用因子 | A 行块 = `num_n_blocks`，B 行块 = `num_m_blocks` | swizzle 与 L2 hint 的设计依据 | 第 3 天 |
| `actual_m/n/k` | 边界 Tile / K 步的有效尺寸 | `scheduler.hpp` 第 129～139 行、`bf16_gemm.hpp:114` | 第 3 天 |
| BF16/FP8/FP4 | 16/8/4 位；指数定范围、尾数定精度；FP4 两值打包一个 int8 | `kPackedFP4`（`csrc/utils/dtype.hpp:9`） | 第 4 天 |
| FP32 累加 | L0C 恒为 `l0c_ptr<float>`；低精度乘积在 FP32 中精确 | `bf16_gemm.hpp:64`、`fp8_gemm.hpp:87` | 第 4 天 |
| 六层存储 | GM / L1(512K) / L0A(64K) / L0B(64K) / L0C(256K) / UB(256K) 每核 | `common.hpp` 第 50～58 行 | 第 5 天 |
| AIC / AIV | `__mix__(1, 2)`：AIC 搬运+MAD，AIV epilogue 写回 | `bf16_gemm.hpp:27`、`ASCEND_IS_AIC`（第 81 行） | 第 5 天 |
| 六条 pipe | MTE2（GM→L1）、MTE1（L1→L0）、M（乘加）、FIX（L0C→UB）、V（向量）、MTE3（UB→GM） | `ascend/copy.hpp` 各封装 | 第 5 天 |
| 事件握手 | 满事件（写完报给读）/ 空事件（读完释放给写） | `ascend/sync.hpp` 第 15～27 行 | 第 5 天 |
| 多级 stage | 同一缓冲切成轮转的乒乓份，掩盖搬运延迟 | `l1a[s]`、`l0a[s_ab]`、`l0c[s_cd]` | 第 5 天 |

### 2.5 易错点盘点

以下是前五天最容易混淆的地方，复盘时逐条自查：

1. **K 与 M/N 的地位不对称**：K 是唯一"被消耗"的维度（`k == 0` 时没有乘法可做），且必须串行累加；M/N 方向的 Tile 互相独立、可任意并行。调度器只按 M×N 网格发任务、K 循环在核内走，根源就在这。
2. **N/T 与 major 是两个层面**：N/T 说"逻辑形状与数学形状的关系"（`nt` 的 b 传 `[N, K]`），major 说"物理上哪根轴连续"（`[N, K]` 可以是 K-major 也可以是 MN-major）。不要用"是不是转置过"来推断 major。
3. **transpose 不搬数据**：`transpose()` 只交换 shape/stride，`data_ptr` 不变；`.contiguous()` 才真搬。判断口诀：视图能表达的（转置、切片、加间隙）用视图，视图表达不了的（打包、分形重排）才搬。
4. **复用因子方向别记反**：A 行块被 **N 方向**的每个 Tile 用一次（因子 = `num_n_blocks`），B 行块被 **M 方向**的每个 Tile 用一次（因子 = `num_m_blocks`）。
5. **L0C 恒为 FP32**：与操作数是 BF16、FP8 还是 FP4 无关（三个 kernel 的声明都是 `l0c_ptr<float>`）；精度转换发生在输出侧（FIX 搬运的量化模式或 epilogue 算子）。
6. **BLOCK 与 MAD 是两级尺寸**：BLOCK 是 L1 级的搬运/调度单位，MAD 是 L0 级、真正喂给 cube 单元的计算子块，约束 `BLOCK_K % MAD_K == 0`（`bf16_gemm.hpp:70`）。
7. **多级 stage 不是硬件 cache**：它是一块缓冲按 `offset()` 切成的轮转份，何时可读、何时可写完全靠 `asc_sync_notify/wait` 的满/空事件控制——软件显式管理，不是自动缓存。
8. **容量账要记牢**：每核 L1 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB、UB 256 KiB，32 核且每核私有。看到任何 BLOCK 配置，第一反应是算它占多少 L1/L0C。

## 三、综合练习

以下练习对应学习计划的三项任务，全部**不需要 Ascend 硬件**。

### 练习 1：实现一个小矩阵分块计算（约 35 分钟）

这是今天的主练习：用代码把"一个输出 Tile 的一生"实现一遍。约定与仓库一致——B 采用 `[N, K]` 视角（`nt` 约定，计算 `D = A @ B^T`），这样 `b[j][p]` 直接对应"B 的第 j 行"。

在自己的练习目录新建 `tiled_gemm.py`（不要改动仓库文件），写入：

```python
def ceil_div(a, b):
    return -(-a // b)            # 向上取整，对应 kernel 的 ceil_div（common.hpp:41）

def gemm_naive(a, b):
    """参考实现：a [M, K]、b [N, K]（仓库 nt 视角），D = A @ B^T。"""
    m, n, k = len(a), len(b), len(b[0])
    d = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                d[i][j] += a[i][p] * b[j][p]
    return d

def gemm_tiled(a, b, bm, bn, bk, trace=False):
    """分块实现：外两层 = 输出 Tile 网格，内层 = K 维分步累加。"""
    m, n, k = len(a), len(b), len(b[0])
    d = [[0] * n for _ in range(m)]
    num_mb, num_nb = ceil_div(m, bm), ceil_div(n, bn)
    a_used = [set() for _ in range(num_mb)]   # 每个 A 行块被哪些 n 方向 Tile 使用
    b_used = [set() for _ in range(num_nb)]   # 每个 B 行块被哪些 m 方向 Tile 使用
    for m_blk in range(num_mb):
        m_idx = m_blk * bm                       # ← scheduler.get_global_idx（scheduler.hpp:113）
        am = min(m - m_idx, bm)                  # ← scheduler.get_actual_m（scheduler.hpp:129）
        for n_blk in range(num_nb):
            n_idx = n_blk * bn
            an = min(n - n_idx, bn)              # ← scheduler.get_actual_n（scheduler.hpp:137）
            a_used[m_blk].add(n_blk)
            b_used[n_blk].add(m_blk)
            acc = [[0] * an for _ in range(am)]  # 累加器：对应 L0C（FP32）
            for k_blk in range(ceil_div(k, bk)): # K 维循环（bf16_gemm.hpp:112）
                k_idx = k_blk * bk
                ak = min(k - k_idx, bk)          # actual_k（bf16_gemm.hpp:114）
                for ii in range(am):             # 一次“MAD”：本 K 片段的部分和
                    for jj in range(an):
                        s = 0
                        for pp in range(ak):
                            s += a[m_idx + ii][k_idx + pp] * b[n_idx + jj][k_idx + pp]
                        acc[ii][jj] += s         # 首步清零（is_first）、后续累加（ENABLE_KEEP）
            for ii in range(am):                 # K 走完，写回 D
                for jj in range(an):
                    d[m_idx + ii][n_idx + jj] = acc[ii][jj]
            if trace:
                print(f"Tile({m_blk},{n_blk}) 计算 D[{m_idx}:{m_idx+am},{n_idx}:{n_idx+an}]"
                      f"  K 步数 = {ceil_div(k, bk)}  acc = {acc}")
    return d, a_used, b_used

if __name__ == "__main__":
    M, N, K = 5, 5, 10
    BM, BN, BK = 2, 2, 4
    a = [[10 * i + k + 1 for k in range(K)] for i in range(M)]
    b = [[10 * j + k + 1 for k in range(K)] for j in range(N)]

    d_ref = gemm_naive(a, b)
    d_tiled, a_used, b_used = gemm_tiled(a, b, BM, BN, BK, trace=True)
    assert d_tiled == d_ref, "分块实现与参考实现不一致"
    print("分块实现与参考实现逐元素一致")
    print(f"num_m_blocks={ceil_div(M, BM)}  num_n_blocks={ceil_div(N, BN)}  "
          f"num_k_blocks={ceil_div(K, BK)}  输出 Tile 数={ceil_div(M, BM) * ceil_div(N, BN)}")
    print("actual_m:", [min(M - i * BM, BM) for i in range(ceil_div(M, BM))])
    print("actual_n:", [min(N - j * BN, BN) for j in range(ceil_div(N, BN))])
    print("actual_k:", [min(K - t * BK, BK) for t in range(ceil_div(K, BK))])
    print("A 行块复用因子（被几个 Tile 使用）:", [len(s) for s in a_used])
    print("B 行块复用因子:", [len(s) for s in b_used])
```

运行 `python tiled_gemm.py`，预期输出（9 行 Tile 轨迹，首尾两行如下）：

```text
Tile(0,0) 计算 D[0:2,0:2]  K 步数 = 3  acc = [[385, 935], [935, 2485]]
Tile(0,1) ...（中间 7 行省略）
Tile(2,2) 计算 D[4:5,4:5]  K 步数 = 3  acc = [[20785]]
分块实现与参考实现逐元素一致
num_m_blocks=3  num_n_blocks=3  num_k_blocks=3  输出 Tile 数=9
actual_m: [2, 2, 1]
actual_n: [2, 2, 1]
actual_k: [4, 4, 2]
A 行块复用因子（被几个 Tile 使用）: [3, 3, 3]
B 行块复用因子: [3, 3, 3]
```

（抽查：`Tile(2,2)` 的 `acc = [[20785]]`，即 `D[4][4] = Σ v²（v = 41..50） = 42925 − 22140 = 20785`，三个 K 步的部分和分别是 7230、8654、4901。）

对照输出回答四个问题（答案见括号）：

1. `acc` 什么时候清零、什么时候写回 D？（每个 Tile 开始时归零——对应 `is_first`/`clear_C`；`ceil_div(K, BK)` 个 K 步走完才写回——"K 是累加维"的代码体现。）
2. A/B 行块的复用因子各是多少，与第 3 天的公式一致吗？（都是 3，即 A 侧 = `num_n_blocks`、B 侧 = `num_m_blocks`，本例两者恰相等。）
3. 把 `BK` 改成 3（K=10 不再整除，`actual_k = [3,3,3,1]`），assert 还通过吗？这说明什么？（通过。分块方式的正确性不依赖 BLOCK 取值——这正是真实 kernel 能按 shape 自由搜索配置的前提。）
4. 把 `M` 改成 1 呢？（`num_m_blocks=1`、`actual_m=[1]`，退化为 GEMV，对应仓库测试里的 `m=1` 解码场景。）

**纸笔替代任务**（无 Python 环境时必做，有环境也建议做）：用 `BLOCK_M=2、BLOCK_N=2、BLOCK_K=3` 手工演算下面的例子，画出 Tile 网格、列出每个 Tile 每个 K 步搬运的 A/B 片段和累加过程，最后与附录 A 的参考答案对照。

```text
A = │ 1  2  3  4 │      B = │ 2  0  1  1 │      （A 是 3×4，B 是 2×4，nt 视角）
    │ 5  6  7  8 │          │ 0  1  3  2 │
    │ 9 10 11 12 │
```

### 练习 2：闭卷解释四个概念（约 20 分钟）

合上本教程和前五天的全部笔记，对 **Tiling、stride、MAD、Scale** 各写 3～5 句话，要求覆盖：它是什么、为什么需要它、在仓库哪个环节出现。写完后对照 2.3 节修订，用红笔标出遗漏或说错的点。

再闭卷回答四个追问（答案见括号）：

1. Tiling：为什么 K 维的循环不能拆给不同核并行，而 M/N 方向的 Tile 可以？（K 是累加维，部分和必须累加进同一块 L0C；M/N 方向的 Tile 之间没有累加关系。）
2. stride：逻辑形状都是 `[64, 32]` 的两个张量，stride 分别为 `(32, 1)` 和 `(1, 64)`，区别是什么？各是什么 major？（前者行优先紧凑、K-major、contiguous；后者 MN-major、非 contiguous——但同样合法，kernel 走另一个寻址分支。）
3. MAD：`clear_C` 与 `ENABLE_KEEP` 分别在 K 循环的什么时刻起作用？（首个 K 片段清零累加器，其余片段继续累加进同一块 L0C；这样输出数据在 K 循环期间不用反复读写。）
4. Scale：为什么 SF 取"不小于 `amax/448` 的最小 2 的幂"，而不是 `amax/448` 本身？（SF 是 UE8M0、只有 8 位指数，只能取 2 的幂；向上取整保证 `amax/sf ≤ 448` 恒成立、量化永不饱和，代价是最多浪费一倍头部空间。）

### 练习 3：一个输出 Tile 需要哪些输入数据、为什么要分层搬运（约 25 分钟，通过标准的核心演练）

不看 2.2 节，独立完成下面的演算，再对照附录 B。

**Part A：Tile 输入账。** 设 `M=130、N=100、K=320`（BF16，Normal GEMM，B 为 `[N, K]` 视角），`BLOCK_M=64、BLOCK_N=64、BLOCK_K=128`：

1. 写出 `num_m_blocks / num_n_blocks / num_k_blocks`、输出 Tile 总数，以及每级行块/列块/K 步的 `actual` 尺寸；
2. 输出 Tile `(1, 1)` 计算的是 D 的哪个区域？它需要 A 的哪一块、B 的哪一块（用行号区间表示，注意沿 K 全长）？各多少字节？
3. 它的 K 循环分几步？每步搬运的 A/B 片段多大？
4. A 行块 1 和 B 行块 1 分别被哪些 Tile 使用？复用因子各是多少？
5. 计算这个 shape 的无复用 GM 读取量与完美复用读取量（BF16），求放大倍数。

**Part B：为什么分层搬运。** 用三条理由回答"数据为什么要走 `GM → L1 → L0A/L0B → L0C → … → GM`，而不是一步到位"，要求至少一条带具体数字。提示（先自己写再看提示）：容量（用真实 shape `(4096, 7168, 16384)` 算）、计算单元的取数约束（MAD 从哪取数）、流水重叠（多级 stage 的作用）。本例 `M=130` 的 shape 故意选得很小——它的 A+B 只有约 144 KiB，甚至塞得进单核 L1；想一想即便如此，数据为什么仍然必须经过 L1 → L0 这条路径。

### 练习 4：补齐前五天的疑问（约 15 分钟）

把前五天笔记末尾"遗留问题记录"里的条目翻出来逐条处理：

- 已解决的：用一句话写出答案，划掉；
- 未解决的：填进下表对应的"后续主题"，下周开始按天消化；
- 表里没有对应天数的：记为第 7 天（机动日）或第 14/21/28 天的任务。

| 疑问（示例） | 后续主题 |
| :--- | :--- |
| `asc_mmad` / `asc_mmad_mx` 指令内部如何乘加、宽度多少 | 第 24 天（MAD 专题） |
| NZ 分形布局的位级排布、`FRAC_K` 的由来 | 第 18～19 天（地址与搬运） |
| `GROUP_M` swizzle 把任务编号映射成 Tile 的具体规则 | 第 17 天（调度专题） |
| `BLOCK_K` 与 `MAD_K` 两层 K 循环如何配合 | 第 22 天 |
| JIT 如何生成模板实例化代码、为什么需要 | 第 12 天 |
| SF 布局转换 kernel（`transform_sf`）的流水细节 | 第 33 天 |
| L1/L0 多级 stage 与事件握手的完整时序 | 第 23 天 |
| Epilogue / AIC-AIV 如何协作写回 | 第 25 天 |
| `kResidentA/B`、`kDirectStore` 特殊路径 | 第 26 天 |
| K-Grouped 为什么强制 A、B 都是 MN-major | 第 37 天 |

同时执行计划的另一条要求——**暂不深究硬件指令细节**：本周遗留的 `asc_copy_gm2l1_nd2nz` 参数含义、事件编号规则、分形对齐约束这类问题，只要能说出"它在数据流的哪一段、由谁发起"就算达标，位级细节留给第 3～4 周的专题日。

### （可选）有 Ascend 环境时

在仓库根目录执行：

```bash
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
```

`DG_PRINT_CONFIGS=1` 会让每个 shape 先打印一行 `GemmDesc(...) GemmConfig(...)`（打印逻辑在 `csrc/jit_kernels/heuristics.hpp` 的 `select_gemm_config` 末尾）。挑两个 shape，把打印出的 `block=` 代入练习 3 的方法，心算它的 Tile 数与 K 步数。`--skip-prof` 表示只测正确性、跳过性能测量（见 `tests/test_bf16_gemm.py` 的 argparse 定义；`select` 可选 `normal / unaligned-k / unaligned-mn / strided / mgrouped / kgrouped`）。

**无硬件替代任务**：静态阅读 `tests/test_bf16_gemm.py` 的 `__main__` 段，列出六种测试场景分别对应本周哪个概念（提示：`unaligned-k` ↔ `actual_k` 边界截断，`strided` ↔ 第 2 天的 stride 寻址）。

## 四、通过标准

### 4.1 自评清单

学习计划的通过标准是"**能解释一个输出 Tile 需要哪些输入数据，以及数据为什么要分层搬运**"。拆成六个检查点，逐条自评"能 / 勉强 / 不能"：

| # | 检查点 | 验证方式 |
| :---: | :--- | :--- |
| 1 | 给定 shape 与 BLOCK，能列出任意输出 Tile 的 D 区域、所需的 A 行块与 B 行块（沿 K 全长）、K 步数与每步 `actual_k` | 练习 3 Part A 第 1～3 问独立做对 |
| 2 | 能说出 A/B 行块的复用因子及来源（`num_n_blocks` / `num_m_blocks`），并解释提高复用为什么减少 GM 访问 | 练习 3 Part A 第 4～5 问；练习 1 输出的复用因子 |
| 3 | 能从**容量**（带数字：单核 L1 512 KiB vs 真实 shape 输入数百 MiB）、**取数约束**（MAD 只认 L0A/L0B）、**流水重叠**（多级 stage 让 MTE2/MTE1/M 并行）三个角度解释分层搬运 | 练习 3 Part B 写出三条理由 |
| 4 | 能完整复述 `GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 数据流，并把每段对应到 `copy.hpp` 的具体函数 | 对照 2.2 节的表 |
| 5 | 闭卷讲清 Tiling、stride、MAD、Scale 四个概念 | 练习 2 闭卷作答后对照 2.3 无重大遗漏 |
| 6 | 分块计算实现正确 | 练习 1 的 assert 通过（或纸笔任务与附录 A 一致） |

### 4.2 评判与补救

- **六条全部"能"**：达标，进入第 7 天机动或休息；下周开始读仓库结构与 JIT 调用链。
- **检查点 1 或 3 "不能"**：这两条正是通过标准本身。回到第 3 天 3.2～3.3 节、第 5 天 3.1 与 3.6 节重读，重做练习 3。
- **检查点 2 "不能"**：重读第 3 天 3.4～3.5 节（复用的量化）。
- **检查点 4 "不能"**：重读第 5 天 3.4～3.6 节，重画数据流图。
- **检查点 5 "不能"**：只补不会的概念对应的当天（Tiling→第 3 天、stride→第 2 天、MAD→第 4/5 天、Scale→第 4 天）。
- **检查点 6 "不能"**：对照附录 A 找出 bug——最常见的是忘了 `actual` 截断（越界）或把"沿 K 全长"写成了"只搬一个 K 片段"。

### 4.3 本周明确不要求掌握的内容

与计划"暂不深究不理解的硬件指令细节"一致，以下内容**不算**本周未达标项：`asc_*` 内建指令的参数与位级语义、NZ 分形布局细节、swizzle 的逐 block 推演、`MAD_K` 内层循环、SF 转换 kernel 的流水、JIT 代码生成、AIC-AIV 事件编号。它们各有专题日（见练习 4 的表）。本周只要能把这些东西**放回 2.2 节那张五步图的正确位置**即可。

## 附录 A：练习 1 纸笔任务参考答案

`A` 为 `[3, 4]`、`B` 为 `[2, 4]`，D 为 `[3, 2]`。`BLOCK_M=2、BLOCK_N=2、BLOCK_K=3`：

```text
num_m_blocks = ceil(3/2) = 2      actual_m = [2, 1]
num_n_blocks = ceil(2/2) = 1      actual_n = [2]
num_k_blocks = ceil(4/3) = 2      actual_k = [3, 1]
输出 Tile 数 = 2 × 1 = 2
```

**Tile (0, 0)**：计算 `D[0:2, 0:2]`，需要 `A[0:2, 0:4]` 与 `B[0:2, 0:4]`（B 行块 0）：

| K 步 | 搬入 A 片段 | 搬入 B 片段 | 部分和（acc 累加后） |
| :---: | :--- | :--- | :--- |
| t=0（actual_k=3） | `[[1,2,3],[5,6,7]]` | `[[2,0,1],[0,1,3]]` | `[[5,11],[17,27]]` |
| t=1（k 从 3 起，actual_k=1） | `[[4],[8]]` | `[[1],[2]]` | `[[9,19],[25,43]]` |

**Tile (1, 0)**：计算 `D[2:3, 0:2]`（`actual_m=1`，边界 Tile），需要 `A[2:3, 0:4]` 与 `B[0:2, 0:4]`：

| K 步 | 搬入 A 片段 | 搬入 B 片段 | 部分和（acc 累加后） |
| :---: | :--- | :--- | :--- |
| t=0 | `[[9,10,11]]` | `[[2,0,1],[0,1,3]]` | `[[29,43]]` |
| t=1 | `[[12]]` | `[[1],[2]]` | `[[41,67]]` |

最终结果（与朴素三重循环一致）：

```text
      ┌        ┐
      │  9  19 │
D  =  │ 25  43 │
      │ 41  67 │
      └        ┘
```

观察三点：B 行块 0 被两个 Tile 各搬了一遍（复用因子 = `num_m_blocks` = 2，无跨 Tile 复用时同样的数据读两次）；两个 Tile 互不依赖、可以分给两个"核"并行；每个 Tile 内部 K 步必须串行累加——这三点合起来就是第 3 天的全部结论。

## 附录 B：练习 3 参考答案

**Part A：**

1. `num_m_blocks = ceil(130/64) = 3`，`num_n_blocks = ceil(100/64) = 2`，`num_k_blocks = ceil(320/128) = 3`；输出 Tile 数 = 6。`actual_m = [64, 64, 2]`，`actual_n = [64, 36]`，`actual_k = [128, 128, 64]`。
2. Tile `(1, 1)` 计算 `D[64:128, 64:100]`。需要 **A 行块 1 = `A[64:128, 0:320]`**（64×320 元素 = 40960 B = 40 KiB）和 **B 行块 1 = `B[64:100, 0:320]`**（36×320 元素 = 23040 B = 22.5 KiB），都沿 K 全长。
3. K 循环 3 步：前两步各搬 `64×128` 的 A 片段（16 KiB）和 `36×128` 的 B 片段（9 KiB）；第三步（边界）搬 `64×64`（8 KiB）和 `36×64`（4.5 KiB）。每步累加进同一块 L0C，走完 K 才写回 `D[64:128, 64:100]`。
4. A 行块 1 被 Tile `(1,0)` 和 `(1,1)` 使用，复用因子 = `num_n_blocks` = 2；B 行块 1 被 Tile `(0,1)` 和 `(1,1)` 使用，复用因子 = `num_m_blocks` = 3。
5. 无复用 GM 读 = `(num_n_blocks·M + num_m_blocks·N)·K·s = (2×130 + 3×100)×320×2 = 358400 B`（350 KiB）；完美复用 = `(M+N)·K·s = 230×320×2 = 147200 B`（约 143.75 KiB）；放大倍数 ≈ 2.43（A 侧 2 倍、B 侧 3 倍的加权）。

**Part B：为什么要分层搬运：**

1. **容量（真实 shape 下是硬约束）**：README 性能表的 `(4096, 7168, 16384)` BF16，A+B 共 352 MiB，是单核 L1（512 KiB）的 704 倍，32 核私有 L1 合计（16 MiB）也差 22 倍，且还要切成多级流水 stage——片上每次只能持有矩阵的一小块，必须分块逐段搬运。本练习的 `M=130` shape 输入只有约 144 KiB，虽塞得进 L1，但真实负载不是这样。
2. **计算单元的取数约束**：MAD 只从 L0A/L0B 这两个专用缓冲（各 64 KiB）取操作数，GM 的数据无论大小都必须经 L1 中转进 L0——`GM → L1 → L0A/L0B` 是唯一通路（`copy_gm_to_l1` → `copy_l1_to_l0a/b`），不是性能选择。
3. **流水重叠（分层搬运的收益）**：L1/L0 切成多级 stage 后，MTE2（GM→L1）预取下一块、MTE1（L1→L0）搬当前块、M（乘加）算上一块三线并行，搬运延迟被计算掩盖；若只有单块缓冲，三级只能互相干等。此外 L0C 专职 FP32 累加（一个 Tile 的所有 K 步累加在同一块，输出不必反复写回）、跨核共享的行块经 L2 复用（调度 swizzle 让 32 核工作在相邻 Tile 上），也都是"分层"带来的复用。

一句话总结（通过标准的标准答法）：**一个输出 Tile 需要一个 A 行块加一个 B 行块（都沿 K 全长，按 `BLOCK_K` 分步搬入、FP32 累加进同一块 L0C）；数据要分层搬运，因为片上每一层的容量、带宽、职责都不同——GM 存全量、L1 做多级中转、L0A/L0B 直接喂乘加单元、L0C 累加、UB 收尾写回，分层既解决"放不下"，又让搬运与计算重叠、让共享数据被复用。**

## 附录 C：第 1 周复盘笔记模板（建议产出）

```text
# 第 1 周复盘笔记

1. 一句话主线
   - GEMM 是 ____ 问题；性能关键是 ____；所以 kernel 的形态是 ____（分块 + 分层搬运 + 流水）

2. 一个输出 Tile 的一生（闭卷默写五步图）
   -（默写 2.2 节的图，标注每步的源码函数名）

3. 四个概念自查
   - Tiling：能 / 勉强 / 不能（不能则回到第 3 天 ____ 节）
   - stride：……    - MAD：……    - Scale：……

4. 我的手算 / 代码记录
   -（贴练习 1 的输出或纸笔演算照片）

5. 遗留问题与去向
   -（抄练习 4 处理后的清单，标注对应的天数）

6. 自评结论
   - 六个检查点：___ 个"能"；未达标项与补救计划：____
```
