# 第 3 周 · 第 3 天（总第 17 天）：理解 Normal GEMM 调度

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前两天我们读完了 `bf16_gemm_impl` 的 26 个模板参数和五行 `constexpr` 的片上内存规划——"一个输出 Tile 的数据要流过哪些缓冲"已经清楚。但还剩一个最基本的问题没有回答：**输出矩阵被切成几十上百个 Tile，到底哪个核（AI Core）在什么时候算哪一个？** 答案全部写在 `deep_gemm/include/deep_gemm/scheduler.hpp`（249 行）里：`bf16_gemm.hpp:73` 用设备内置变量 `block_idx` 构造 `Scheduler`，AIC 侧（第 89 行）和 AIV 侧（第 180 行）各有一个 `while (scheduler.get_next_block(...))`，每次迭代领一个输出 Tile。今天只聚焦 **Normal GEMM** 分支——持久化（persistent）任务分发、`GROUP_M` 混洗、XOR 与蛇形遍历的几何含义、`get_global_idx()` 与 `get_actual_m/n()` 的边界处理；M-Grouped / K-Grouped / Batched 分支今天只扫注释，留到第 36～40 天。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 解释**持久化调度模式**：kernel 恰好以 `kNumCores` 个 block 启动（`csrc/jit_kernels/heuristics.hpp:98` 把 `launch_options.num_blocks` 设为 `desc.num_cores`，源头是 `csrc/apis/gemm.hpp:176` 的 `runtime->get_num_sms()`）；每个核循环调用 `get_next_block()` 直到返回 `false`，第 `t` 次调用领到的全局块号是 `(++current_iter) * kNumCores + core_id`（`scheduler.hpp:142`）；由此定义 **wave**（一轮 32 个核并行处理的 Tile 批次），`num_waves = ceil_div(num_blocks, kNumCores)`（Host 镜像：heuristics.hpp:219）；
2. 说出 `Scheduler` 的 7 个模板参数（`scheduler.hpp:42-48`）各自的作用，以及 `kGroupM=4`、`kXorN=true`、`kSnake=true` 在本仓库所有 GEMM kernel 中都取默认值（`bf16_gemm.hpp:73`、`fp8_gemm.hpp:97`、`fp8_dequant_gemm.hpp:194` 实例化时均未传这三个参数）；
3. 手算构造函数（`scheduler.hpp:87-108`）派生的 5 个网格常量：`num_m_blocks`、`num_n_blocks`、`num_blocks`、`tail_m_start`、`tail_start`，并解释后两个为什么是"能凑满 `kGroupM` 行的混洗区"与"剩余 m 尾部行"的分界；
4. 画出 `get_swizzled_block()`（`scheduler.hpp:202-248`）的三段式区域划分——小 M 快路径、m 尾部行、混洗区（superrow → 4×8 块组 → XOR + 蛇形），并对任意给定 `block_idx` 手工算出 `(m_block_idx, n_block_idx)`；
5. 解释 XOR（`n_local ^= m_local`，第 232 行）与蛇形（`ng = n_groups - 1 - ng`，第 235 行）各自改写哪个下标、不改变什么（工作集仍是 4×8 矩形）；说明每个核的 `m_local` 恒等于 `core_id % 4`，以及 `n_local` 在什么条件下也恒定；再把这套几何与 kernel 里的 `reuse_a/reuse_b`（`bf16_gemm.hpp:92-93`、`170-171`）联系起来；
6. 写出 `get_global_idx()`（`scheduler.hpp:113-115`）与 `get_actual_m/get_actual_n`（129-139 行）的 Normal 公式，指出 kernel 中的消费点（`bf16_gemm.hpp:90-91`、`105-106`），并说明非整块 Tile 如何"少搬少算"；
7. 说明 AIC 与 AIV 如何共享同一调度序列：每个 block 固定 1 个 AIC + 2 个 AIV（`__mix__(1, kDirectStore ? 0 : 2)`，`bf16_gemm.hpp:27`），两侧各有一个 `while (get_next_block)` 循环（89、180 行），两个 AIV 用 `asc_get_sub_block_id()` 区分（`epilogue.hpp:51`）；
8. 把 Host 侧启发式里的 `num_waves`（heuristics.hpp:215-219）、`kBlockSwizzleM = 4`（211、257 行，与设备 `kGroupM` 互为镜像）、epilogue stage 数的 wave 上界（192-193 行）与设备侧调度逐一对应。

**当天产出：一张 M/N Tile 调度示意图**（模板见第六节，参考答案见 3.6 节）。

## 二、预备知识

- **第 15 天（最重要）**：`kNumCores` 模板参数的来源链（`launch_options.num_blocks` ← `desc.num_cores` ← `runtime->get_num_sms()`）；`block_idx` 是设备内置变量（当前核编号），第 73 行作为 `core_id` 传给 `Scheduler`。
- **第 16 天**：L1/L0A/L0B/L0C 的多级缓冲与 stage 游标——调度决定"哪个 Tile 的数据此刻流过这些缓冲"，两者是一体两面。
- **第 12 天**：`GemmConfig` 的字段与启发式搜索；注意 `launch_options.num_blocks`（网格大小）与 `Scheduler::num_blocks`（Tile 总数）是两个不同的量。
- **第 3 天**：`BLOCK_M/N/K` 的 Tiling、输出 Tile 数、每个 Tile 需要读哪些 A/B 数据——今天的"切块"就是那天的内容落地成代码。
- **第 5 天**：AIC（Cube，做矩阵乘）与 AIV（Vector，做后处理）的分工；`GM → L1 → L0 → MAD → L0C → Epilogue → GM` 数据流。
- **C++ 语法**：位运算（`&`、`^`、`>>` 与掩码取模）、构造函数初始化列表与委托构造、局部变量遮蔽外层同名变量、`static_assert`。能对照读懂即可。

## 三、详细讲解

### 3.1 调度问题：从 Tile 网格到核任务

Normal GEMM 把输出 `D[M, N]` 切成 `num_m_blocks × num_n_blocks` 个 `BLOCK_M × BLOCK_N` 的输出 Tile：

```text
num_m_blocks = ceil_div(shape_m, BLOCK_M)     // scheduler.hpp:91
num_n_blocks = ceil_div(shape_n, BLOCK_N)     // 92
num_blocks   = num_m_blocks * num_n_blocks     // 93
```

每个 Tile 由**一个核从头算到尾**——K 维循环完全发生在一个核内部（`bf16_gemm.hpp:112`），不拆给多核。于是调度问题就是三问：**哪个核、按什么顺序、算哪个 Tile？** 把 Tile 任意分完只是正确性问题，顺序还决定了 L2 命不命中、相邻任务能不能复用数据，是性能问题。

本仓库选择**持久化（persistent）**方案，文件头注释（`scheduler.hpp:10-15`）说得很清楚：每个 Cube 核通过 `get_next_block` 独立地遍历全局 `(m_block, n_block)` 网格。三个要素：

1. **固定网格启动**：kernel 恰好以 `kNumCores` 个 block 启动——`make_gemm_candidate` 里 `c.launch_options.num_blocks = d.num_cores`（heuristics.hpp:98）。`d.num_cores` 来自 `runtime->get_num_sms()`（apis/gemm.hpp:176），在 Ascend 950DT 上等于 `kNumAICores = 32`（common.hpp:51）。launch 开销与矩阵大小无关。
2. **无锁领任务**：每个核只依赖自己的 `core_id` 和私有计数器 `current_iter`（初值 −1，第 89 行），第 t 次调用算出全局块号 `(t) * kNumCores + core_id`。核与核之间**没有任何通信或原子操作**——32 个核各自做同一道算术题，天然不重不漏地覆盖 `[0, num_blocks)`。
3. **顺序可控**：既然块号到 Tile 的映射是自己写的函数（`get_swizzled_block`），就可以为 L2 局部性精心设计访问顺序（3.4 节），而不是受制于硬件默认的行主序。

由此得到 **wave** 的概念：wave `t` 里，核 `c` 处理全局块号 `c + 32t`。`num_waves = ceil_div(num_blocks, 32)`——这就是 Host 成本模型里反复出现的 `num_waves`（heuristics.hpp:219、538、665 等）。块号 ≥ `num_blocks` 的核直接退出循环，所以：

- `num_blocks < 32` 时（比如 `M=64, N=64, BLOCK=128×128` 只有 1 个 Tile），只有 0 号核干活，31 个核第一次调用就返回 `false`，空手而归；
- `num_blocks` 不是 32 的倍数时，最后一个 wave 只有部分核有活。

> **阅读陷阱**：代码里有两个 `num_blocks`——`GemmConfig.launch_options.num_blocks` 是**网格大小（=核数）**（heuristics.hpp:98），`Scheduler::num_blocks` 是**Tile 总数**（scheduler.hpp:58、93）。语义完全不同，读代码时别混。

### 3.2 `Scheduler` 的模板参数、状态与构造

模板声明（`scheduler.hpp:42-48`）：

```cpp
template <
    GemmType kGemmType,                    // Normal / MGrouped / KGrouped / Batched（common.hpp:87-92）
    uint32_t BLOCK_M, uint32_t BLOCK_N,    // Tile 尺寸（切块粒度）
    uint32_t kNumCores,                     // 持久化调度的核数（=32）
    uint32_t kAlignment = 256u,             // 分组布局的对齐（Normal 下只是占位，Grouped 用，第 36～37 天）
    uint32_t kGroupM = 4, bool kXorN = true, bool kSnake = true   // 混洗参数
>
struct Scheduler {
    static_assert(kGroupM == 4 || kGroupM == 8, "kGroupM must be 4 or 8");   // 50
```

三个混洗参数今天的主角：`kGroupM`（一个混洗"超行"含几个 m 块）、`kXorN`（n 方向是否做 XOR 置换）、`kSnake`（相邻超行是否蛇形反向）。`static_assert` 只允许 4 或 8；而三个 GEMM kernel 实例化时都没传它们（`bf16_gemm.hpp:73`），所以实际恒为 `4 / true / true`——但读懂这三个开关，才能看懂映射公式的每一行。

**状态字段**（52-80 行）按 GEMM 类型分两组：

| 字段 | 行号 | Normal 下是否使用 | 含义 |
| :-- | :-- | :-- | :-- |
| `current_iter` | 52 | ✓ | 领任务计数器，构造时置 −1 |
| `core_id` | 53 | ✓ | 核编号（设备内置 `block_idx` 传入） |
| `shape_m / shape_n` | 54-55 | ✓ | 运行期 M/N（编译期 shape 的 `SHAPE_M=0` 哨兵已在 kernel 第 47-48 行折算过） |
| `num_m_blocks / num_n_blocks / num_blocks` | 56-58 | ✓ | 网格三常量 |
| `tail_m_start / tail_start` | 59-60 | ✓（见下） | 混洗区与 m 尾部行的分界（m 块号 / 块号） |
| `grouped_layout / num_groups / current_group_idx` | 62-65 | ✗ | M/K-Grouped 用 |
| `last_psum_m / current_psum_m / current_m_block_cumsum` | 68-70 | ✗ | M-Grouped 专用 |
| `current_shape_k / last_psum_k / current_psum_k / ...` | 73-77 | ✗ | K-Grouped 专用 |
| `current_batch_idx` | 80 | ✗ | Batched 专用 |

**构造函数**（87-108 行）用委托构造（83-84 行）让 Normal 版转发到通用版，初始化列表（91-97 行）一次性算出五个派生值：

```cpp
num_m_blocks(ceil_div(shape_m, BLOCK_M)),
num_n_blocks(ceil_div(shape_n, BLOCK_N)),
num_blocks(ceil_div(shape_m, BLOCK_M) * ceil_div(shape_n, BLOCK_N)),
// swizzle covers floor-aligned m_blocks; the remaining < kGroupM rows use row-major
tail_m_start(num_m_blocks >= kGroupM ? (num_m_blocks - num_m_blocks % kGroupM) : 0),
tail_start(tail_m_start * num_n_blocks),
```

`tail_m_start` 是 `num_m_blocks` **向下取整到 `kGroupM` 的倍数**：前面 `tail_m_start` 行 m 块能凑成完整的 4 行混洗"超行"，剩下的不足 4 行（`num_m_blocks % kGroupM` 个）退化为行主序（3.4 节的 b 段）。`tail_start` 是对应的**块号**分界。99-107 行的 `if constexpr` 只在 Grouped 类型下初始化分组游标，Normal 直接跳过。

> **阅读提示**：`get_next_block` 内部声明了局部变量 `uint32_t block_idx`（第 142 行），它**遮蔽**了设备内置的全局 `block_idx`（构造函数收到的那个，已存进 `core_id`）。读到 142 行别以为是内置变量。

### 3.3 `get_next_block`：持久化索引（Normal 主线）

```cpp
__aicore__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    uint32_t block_idx = static_cast<uint32_t>(++current_iter) * kNumCores + core_id;   // 142

    ... // Grouped/Batched 分支（144-189 行），今天跳过
    } else {                                     // Normal（190-195 行）
        if (block_idx >= num_blocks)
            return false;                        // 领完任务，退出 while 循环
        get_swizzled_block(block_idx, num_m_blocks, m_block_idx, n_block_idx);
        return true;
    }
}
```

第一次调用时 `++current_iter` 把 −1 变 0，块号恰好等于 `core_id`——所以 **wave 0 里核 `c` 处理块号 `c`**。此后每轮加 `kNumCores`。

**AIC 与 AIV 两侧共享同一序列。** 每个 block 固定 1 个 AIC + 2 个 AIV（`__mix__(1, kDirectStore ? 0 : 2)`，第 27 行；direct store 路径没有 AIV）。`Scheduler` 在函数体最外层构造一次（72-74 行），随后：

- **AIC 分支**（`if ASCEND_IS_AIC`，81 行）：`while (scheduler.get_next_block(...))`（89 行）逐 Tile 做 K 循环、MAD、把结果送进 epilogue 缓冲；
- **AIV 分支**（173 行起）：**同样的 while 循环**（180 行）走同一串 `(m, n)`，但只做 `aiv_drain`（UB→GM 排水，190 行）。同一 block 的两个 AIV 都执行这份代码，靠 `asc_get_sub_block_id()` 区分（epilogue.hpp:51），按 `s_epi % kNumAIVs` 轮流认领 MAD 结果块（epilogue.hpp:57 的 `continue`）。细节留到第 25 天，今天只需要记住：**同一个 `core_id` 推出的 Tile 序列，驱动了三份执行流（1 AIC 算 + 2 AIV 排）**。

```text
核 c 的块号序列（kNumCores=32, num_blocks=40 的例子）：
  wave 0:  c        （块号 0-31，全部有效）
  wave 1:  32+c     （块号 32-39 有效，核 8-31 得到 ≥40 → 退出）
  wave 2:  64+c     （≥40 → 退出）
```

### 3.4 `get_swizzled_block`：三段式映射（今天的重头戏）

块号 → Tile 的映射分三大段（202-248 行）。先看全景：

```text
block_idx 轴（Normal）
0                     tail_start                    num_blocks
├─────── 混洗区（前 tail_m_start 行 m 块）──────┤├── m 尾部行（行主序）──┤
              │
              └─ superrow（每 superrow = kGroupM=4 行 m 块 × 全部 n 块）
                      ├─ full 区：n_groups × 32 块（每 32 块 = 4×8 矩形）→ XOR + 蛇形
                      └─ n 尾部列：(num_n_blocks % 8) × 4 块           → m 快、无 XOR
```

若 `num_m_blocks < kGroupM`（整个网格不足 4 行 m 块），连混洗区都不存在，全网格走快路径。

**(a) 快路径**（209-211 行）：`local_num_m_blocks < kGroupM` 时直接行主序，**n 是快轴**：

```cpp
m_block_idx = block_idx / num_n_blocks;
n_block_idx = block_idx % num_n_blocks;
```

**(b) m 尾部行**（212-216 行）：块号 ≥ `local_tail_start` 的部分，同样 n 快、m 慢：

```cpp
uint32_t tail = block_idx - local_tail_start;
m_block_idx = local_tail_m_start + tail / num_n_blocks;
n_block_idx = tail % num_n_blocks;
```

注释点明动机：**"reuses A across cores"**——同一 wave 里相邻块号共享同一个 A Tile（m 相同、n 递增），B Tile 沿 n 依次流过 L2。

**(c) 混洗 full 区**（219-238 行）是核心。逐行拆解：

```cpp
constexpr uint32_t kGroupN = 32 / kGroupM;        // 219：= 8，kGroupM×kGroupN=32 恰是一个 wave 的核数
uint32_t n_groups = num_n_blocks / kGroupN;       // 220：n 方向能切成几个 8 列组
uint32_t blocks_per_superrow = kGroupM * num_n_blocks;   // 221：一个 superrow 的块数
uint32_t full_blocks = n_groups * 32;             // 222：superrow 内 full 区块数

uint32_t superrow = block_idx / blocks_per_superrow;      // 224：第几个"4 行带"
uint32_t within   = block_idx - superrow * blocks_per_superrow;   // 225：带内序号

if (within < full_blocks) {
    uint32_t grp   = within / 32;                // 228：第几个 4×8 块组
    uint32_t local = within - grp * 32;          // 229：组内 0-31
    uint32_t m_local = local & (kGroupM - 1);    // 230：local % kGroupM——m 是快轴
    uint32_t n_local = local / kGroupM;          // 231：n 是慢轴（0-7）
    if constexpr (kXorN) n_local ^= m_local;     // 232：XOR 置换

    uint32_t ng = grp;
    if (kSnake && (superrow & 1)) ng = n_groups - 1 - ng;    // 235：奇数 superrow 蛇形反向

    m_block_idx = superrow * kGroupM + m_local;  // 237
    n_block_idx = ng * kGroupN + n_local;        // 238
}
```

三个几何要点：

1. **32 块一组 = 4×8 矩形，正好压住一个 wave**。连续 32 个块号（= 同一 wave 的 32 个核）落在 4 个连续 m 块 × 8 个连续 n 块的矩形里。这一 wave 的 A 工作集只有 4 个 A Tile、B 只有 8 个 B Tile——共 12 个，而不是行主序大 N 扫描时一个 wave 可能碰到的几十个。`kGroupM × kGroupN = 32 = kNumAICores` 显然不是巧合。Host 成本模型**按完全相同的矩形**估算 L2 驻留：`kBlockSwizzleM = 4`（heuristics.hpp:211、257，与设备 `kGroupM` 镜像）、`block_swizzle_n = min(num_n_blocks, 32/4)`（317 行，与 `kGroupN` 镜像）、一个混洗组的 L2 字节 = `min(m, 4*block_m) 行 × K` + `min(n, 8*block_n) 行 × K`（318-319 行）；若矩形装不进 L2，还要按 swizzle 边界给 A/B 计重载（321-329 行）。设备侧的设计意图与 Host 侧的估价公式严丝合缝。
2. **XOR 只做矩形内置换**。`n_local = n_pre ^ m_local` 对固定的 `m_local` 是 `[0,8)` 上的双射，所以它不改变工作集（还是那 4 A + 8 B），只改变"同一时刻 32 个核各自在读哪个 B Tile"。不做 XOR 时，块号相邻的 4 个核（`local/4` 相同）在同一瞬间读**同一个** B Tile；做了 XOR，这 4 个核的 B Tile 各不相同，访问被摊开到矩形的 4 个不同列。这与共享内存的 XOR swizzle 是同一思想：不改变集合，只消除"同刻同址"。
3. **蛇形让相邻 superrow 首尾相接**。偶数 superrow 沿 n 组 0→`n_groups-1` 正向走，奇数反向：`ng = n_groups-1-grp`。于是 superrow `s` 结束时的最后一个 n 组（`ng = n_groups-1`）恰好是 superrow `s+1` **开始**时的第一个 n 组——跨带边界时这 8 列 B Tile 更可能还留在 L2 里。这就是注释（第 14 行）"N-first snake for L2 locality" 的含义。

**(d) n 尾部列**（239-246 行）：`num_n_blocks % 8` 的余数列不在 full 区里，单独用 m 快、无 XOR/蛇形的行主序补齐：

```cpp
uint32_t rem_within = within - full_blocks;
m_block_idx = superrow * kGroupM + (rem_within & (kGroupM - 1));
n_block_idx = n_groups * kGroupN + rem_within / kGroupM;
```

**每个核的固定偏移（重要性质）**。混洗区里 `local = block_idx − superrow·blocks_per_superrow − grp·32`。核 `c` 的块号是 `32t + c`（`kNumCores = 32`），代入后含 `t` 和 `grp` 的两个 32 相消，得到 `local ≡ c − superrow·blocks_per_superrow (mod 32)`。由此推出：

- **`m_local = c % 4` 恒定**（`blocks_per_superrow = 4·num_n_blocks` 是 4 的倍数，superrow 平移不影响 `local` 的低两位）；
- **`n_local = (c/4) ^ (c%4)` 也恒定**，只要 `num_n_blocks` 是 8 的倍数（此时 `superrow·blocks_per_superrow ≡ 0 (mod 32)`，平移为零；一般情形 `n_local` 会随 superrow 平移 `num_n_blocks mod 8`）。

换句话说，**每个核在自己的 4×8 组内有一个固定的"座位"，任务序列 = 扫过所有 (superrow, n 组)**：`m = superrow*4 + m_local`，`n = ng*8 + n_local`，`ng` 方向由蛇形决定。这个视角直接解释了 kernel 里的复用判断（下一节）。

> **小字**：构造函数里算好的成员 `tail_m_start/tail_start`（59-60 行）在 Normal 路径下其实没有被 `get_swizzled_block` 消费——后者用 `local_num_m_blocks` **局部重算**了一份（204-206 行）。两者在 Normal 下相等；m-grouped 路径每组都会改写 `num_m_blocks`（152-155 行），所以映射函数必须拿"当前组的行数"当参数局部重算。成员字段可以当作构造期就给你看答案的注释。

### 3.5 `get_global_idx` 与边界处理

**`get_global_idx`**（113-115 行）把块号换算成全局元素偏移：

```cpp
__aicore__ uint32_t get_global_idx(uint32_t block_size, uint32_t block_idx) const {
    return block_idx * block_size;    // m_idx = m_block_idx * BLOCK_M；n_idx 同理
}
```

kernel 在拿到 `(m_block_idx, n_block_idx)` 后立即调用两次（`bf16_gemm.hpp:90-91`），得到 `m_idx / n_idx`，再喂给 `a.index(m_idx, k_idx)`、`b.index(n_idx, k_idx)`、`d.index(...)` 做地址计算（102-103、157-158、188 行）——`gm_ptr::index()` 的细节是明天（第 18 天）的主题。AIV 侧同样调用（181-182 行）。

**`get_actual_m / get_actual_n`**（129-139 行）处理"shape 不是 BLOCK 整数倍"的尾部 Tile：

```cpp
__aicore__ uint32_t get_actual_m(uint32_t m_block_idx) const {
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        return BLOCK_M;                 // TODO：m-grouped 恒返回 BLOCK_M（131-133 行），第 36 天再谈
    } else {
        return min(shape_m - m_block_idx * BLOCK_M, BLOCK_M);   // 134：Normal
    }
}
__aicore__ uint32_t get_actual_n(uint32_t n_block_idx) const {
    return min(shape_n - n_block_idx * BLOCK_N, BLOCK_N);        // 138
}
```

消费点：`actual_m` 同时界定**搬多少**（`copy_gm_to_l1` 的行数，110、117-118 行）和**算多少**（`m_mad_idx` 循环上界，123 行）；`actual_n` 同理（118、124 行）。AIV 侧的 drain 也用它们截断输出（183-184 行）。边界 Tile 于是"少搬少算"：比如 `M=513, BLOCK_M=128` 的最后一个 m 块只有 `actual_m = 1` 行。仓库的 `unaligned-mn` 测试场景（`tests/generators.py:491-516`，含 `(513, 257, 4096)` 这类 shape）专门覆盖这条路径。

最后明确边界：**K 维不归调度器管**。`num_k_blocks = ceil_div(shape_k, BLOCK_K)` 在 kernel 里按 `shape_k` 现算（95-96 行），K 循环（112 行）在单个核的 Tile 内部完成。调度器只回答"M/N 网格"的问题。

### 3.6 完整演算：`M=600, N=1000`（当天产出·参考答案）

沿用第 15、16 天的假设风格：Normal BF16 GEMM，`M=600, N=1000, K=256`，启发式**假设**选中 `BLOCK_M=128, BLOCK_N=128`，`kNumCores=32`，混洗参数取默认 `kGroupM=4 / kXorN / kSnake`。

**第一步：网格与区域常量。**

| 量 | 公式 | 值 |
| :-- | :-- | :-- |
| `num_m_blocks` | ceil(600/128) | 5 |
| `num_n_blocks` | ceil(1000/128) | 8 |
| `num_blocks` | 5×8 | 40 |
| `tail_m_start` | 5 − 5%4 | 4（前 4 行凑成一个完整 superrow） |
| `tail_start` | 4×8 | 32 |
| `n_groups` | 8 / 8 | 1 |
| `blocks_per_superrow` | 4×8 | 32 |
| `full_blocks` | 1×32 | 32（无 n 尾部列：8 % 8 = 0） |
| `num_waves` | ceil(40/32) | 2 |

**第二步：wave 0（块号 0-31 = 核 0-31）。** 全部落入混洗 full 区、superrow 0、`grp=0`（`n_groups=1`，蛇形无效果）。`local = 块号`：

```text
m_local = local % 4        （m 快）
n_local = (local / 4) ^ (local % 4)   （XOR，232 行）
m = 0*4 + m_local，n = 0*8 + n_local
```

把 32 个核填进 4×8 矩形（行 = m 块，列 = n 块，格子 = 核号）：

```text
        n=0   n=1   n=2   n=3   n=4   n=5   n=6   n=7
m=0      0     4     8    12    16    20    24    28
m=1      5     1    13     9    21    17    29    25
m=2     10    14     2     6    26    30    18    22
m=3     15    11     7     3    31    27    23    19
```

抽查验证：核 21 → `local=21`，`m_local = 21%4 = 1`，`n_local = (21/4) ^ 1 = 5^1 = 4` → Tile (m1, n4) ✓。每行、每列的核号都恰好不重复——32 个核瓜分 32 个 Tile。**本 wave 的 L2 工作集：4 个 A Tile（m 0-3）+ 8 个 B Tile（n 0-7）= 12 个 Tile**（K=256、BF16 下每个 Tile 128×256×2 = 64 KB，A 共 256 KB、B 共 512 KB）。

**第三步：wave 1（块号 32-63）。** 块号 32-39 落进 m 尾部区（`tail = 0-7`）：`m = 4 + tail/8 = 4`，`n = tail%8 = tail`——**核 0-7 依次领 (m4, n0..n7)**。块号 40-63 ≥ `num_blocks` → **核 8-31 退出**。总账：32 + 8 = 40 ✓。

**第四步：边界量。** `actual_m(4) = min(600 − 4×128, 128) = min(88, 128) = 88`——最后一个 m 块只搬 88 行、MAD 循环只跑 88 行；`actual_n(7) = min(1000 − 7×128, 128) = 104`（其余 n 块 128）。全局偏移举例：核 0 的 wave-1 Tile 是 `m_idx = 4×128 = 512`、`n_idx = 0`，输出区 `D[512:600, 0:128]`。

**第五步：每核序列与复用。** 核 `c` 的两跳：wave 0 是 `(c%4, (c/4)^(c%4))`，wave 1 是 `(4, c)`（仅 `c ≤ 7`）。对照 kernel 的 `reuse_a = (m_idx == last_m_idx)`、`reuse_b = (n_idx == last_n_idx)`（92-93 行；`last_*` 初值为 `shape_m/shape_n`，87 行，所以第一跳永不"复用"；每处理完一个 Tile 更新，170-171 行）：

| 核 | wave 0 Tile | wave 1 Tile | 第二跳 reuse_a | 第二跳 reuse_b |
| :-- | :-- | :-- | :-- | :-- |
| 0 | (0, 0) | (4, 0) | ✗（m 0→512） | ✓（n 0→0） |
| 3 | (3, 3) | (4, 3) | ✗ | ✓ |
| 4 | (0, 1) | (4, 4) | ✗ | ✗（n 128→512） |
| 8 | (0, 2) | （退出） | — | — |

`num_n_blocks = 8`（是 8 的倍数）⇒ 每核 `n_local` 恒定 ⇒ 本例里 0-3 号核第二跳 `n` 不变、`reuse_b = true`——B Tile 还留在 L0B/L2 里，kernel 就能跳过或减轻 B 的重搬（`bf16_gemm.hpp:118` 的 `!reuse_b` 条件；resident 路径第 26 天展开）。

**第六步：调度示意图（当天产出）。** 把上面所有结论画成一张图（模板见第六节）：

```text
M=600, N=1000, BLOCK_M=128, BLOCK_N=128, kNumCores=32   （K=256，K 不参与调度）

      n块    0      1      2      3      4      5      6      7     actual_n
m块 0  ┌  0 │   4  │   8  │  12  │  16  │  20  │  24  │  28  ┐    128
     1  │  5 │   1  │  13  │   9  │  21  │  17  │  29  │  25  │    128
     2  │ 10 │  14  │   2  │   6  │  26  │  30  │  18  │  22  │    128
     3  └ 15 │  11  │   7  │   3  │  31  │  27  │  23  │  19  ┘    128
        ↑ wave 0：块号 0-31 = 核 0-31（混洗区 superrow 0：4×8 矩形，n = n_pre ⊕ m_local）
     4  ┌  0    1    2    3    4    5    6    7  ┐    actual_m=88
        └────────────────────────────────────────┘
        ↑ wave 1：块号 32-39 = 核 0-7（m 尾部行、n 快）；核 8-31 无任务退出
        列 0-6 的 actual_n=128，列 7 的 actual_n=104
```

### 3.7（延伸）N 再大一倍：蛇形与 `reuse_a` 登场

主例 `n_groups = 1`，蛇形和"同 superrow 内 m 不变"都没机会表演。把 N 翻到 2048（`M=1024, N=2048`，同一 BLOCK）：`num_m_blocks = 8`、`num_n_blocks = 16`、`num_blocks = 128`、`n_groups = 2`、`blocks_per_superrow = 64`、`num_waves = 4`。核 0（`m_local = 0`、`n_local = 0`）的完整序列：

```text
块号   superrow  grp  snake→ng   Tile      相对上一跳
  0        0       0    0→0      (0, 0)     —
 32        0       1    0→1      (0, 8)     reuse_a（m 不变，n 组 +8）
 64        1       0    1→1      (4, 8)     reuse_b（蛇形首尾相接：n 组不变，m 带 +4）
 96        1       1    1→0      (4, 0)     reuse_a
```

三个观察：同一 superrow 内核 0 的 m 恒为 0（`reuse_a`）；superrow 边界处蛇形让 n 组**不变**（`reuse_b`，且边界两侧正是同一批 B Tile）；8×16 的网格被切成 4 个 wave，每个 wave 一个 4×8 矩形（wave 2 之所以从 n 组 1 开始，就是蛇形的功劳）。`num_n_blocks = 16` 是 8 的倍数 ⇒ 所有核的 `(m_local, n_local)` 固定，每核都在"(superrow × n 组)"网格上按自己的座位走一条蛇形折线。

## 四、源码导读

今天的目标是**把 3.4/3.6 的每个公式落到真实代码行上**。每站 5～10 分钟。

### 站点 1：`scheduler.hpp` 顶部注释（10-41 行，精读 Normal 段）

- 第 10-15 行：持久化索引公式与 Normal 的设计描述（"GROUP_M swizzle + N-first snake for L2 locality; m/n tails fall back to row-major"）——把这三句抄进笔记；
- 第 17-41 行：M-Grouped / K-Grouped / Batched 的注释**扫读**即可，只需记住三点：它们复用同一个 `get_swizzled_block`；`grouped_layout` 是前缀和数组；每核块号单调递增保证了组游标只需前进。细节留给第 36、37 天。

### 站点 2：模板与构造（42-108 行，精读）

- 42-50 行：7 个模板参数与 `static_assert`；问自己：`kAlignment` 在 Normal 路径里被谁消费？（提示：`get_actual_m` 的 m-grouped 分支用了它，146 行的 `static_assert(kAlignment == BLOCK_M)`——Normal 下它只是随模板传入的占位）；
- 87-108 行：委托构造（83-84）；初始化列表里五个派生值逐个对照 3.2 的表格；注意 94-95 行注释对 `tail_m_start` 的解释。

### 站点 3：`get_next_block`（141-196 行，精读 Normal 分支）

- 142 行：先确认这行里 `block_idx` 是**局部变量**（遮蔽内置变量），再写出公式；
- 191-194 行：Normal 分支的界检查与转发；
- 147-162 行（m-grouped）：只看两点——`while(true)` 如何推进组游标、第 160 行如何把组内块号交给**同一个** `get_swizzled_block`（第 159 行传的 `num_m_blocks` 是当前组的行数——这正是 3.4 节"局部重算"的原因）。

### 站点 4：`get_swizzled_block`（202-248 行，逐行）

对照 3.4 的 (a)(b)(c)(d) 四段，逐行核对：快路径条件（209）、m 尾部（212-216）、`kGroupN`（219）、`superrow/within`（224-225）、`grp/local`（228-229）、XOR（232）、蛇形（235）、n 尾部（239-246）。最后用主例块号 21、35 手算一遍。

### 站点 5：kernel 中的消费点（`bf16_gemm.hpp`，扫读）

- 27 行 `__mix__(1, kDirectStore ? 0 : 2)`：每 block 1 AIC + 2 AIV；
- 72-74 行：`Scheduler` 构造，`block_idx` 作 `core_id`；
- 89-93 行：AIC 的 while 循环、`m_idx/n_idx`、`reuse_a/reuse_b`；
- 105-106 行：`actual_m/actual_n` 的取用；
- 170-171 行：`last_m_idx/last_n_idx` 的更新（每个 Tile 结束时）；
- 180-191 行：AIV 的同款 while 循环（direct store 时 AIV 直接 return，174 行）。

### 站点 6：`common.hpp` 与 `epilogue.hpp`（工具与 AIV 分工）

- `common.hpp:41` 设备版 `ceil_div`、51 行 `kNumAICores = 32`、87-92 行 `GemmType` 枚举；
- `epilogue.hpp:49-57`：`aiv_drain` 里 `aiv_id = asc_get_sub_block_id()`（51 行）与 `continue`（57 行）——两个 AIV 如何瓜分同一 Tile 的 MAD 块；112 行 `aiv_idx = block_idx * kNumAIVs + ...` 印证"block 内共享 block_idx"。

### 站点 7：Host 镜像（`csrc/jit_kernels/heuristics.hpp`，验证对号）

- 98 行：`launch_options.num_blocks = d.num_cores`；
- 211、257 行：`kBlockSwizzleM = 4`（与设备 `kGroupM` 镜像）；316-319 行：混洗矩形的 L2 字节估算；321-329 行：矩形超出 L2 时按 swizzle 边界计重载；
- 215-219 行：`num_waves = ceil_div(num_blocks * num_batches, num_cores)`；
- 192-193 行：`num_epilogue_stages ≤ ceil_div(num_blocks, num_cores) * num_cd_stages`——epilogue 的 stage 数以"每 wave × 每 Tile 的 CD 块数"封顶，调度直接约束了流水线深度（第 16 天 3.6 的伏笔在此收线）；
- 537-538、552-557 行：resident A/B 的启用条件依赖 `num_m_blocks/num_n_blocks/num_waves`——扫读即可，第 26 天展开。

**阅读检查点**（能回答再进入实践）：

1. 两个 `num_blocks`（launch 网格 vs Tile 总数）分别在哪一行定义、各等于什么？
2. 为什么 `kGroupM × kGroupN` 必须等于 32？如果不等，wave 与矩形的关系会发生什么变化？
3. m 尾部行为什么选"n 快"而不是"m 快"？两句话说明它对 A/B 复用的影响。
4. 主例中核 21 在 wave 0 算哪个 Tile？写出 `m_idx/n_idx/actual_m/actual_n`。

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：打印配置并手算网格（约 20 分钟）。**

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day17_run1.log
grep -E 'GemmDesc|GemmConfig' /tmp/day17_run1.log | head -10
```

每行输出形如 `GemmDesc(m=..., n=..., ...) GemmConfig(block=MxNxK, ..., launch_options=LaunchOptions(num_blocks=...))`（格式定义在 `csrc/jit_kernels/gemm_config.hpp:145-172`）。挑 **3 个不同 shape**，各算一遍：`num_m/n_blocks`、`num_blocks`、`num_waves`、`tail_m_start/tail_start`、`n_groups`，核对 `LaunchOptions.num_blocks` 恒等于核数（32），并检查 `epi_stages ≤ num_waves × num_cd_stages`（heuristics.hpp:193）是否成立。

**步骤 2：跑非对齐场景，对照边界处理（约 10 分钟）。**

```bash
python tests/test_bf16_gemm.py unaligned-mn --skip-prof
```

`unaligned-mn` 生成的 shape（`tests/generators.py:491-500`，如 `(513, 257, 4096)`、`(17, 4096, 4096)`）大量出现 `actual_m/actual_n < BLOCK` 的尾部 Tile。任选一个 shape 手算最后一个 m 块/n 块的 `actual_m/actual_n`，并回答：为什么这些用例不需要给输出先清零也能通过？（提示：`m_mad_idx < actual_m` 同时限制了计算与写回范围。）

**步骤 3：复现主例（约 10 分钟）。**

```bash
DG_PRINT_CONFIGS=1 python - <<'EOF' 2>&1 | tee /tmp/day17_run2.log
import torch, torch_npu  # noqa
import deep_gemm

a = torch.randn(600, 256, dtype=torch.bfloat16, device='npu')
b = torch.randn(1000, 256, dtype=torch.bfloat16, device='npu')
d = torch.empty(600, 1000, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d)
torch.npu.synchronize()
assert torch.allclose(d.float(), (a.float() @ b.float().T), atol=1, rtol=1e-2), "mismatch"
print("M=600,N=1000 OK")
EOF
```

记录打印的 `block_m/block_n`，按 3.6 重算网格；若启发式选的不是 128×128，就用它的实际值重画一遍调度图——演算方法不变。

**步骤 4（可选）：改变核数观察调度（约 10 分钟）。** 在脚本开头加 `deep_gemm._C.set_num_sms(16)`（`csrc/apis/runtime.hpp:17-19` 注册）再跑步骤 3：`kNumCores` 变 16，块号公式变成 `t*16 + c`，`num_waves = ceil(40/16) = 3`。先在纸上预测哪些核在哪些 wave 干活，再确认测试依然通过（调度与核数解耦，正确性不该受影响）。

### B. 无 Ascend 环境：静态替代任务

**B-1：闭卷重算主例（约 20 分钟）。** 合上教程，只看 `scheduler.hpp:87-108、141-194、202-248`，对 `M=600, N=1000, BLOCK=128×128` 重算 3.6 的全部六步（网格常量、wave 0 的 4×8 核号矩阵、wave 1 的分配、`actual_m/n`、每核两跳的 reuse、示意图），对照订正。

**B-2：用 Python 复刻调度器并验证（约 25 分钟）。** 把 Normal 路线翻译成下面这个骨架，补齐三处空格（**对照源码填，不要凭记忆**），然后打印 `M=600, N=1000` 下 0-7 号核与 8-11 号核的任务序列，与 3.6 的表格互相验证：

```python
import math

def swizzle(block_idx, num_m_blocks, num_n_blocks, kGroupM=4, kXorN=True, kSnake=True):
    if num_m_blocks < kGroupM:                       # 快路径（scheduler.hpp:209-211）
        return block_idx // num_n_blocks, block_idx % num_n_blocks
    tail_m_start = num_m_blocks - num_m_blocks % kGroupM
    tail_start = tail_m_start * num_n_blocks
    if block_idx >= tail_start:                      # m 尾部（212-216）
        tail = block_idx - tail_start
        return tail_m_start + tail // num_n_blocks, tail % num_n_blocks
    kGroupN = 32 // kGroupM                          # 混洗区（219-246）
    n_groups = num_n_blocks // kGroupN
    per_super = kGroupM * num_n_blocks
    full = n_groups * 32
    superrow, within = divmod(block_idx, per_super)
    if within < full:
        grp, local = divmod(within, 32)
        m_local = local & (kGroupM - 1)
        n_local = local // kGroupM
        if kXorN:
            n_local = ____                           # 232 行
        ng = grp
        if kSnake and (superrow & 1):
            ng = ____                                # 235 行
        return superrow * kGroupM + m_local, ng * kGroupN + n_local
    rem = within - full
    return superrow * kGroupM + (rem & (kGroupM - 1)), n_groups * kGroupN + rem // kGroupM

def schedule(M, N, BM, BN, cores=32):
    nmb, nnb = math.ceil(M / BM), math.ceil(N / BN)
    total = nmb * nnb
    for c in range(cores):
        t, seq = 0, []
        while t * cores + c < total:
            seq.append(swizzle(t * cores + c, nmb, nnb))
            t += 1
        if seq:
            print(f"core {c:2d}: {seq}")

schedule(600, 1000, 128, 128)
```

**B-3：新 shape 演算——快路径与空转核（约 15 分钟）。** 取 `M=300, N=512, BLOCK=128×128`：`num_m_blocks = 3 < kGroupM` → 整个网格走快路径。手算：`num_n_blocks`、`num_blocks`、哪些核有任务（应有 12 个核、其余 20 个空转）、核 0-11 各算哪个 Tile、`actual_m(2) = 300 - 256 = 44` 对应哪个核。再用 B-2 的脚本核对。

**B-4：大 shape 演算——蛇形与交替复用（约 15 分钟）。** 取 3.7 的 `M=1024, N=2048`：写出核 0 与核 1 的完整 4-Tile 序列（核 1 应为 `(1,1) → (1,9) → (5,9) → (5,1)`），逐跳标注 `reuse_a/reuse_b`；再回答：wave 2 覆盖哪个 4×8 矩形？为什么它的 n 块从 8 开始而不是 0？

**B-5：XOR 双射验证（约 10 分钟）。** 证明 `(m_local, n_local) = (local%4, (local/4)^(local%4))` 是 32 个 `local` 值到 4×8 矩形的双射（提示：固定 `m_local` 时 `n_local` 遍历 `[0,8)`）。然后反向查询：主例 wave 0 中哪些核算 `n=5` 列的 Tile？（答案应为 17、20、27、30 四个核——对照 3.6 矩阵第 n=5 列验证。）

**B-6：闭卷画调度图（约 15 分钟）。** 凭记忆重画 3.6 第六步的示意图（含 wave 划分、核号矩阵、m 尾部行、`actual_m/n` 标注），对照订正——订正后的图就是当天产出。

## 六、当天产出

学习计划要求今天的产出是**一张 M/N Tile 调度示意图**。按下述模板填写（B-1～B-5 的演算附在后面）：

```text
# DeepGEMM-Ascend 第 17 天记录：理解 Normal GEMM 调度

## A. 调度示意图（闭卷填写后对照 3.6 订正）
配置：M=＿, N=＿, BLOCK=＿×＿, kNumCores=＿（kGroupM=＿, kXorN=＿, kSnake=＿）
网格：num_m_blocks=＿, num_n_blocks=＿, num_blocks=＿, num_waves=＿
区域：tail_m_start=＿, tail_start=＿, n_groups=＿, blocks_per_superrow=＿
wave 0（块号＿～＿）：覆盖 m 块＿～＿ × n 块＿～＿；核 c → (m, n) = (＿＿, ＿＿＿＿)
wave 1（块号＿～＿）：＿＿＿＿＿＿＿＿；退出/空转的核：＿＿＿＿
边界：actual_m(＿)=＿；actual_n(＿)=＿；它们限定 kernel 的哪两件事？＿＿、＿＿
（把 wave 0 的核号矩阵画成 m×n 网格，m 尾部行单独画一行）

## B. 每核序列与复用（挑 3 个核）
核＿：wave0 (＿,＿) → wave1 (＿,＿)；第二跳 reuse_a=＿（m ___→___）, reuse_b=＿（n ___→___）
核＿：……
核＿：……

## C. 公式核对
- 持久化块号：block_idx = ＿＿＿＿＿＿＿（scheduler.hpp:142）
- 快路径条件：＿＿＿＿＿；映射 m=＿＿, n=＿＿
- m 尾部：m=＿＿＿＿, n=＿＿＿＿（行＿＿）
- 混洗 full 区：m_local=＿＿, n_local=＿＿＿＿, snake 后 ng=＿＿＿＿（行＿＿）
- get_global_idx：＿＿＿＿（行＿＿）；get_actual_m（Normal）：＿＿＿＿（行＿＿）

## D. 实验记录（有环境时）
- 3 个 shape 的网格/wave/epi_stages 核对表
- unaligned-mn 的 shape 与手算 actual_m/n
- set_num_sms(16) 后的 wave 划分预测与结果

## E. 遗留问题（交给后续天）
- gm_ptr::index 的地址计算（第 18 天）
- actual_m/actual_k 如何流向 copy_gm_to_l1（第 19-20 天）
- resident A/B 与 reuse 的完整配合（第 26 天）
- M/K-Grouped 分支的组游标推进（第 36-37 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. kernel 以多少个 block 启动？写出这条链路上的三个关键赋值（Python 可影响的接口、Host 写入、设备消费）。
2. `Scheduler` 的 7 个模板参数是什么？本仓库的 GEMM kernel 实际传入几个？混洗参数的实际取值是什么？
3. `M=600, BLOCK_M=128` 时 `tail_m_start` 和 `tail_start`（设 `num_n_blocks=8`）各是多少？这两个字段在 Normal 路径里被哪段代码真正消费？
4. 核 `c` 第 `t` 次调用 `get_next_block` 得到的全局块号是多少？什么时候返回 `false`？
5. `get_swizzled_block` 的三段各是什么？进入条件分别是什么？混洗区里的 n 尾部列走哪一段、用什么映射？
6. 主例（`M=600, N=1000`）中 17 号核在 wave 0 算哪个 Tile？写出它的 `m_idx/n_idx`。
7. XOR 与蛇形分别改写哪个量？它们各自**不改变**什么？为什么说"工作集仍是 4×8 矩形"？
8. 为什么每核的 `m_local` 恒等于 `core_id % 4`？`n_local` 恒定还需要什么额外条件？
9. `get_actual_m` 的 Normal 公式是什么？它同时限定了 kernel 里的哪两类操作？K 维归调度器管吗？
10. AIC 与 AIV 如何共享调度结果？`__mix__(1, kDirectStore ? 0 : 2)` 的两个数字是什么含义？
11. （思考）Host 成本模型里哪个常量与 `kGroupM` 互为镜像？`num_epilogue_stages` 的上界公式（heuristics.hpp:193）说明了"调度"对"流水线深度"的什么约束？XOR 到底优化了什么——如果关掉它（`kXorN=false`），你觉得哪项指标最先变化？

**参考答案：**

1. 恰好 `kNumCores` 个（950DT 上 32）。链路：`runtime->get_num_sms()`（apis/gemm.hpp:176，Python 侧 `deep_gemm._C.set_num_sms(n)` 可覆盖）→ `c.launch_options.num_blocks = d.num_cores`（heuristics.hpp:98）→ 设备上成为内置 `block_idx` 的取值范围，`bf16_gemm.hpp:73` 把它作为 `core_id` 传给 `Scheduler`。
2. `kGemmType / BLOCK_M / BLOCK_N / kNumCores / kAlignment(=256) / kGroupM(=4) / kXorN(=true) / kSnake(=true)`——7 个参数带 4 个默认值（42-48 行）。三个 GEMM kernel（`bf16_gemm.hpp:73`、`fp8_gemm.hpp:97`、`fp8_dequant_gemm.hpp:194`）实例化时都只传前 5 个，混洗参数恒为 `4/true/true`。
3. `tail_m_start = 5 − 5%4 = 4`；`tail_start = 4×8 = 32`。Normal 路径下真正消费分界的是 `get_swizzled_block` 里**局部重算**的 `local_tail_m_start/local_tail_start`（204-206 行）——成员字段与之一致，可视为构造期给出的"答案注释"；m-grouped 因每组 `num_m_blocks` 会变，必须局部重算。
4. `block_idx = t * kNumCores + core_id`（142 行，`current_iter` 从 −1 起算，`++` 后即 t）。当 `block_idx >= num_blocks` 时返回 `false`（191 行）。
5. 三段：(a) 快路径 `num_m_blocks < kGroupM` → 行主序 n 快；(b) m 尾部行 `block_idx >= tail_start` → 行主序 n 快（"reuses A across cores"）；(c) 混洗区：superrow → 32 块组（4×8）→ `m_local = local%4`、`n_local = (local/4)^m_local`、`ng` 受蛇形调制。n 尾部列走混洗区内的 else 分支（239-246 行）：`m_local = rem%4`、`n_local = rem/4`，无 XOR、无蛇形。
6. 核 17：`local=17`，`m_local = 1`，`n_local = (17/4)^1 = 4^1 = 5` → Tile (m1, n5)。`m_idx = 128`，`n_idx = 640`。
7. XOR 改写 `n_local`（232 行）；蛇形改写 `ng`（235 行）。都不改变一个 wave 覆盖的 Tile 集合：XOR 对固定 `m_local` 是 `[0,8)` 上的双射，蛇形只是反转 n 组的遍历方向——工作集始终是 4 个 A Tile + 8 个 B Tile 的 4×8 矩形。XOR 消除"相邻 4 核同刻读同一 B Tile"，蛇形让相邻 superrow 的 n 组首尾相接、跨带时 B Tile 更可能仍在 L2。
8. `m_local = local & 3`，而核 `c` 的块号 ≡ `c (mod 32)`；`local` 跨 superrow 的平移量是 `4*num_n_blocks mod 32`，恒为 4 的倍数，对 `local & 3` 无影响。`n_local` 恒定还需要 `num_n_blocks` 是 8 的倍数（平移量为 0）。
9. `get_actual_m = min(shape_m − m_block_idx*BLOCK_M, BLOCK_M)`（134 行）。它同时限定：① 搬运行数（`copy_gm_to_l1`，110/117-118 行）；② MAD/写回范围（`m_mad_idx < actual_m`，123 行）。K 维不归调度器管——`num_k_blocks` 在 kernel 里按 `shape_k` 现算（95-96 行）。
10. 同一个 `Scheduler` 实例驱动两个 while 循环：AIC 侧（89 行）算并送 epilogue，AIV 侧（180 行）drain；两个 AIV 执行同一份循环、走同一串 Tile，靠 `asc_get_sub_block_id()` 区分（epilogue.hpp:51、57）。`__mix__` 的两个数字依次声明每 block 的 AIC 数与 AIV 数——普通路径 1 AIC + 2 AIV，direct store 时 AIV 数为 0（AIV 分支直接 return，174 行）。
11. `kBlockSwizzleM = 4`（heuristics.hpp:211、257）；成本模型还用 `block_swizzle_n = min(num_n_blocks, 32/4)`（317 行）拼出同一个 4×8 矩形估 L2（318-319 行），矩形装不下就按 swizzle 边界计重载（321-329 行）。`num_epilogue_stages ≤ ceil_div(num_blocks, num_cores) * num_cd_stages`（193 行）：epilogue 缓冲只需容纳"一个核在一个 wave 里产生的 CD 块"，wave 数（调度）直接封顶了流水线深度（存储）。XOR 的作用是开放性问题，合理的回答方向：它不改变 GM 读总量与工作集，改变的是**同一时刻**各核访问的 B Tile 分布——预期最先变化的是 L2/GM 的瞬时带宽压力与 A/B 装载的突发性，可通过 `DG_PRINT_CONFIGS` 对比配置选择（成本模型对 `num_a/b_gm_blocks` 的估计）或 profiling 的带宽曲线去验证。

---

今天结束时，你手里应该有一张标注了 wave 划分、核号矩阵、m 尾部行和 `actual_m/n` 的 M/N Tile 调度示意图，并且能从"核 c 的第 t 次调用"一路手算到"它正在写 `D` 的哪一块"。调度给了每个 Tile 一个全局坐标 `(m_idx, n_idx)`，明天（第 18 天）读 `ascend/tile_ptr.hpp`，看这些坐标如何变成 `gm_ptr/l1_ptr` 上的真实地址。
