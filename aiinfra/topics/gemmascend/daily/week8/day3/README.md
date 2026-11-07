# 第 8 周 · 第 3 天（总第 52 天）：完成单变量性能实验

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 50 分钟，源码导读约 30 分钟，实践步骤约 70 分钟，自测题约 15 分钟。

前两天搭好了舞台：第 50 天有了"度量衡"（TFLOPS、带宽、算术强度、`bench_msprof`），第 51 天有了"观察窗口"和**基线记录表**（`DG_PRINT_CONFIGS` 打印的 `GemmConfig` 全套字段 + perf 行）。今天正式**拧旋钮**。学习计划对本天的要求非常明确：固定一个 shape，每次只改变一个因素——Block 大小、L1/L0 stage 数、resident A/B、direct store、AI Core 数量——**每次实验都先验证正确性，再比较性能，避免同时修改多个参数**；通过标准不是"扫一圈参数挑最快的"，而是**能根据结果提出并验证一个性能假设**。因此今天的核心方法是"假设 → 预测 → 单变量改动 → 正确性 → 测量 → 判定"的闭环。五个旋钮里只有 AI Core 数量是纯运行时的（`deep_gemm.set_num_sms`），其余四个都要改 `csrc/` 里的启发式源码并重新构建——我们会给出精确到行号的插入点和最小补丁。没有 Ascend 环境时，B 组任务同样完整：用仓库自带的硬件模型和代价公式做"纸面实验"（波形推演、roofline 手算、合法性判定），产出一张可被未来实验检验的**预测表**。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说清楚"单变量"实验纪律的三条规则：固定 shape 与口径、每次只改一个因素、先正确性后性能，并能解释违反每条规则分别会引入什么混淆；
2. 画出五个旋钮的代码地图：每个因素默认由哪段代码决定、用什么方式修改、受哪些合法性约束（容量公式、`static_assert`、前置条件）；
3. 区分两类改法并正确执行：运行时旋钮 `deep_gemm.set_num_sms()`（含其取值边界与"配置联动"副作用），以及源码旋钮（改 `csrc/jit_kernels/heuristics.hpp` → `./develop.sh` 重新构建 → 用 `DG_PRINT_CONFIGS` 核对实际生效的配置）；
4. 解释为什么"强制 BLOCK 大小"要改搜索范围（让 `make_gemm_candidate` 联动重算 MAD/stages），而"强制标志位"可以在打印块之前直接覆盖；并理解配置值会被 `std::format` 写进 JIT 生成代码，因此不同配置自动触发新编译、无需手动清缓存；
5. 按模板写下一个可证伪的性能假设（机制 → 预测方向 → 证据字段），并用 `KernelProfile` 的利用率字段（mad/mte2/mte1/fixpipe/aiv_*）为判定找证据；
6. 有环境时：完成至少两个完整的实验闭环（建议：AI Core 数量 + 一个源码旋钮），产出实验日志；无环境时：完成波形推演与 roofline 手算预测表，以及一份实验设计文档。

## 二、预备知识

- **第 50 天**：`bench_msprof` / `KernelProfile` 的设备侧计时、TFLOPS/带宽/算术强度公式、冷 L2 口径、受限类型判别。
- **第 51 天**：`DG_PRINT_CONFIGS` 配置行语法（`GemmDesc` + `GemmConfig`）、配置与 JIT 代码生成的对应关系、基线记录表——**今天的每一次对照都以它为基准**。
- **第 12 天**：`select_gemm_config()` → 模板代码生成 → `jit->compile()` → `jit->launch()` 的调用链。
- **第 16 天**：L1/L0A/L0B/L0C/UB 容量常量（`deep_gemm/include/deep_gemm/common.hpp:54-58`）与多阶段缓冲的 offset 分配。
- **第 23 天**：L1 stages（MTE2↔MTE1）与 L0 stages（MTE1↔Cube）的流水线语义与同步。
- **第 26 天**：`kDirectStore` 与 `kResidentA/B` 两条特殊路径的行为与约束。
- **工具**：会用 `git diff` / `git checkout -- <file>` 管理实验补丁（今天的修改**不应提交**，实验结束要还原）。
- **环境说明**：第五节 A 组任务需要 Ascend 环境（CANN 9.20 + torch_npu，且已按第 9 天完成 `./develop.sh` 构建）；无环境时完成 B 组静态任务即可达到当天目标。

## 三、详细讲解

### 3.1 单变量纪律与实验闭环

为什么强调"每次只改一个因素"？因为性能是多个机制的函数。以一个 BF16 GEMM 为例，耗时同时受 Tile 数量与 wave 划分（调度）、GM 流量（带宽）、L1/L0 流水深度（延迟隐藏）、epilogue 路径（回写）影响。如果一次改两个参数，性能变化就无法归因——这正是"给出一组调参结果"与"验证一个假设"的分界线。

今天的实验闭环固定为五步：

```text
① 写下假设：在 shape S 上，把因子 X 从 a 改到 b，
   预期指标 Y 向方向 Z 变化，因为机制 M；
② 预测：用手推（wave 算术 / roofline / 代价公式）给出数量级或方向；
③ 单变量改动：只改 X，其余一切与基线相同（用 DG_PRINT_CONFIGS 核对）；
④ 正确性：先跑正确性测试，通过才有资格进入性能对比；
⑤ 测量与判定：bench_msprof 采集，与基线对照，用利用率字段解释结果，
   接受 / 拒绝 / 修正假设 M。
```

第 ③ 步的"核对"是今天的新要求：改完之后**实际生效的配置**必须与你的意图一致。配置行（`heuristics.hpp:749-753` 打印）是最终裁判——尤其要注意，选择流程有多个后续阶段可能改写你动过的字段（见 3.3 节的"改写链"清单）。

### 3.2 五个旋钮的代码地图

| # | 因素 | 默认由哪里决定 | 修改方式 | 主要合法性约束 |
|---|---|---|---|---|
| 1 | AI Core 数量（`num_cores`/`num_blocks`/`kNumCores`） | `runtime->get_num_sms()` 查设备（950 为 32），写入 desc（`csrc/apis/gemm.hpp:176`）→ launch 块数（`heuristics.hpp:98`）→ 模板 `kNumCores`（`csrc/jit_kernels/bf16_gemm.hpp:57`）→ `Scheduler`（`deep_gemm/include/deep_gemm/scheduler.hpp:42-47`） | **运行时**：`deep_gemm.set_num_sms(n)`，无需重编译 | `1 ≤ n ≤ 设备核数`（`csrc/jit/jit.hpp:39` 的断言）；**不能设 0 恢复默认**，须先存 `get_num_sms()` |
| 2 | BLOCK_M/N | 搜索范围 `m_range/n_range`（`heuristics.hpp:608-616`）+ 双层循环（`:626-657`） | 源码：收窄 range | L0C：`BLOCK_M×BLOCK_N×4B ≤ 256KB`（`:100-102`）；`block_cap=512`（`:604`）；M/N 对齐 16（K-major 时） |
| 3 | BLOCK_K | `min_block_k=128 / max_block_k=256`（BF16，`:620-621`）+ K 精修（`:672-694`） | 源码：钉死上下界并门控精修 | `block_k % 16 == 0`（`:108-109`）；L1 容量决定 stage 数 |
| 4 | L1 stage 数 | 容量公式 `min(8, (512KB−SF)/ab_bytes)`（`:168-174`）；L0 stage 数恒为 2（`:111-112`） | 源码：L1 可在打印前覆盖（**只宜调小**）；L0 改 `:112`（进阶） | L1 容量 + 流水事件上限 8（`common.hpp:61` 的 `kNumPipeEvents`）；≥ 2（`:173`） |
| 5 | resident A/B | `refine_gemm_pipeline`（`heuristics.hpp:546-565`） | 源码：打印前覆盖（**关**安全；**开**须查条件） | Normal + K 为编译常量（`:546`，`bf16_gemm_nt` 的 `compiled_dims="nk"` 满足）；BF16 下 resident A 还要求整块 A 装进 L1（`:549-552`）；设备侧 `static_assert`（device `bf16_gemm.hpp:40/:45`） |
| 6 | direct store | `can_direct_store` 资格检查（`:521-533`）+ 代价比较（`:539-543`） | 源码：打印前覆盖（**关**安全；**开**必须先过 `can_direct_store` 全部条件） | Normal + Identity epilogue（无 bias/alpha）；`acc` 仅 fp32；D 指针与行 stride 按 32B 对齐；`blocks ≤ cores` 或单 M 块（`:526-532`） |

两点补充说明：

- **"关安全、开须查"**：`direct_store=false` / `resident=false` 回到通用路径，永远合法；反向强制则可能违反正确性前提（例如 direct store 下累加只支持 fp32，见 device `bf16_gemm.hpp:37-39` 的 `static_assert`）。做实验时优先做"把启发式打开的开关关掉"，观察代价。
- **L0 stage 数**：`:111-112` 把 `num_l0_stages` 钉在 2，MAD 搜索（`:133-137`）与代价模型（`:586-587`）虽以它为参数，但整个流水线设计都以 2 级为前提。把它作为实验因素属于进阶操作，今天以 L1 stage 数为主。

### 3.3 两种改法：运行时旋钮与源码旋钮

**旋钮 1：AI Core 数量（纯运行时）。**

```python
import deep_gemm
full = deep_gemm.get_num_sms()   # 默认查设备（950 为 32）；set_num_sms 不能设 0 恢复，先存下来
deep_gemm.set_num_sms(16)        # 之后所有 kernel 只用 16 个 AI Core
```

链路：`set_num_sms`（`csrc/jit/jit.hpp:38-41`，注意断言 `n > 0 and n <= 设备核数`，即**只能减不能加**）→ `get_gemm_desc` 把它写进 `desc.num_cores`（`csrc/apis/gemm.hpp:176`）→ 候选生成时成为 `launch_options.num_blocks`（`heuristics.hpp:98`）并作为模板实参 `kNumCores` 进入 kernel（`csrc/jit_kernels/bf16_gemm.hpp:57`），设备侧 `Scheduler` 用它划分 Tile（`scheduler.hpp:42-47`）。

一个必须记录的副作用：**启发式自己也在用 `num_cores` 做决策**——wave 数（`heuristics.hpp:219/:631`）、epilogue stage 数（`:193`）、direct store 资格（`:531` 的 `blocks ≤ num_cores`）、resident B 资格（`:556` 的 `num_waves > 1`）。所以改核数后配置行可能跟着变。这不破坏"单变量"（你改的变量就是核数），但要**把每次的配置行记进日志**；如果想剥离配置联动的干扰、只看调度本身的影响，可以配合源码旋钮把配置钉死。

**旋钮 2～6：源码旋钮。** 统一流程：

```bash
cd DeepGEMM-Ascend
# 1) 打补丁（见下）
# 2) 重新构建 C++ 扩展（develop.sh 会清掉旧 build 并把新 .so 链接进 deep_gemm/）
./develop.sh
# 3) 导出观察开关后运行
export DG_PRINT_CONFIGS=1
DG_EXP_BLOCK_N=224 python week8_day3_exp.py   # 示例：环境变量选择实验条件
# 4) 实验结束还原
git checkout -- csrc/jit_kernels/heuristics.hpp && ./develop.sh
```

插入点选择原则：**块大小类旋钮改"搜索范围"（早），标志类旋钮改"最终配置"（晚）**。

- 强制 BLOCK_N 要改 `n_range`（`heuristics.hpp:609` 之后插一行），这样 `make_gemm_candidate` 会为强制出的 block 联动重算 MAD、L1/L0 stage、epilogue stage——配置内部保持自洽。若在函数末尾直接改 `config.block_n`，MAD/stages 还是旧 block 的值，属于自造非法配置。
- 强制 `direct_store`/`resident_*`/`num_l1_stages` 这类正交标志，可以插在打印块（`:749-753`）**之前**——这是 `select_gemm_config` 的最后一站，打印行会如实反映你的覆盖。

推荐用**按 shape 门控 + 环境变量**的写法，一次构建支持多组实验条件（环境变量名自取，避开仓库已有的 `DG_*` 语义即可；注意**不要**用 `DG_JIT_` 前缀——那类变量在 JIT 运行时首次创建时被快照（`README.md:85`），而像 `DG_PRINT_CONFIGS`（`heuristics.hpp:749`）、`DG_TRANSFORM_SF_LD_CACHE_HINT`（`transform_sf.hpp:70-71`）这类不带该前缀的变量是每次调用实时读取的）：

```cpp
// ===== [第 52 天实验] 插在 heuristics.hpp:609 之后（n_range 定义之后）=====
if (const int forced_n = deep_jit::get_env<int>("DG_EXP_BLOCK_N"))
    if (d.gemm_type == GemmType::Normal && d.m == 128 && d.n == 7168 && d.k == 2048)
        n_range = {forced_n, forced_n};          // 实验 2：固定 BLOCK_N

// ===== [第 52 天实验] 插在 heuristics.hpp:749（打印块）之前 =====
if (d.gemm_type == GemmType::Normal && d.m == 128 && d.n == 7168 && d.k == 2048) {
    if (deep_jit::get_env<int>("DG_EXP_NO_DIRECT_STORE")) config.direct_store = false;  // 实验 3
    if (deep_jit::get_env<int>("DG_EXP_NO_RESIDENT_A"))   config.resident_a = false;    // 实验 4
    if (const int s = deep_jit::get_env<int>("DG_EXP_L1_STAGES"))                      // 实验 5
        config.num_l1_stages = std::min(config.num_l1_stages, static_cast<uint32_t>(s)); // 只允许调小
}
```

（`get_env<int>` 未设置时返回 0、视为关闭——与 `heuristics.hpp:749` 对 `DG_JIT_DEBUG` 的用法一致；`<deep_jit/utils/env.hpp>` 已在 `heuristics.hpp:13` 引入。这两段是**你自己添加的实验代码**，不属于仓库原有接口，实验后必须删除。）

三个工程细节：

1. **改写链**：在你动过手脚之后，选择流程的后续阶段仍可能改写字段——更宽 K 候选（`:648-651`）、K 精修（`:672-694`，会改 `block_k`）、bias 双缓冲（`:697-711`，会缩 block_m/n）、FP4 整块 MAD（`:713-724`）、双 AIV 解码（`:727-737`）。例如强制 `n_range={128,128}` 后，若 `k % 256 == 0` 且更宽 K 候选的 L1 stage ≥ 3，`block_k` 可能从 128 变 256。**打印行是唯一裁判**：对照两次运行的完整 `GemmConfig` 行，把协变字段记进日志；想让 `block_k` 也固定，就同时门控 `:620-621` 的 `min/max_block_k` 与 `:672` 的精修条件。
2. **JIT 缓存无需手动清理**：配置值通过 `std::format` 写进模板实例化代码（`csrc/jit_kernels/bf16_gemm.hpp:18-66`），配置不同 ⇒ 生成代码不同 ⇒ DeepJIT 视为新的编译单元（缓存语义见 `README.md:91`，默认目录 `$HOME/.dj`）。只改 `csrc/`（host 侧）时如此；如果你动了 `deep_gemm/include/` 下的设备头文件，建议把 `DG_JIT_CACHE_DIR` 指到新目录以避免读到旧缓存 kernel（第 51 天用过的"空目录强制重编译"技巧）。
3. **门控的意义**：按 shape 门控后，`python tests/test_bf16_gemm.py normal --skip-prof` 里其余 shape 全部走默认路径，只有目标 shape 吃到实验配置——正确性回归既全面又聚焦。

### 3.4 合法性约束：让非法配置"响亮地失败"

单变量实验的前提是改动合法。仓库在这方面很友好——大多数非法配置会在**编译期**被设备侧 `static_assert` 拦下（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:66-70`）：

```cpp
static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes,  "L1 overflow");
static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
static_assert(l0c.offset(num_mad_cd_stages) <= L0CSizeBytes, "L0C overflow");
static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");
```

也就是说，如果你强制了一个装不进片上内存的配置，JIT 编译会直接报错（报错文本含 "L1 overflow" 等），而不是静默算错。这仍是"失败"，但属于**响亮的失败**——记录下来即可。真正危险的是语义级违规（例如给非 Identity epilogue 强开 direct store，`device bf16_gemm.hpp:56-57` 的 `static_assert` 也会拦住）。手工核对容量的公式（BF16、无 SF）：

```text
ab_bytes      = (BLOCK_M + BLOCK_N) × BLOCK_K × 2 B          （heuristics.hpp:171）
num_l1_stages ≤ 512KB / ab_bytes，且 ≤ 8（事件上限），且 ≥ 2   （:168-174，common.hpp:54/:61）
L0A           ≥ MAD_M × MAD_K × 2B × 2（双缓冲）              （:133-134）
L0C           ≥ BLOCK_M × BLOCK_N × 4B                       （:100-102）
```

### 3.5 把假设写下来：三个示范

**模板**：`在 shape S 上，把 X 从 a 改到 b，预期 Y 变化方向 Z，因为机制 M；若实测相反，优先怀疑 N。`下面三个示范全部可以用源码里的常数手工推出（`get_hw_desc()`，`heuristics.hpp:52-71`：`aic_clock_mhz=1650`、`bf16_flops_per_cycle=8192`、`num_ai_cores=32`、`gm_bw_gbps=3500`、`gm_to_l1_bytes_per_cycle=256`）。

**示范 1：AI Core 数量（计算受限 shape）。** 取 `4096×7168×16384` BF16（README Dense 表 `README.md:113`：2229.9 µs / 431 TFLOPS / 99.8%）。假设第 51 天基线表记录的配置是 `block=512×128`（以你自己的记录为准），则：

```text
总 Tile 数 = ceil(4096/512) × ceil(7168/128) = 8 × 56 = 448
单 Tile 计算量 = 2 × 512 × 128 × 16384 FLOP = 2.147 GFLOP
单核速率 = 8192 FLOP/cycle × 1.65 GHz = 13.52 TFLOPS → 单 Tile ≈ 158.9 µs
32 核：waves = ceil(448/32) = 14 → 预测 ≈ 14 × 158.9 = 2224 µs（README 实测 2229.9，吻合）
```

假设：`set_num_sms(28)` 后时长增至约 16/14 ≈ +14%（waves = ceil(448/28) = 16），`set_num_sms(16)` 后约 2 倍——**只要仍是计算受限**。可行性核查：该 shape 最少 GM 流量 ≈ A 134MB + B 235MB ≈ 369MB，AI ≈ 2600 FLOP/B ≫ 交叉点 123（432T / 3.5TB/s），即使 8 核（发射能力 8 × 256B/cycle × 1.65GHz ≈ 3.4TB/s）也不会先撞带宽墙。证据字段：`aic_mad` 应保持高位；若时长不按 waves 比例增长，查 `aic_mte2` 是否抬升。这正是学习计划说的"根据结果验证假设"——预测有具体数值和方向，可被实测推翻。

**示范 2：BLOCK_N（边界 shape）。** 取 `128×7168×2048` BF16（测试套件 `normal` 的 fwd 用例之一，`tests/generators.py:560-570`）。roofline 手算：

```text
计算 = 2×128×7168×2048 = 3.76 GFLOP → /432T = 8.70 µs
字节 = A 0.52MB + B 29.4MB + D 1.75MB ≈ 31.7MB → /3.5TB/s = 9.06 µs
AI ≈ 119 FLOP/B，略低于交叉点 123 → 名义上偏访存受限（边界形态）
```

`m_limit = align(128,16) = 128`，BLOCK_M 只能取 128；BLOCK_N 是自由变量。按"每核串行时间 = num_waves × 单 Tile 面积"推演（这正是 `get_gemm_compute_cost` 的 `compute_mn = num_waves × compute_m × compute_n`，`heuristics.hpp:231`）：

| BLOCK_N | Tile 数 | waves(32 核) | 计算侧预测 | 备注 |
|---|---|---|---|---|
| 512 | 14 | 1 | 128×512 单 Tile ≈ 19.9 µs | 只有 14 核干活，单 Tile 串行时间成为下界 |
| 256 | 28 | 1 | ≈ 9.9 µs | 28/32 核忙 |
| 224 | 32 | 1 | ≈ 8.7 µs，取 max(计算, 访存) ≈ 9.1 µs | 恰好铺满 32 核 |
| 128 | 56 | 2 | 2 × 4.96 ≈ 9.9 µs | 第二波只有 24 核忙（尾部浪费，代价模型 `:233-249` 会再修一点） |

假设：`BLOCK_N=224 附近最优；512 明显变慢`，机制是"单 Tile 串行时间 vs wave 尾部浪费"的权衡；预测 `BLOCK_N=512` 时 `aic_mad` 下降（大量核空闲）。以上全部是**纸面推演，不是测量值**——A 组用它当假设，B 组用它当产出（还可以进一步预测启发式自己会选哪个 `block_n`，再用 `DG_PRINT_CONFIGS` 检验）。

**示范 3：direct store（输出路径）。** 对 `128×7168×2048`（Identity、bf16 输出、无 bias），若基线配置是 `block=128×512`（14 个 Tile ≤ 32 核），`can_direct_store`（`heuristics.hpp:521-533`）的各条件均满足（`blocks ≤ num_cores` 分支），启发式可能在代价比较后打开它。关掉它（`DG_EXP_NO_DIRECT_STORE=1`）会带来两个可观察变化：

1. **执行结构**：kernel 声明是 `__mix__(1, kDirectStore ? 0 : 2)`（device `bf16_gemm.hpp:27`）——direct store 时每个 block **没有 AIV**，AIV 分支整个 `return`（`:174`）；普通路径则由 AIC 把 L0C 存进 UB（`:161`），AIV 再 drain 到 GM（`:190`）。证据：profile 里 `aiv_vec/aiv_mte3` 等字段在有/无 direct store 之间的差异。
2. **输出耗时模型**：`get_gemm_output_cost`（`heuristics.hpp:491-493`）里，direct 把"fixpipe 拷贝"与"GM 写回"从**串行求和**变为**取最大**（`max(fix_ns, gm_ns)` vs `max(fix_ns, bias_ns) + gm_ns + vector_ns`）。

假设：`关掉 direct store 后该 shape 略变慢，增幅集中在输出阶段`；若实测几乎不变，说明该 shape 输出阶段本来就不是瓶颈（fixpipe 与写回早已被主循环掩盖）——这同样是有效结论。对比组建议选 `128×32768×512`（输出占比更小）感受差异。

**证据字段速查**：计算压力 → `aic_mad`；GM 搬运 → `aic_mte2`；L1→L0 → `aic_mte1`；回写/epilogue → `aic_fixpipe` 与 `aiv_mte3`；wave 空洞无直接字段，用手推 waves + `aic_mad` 降幅佐证。

### 3.6 实验日志与判定

每次实验记录一行（沿用第 51 天基线表的口径头）：

| # | 假设（一句话） | 因子/取值 | 正确性 | 配置行（block/mad/stages/direct_store/resident/num_blocks） | dur (µs) | TFLOPS | GB/s | mad% | mte2% | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|

判定三档：**证实**（方向与机制均符合）、**证伪**（方向不符——回到机制找漏项，例如 shape 其实不在你以为的受限区）、**无法判定**（差异小于噪声——加大 `num_tests` 或换更敏感的 shape）。证伪不是失败：它修正的是你对机制的理解，这正是通过标准要的"提出并验证假设"。

## 四、源码导读

| # | 文件 | 关键符号 | 职责 |
|---|---|---|---|
| 1 | `csrc/jit_kernels/heuristics.hpp` | `select_gemm_config` `:592-755`、范围构造 `:604-621`、搜索 `:626-657`、`make_gemm_candidate` `:92-204`、`refine_gemm_pipeline` `:535-574`、`can_direct_store` `:521-533`、打印块 `:749-753` | 今天的主战场：五个旋钮的默认决策点 |
| 2 | `csrc/jit_kernels/heuristics.hpp`（代价模型） | `get_gemm_compute_cost` `:206-254`（waves `:219`、`compute_mn` `:231`）、`get_gemm_cost` `:256-468`（GM 字节 `:315-337`、装载时间 `:340-361`、prefetch `:427`、双界取 max `:459-467`）、`get_gemm_output_cost` `:474-494` | 假设的"推理引擎"：waves、流量、预取深度、输出串行/重叠 |
| 3 | `csrc/jit_kernels/gemm_config.hpp` | `GemmConfig` `:97-121`、formatter `:163-172` | 旋钮的宿主结构与打印语法 |
| 4 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78`（代码生成 `:18-66`、`kNumCores` 实参 `:57`、compile/launch `:71-77`） | 配置 → 模板实例化 → 编译启动；配置进代码字符串的实证 |
| 5 | `csrc/jit/jit.hpp` | `get_num_sms/set_num_sms` `:34-41` | 旋钮 1 的边界（只能减不能加，不能设 0） |
| 6 | `csrc/apis/gemm.hpp` | `num_cores` `:176`、`bf16_gemm_nt` `:450`、绑定默认 `compiled_dims="nk"` `:632-637` | 核数进入 desc；K 编译常量是 resident 的前提 |
| 7 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp`（设备） | 模板 `:9-26`、`__mix__` `:27`、`static_assert` `:37-45/:66-70`、resident 装载 `:109-118`、MAD 循环 `:133-153`、direct store 写回 `:155-163`、AIV drain `:173-191` | 旋钮在设备侧的真实行为与合法性拦截 |
| 8 | `deep_gemm/include/deep_gemm/scheduler.hpp` | `Scheduler` 模板 `:42-47` | `kNumCores` 如何划分 Tile |
| 9 | `deep_gemm/include/deep_gemm/common.hpp` | 容量常量 `:51-65` | 合法性公式的全部常数 |
| 10 | `deep_gemm/testing/bench.py` / `tests/utils.py` / `tests/common.py` | `bench_msprof` `:284-317`、`KernelProfile` `:27-71`、`calc_diff` `utils.py:36-43`、perf 段 `common.py:223-236` | 测量与正确性口径（第 50/51 天已精读） |
| 11 | `develop.sh` / `README.md` | 构建 `:10-19`、环境变量 `:81-98`、性能数据 `:100-133` | 重建流程与对照数据 |

**重点读四段**：

- **`heuristics.hpp:592-657`（选择主循环）**：看清 `m_range/n_range`（`:608-616`）与 `min/max_block_k`（`:620-621`）如何框定搜索空间——你的 BLOCK 实验补丁就插在这里；`:626-628` 从大到小的遍历顺序与 `:635-641` 的代价剪枝。
- **`heuristics.hpp:535-574`（`refine_gemm_pipeline`）**：direct store（`:539-543`）与 resident（`:546-565`）的启用条件与代价比较。BF16 的 resident A 要求"整块 A 装进 L1"（`a_bytes < 512KB`，`:549-552`）——用两个 shape 走一遍条件：`128×7168×K`，K=512 时 `a_bytes = 128×512×2 = 131KB < 512KB`（可启用，剩余空间还能容纳 3 级 B 缓冲）；K=2048 时 `a_bytes = 512KB` 不小于 512KB（不可启用）。**同样的 M/N，K 从 512 涨到 2048 就翻转 resident A 资格**——这是选实验 shape 的关键直觉。
- **`heuristics.hpp:206-254`（`get_gemm_compute_cost`）**：`:219` 的 `num_waves = ceil_div(num_blocks, d.num_cores)` 与 `:231` 的 `compute_mn = num_waves × compute_m × compute_n` 是 3.5 节全部波形手推的公式来源；`:233-249` 的尾部修正解释了为什么手推会略微高估。
- **设备 `bf16_gemm.hpp:109-163`（主循环）**：`kResidentA` 如何改变 A 的装载（`:110/:117/:143`，常驻后只搬一次、L0A 从 L1 的固定槽取数），`kDirectStore` 如何改变写回（`:156-159` 直接 `copy_l0c_to_gm`）。把第 26 天的路径图与今天的旋钮一一对应。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **准备**：确认 `git status` 干净（实验补丁可随时还原）；`export DG_PRINT_CONFIGS=1`；备好第 51 天的基线表。

2. **实验 1：AI Core 数量（无源码改动）**。保存为 `week8_day3_cores.py`：

   ```python
   import torch, deep_gemm
   from deep_gemm.testing import bench_msprof

   M, N, K = 4096, 7168, 16384
   a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')
   b = torch.randn(N, K, dtype=torch.bfloat16, device='npu')
   d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
   ref = (a.float() @ b.float().t()).to(torch.bfloat16)
   full = deep_gemm.get_num_sms()                     # set_num_sms 不能设 0 恢复，先存
   flops, moved = 2.0 * M * N * K, (a.numel() + b.numel() + d.numel()) * 2

   for cores in (full, 28, 16, 8):
       deep_gemm.set_num_sms(cores)
       deep_gemm.bf16_gemm_nt(a, b, d)                # 首调编译；打印该核数下的配置行
       torch.npu.synchronize()
       x, y = d.float(), ref.float()                  # calc_diff（tests/utils.py:36-43）
       diff = 1 - 2 * (x * y).sum() / (x * x + y * y).sum()
       assert diff < 1e-5, (cores, diff)              # 阈值同 tests/test_bf16_gemm.py:17
       prof = bench_msprof(lambda: deep_gemm.bf16_gemm_nt(a, b, d), kernel_names='gemm_impl')
       print(f'cores={cores:2d}: {prof.dur_us:8.1f} us | {prof.tflops(flops):4.0f} TFLOPS | '
             f'{prof.gbps(moved):5.0f} GB/s | mad={prof.aic_mad*100:4.1f}% | mte2={prof.aic_mte2*100:4.1f}%')
   deep_gemm.set_num_sms(full)
   ```

   - 先写下示范 1 的预测（waves 与 µs 数值），再运行对照；把每个核数打印的配置行记入日志——`block=` 是否随核数变化？`num_blocks` 是否等于核数？
   - 把 shape 换成 `128×7168×2048` 再跑一遍：预测"几乎不变（访存受限）"，用 `mte2%` 与 `mad%` 解释差异。
   - 注意 `set_num_sms` 只能减不能加（`jit.hpp:39`），进程内还原用存下的 `full`。

3. **实验 2：BLOCK_N（源码旋钮）**。按 3.3 节打补丁（`n_range` 门控 + `DG_EXP_BLOCK_N`），`./develop.sh` 重建，然后：

   ```bash
   for BN in 128 224 256 512; do
       DG_EXP_BLOCK_N=$BN python tests/test_bf16_gemm.py normal --skip-prof   # 正确性优先
       DG_EXP_BLOCK_N=$BN python week8_day3_block.py                            # 性能
   done
   ```

   `week8_day3_block.py` 与实验 1 脚本同构（shape 换成 `128×7168×2048`，去掉核数循环）。每次运行核对：配置行的 `block=128×{BN}×…` 是否符合意图；`mad=`/`l1_stages=` 等字段随 BLOCK_N 如何联动；`block_k` 有没有被 `:648-651` 的更宽 K 候选改写（若有且想固定，按 3.3 节同时门控 `:620-621`）。把结果与 3.5 节示范 2 的预测表对照，写下判定。

4. **实验 3/4（二选一）：direct store 或 resident A**。用 3.3 节的标志位补丁：

   ```bash
   DG_EXP_NO_DIRECT_STORE=1 python tests/test_bf16_gemm.py normal --skip-prof
   DG_EXP_NO_DIRECT_STORE=1 python week8_day3_block.py        # 对比基线（不设该变量）
   ```

   先确认基线配置行里 `direct_store=1`（或 `resident=1/1/...`），再对比开关两态的 dur 与 `aiv_*` 字段；若基线本来就是 0，换 shape（direct store 见 3.5 节示范 3；resident A 用 `128×7168×512` 一类小 K shape，见源码导读第 2 条的走查）。

5. **整理与还原**：填写 3.6 节日志表；`git checkout -- csrc/jit_kernels/heuristics.hpp && ./develop.sh` 还原；重跑一次基线 shape 确认回到第 51 天的数值。

### B 组：无 Ascend 环境（静态任务）

1. **合法性枚举**：对 `128×7168×2048` BF16 NT（K-major、无 bias）：推出 `m_range={16..128}`、`n_alignment=16`、`block_cap=512`；枚举 BLOCK_N ∈ {128, 224, 256, 512} 的合法性（L0C、对齐）并按 3.4 节公式算 `ab_bytes` 与 `num_l1_stages`（block_k=128 时分别为 65,536/90,112/98,304/163,840 B → 8/5/5/3 级）。
2. **roofline 手算**：完成 3.5 节示范 2 的完整版（计算 8.70 µs、访存 9.06 µs、AI≈119），并解释为什么它是"边界形态"、实验对这个 shape 特别有价值。
3. **条件走查**：仿照源码导读第 2 条，对 `128×7168×512` 与 `128×7168×2048` 手工走 `refine_gemm_pipeline`（`:546-565`），判定 resident A/B 的开合；再对 `128×7168×2048`（block 128×512）与 `4096×7168×16384`（448 blocks）走 `can_direct_store`（`:521-533`），后者因 `448 > 32` 且 `4096 > 512` 不具资格。
4. **波形预测表**：完成示范 1 的推演并把核数扩到 {32, 28, 16, 8}（waves = 14/16/28/56，计算侧预测 ≈ 2224/2542/4449/8898 µs），附带宽可行性核查（8 核发射能力 ≈ 3.4TB/s）；这张表就是你的"纸面实验结果"，第 53 天或未来有环境时逐格验证。
5. **实验设计文档**（当天产出）：从 BLOCK_N、L1 stages、direct store、resident 中选一个，按 3.1 节闭环写一份完整设计：假设、预测（含手推）、改动点（精确到行号）、正确性方案、测量方案、判定标准。

## 六、当天产出

**一份单变量实验记录**（`week8_day3_log.md`），至少包含：

1. **口径头**：沿用第 51 天基线表（设备、CANN、冷 L2、`bench_msprof` 默认参数、核数）；
2. **实验日志表**（3.6 节格式）：A 组 ≥ 2 个完整闭环（建议核数 + 一个源码旋钮），每行含假设、因子取值、正确性结果、完整配置行、dur/TFLOPS/GB/s/利用率、判定；
3. **假设闭环记录**：至少一个"假设 → 预测 → 结果 → 判定与解释"的完整叙述——这是**通过标准**的直接证据；
4. **B 组替代产出**：预测表（波形 + roofline）、合法性演算与一份实验设计文档；
5. **还原确认**：`git diff` 为空、基线 shape 复测回到第 51 天数值。

## 七、自测题

1. 为什么每次性能对比前必须先跑正确性？非法配置通常如何暴露？
   > 性能数字只有在结果正确时才有意义。多数非法配置会在 JIT 编译期被设备侧 `static_assert` 拦下（device `bf16_gemm.hpp:66-70`，报错含 "L1 overflow" 等）——响亮失败；语义级违规（如非 Identity 强开 direct store）也被 `:56-57` 拦住。按 shape 门控补丁可让官方测试套件只对目标 shape 施加实验配置。
2. 五个因素里哪个不需要改源码？边界是什么？
   > AI Core 数量：`deep_gemm.set_num_sms(n)`（`deep_gemm/__init__.py:27`）。断言 `1 ≤ n ≤ 设备核数`（`csrc/jit/jit.hpp:39`），只能减不能加；且不能设 0 恢复默认，须先存 `get_num_sms()`。
3. `set_num_sms(28)` 之后选中的 `GemmConfig` 一定不变吗？为什么？
   > 不一定。启发式用 `num_cores` 计算 waves（`heuristics.hpp:219`）、epilogue stage 数（`:193`）、direct store 资格（`:531`）、resident B 资格（`:556`）。所以每次都要记录配置行；配置协变本身也是实验观察。
4. 强制 BLOCK_N 为什么应改 `n_range` 而不是在函数末尾覆盖 `config.block_n`？
   > 改 `n_range`（`:609` 附近）让 `make_gemm_candidate`（`:92-204`）为该 block 联动重算 MAD、L1/L0/epilogue stage，配置内部自洽；末尾覆盖会留下与 block 不匹配的 MAD/stages，等于自造非法配置。
5. BF16 `block=128×256×128` 的 `num_l1_stages` 是多少？
   > `ab_bytes = (128+256)×128×2 = 98,304 B`；`524,288/98,304 = 5`（整数除法，`heuristics.hpp:170-174`），5×98,304 = 491,520 ≤ 512KB，6 级则超。
6. direct store 打开后，kernel 的 AIV 数量与 AIV 分支行为如何变化？
   > kernel 声明为 `__mix__(1, kDirectStore ? 0 : 2)`（device `bf16_gemm.hpp:27`）：direct store 时每个 block 不带 AIV，AIV 分支 `if constexpr (kDirectStore) return;`（`:174`）；写回由 AIC 直接 `copy_l0c_to_gm`（`:156-159`）。
7. `4096×7168×16384`（block=512×128）在 32 核与 28 核下各多少个 wave？预测时长比例？
   > 448 个 Tile：32 核 14 个 wave，28 核 16 个 wave（`ceil(448/n)`）；计算受限下时长比 ≈ 16/14 ≈ 1.14（单 Tile ≈ 158.9 µs，14 wave ≈ 2224 µs，与 README 2229.9 µs 吻合）。
8. 什么才算"验证了一个性能假设"，而不是"给出一组调参结果"？
   > 先写下机制与可证伪的预测（方向乃至数值），只改一个变量，正确性通过后测量，用数据和利用率字段判定假设被证实/证伪/修正——证伪也修正了对机制的理解。反例：扫一圈参数挑最快，无归因、无预测、无判定（学习计划第 52 天通过标准）。

---

明天（第 53 天）进入**综合小项目**：从新增测试场景、修改启发式配置、增加 Epilogue 变体、撰写源码讲解中选一项实施——今天练成的"单变量 + 假设闭环"方法正是这些项目的验证工具。
