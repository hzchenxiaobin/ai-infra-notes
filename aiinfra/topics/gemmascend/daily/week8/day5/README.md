# 第 8 周 · 第 5 天（总第 54 天）：验证与修正

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 45 分钟，源码导读约 20 分钟，实践步骤约 60～80 分钟，自测题约 15 分钟。

昨天（第 53 天）你完成了一次综合小项目的**立项、实施和首轮验证**：A＝新增非对齐/strided 测试场景，B＝针对一个 shape 修改启发式配置，C＝新增 ReLU Epilogue 变体，D＝撰写 BF16 Kernel 源码讲解。首轮验证只要求"至少一条正确性证据"，完整的工作还没有做。今天学习计划的四条要求正好构成四道工序：**① 运行与改动相关的正确性测试；② 检查新增测试是否真正覆盖目标场景；③ 有性能改动时，与第 51 天的基线比较；④ 记录失败实验，不为好结果省略不利数据**。验证发现问题就进入"修正→重验"的循环，最后做出"保留还是还原"的决定。当天产出是三件套：**修改说明、测试结果和性能对比**——它们同时也是明天（第 55 天）最终复盘与分享的最后一块素材。无 Ascend 环境时，今天同样有完整的静态替代：回归审查、覆盖性手算、配置指纹预测与失败日志撰写。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 为自己的项目列出"直接相关测试 + 回归测试"两层清单，说明为什么改动了共用路径（如 `tests/common.py` 的运行阶段）就必须扩大回归范围；
2. 按三层分诊一次失败——编译期 `static_assert`、host 侧 `DJ_HOST_ASSERT`、结果层 diff 超阈值——并说出每层失败在哪里暴露、错误信息长什么样；
3. 用"三问法"检查一个新增测试是否名副其实：**跑到了吗**（计数核对）、**数据具备目标属性吗**（属性核对）、**出错时它会发现吗**（故障注入），并用"真视图通过 + 假视图失败"的对称自证法证明 strided 测试的检测力；
4. 与第 51 天基线做同口径性能对比（冷 L2、30 次迭代、设备侧计时、配置行同记），用第 52 天的三档判定（证实/证伪/无法判定）给出结论，并处理噪声；
5. 用固定模板如实记录失败实验（预期、实际、归因、后续），解释为什么"证伪"和"不利数据"同样是有效产出；
6. 做出并执行"保留或还原"的决定：还原后用一次基线复测确认回到第 51 天数值。

## 二、预备知识

- **第 53 天**：立项文档（修改范围/预期结果/验证方法）、四个项目的实施与首轮验证、失败三层级的初步认识——今天的全部工作以它为输入。
- **第 51 天**：基线记录表（shape、配置行、dur、TFLOPS、GB/s、mad%/mte2%）与口径头（冷 L2、`bench_msprof`、设备侧计时）——性能对比的对照组。
- **第 52 天**：单变量纪律、实验日志表、三档判定——今天性能对比的判定框架原样沿用。
- **第 29～31 天**：测试体系数据流（`GemmTestDesc` → `enumerate_gemm_tests` → `run_gemm_test` 的五个阶段）、`assert_equal`/`calc_diff` 的误差口径。
- **第 34 天（可选）**：若你的项目涉及低精度，回顾 recipe 与误差验证。
- **工具**：`git diff` / `git checkout -- <file>` / `./develop.sh`；配置行是 C 侧 `std::puts`、perf 行是 Python `print`，重定向留档时建议 `stdbuf -oL python -u ... | tee log`。
- **环境说明**：A 组（运行测试与测量）需要 Ascend 环境（CANN 9.20 + torch_npu，已按第 9 天 `./develop.sh` 构建）；B 组静态任务（第五节）不需要硬件。

## 三、详细讲解

### 3.1 今天做什么：四道工序与一个循环

把学习计划的四条要求排成流水线，每道工序有明确的输入、动作与"完成判据"：

| 工序 | 计划原文 | 输入 | 核心动作 | 完成判据 |
|---|---|---|---|---|
| ① 正确性回归 | "运行与改动相关的正确性测试" | 昨天的改动 + 测试清单 | 跑直接相关与回归两层测试；失败则分诊 | 相关命令全绿，或失败已被定位并修正 |
| ② 覆盖检查 | "检查新增测试是否真正覆盖目标场景" | 新增测试（项目 A/C） | 三问法 + 故障注入 | 能证明"测试该失败时会失败" |
| ③ 性能对比 | "有性能改动时，与第 51 天的基线比较" | 基线表 + 改动后的测量 | 同口径重测、配置指纹 diff、填对比表 | 每个改动的 shape 有一行三档判定结论 |
| ④ 记录与修正 | "记录失败实验，不要为了得到好结果而省略不利数据" | 全天所有现象（含失败） | 失败日志 + 修正循环 + 保留/还原决策 | 报告里失败与成功同样在册 |

两点先说清，避免误用工序③：

- **"性能改动"主要指项目 B**（改了启发式 ⇒ 改了 kernel 的编译参数）。项目 A 是测试基建，strided 张量上的 perf 行是**新数据采集**，不是与基线的对照；项目 C 的 ReLU 与 Identity 可以在**同一脚本内做内部 A/B**（同 shape 两次调用），这属于新实验而非基线回归。
- 四个项目的验证重点不同：A 的重心在工序②，B 的重心在工序①③，C 在①②，D 的"验证"是事实核对（每条论断回源码复核），修正就是把讲错的地方改对。

### 3.2 工序一：正确性回归——测试矩阵与失败分诊

**先列清单再跑。** 两层测试的选择依据是"改动碰到了哪条共用路径"：

| 项目 | 直接相关测试 | 回归测试（防波及） | 理由 |
|---|---|---|---|
| A（strided 接线 / A1 加 shape） | `python tests/test_bf16_gemm.py strided --skip-prof`（A1 另跑 `unaligned-k`） | `python tests/test_bf16_gemm.py normal --skip-prof` | 接线点在 `run_gemm_test` 的运行阶段（`tests/common.py:190/:209` 的两处 `d` 创建之后），是**所有**测试共用的路径——`if test.strided` 分支之外的一行笔误也会波及全部用例 |
| B（启发式门控） | `python tests/test_bf16_gemm.py normal --skip-prof` | 其余 selector 可选；配置行零泄漏检查（见 3.4） | 门控按 shape 命中，目标 shape `128×7168×2048` 本身就在 normal 套件里（`tests/generators.py:559-570` 的 fwd 用例，m=128 × (7168, 2048)），同一命令既是聚焦验证又是全量回归 |
| C（ReLU epilogue） | 自写 ReLU 脚本 + 两个"预期报错"用例（见 3.3） | `python tests/test_bf16_gemm.py normal --skip-prof` | 默认 Identity 路径未被改写，但设备头文件被触碰，需确认默认路径无损 |
| D（讲解文档） | 事实核对清单 | — | 无代码改动 |

时间充裕时，`python tests/test_bf16_gemm.py all --skip-prof`（`tests/test_bf16_gemm.py:43-50`，依次跑 normal、unaligned-mn、unaligned-k、strided、mgrouped、kgrouped）是最重的完整回归。

**失败分诊：三层失败在哪里暴露。**

1. **编译期拦截（最响亮）**：设备侧 `static_assert` 在 JIT 编译时触发，典型报错文本含 "L1 overflow"、"L0A overflow" 等（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:66-70`）。因为测试先在 `Compiling` 进度条阶段做 dry-run 预编译（`deep_gemm/testing/par_compile.py:6-13` 的 `set_dry_run(True)` + `tests/common.py:152-175`），这类失败通常在进度条阶段就集中爆发。记录即可，不要"绕过"。
2. **Host 侧拦截**：`DJ_HOST_ASSERT` 在 launch 前抛出，Python 侧表现为 `RuntimeError`，信息即断言文本——如 "Bias must be contiguous"（`csrc/apis/gemm.hpp:114`）、"alpha and bias cannot be used together"（`:116`）。**有些失败是预期内的**：测试套件本身就用 try/except 断言 alpha+bias 必须被拒（`tests/common.py:191-198`）——"预期报错"是一类正当的测试。
3. **结果层失败（最隐蔽）**：`assert_equal` 的 diff 超过阈值，报错信息带行区间和 diff 值（`tests/utils.py:55-58` 的 `f'{msg} (rows {s}, diff={diff:.2e})'`）。注意 BF16 的阈值是 `1e-5` 的余弦相似度 diff 而非逐位相等——`tests/test_bf16_gemm.py:17` 传入 `threshold=1e-5`，`assert_equal` 的注释解释了原因：cube kernel 与参考在不同 K 累加顺序下可差约 1 ULP（`tests/utils.py:49-51`）；`threshold=0` 才走 `torch.equal` 逐位比较（FP8 MX 路径）。

三个让失败"更响亮、更可复现"的基础设施，分诊时要主动利用：

- **禁用 CPU 回退**：`run_gemm_test` 第一行就调用 `disable_cpu_fallback()`（`tests/common.py:142`）。它给所有没有 NPU 实现的算子注册抛错 kernel（`deep_gemm/testing/disable_fallback.py:44-73`），错误信息形如 `[npu_no_fallback] operator '...' has no NPU kernel and would fall back to the CPU`（`:33-39`）。否则一个静默回退到 CPU 的测试会"通过"，却什么都没验证。
- **同步化设备错误**：设备侧异步错误默认在提交后才浮出，栈信息缺失。`ASCEND_LAUNCH_BLOCKING=1` 让每次 launch 后同步，把异步错误变成行内失败（`README.md:98`）；配套的 `DG_JIT_LAUNCH_TIMEOUT`（默认 300 秒，`README.md:97`）管 kernel 启动超时。
- **确定性种子**：每个用例的 `generate()` 都先 `reset_seed()`（`tests/generators.py:429-430`，函数在 `:9-12`，固定 seed=0），`enumerate_gemm_tests` 开头同样重置（`:661`）；alpha 值来自 `random.Random(0)` 且运行前重新 `seed(0)`（`tests/common.py:143/:180`）。因此**失败是可精确复现的**——修正循环依赖这一点。

### 3.3 工序二：覆盖检查——"看起来在测"不等于"在测"

第 53 天已经见过反例：`python tests/test_bf16_gemm.py strided` 曾经**全部通过**，但 `make_strided` 无任何调用方、`GemmTestDesc.strided` 无人读取——通过的是"连续张量 + accumulate"。这就是覆盖检查要防的病。用三问法逐项过：

**第一问：跑到了吗？（接线与计数）**

- **计数核对**：`Num tests` 打印在枚举之后（`tests/common.py:150`），数量 = shape 数 × major 组合数（`product(shapes, [Major.K, Major.MN], [Major.K, Major.MN])`，`tests/generators.py:527`）。可核对的事实：`strided` 选择器 10 个 shape × 4 组合 = 40；`unaligned-k` 5 个 shape × 4 = 20，昨天 A1 加了 `(256, 512, 1000)` 后应为 24。**若计数没变，说明加错了列表或 selector 没走到**——这是最常见的"假覆盖"。
- **输出核对**：新用例的行（如 `m=  256, n=  512, k= 1000`）必须出现在输出里；strided 用例的 perf 行标签含 `strided`（tag 经 `test.tag` 拼进 perf 行，`tests/common.py:232`、`tests/generators.py:487`）。

**第二问：数据具备目标属性吗？（属性核对）**

测试数据必须真的具备你要覆盖的性质，否则断言通过只是"碰巧没测到"：

- strided：`a_s.stride()` 应为 `(K+pad, 1)` 这类非紧凑值（第 53 天脚本里打印过）；
- 非对齐 K：`k % 16 != 0`（BF16 的 `k_alignment`）且 `k % 64 != 0`（FP8 的 SF 粒度）——K=1000 两者都满足；
- ReLU：**乘积里必须存在负值**，否则 ReLU 与 Identity 不可区分。这是第 53 天预告的针对性用例：`torch.randn` 数据下乘积约一半为负，但要用代码固化这个前提（见第五节项目 C 步骤 1），不能靠运气。

仓库自己的测试数据生成就是这种"对抗性构造"的范本，值得对照学习：

| 手法 | 位置 | 目的 |
|---|---|---|
| 向 padding 区填随机垃圾 | `tests/generators.py:342-346`（M-Grouped SF 的 psum 空间）、`:384-389`（K-Grouped） | 若 kernel 越界读到 padding，结果立刻错——垃圾是"越界读"的探针 |
| FP8 覆盖全部非 NaN 编码 | `generate_random_quantized_with_major`，注释 "Cover all non-NaN E4M3 codes independently at every tensor element"（`:149-151`） | 不让随机采样漏掉罕见码字 |
| 数据与参考同步置零 | `zero_operand_slices_`（`:194-205`）同时改 operand 存储与参考 | 保证比较的基准与输入一致 |

**第三问：出错时它会发现吗？（故障注入）**

通过的测试可能只是"没有检测力"。证明检测力的办法是**故意让它失败**。以 strided 为例，做一对对称实验：

```text
实验 1（真视图，应通过）：数据按 padded 布局写入，声明 padded stride → kernel 读到正确数据 → diff < 1e-5
实验 2（假视图，应失败）：数据按紧凑布局写入，仍声明 padded stride → kernel 按声明 stride 读 → 除第 0 行外全部错位 → diff 爆炸
```

实验 2 精确模拟了"kernel 忽略声明的 outer stride、按紧凑布局寻址"的效果。两个实验合起来是一组**对称自证**：

- 若 kernel 忽略声明的 stride，实验 1 会失败、实验 2 反而通过——与观察相反；
- 观察到的"1 通过、2 失败"恰好证明：kernel 遵守声明的 stride（实验 1），且测试链路（参考计算 + 比较器）能发现 stride 错误（实验 2）。

再加两个廉价的健全性检查：**比较器健全性**（把参考结果的一行取反 `ref[0] = -ref[0]`，断言必须失败——证明你没有在拿 `d` 和 `d` 自己比）；**NaN 墓碑**（把 `pad_outer_stride` 里的 `torch.empty` 换成 `torch.full(..., float('nan'))`，测试仍应通过——正确 kernel 只读每行的 K 个元素，不碰 padding；若 padding 的 NaN 泄入结果，`diff < 1e-5` 对 NaN 恒为 False，断言立刻失败）。

对项目 C，"预期报错"用例同样是覆盖检查的一部分：带 `c=` 调用 ReLU 必须抛出 `check()` 里写的 "ReLU requires a direct output without accumulation"（第 53 天添加的 `ReluEpilogue::check`），alpha 与 epilogue 互斥必须抛错（`csrc/jit_kernels/epilogue_class.hpp:117` 的互斥检查）——写法直接仿照 `tests/common.py:191-198` 的 try/except 模式。**一个只在"正常输入"下通过的测试，只覆盖了一半语义。**

### 3.4 工序三：性能对比——回到第 51 天的基线

性能对比只对"真的改了 kernel 行为"的改动有意义（主要是项目 B）。三条纪律：

**1. 同口径。** 基线怎么测，今天就怎么测：`bench_msprof` 默认 `num_warmups=10, num_tests=30, flush_l2=True`（`deep_gemm/testing/bench.py:288-290`）——每次迭代前清 2×L2 缓冲（`_make_l2_flush_buffer`，`:263-268`；`profile_fn` 里的 `flush_buffer.zero_()`，`:342-351`），耗时取 30 次迭代的设备侧 PMU 均值。口径头（设备、CANN 版本、核数）原样抄第 51 天的。噪声处理：同一条件重复测 2～3 次看波动，或显式加大迭代 `bench_msprof(fn, kernel_names='gemm_impl', num_tests=100)`；波动大于效应时判"无法判定"，不要硬解读。

**2. 配置指纹一起比。** 性能数字必须和"实际生效的配置"绑定（第 52 天的教训：改写链可能让实际配置偏离意图）。配置行日志可以当**指纹**用：`select_gemm_config` 末尾按整行 `(desc, config)` 字符串去重打印（进程级 `static std::unordered_set`，`csrc/jit_kernels/heuristics.hpp:749-753`），且集中在 dry-run 编译阶段出现（`par_compile`）——同一 selector 的两次运行，日志可直接 `diff`：

```bash
export DG_PRINT_CONFIGS=1
stdbuf -oL python -u tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee day54_configs.log
diff day51_configs.log day54_configs.log     # 若第 51 天留过同命令的日志
```

对项目 B 的预期：**目标 shape `128×7168×2048` 的一族 desc 行变化**（同一 shape 有 nt/tn/nn/tt 四种布局 × plain/alpha/bias 等变体，每个变体是一条独立的 desc+config 行），**其余所有行零差异**。若目标 shape 之外的行也变了，说明门控条件写宽了（泄漏）——这是必须回去修的缺陷，而不是"意外收获"。

**3. 三档判定。** 沿用第 52 天的日志表与判定：每行记 假设 / 因子取值 / 正确性 / 完整配置行 / dur / TFLOPS / GB/s / mad% / mte2% / 判定（**证实 / 证伪 / 无法判定**）。证伪不是失败——它修正的是你对机制的理解。表外再挂一个锚点：与 README 已发表数据对数（如 BF16 `4096×7168×16384` = 2229.9 µs / 431 TFLOPS，`README.md:113`），确认测量环境没有漂移。

### 3.5 工序四：失败实验的记录与"修正"

学习计划原话："记录失败实验，不要为了得到好结果而省略不利数据。"落成一张表，每条失败实验一行：

| 字段 | 要求 |
|---|---|
| 实验 | 做了什么改动、跑了什么命令 |
| 预期 | 立项文档里**事先写下**的预期（没有预期就承认没有） |
| 实际 | 观察到的现象（含数值、报错原文） |
| 归因 | 一句话：机制理解错 / 门控没命中 / 改写链改写 / 环境噪声 / 测试盲区 |
| 后续 | 修正动作或"保留为已知问题" |

仓库自身就是"诚实记录"的范本：`gemm_aclnn` 里每一处跳过交叉验证都写明原因——"ACLNN requires K to be a multiple of 64"（`tests/common.py:98-99`）、MX 路径只支持 per-row 缩放（`:100-105`）、K≤64 的 MN-major scale 反例（`:106-112`），并注明 "the DeepGEMM kernel is still validated against d_ref above"；启发式里也有明文的自我批评注释——"TODO: add a memorization cache"（`heuristics.hpp:593`）、"We do not check the cost of the wider config, because the cost model doesn't consider the issue latency and scalar overhead"（`:645-647`）。**没被覆盖的角落被写下来，而不是被藏起来**——这是测试基建可信的前提，也是你今天报告应有的样子。

**修正循环**（验证发现问题时进入）：

```text
复现（确定性种子保证可复现，见 3.2）
→ 分诊（三层失败定位到层）
→ 单一假设修正（一次只改一处，回到第 52 天的单变量纪律）
→ 重跑全部相关验证（不是只跑失败的那一条）
→ 更新记录（预期-实际对照表更新，失败日志保留）
```

**保留还是还原。** 全部通过且结论成立（A 的覆盖真实存在、B 的改进被证实、C 的语义与联动符合预期）：改动保留在工作区，报告写清改进幅度与适用范围（本学习计划不要求提交；真实合入上游需走仓库的 PR 流程）。证伪或回归：`git checkout -- <files> && ./develop.sh` 还原，并**复测一个基线 shape 确认回到第 51 天数值**（还原也要被验证，这是第 52 天实验纪律的最后一步）。无论去留，失败记录都进报告。项目 C 若保留，记得设备头文件改动过的构建要配独立的 `DG_JIT_CACHE_DIR`，避免读到旧缓存 kernel（第 52/53 天的工程细节）。

## 四、源码导读

今天按"验证工具箱"读源码，行号以当前仓库为准：

| # | 文件 | 关键符号 | 与今天的关系 |
|---|---|---|---|
| 1 | `tests/common.py` | `run_gemm_test` `:131-237`（`disable_cpu_fallback` `:142`、`Num tests` `:150`、meta 预编译 `:152-175`、运行段 `:180-221`、alpha+bias 预期报错 `:191-198`、aclnn 交叉验证 `:208-213`、随机舍入 `:215-221`、perf 段 `:223-236`） | 工序①的执行器：五阶段结构决定"改动波及哪些阶段" |
| 2 | `tests/test_bf16_gemm.py` | CLI choices `:25`、`all` 分支 `:43-50`、`threshold=1e-5` `:17` | 测试矩阵的入口与误差口径 |
| 3 | `tests/utils.py` | `calc_diff` `:36-43`、`assert_equal` `:46-58`（阈值语义注释 `:49-51`）、`print_perf` `:106-115`（prof 为 None 时打 `OK` `:113-114`） | 工序③的指标口径；失败信息的字段来源 |
| 4 | `tests/generators.py` | `reset_seed` `:9-12`、`zero_operand_slices_` `:194-205`、`make_strided` `:238-246`、`GemmTestDesc` `:249-269`（`strided` `:261`、`profile` `:269`）、`_enumerate_normal_strided` `:466-488`、`_enumerate_normal_unaligned_k` `:519-541`、垃圾填充 `:342-346/:384-389`、全码覆盖 `:149-151` | 工序②的范本：对抗性数据构造与确定性 |
| 5 | `deep_gemm/testing/bench.py` | `bench_msprof` `:284-317`（默认参数 `:288-290`）、L2 清缓冲 `:263-268/:342-351`、`KernelProfile` `:27-71` | 工序③的口径与利用率证据字段 |
| 6 | `deep_gemm/testing/par_compile.py` | `dry_run` `:6-13`、`par_compile` `:16-36` | 配置行为何集中在 Compiling 阶段（日志可 diff 的前提） |
| 7 | `deep_gemm/testing/disable_fallback.py` | `disable_cpu_fallback` `:44-73`、抛错信息 `:33-39` | 让"静默回退"变成响亮失败 |
| 8 | `csrc/jit_kernels/heuristics.hpp` | 打印块 `:749-753`、`select_gemm_config` `:592-755`、诚实的 TODO `:593/:645-647` | 配置指纹的实现；门控泄漏检查的对照物 |
| 9 | `csrc/jit_kernels/gemm_config.hpp` | 两个 formatter `:145-172`、`get_outer_stride` `:27-34` | 配置行"语法"与 stride 的 host 侧来源 |
| 10 | `csrc/apis/gemm.hpp` | 断言群 `:106-123`（bias 连续 `:114`、alpha+bias `:115-116`、BiasEpilogue 构造 `:122-123`）、`num_cores` `:176`、outer stride `:191-193` | 工序①第二层失败的报错原文出处 |
| 11 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78`（dry-run `:67-70`、launch 传 stride `:73-77`） | stride 如何真实进入 kernel（假视图实验的机理） |
| 12 | `README.md` | 环境变量 `:81-98`（`DG_JIT_LAUNCH_TIMEOUT` `:97`、`ASCEND_LAUNCH_BLOCKING` `:98`）、Performance `:100-133` | 分诊工具开关与对数锚点 |

**重点读三段**：

- **`tests/common.py:180-221`（运行段）**：逐行标出昨天接线点（`:190` 与 `:209` 之后）影响哪些阶段——运行、aclnn、perf 三段都拿到 strided 张量，meta 编译段不经过这里（stride 不参与 `GemmConfig`，第 53 天已论证）。再看 `:226` 的 `if not skip_prof and test.profile and k:`——`test.profile` 默认 True（`generators.py:269`），K-Grouped 的非 normal 变体显式关掉（`:648`），这就是"哪些用例有 perf 行"的判据。
- **`tests/generators.py:124-166` + `:342-346`**：看仓库如何构造"必然能暴露错误"的数据——全码覆盖、垃圾填充、同步置零。对照你自己的新测试问一句：我的数据里有没有"如果 kernel 犯错就必然显现"的探针？
- **`csrc/jit_kernels/heuristics.hpp:749-753`（打印块）**：5 行读完，确认三件事——整行去重、`std::puts` 走 C stdout（重定向时的缓冲差异）、在 dry-run 也会执行。这是把配置日志当"指纹"diff 的全部依据。

## 五、实践步骤

### 第 0 步：盘点昨天的改动（所有项目、所有组别，约 10 分钟）

`git status` + `git diff`，逐块对照 `week8_day4_plan.md` 的"修改范围"：**每一行改动都应能追溯到立项文档**；超范围的多余改动今天清理掉（手术式改动原则）。同时取出昨天"给明天的交接"清单：待跑测试、基线对照项、已知风险。

### A 组：有 Ascend 环境

**项目 A（strided 接线 / A1 新 shape）**

1. **回归**：`python tests/test_bf16_gemm.py normal --skip-prof` 先跑（证明共用路径未波及），再 `python tests/test_bf16_gemm.py strided --skip-prof`；A1 另跑 `python tests/test_bf16_gemm.py unaligned-k --skip-prof`。
2. **计数与属性**：核对 `Num tests`（strided=40；unaligned-k 做 A1 后=24）；确认输出含新 shape 行、perf 行带 `strided` 标签；最小脚本打印的 `stride()` 非紧凑。
3. **故障注入三连**（扩展 `week8_day4_strided.py`）：

   ```python
   # (a) 比较器健全性：参考被破坏时必须失败
   ref = a_s.float() @ b_s.float().t()
   ref[0] = -ref[0]
   bad = 1 - 2 * (d_s.float() * ref).sum() / (d_s.float() ** 2 + ref ** 2).sum()
   assert not bad < 1e-5, 'comparator is blind!'

   # (b) 假视图：紧凑数据 + 声明 padded stride → 必须失败
   buf = torch.empty(M * (K + PAD), dtype=torch.bfloat16, device='npu')
   buf[: M * K].view(M, K).copy_(a)               # 数据按紧凑布局写入
   fake = buf.as_strided((M, K), (K + PAD, 1))    # 声明 padded stride
   d_fake = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
   deep_gemm.bf16_gemm_nt(fake, b, d_fake); torch.npu.synchronize()
   x, y = d_fake.float(), a.float() @ b.float().t()
   diff = 1 - 2 * (x * y).sum() / (x * x + y * y).sum()
   print('fake-view diff:', float(diff))          # 预期远大于 1e-5

   # (c) NaN 墓碑：把 pad_outer_stride 的 torch.empty 换成
   #     torch.full(shape, float('nan'))，重跑 strided 选择器，应仍然全部通过
   ```

   把 (a)(b)(c) 三个结果记入报告——这就是"新增测试真正覆盖目标场景"的证据链：(a) 证明比较器有视力，(b) 证明 stride 错误必然被抓住（配合真视图通过，构成 3.3 节的对称自证），(c) 证明正确 kernel 不读 padding。实验后把 (a)(c) 的临时代码撤掉，(c) 可作为更严的默认构造保留。
4. **记录并保留改动**；纯 Python 改动，无需重建。

**项目 B（启发式门控）**

1. **配置指纹**：按 3.4 节导出 `DG_PRINT_CONFIGS=1` 的 `normal --skip-prof` 日志，与第 51 天同命令日志 `diff`（没有留档则用第 51 天基线表逐 shape 核对）。预期：目标 shape 一族行变化、其余零差异；任何泄漏行都要回去收窄门控，然后重跑本步。
2. **正确性**：同一条 `normal --skip-prof` 全绿（目标 shape 在套件内，见 3.2 的矩阵表）。
3. **性能**：用第 51 天的最小脚本 + `bench_msprof`（默认口径）测目标 shape，填对比表（基线行 vs 门控行：配置、dur、TFLOPS、GB/s、mad%、mte2%），给出三档判定；重复测 2～3 次确认效应超出噪声。
4. **决策**：证实→保留补丁与构建产物，写明改进幅度与适用范围（只对目标 shape 有效）；证伪/回归→`git checkout -- csrc/jit_kernels/heuristics.hpp && ./develop.sh`，复测一个基线 shape 确认回到第 51 天数值。失败日志照写。

**项目 C（ReLU epilogue）**

1. **功能 + 负值占比**（补第 53 天预告的针对性用例）：

   ```python
   prod = a.float() @ b.float().t()
   neg_ratio = (prod < 0).float().mean().item()
   print('negative ratio:', neg_ratio)            # randn 数据下约 0.5；必须 > 0
   assert (prod < 0).any() and (prod > 0).any()

   # 全负构造：取反 A，乘积取反，正负交换 → ReLU 输出必然含 0
   d_neg = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
   deep_gemm.bf16_gemm_nt(-a, b, d_neg); torch.npu.synchronize()
   ref_neg = torch.relu((-a).float() @ b.float().t()).to(torch.bfloat16)
   # 用 calc_diff 口径比较 d_neg 与 ref_neg，阈值 1e-5

   # 手工小用例：M=1, N=2, K=4，期望输出 [[4, 0]]（一个正值一个零）
   # a = [[1, 1, 1, 1]]，b = [[1, 1, 1, 1], [-1, -1, -1, -1]]
   ```

2. **预期报错用例**：带 `c=`（形状 `[M, N]`）调用 ReLU 应抛 "ReLU requires a direct output without accumulation"；同时传 `alpha=` 与 `epilogue=` 应抛互斥错误——均按 `tests/common.py:191-198` 的 try/except 写法断言"必须失败"。
3. **配置联动核对**：`DG_PRINT_CONFIGS=1` 下对比同 shape 的 Identity 与 ReLU 两次调用：`epilogue_type=epilogue::operators::ReLU`、`direct_store=0`、`epi_stages` 变小（第 53 天预测的联动，今天验证；若不符，回到 3.5 的修正循环归因）。
4. **回归与内部 A/B**：`python tests/test_bf16_gemm.py normal --skip-prof` 确认默认路径无损；可选：同脚本内测 Identity vs ReLU 的 dur 与 `aiv_vec%`，记录 epilogue 的向量开销。
5. 保留改动（设备头文件改动的构建记得配独立 `DG_JIT_CACHE_DIR`）。

**项目 D（源码讲解）**

1. **事实核对**：把文档里每条带 `文件:行号` 的论断逐条回源码复核，标 ✓/✗；✗ 的今天改对——这就是项目 D 的"验证与修正"。
2. 若昨天没完成试讲，补一次 15 分钟试讲（或自述录音回听），记录没讲清的地方并修订。

### B 组：无 Ascend 环境（静态任务）

1. **静态回归审查**：通读昨天的 `git diff`（或设计文档），对每个改动块标注它影响的测试阶段（compile / run / aclnn / perf），论证非目标测试不受影响——这是 A 组回归步骤的纸面版本。
2. **假视图手算**（覆盖检查的静态版）：设 A 为 `[4, 8]` BF16、`pad=4`（声明 stride `(12, 1)`），数据按紧凑布局写入 44 元素缓冲（`3×12+8=44`）。计算 kernel 在声明 stride 下读到的"行 1"＝缓冲元素 12..19 ＝ `[row1[4:8], row2[0:4]]`，而正确的行 1 是元素 8..15——从第 12 个元素起全部错位，diff 必然爆炸；再写出 NaN 墓碑的论证（padding 含 NaN，若被读入结果则 diff 为 NaN，`diff < 1e-5` 恒为 False）。这两段手算就是你的"覆盖检查记录"。
3. **项目 B 静态版**：列出门控条件命中的 desc 家族（同一 shape 的 nt/tn/nn/tt × plain/alpha/bias 变体），预测每族配置行的变化方向（对照第 52 天的候选枚举与代价公式手推）；再写一份"未来验证清单"：每行一个 shape + 预测配置 + 验证命令 + 通过标准，将来有环境时逐格核对。
4. **项目 C 静态版**：手算 ReLU 语义边界的反例——`relu(1 + (-2)) = 0` 而 `relu(1) + (-2) = -1`，说明 ReLU 与累加 c 不可交换，这正是 `check()` 排除 c 的理由；写出两个预期报错与判据（报错文本原文）；复核 UB 布局手推（第 53 天 B 组已做，结论应为 fp32 变换使每 stage 占用翻倍、`epi_stages` 相应变小）。
5. **项目 D**：与 A 组相同（事实核对不需要硬件）。
6. **失败实验记录**：从本周挑一次"预期落空"（例如第 52 天某个手推预测与 README 数据的偏差、或第 53 天发现 strided 选择器名不副实），按 3.5 节模板写成一条完整的失败日志。

## 六、当天产出

**一份验证与修正报告**（`week8_day5_report.md`），对应学习计划要求的"修改说明、测试结果和性能对比"三件套，建议结构：

1. **修改说明**：最终 `git diff` 摘要 + 每块改动的目的（回链立项文档的"修改范围"）；与昨天相比今天新做的修正（若有），以及修正的原因；
2. **测试结果**：跑过的每条命令 + 通过/失败状态；覆盖检查证据链（计数核对、属性核对、故障注入三连的输出）；失败的分诊记录（层、报错原文、处置）；
3. **性能对比**（有性能改动时）：基线 vs 改动后的对照表（口径头 + 配置行 + dur/TFLOPS/GB/s/利用率）+ 三档判定结论；项目 C 的 Identity/ReLU 内部 A/B（可选）；
4. **失败实验记录**：3.5 节模板的表格，不利数据全部在册；
5. **保留/还原决定**：结论 + 理由；若还原，附基线复测结果证明回到第 51 天数值；
6. **B 组替代产出**：静态回归审查、假视图手算与 NaN 墓碑论证、配置预测与未来验证清单、失败日志条目。

写完后自查两个问题：**不看立项文档，能否从报告看出"改了什么、为什么、证据是什么"？** 若明天有人质疑你的某个结论，报告里是否有可以翻给他看的数据？

## 七、自测题

1. 为什么"新增测试通过"不等于"覆盖了目标场景"？举出让通过变得无意义的两种方式。
   > 第 53 天的实例：`strided` 选择器全部通过，但 `make_strided` 无调用方、`GemmTestDesc.strided`（`tests/generators.py:261`）无人读取。两种方式：① 接线遗漏，数据根本没有目标属性（跑到了吗）；② 数据缺少可观测性，如 ReLU 测试里乘积无负值，ReLU 与 Identity 不可区分（属性核对）。对策是计数/属性核对加故障注入。
2. 故障注入的"对称自证"怎么做？两个实验各证明什么？
   > 实验 1（真 padded 视图）通过 + 实验 2（紧凑数据、声明 padded stride 的假视图）失败。若 kernel 忽略声明的 stride，两实验结果会互换；观察到的组合证明 kernel 遵守声明 stride（1）且测试链路能发现 stride 错误（2）。
3. 三层失败各在哪里暴露？举一个"预期失败"的测试先例。
   > 编译期 `static_assert` 在 JIT 编译（Compiling 进度条阶段）爆发，如 "L1 overflow"（设备 `bf16_gemm.hpp:66-70`）；host 侧 `DJ_HOST_ASSERT` 是 Python `RuntimeError`，如 "Bias must be contiguous"（`csrc/apis/gemm.hpp:114`）；结果层是 `assert_equal` 的 diff 超阈值，报错带行区间与 diff 值（`tests/utils.py:55-58`）。先例：alpha+bias 互斥测试断言必须抛 'alpha and bias cannot be used together'（`tests/common.py:191-198`）。
4. BF16 测试为什么用 `threshold=1e-5` 的余弦 diff 而不是逐位相等？
   > cube kernel 与参考在不同 K 累加顺序下可差约 1 ULP，逐位相等不可期（`tests/utils.py:49-51` 的注释）；`threshold=0` 的 `torch.equal` 逐位比较只用于 FP8 MX 这类"相同 cube + cast 路径"的情形。BF16 的 `1e-5` 由 `tests/test_bf16_gemm.py:17` 传入。
5. `bench_msprof` 的默认口径是什么？为什么对照必须同口径？
   > `num_warmups=10, num_tests=30, flush_l2=True`（`bench.py:288-290`）：每次迭代前清 2×L2 缓冲（`:263-268/:342-351`），耗时取 30 次设备侧 PMU 均值。冷/热 L2 改变带宽观感，迭代均值排噪声——第 51 天基线如此取得，口径不同则对照无效。
6. 配置行日志为什么可以当"指纹"做 `diff`？
   > `select_gemm_config` 末尾按整行 `(desc, config)` 字符串去重打印（进程级 `static std::unordered_set`，`heuristics.hpp:749-753`），且 dry-run 预编译阶段就会执行（`par_compile.py:6-13`）——同一 selector 的两次运行各产生一份稳定的唯一行集合，差异行即配置变化。
7. 项目 B 的配置日志 diff 里，若目标 shape 之外的行也变了，说明什么？该怎么办？
   > 门控条件写宽了（泄漏）——改动影响了非目标 shape，违反"最小改动"意图。回到门控条件收窄它（回到 3.5 的修正循环），重跑指纹 diff 直到只有目标 shape 一族行变化。
8. 项目 B 证伪后应该做什么？失败记录要保留吗？
   > 还原：`git checkout -- csrc/jit_kernels/heuristics.hpp && ./develop.sh`，并复测一个基线 shape 确认回到第 51 天数值（还原本身也要被验证）。失败记录必须保留——学习计划第 54 天明确"记录失败实验，不要为了得到好结果而省略不利数据"；证伪修正的是对机制的理解，是有效产出。

---

明天（第 55 天）进入**最终复盘与分享**：准备 15 分钟分享或 2～3 页总结（仓库架构、调用链、BF16 数据流、测试验证方法、一个扩展算子、一次性能分析或综合实践）——今天的报告（修改说明、测试结果、性能对比）正是其中"综合实践"一节的现成素材。

