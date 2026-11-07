# 第 5 周 · 第 6 天（总第 34 天）：串联低精度调用链

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 6 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前两天我们把低精度世界的两块拼图各自看清了：第 32 天拆开了 FP8/FP4 的三条执行路径（SF 在 Cube 的 `asc_mmad_mx` 内部参与计算），第 33 天打开了 Scale 布局转换的黑盒（Host 用一个 AIV Kernel 把任意输入规整成 int16 MN-major 契约）。今天把它们**串成一条完整调用链**：选择 `fp8_fp4_gemm_nn`，从"输入与 Scale"一路追到"结果验证"，每一站都在源码里找到确切位置。当天的**通过标准**只有一句话：**能解释 Scaling Factor 在哪里转换、在哪里参与计算**——3.8 节给出完整答案，当天产出是一张 `fp8_fp4_gemm_nn` 全链路流程图（参考版本见 3.1 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出 `fp8_fp4_gemm_nn` 的六站调用链——**输入与 Scale → recipe 检查 → Scale 布局转换 → JIT 配置 → 低精度 Kernel → 结果验证**——并说出每一站对应的文件与函数：`fp8_fp4_gemm<false, true>`（`csrc/apis/gemm.hpp:225`）、`get_gemm_desc`（`:60`）、`layout::get_recipe`（`csrc/apis/layout.hpp:113`）、`transform_sf_into_required_layout`（`:68`）、`select_gemm_config`（`csrc/jit_kernels/heuristics.hpp:592`）、`fp8_gemm_impl`（`deep_gemm/include/deep_gemm/fp8_gemm.hpp:36`）、`run_gemm_test`（`tests/common.py:131`）；
2. 解释 `nn` 后缀的确切语义：`kTransA=false, kTransB=true`（`gemm.hpp:255`），`get_transposed` 把 **(B, sfB) 二元组一起做逻辑转置**（`gemm.hpp:239-240`、`csrc/utils/layout.hpp:88-99`）——sfB 的两个轴始终跟随 B 的两个轴，转置只改 stride 视图不搬数据，随后 `get_major_type_ab` 按实际 stride 判定 major（`utils/layout.hpp:44-47`）；
3. 复述 `get_gemm_desc` Normal 分支的检查顺序：`get_shape_by_spec<"mnk">` 从 A/B/D 的标注推断 m/n/k（`gemm.hpp:83`）、D 必须 N-major（`:99`）、C 的"累加别名 vs [1, N] bias"判别（`:106-117`）、`gemm_early_return` 的空 shape/零 K 短路（`:118`、`utils/layout.hpp:54-86`），以及 `GemmDesc` 里 `sfa/sfb` 两个 optional 张量"持有防释放"的作用（`gemm.hpp:194-195`、`csrc/jit_kernels/gemm_config.hpp:77-78`）；
4. 说出 recipe 检查的三条规则（`layout.hpp:113-128`）：`recipe` 三元组与 `recipe_a/recipe_b` 二元组互斥；两侧 `gran_k` 必须一致；缺省值为 `(1, 32)`——并解释 `gran_k == 32` 最终由 `check_sf_layout` 钉死（`:55`）；
5. 指出 Scale 布局转换发生在 `get_gemm_desc` 内部、`select_gemm_config` **之前**（`gemm.hpp:127-154`），命中快速路径时零拷贝返回、否则 `launch_transform_sf` 启动 AIV Kernel（`csrc/jit_kernels/transform_sf.hpp:15-105`）产出 int16 MN-major 契约布局（详见第 33 天）；
6. 列出 JIT 配置的低精度专属逻辑（`heuristics.hpp`）：`has_sf` 时 `BLOCK_K` 必须是 64 的倍数（`k_alignment = sf_elements_per_pair = 64`，`:108-109`）、`sf_k_blocks = min(4, ceil(k/BLOCK_K))` 再按 L1 余量收紧（`:156-164, :177-183`）、`num_l1_sf_stages = 2`、L1 容量在 A/B 与 SF 之间共享且上限 4 级（`:166-174`）；并数出 `launch_fp8_gemm` 生成的 `fp8_gemm_impl<...>` 共 27 个模板实参（`csrc/jit_kernels/fp8_gemm.hpp:13-38`）；
7. 追踪 Kernel 内 SF 的消费链：`gm_sfa/gm_sfb` 固定 MN-major（`fp8_gemm.hpp:65-66`）→ `copy_gm_to_l1_mx` 每 `kSFKBlocks` 个 K 块搬一个 SF chunk（`:173-187`）→ `copy_l1_to_l0a/b_mx` 用 `sf_y` 把 UE8M0 与数据一同装入 L0（`:209, :221-224`）→ `mad()` 对 8 位操作数调用 `asc_mmad_mx`，**scale 在 Cube 的 MAD 内部以指数加法参与计算，部分和以 FP32 累加在 L0C**（`ascend/copy.hpp:370-381`、`fp8_gemm.hpp:87`）；
8. 描述结果验证的完整流程（`tests/common.py:131-237`）：meta 设备 `par_compile` 并行预编译（`:152-175`）、同一份数据在 nt/tn/nn/tt 四种绑定下逐一执行（`:185-199`）、`assert_equal` 以 1e-5 的余弦相似度阈值比较（`tests/utils.py:36-58`）、ACLNN 交叉验证及其四类跳过条件（`common.py:98-122`）、`bench_msprof` 性能行（`:225-236`）；有环境时运行 `python tests/test_fp8_gemm.py normal --skip-prof` 并解读输出。

## 二、预备知识

- **第 2 天**：逻辑转置与物理搬移的区别——`nn` 后缀的全部魔法就是一次 `transpose` 视图。
- **第 10～12 天**：Python binding、`GemmDesc` 与 `GemmConfig` 的区别、`jit->compile()/launch()` 与模板代码生成——今天的六站链路是它们的低精度版本。
- **第 17 天**：`Scheduler` 的 `get_next_block/get_global_idx`——Kernel 侧循环的骨架。
- **第 19、22～25 天**：`copy_gm_to_l1 / copy_l1_to_l0a/b`、K 维两层循环、MAD 累加与 AIC/AIV Epilogue——低精度 Kernel 原样复用这些结构。
- **第 29～30 天**：`GemmTestDesc.generate()`、参考结果构造、`par_compile` 与 `run_gemm_test` 的两阶段编排——今天的"结果验证"站直接消费。
- **第 32 天**：FP8/FP4 三条路径、SF 搬运链（`copy_gm_to_l1_mx → copy_l1_to_l0a/b_mx → asc_mmad_mx`）——今天是它的"全链路版"。
- **第 33 天**：SF 目标布局契约（int16 ue8m0x2、`[MN, ceil(K/64)]`、MN-major、按 `gran_mn` 广播）与 `transform_sf` Kernel 内部——今天只引用结论，不重复推导。

另注：第五节 A 组任务需要 Ascend 环境；无环境时完成 B 组静态任务（形状推演、sf_y 手算、调用链重画、测试覆盖分析）即可达到当天目标。

## 三、详细讲解

### 3.1 全链路地图：六站串成一条线

先给全景。`fp8_fp4_gemm_nn` 的一次调用依次经过（行号见括号）：

```text
Python: deep_gemm.fp8_fp4_gemm_nn((A, sfA), (B, sfB), D, recipe=…)
  │  导出：deep_gemm/__init__.py:47-48    注册：csrc/python_api.cpp:35 → csrc/apis/gemm.hpp:562-569
  ▼
① fp8_fp4_gemm<false, true>                          (csrc/apis/gemm.hpp:225)
  │   ├─ get_transposed<false>(a_) / get_transposed<true>(b_)   (:239-240)  ← 输入归一化（nn：B、sfB 视图转回 [N, K]）
  │   └─ get_gemm_desc(GemmType::Normal, …)           (:242-248)
  │        ├─ get_shape_by_spec<"mnk">                (:83)      ← m/n/k 推断 + major 判定
  │        ├─ C/bias 检查、gemm_early_return           (:99-120)
  │        ├─ ② layout::get_recipe(…)                  (:127)    ← recipe 检查
  │        ├─ ③ sfa/sfb = transform_sf_into_required_layout × 2 (:128-154) ← Scale 布局转换（AIV Kernel）
  │        └─ 构造 GemmDesc（sfa/sfb 持有防释放）       (:164-196)
  ▼
④ dispatch_fp8_fp4_gemm(desc)                         (gemm.hpp:22-33)  ← 按 (a_dtype, b_dtype) 三分
  └─ launch_fp8_gemm(desc)                            (csrc/jit_kernels/fp8_gemm.hpp:11)
       ├─ select_gemm_config(desc)                    (heuristics.hpp:592) ← JIT 配置（低精度专属项见 3.5）
       ├─ std::format 生成 fp8_gemm_impl<27 个模板实参> (fp8_gemm.hpp:13-38)
       ├─ jit->compile("fp8_gemm", code)               (fp8_gemm.hpp:71)
       └─ jit->launch(kernel, …, sfa_ptr, sfb_ptr, …)  (fp8_gemm.hpp:73-78)
  ▼
⑤ 设备 Kernel fp8_gemm_impl                            (deep_gemm/include/deep_gemm/fp8_gemm.hpp:36)
  │  AIC：Scheduler 循环 → GM→L1（A/B 数据 + SF chunk）→ L1→L0（数据 + _mx SF）→ asc_mmad_mx → L0C(FP32)
  │        └→ 最后一个 K 块：Epilogue aic_store（或 direct store 直写 GM）
  │  AIV ×2：aiv_drain → UB→GM(D)
  ▼
⑥ 结果验证：python tests/test_fp8_gemm.py normal --skip-prof
     └─ run_gemm_test → gemm_dg(四种绑定) → assert_equal(1e-5) / aclnn 交叉验证 / bench_msprof
```

本节余下部分按六站逐一展开。贯穿全文的例子取测试里的真实 shape：`M=4096, N=7168, K=7168`、`recipe=(1, 1, 32)`（`tests/generators.py:556` 的默认值）、输出 bf16。

### 3.2 第 ① 站：输入与 Scale——`nn` 后缀到底改了什么

`fp8_fp4_gemm_nn` 是模板函数 `fp8_fp4_gemm<kTransA, kTransB>` 的 `kTransA=false, kTransB=true` 实例（`gemm.hpp:255`）。它做的第一件事是把两个输入对分别"归一化"：

```cpp
auto [a, sfa_in] = get_transposed<kTransA>(a_, 0, 1);   // false：原样返回
auto [b, sfb_in] = get_transposed<kTransB>(b_, 0, 1);   // true ：B 和 sfB 一起 transpose(0, 1)
```

`get_transposed`（`csrc/utils/layout.hpp:88-99`）对 `std::pair<Tensor, Tensor>` 的行为是**数据与 SF 一起转**。所以四种绑定的用户视角是：

| 绑定 | kTransA/B | 用户传 A | 用户传 B | 归一化动作 |
|---|---|---|---|---|
| nt | false/false | [M, K] | [N, K] | 无 |
| **nn** | **false/true** | **[M, K]** | **[K, N]** | **B、sfB 视图转回 [N, K]** |
| tn | true/true | [K, M] | [K, N] | A、sfA、B、sfB 全部转回 |
| tt | true/false | [K, M] | [N, K] | A、sfA 转回 [M, K] |

两个要点：

1. **sfB 的轴永远跟随 B 的轴**。`nn` 下用户传 B: [K, N]、sfB: [ceil(K/gran_k), ceil(N/gran_n)]；归一化后两者一起变成 [N, K] 与 [ceil(N/gran_n), ceil(K/gran_k)]——这正是 `check_sf_layout` 期望的"倒数第二维 = MN、最后一维 = K"框架（`layout.hpp:53-54`）。接口文档（`gemm.hpp:212-213`）描述的 `[ceil(N/gran_n), ceil(K/gran_k)]` 即归一化后的框架。
2. **转置是纯视图**（第 2 天的"逻辑转置"）：只改 shape/stride，不搬数据。随后的 `get_major_type_ab`（`utils/layout.hpp:44-47`）按实际 stride 判定 major——`stride(-1) == 1 → Major::K`，`stride(-2) == 1 → Major::MN`。于是同一块物理 B 存储，在 nt 视角下是 K-major，在 nn 视角下（转回 [N, K] 视图后 stride(-2)=1）就是 MN-major——**四种绑定天然覆盖了不同的 major 组合**，这是测试能让同一份数据跑四遍的原因（见 3.7）。

输入侧的形态约束在函数开头与 `get_gemm_desc` 里断言：D 必须 bf16/fp32（`gemm.hpp:241`），fp32 SF 只允许 8 位指数、尾数为 0（`gemm.hpp:209-215`，设备侧 `transform_sf.hpp` 的 `check_validation` 兜底）。

### 3.3 第 ① 站（续）：`get_gemm_desc` 的 Normal 分支

进入 `get_gemm_desc`（`gemm.hpp:60-197`）后，Normal 分支依次做五件事：

**（1）shape 推断**。`std::tie(m, n, k) = get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);`（`:83`）。标注机制（`utils/layout.hpp:140-219`）要求每个张量至少有一个单位 stride 轴，同名维度跨张量必须一致——A 标 "mk"、B 标 "nk"、D 标 "mn"，冲突即 panic。FP4 有个特例：打包维度在 major 轴上逻辑长度要 ×2（`:185-187`）。

**（2）D 与 C 的检查**。`check_major_type_cd(d)` 断言 D 的 `stride(-1) == 1`（D 永远 N-major）；C 通过指针判别两种身份（`:106-107`）：与 D 同指针同形状 → **累加**（`acc = true`），否则必须是连续的 [1, N] → **bias**（换成 `BiasEpilogue`，`:122-123`）；alpha 与 bias 互斥（`:113-117`）。

**（3）早退**。`gemm_early_return`（`utils/layout.hpp:54-86`）：m 或 n 为 0 直接返回；k 为 0 时 `d.copy_(c)` 或 `d.zero_()` 后返回——`get_gemm_desc` 返回 `nullopt`，`fp8_fp4_gemm` 检测到后直接 return（`gemm.hpp:249`），**连 SF 转换都不会发生**。

**（4）recipe 检查与 SF 转换**（第 ② ③ 站，见 3.4）。

**（5）构造 `GemmDesc`**（`:164-196`）。关键字段：`sfa_ptr/sfb_ptr` 取自**转换后**的张量（`:171-172`）；`outer_stride_a/b/d` 携带非连续轴的真实 stride（`gemm_config.hpp:27-34`，K-major 取 `stride(-2)`、MN-major 取 `stride(-1)`）；`num_cores = runtime->get_num_sms()`（`:176`）；以及两个 optional 成员 `sfa/sfb` **持有张量本身**——JIT launch 是异步的，若不持有，转换产生的临时 SF 张量可能在 Kernel 真正执行前被释放（`gemm_config.hpp:77-78` 的注释写明 "used to hold tensor to prevent it from being freed before the kernel is launched"）。

随后 `dispatch_fp8_fp4_gemm`（`gemm.hpp:22-33`）按 `(a_dtype, b_dtype)` 三分：fp8×fp8 → `launch_fp8_gemm`，fp4×fp4 → `launch_fp4_gemm`（同一个 `fp8_gemm_impl`，只是 `ab_dtype_t = float4_e2m1x2_t`），fp8×fp4 → `launch_fp8_dequant_gemm`（第 32 天的 AIV 解码路径）。本文的例子走 `launch_fp8_gemm`。

### 3.4 第 ② ③ 站：recipe 检查与 Scale 布局转换

**recipe 检查**由 `layout::get_recipe`（`layout.hpp:113-128`）完成，规则三条：

- `recipe`（三元组 `(gran_m, gran_n, gran_k)`）与 `recipe_a/recipe_b`（二元组 `(gran_mn, gran_k)`）**互斥**（`:118`）；
- 传二元组时各自可缺省 `(1, 32)`，但两侧 `gran_k` 必须相等（`:125`）；
- 合并结果统一为三元组。`gran_k == 32` 这条硬约束由 `check_sf_layout` 钉死（`:55`）——MX 标准粒度。

**Scale 布局转换**紧跟其后（`gemm.hpp:128-154`）：sfa 用 `(m, k, is_sfa=true)`、sfb 用 `(n, k, is_sfa=false)` 分别调用 `transform_sf_into_required_layout`（`layout.hpp:68-92`）。内部两步（第 33 天的结论，此处只列要点）：

1. `check_sf_layout`（`:37-65`）探测输入是否已满足契约（int16、`gran_mn == 1`、紧凑 MN-major、稠密 batch）——满足则**零拷贝原样返回**；
2. 否则 `launch_transform_sf`（`csrc/jit_kernels/transform_sf.hpp:15-105`）JIT 实例化 AIV Kernel `transform_sf`，在 MTE2/V/MTE3 三级流水里完成"抽指数 → 打包 ue8m0x2 → 转置 → 按 `gran_mn` 广播"，产出逻辑 `[..., MN, ceil(K/64)]`、物理 MN-major 的 int16 张量。

**在链路中的位置**是今天要强调的：转换发生在 `get_gemm_desc` 内部，**先于** `select_gemm_config`、更先于 Kernel launch。这意味着：

- 转换输出**已经确定**了 Kernel 将看到的 SF 布局——`GemmDesc` 注释把这条约定写成仓库级规范："D is always N-major and SF is always MN-major"（`gemm_config.hpp:36-38`）；
- `sfa_ptr/sfb_ptr` 是转换后张量的 `data_ptr()`，launch 时作为裸指针传入（`fp8_gemm.hpp:76`），生命周期由 `GemmDesc.sfa/sfb` 保底；
- 对例 shape（M=4096, N=7168, K=7168, recipe (1,1,32)）：用户 fp32 sfA `[4096, 224]` → 转换后 int16 `[4096, 112]`（物理 `[112, 4096]` 转置视图），sfB 同理 `[7168, 112]`——`ceil(7168/64) = 112`，与 Kernel 侧的 `shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR)`（`fp8_gemm.hpp:63`）严丝合缝。

### 3.5 第 ④ 站：JIT 配置——`select_gemm_config` 的低精度专属逻辑

`launch_fp8_gemm`（`csrc/jit_kernels/fp8_gemm.hpp:11-79`）第一步 `select_gemm_config(desc)`。通用逻辑（块搜索 + 代价模型，第 12 天已读）不变，低精度相关的差异集中在这几处（`heuristics.hpp`）：

| 决策 | 规则 | 位置 |
|---|---|---|
| dtype 摘要 | `get_gemm_dtype`：fp8×fp8 → `elem_bits=8, has_sf=true` | `:80-90` |
| BLOCK_K 对齐 | `has_sf` 时 `k_alignment = hw.sf_elements_per_pair = 64`——**BLOCK_K 必须是 64 的倍数**（一个 SF pair 覆盖 64 个 K） | `:108-109` |
| BLOCK_K 范围 | `min_block_k = 256×8/8 = 256`，`max_block_k = 512`（fp8）；块上限 512 | `:604, :620-621` |
| SF 流水 | `sf_k_blocks = min(4, ceil(k/BLOCK_K))`、`num_l1_sf_stages = 2`，再按 L1 余量收紧上限 | `:156-164, :177-183` |
| L1 预算共享 | `(BLOCK_M+BLOCK_N) × BLOCK_K` 的 A/B 字节与 `(BLOCK_M+BLOCK_N) × (BLOCK_K/64) × 2B × sf_k_blocks × 2` 的 SF 字节**共用 512KB L1**，`num_l1_stages = min(4, 余量/每级字节)` | `:166-174` |
| AIV 数量 | 非 dequant 的 fp8 路径 `num_aivs = 2`（两个 AIV 做 Epilogue drain） | `:186` |
| L2 hint | A/B 按 M/N 块数比选 `NOTALLOC_KEEP` 或 `NORMAL_FIRST_VICTIM`；store 默认 `NOTALLOC_CLEAN`，`acc && waves≥4 && K 块数≥16` 时改 `NORMAL_FIRST_VICTIM` | `:200-201, :664-667` |
| 精化 | `refine_gemm_pipeline`：`can_direct_store` 判定直写路径、compiled K 时尝试 resident A/B/SF | `:521-574` |

用例 shape 估算一组典型值：`BLOCK_M=128, BLOCK_N=256, BLOCK_K=256` 时，`sf_bytes = (128+256)×(256/64)×2×4×2 = 24KB`，`ab_bytes = 384×256 = 96KB`，`num_l1_stages = min(4, (512−24)/96) = 4`——四组 AB 缓冲与两份 SF chunk 缓冲同时在 L1 里滚动。

随后是**代码生成与启动**：`std::format` 拼出 `fp8_gemm_impl<...>` 的实例化代码，模板实参共 **27 个**（`fp8_gemm.hpp:13-38` 的占位符与 Kernel 声明 `fp8_gemm.hpp:18-35` 一一对应）：major_a/b、编译期 m/n/k（由 `compiled_dims` 决定，`nn` 默认 `"nk"`，`gemm_config.hpp:17-21`）、BLOCK/MAD 九个、五个流水深度（含 `sf_k_blocks/num_l1_sf_stages` 两个 SF 专属项）、num_blocks、三个 L2 模式、acc、gemm_type、ab/cd dtype、epilogue operator、对齐常量 256、direct_store 与 single_m_block。`jit->compile("fp8_gemm", code)` 走缓存（第 12 天），`jit->launch` 把 `gm_ptr` 包装的 A/B/D 指针、`sfa_ptr/sfb_ptr` 裸指针、运行时 shape 与 epilogue 参数一并传入（`:73-78`）。`runtime->get_dry_run()` 为真时只编译不加载——`par_compile` 正是靠它做并行预编译（`deep_gemm/testing/par_compile.py:18-31`）。

想观察配置：`export DG_PRINT_CONFIGS=1` 会让 `select_gemm_config` 末尾打印一行 `GemmDesc(...) GemmConfig(...)`（`heuristics.hpp:749-753`）。

### 3.6 第 ⑤ 站：低精度 Kernel——SF 的搬运与消费

`fp8_gemm_impl`（`fp8_gemm.hpp:36-287`）是 `__global__ __mix__(1, kDirectStore ? 0 : 2)` Kernel：每个 block 1 个 AIC + 2 个 AIV（直写路径 0 个 AIV）。SF 相关的主线如下（AIC 分支）：

**（1）入口包装**（`:60-66`）。运行时 shape 取回后计算 `shape_sf_k = ceil_div(shape_k, 64)`，把两个裸指针包装成固定 MN-major 的 `gm_ptr<int16_t>`——stride_outer 是 MN、batch stride 是稠密的 `shape_mn × shape_sf_k`（`gemm_config.hpp:62-65` 的约定）。**从这里开始，Kernel 对 SF 的全部假设就是第 33 天的契约**。

**（2）片上规划**（`:68-94`）。L1 里有四块区域：`l1a/l1b`（各 `kNumL1Stages` 级）和 `l1_sfa/l1_sfb`（各 `kNumL1SFStages` 级，形状 `(BLOCK_M/N, kSFPairsPerChunk)`，`kSFPairsPerChunk = (BLOCK_K/64) × kSFKBlocks`）。SF 事件从独立基址 `kSFEventBase = kNumL1Stages` 开始编号（`:108-113`）——**A/B 与 SF 的 stage 环各自旋转**，因为一个 SF chunk 覆盖多个 K 块，两个环的转速不同。

**（3）块循环与地址计算**（`:127-155`）。`Scheduler::get_next_block` 发放 (m, n) 块，`get_global_idx` 算全局起点；`gm_a/gm_b/gm_sfa/gm_sfb` 经 `.batch().group().index()` 定位到本块起点（`:137-140`，Normal 下 batch/group 都是 0）。当 `shape_m ≤ BLOCK_M` 且 N 块数多于核数时，`reuse_a/reuse_sfa` 等标志会跳过重复搬运（`:141-151`）。

**（4）K 循环内的三段流水**（`:157-260`）：

```text
对每个 k_block_idx：
  ├─ GM→L1：copy_gm_to_l1(l1a/l1b[stage], a/b.index(0, k_idx), actual_m/n, actual_k)
  │          K 尾用 fill_tail_block 补零到 k64 = aligned(actual_k, 64)          (:164-171)
  ├─ SF chunk（仅 k_within_chunk == 0 时）：
  │     chunk_k_blocks = min(剩余 K 块数, kSFKBlocks)
  │     chunk_sf_pairs = ceil((chunk 末 − k_idx) / 64)
  │     copy_gm_to_l1_mx(l1_sfa[stage], sfa.index(0, k_block_idx × kSFPairsPerKBlock),
  │                      actual_m, chunk_sf_pairs, kL2CtrlA)                     (:173-187)
  └─ MAD 双层循环（m_mad_idx × n_mad_idx）内：
        对每个 k_mad_idx（步长 MAD_K）：
          copy_l1_to_l0a/l0b      （数据，L1→L0，MN-major 时带转置）            (:215-217)
          copy_l1_to_l0a_mx/l0b_mx（SF：用 sf_y 在 chunk 内定位本 MAD 的 pair） (:221-224)
          mad(l0c_tile, l0a_tile, l0b_tile, …)                                   (:228-230)
```

`sf_y = k_within_chunk × kSFPairsPerKBlock + k_mad_idx / MX_SF_DIVISOR`（`:209`）——chunk 内的 SF pair 列号。以 `BLOCK_K=256, MAD_K=128` 为例：`kSFPairsPerKBlock = 4`，chunk 内第 1 个 K 块（`k_within_chunk=0`）的两个 MAD 分别取 `sf_y = 0` 和 `sf_y = 2`（每个 MAD 的 128 个 K 恰好消耗 2 个 pair）。`copy_l1_to_l0a_mx`（`ascend/copy.hpp:199-213`）把它作为 `y_start` 传给 `asc_copy_l12l0a_mx`，目标地址除以 `MX_ADDR_DIV = 16` 做 MX 编码，SF 以 `fp8_e8m0_t` 视角与数据一起进 L0A/L0B。

**（5）计算与写出**。`mad()` 对 8 位操作数自动切换到 `asc_mmad_mx`（`copy.hpp:370-381`）：**缩放因子在 Cube 的 MAD 内部以 UE8M0 指数加法参与乘法，乘积在 L0C 以 FP32 累加**（`l0c_ptr<float> l0c`，`fp8_gemm.hpp:87`）。`is_first` 时清零 L0C、`is_last` 时 `ENABLE_UPDATE` 写回（`:228-230`，第 24 天）。最后一个 K 块结束时：普通路径 `epilogue.aic_store` 把 L0C 搬进 UB 并通知 AIV（`:240`）；`kDirectStore` 路径直接 `copy_l0c_to_gm` 写 GM（`:235-238`）。AIV 分支（`:267-286`）执行 `aiv_initialize + aiv_drain`，把 UB 里的 bf16/fp32 结果写回 D。

### 3.7 第 ⑥ 站：结果验证——测试如何断言这条链是对的

`python tests/test_fp8_gemm.py normal --skip-prof` 进入 `test_fp8_gemm(GemmType.Normal, skip_prof=True)`（`tests/test_fp8_gemm.py:42`），转手调用 `run_gemm_test(fp8, fp8, Normal, …, threshold=1e-5)`（`:16-22`）。`run_gemm_test`（`tests/common.py:131-237`）分两阶段：

**阶段一：并行预编译**（`:152-175`）。对每个用例 × with_bias × 四种 `trans_ab`（× alpha 变体），在 **meta 设备**上 `generate()`（不分配 NPU 存储），把 `partial(gemm_dg, …)` 收进列表后 `par_compile(kernels)`——`dry_run` 上下文里开线程池批量触发 JIT 编译（`deep_gemm/testing/par_compile.py:18-31`）。`normal` 场景的用例来自 `_enumerate_normal`（`generators.py:544-576`）：3 个 fwd M × 7 组 (N, K) + 1 个 bwd M × 7 组 × 3 种（dgrad/wgrad-acc/wgrad），共 **42 个用例**（可静态数出），默认 `recipe=(1,1,32)`。

**阶段二：正确性与性能**（`:180-236`）。对每个用例重置随机种子后真实生成数据，然后：

1. **四种绑定逐一执行**：`for trans_ab in test.get_available_trans_ab()`（Normal 返回 `['nt','tn','nn','tt']`，`generators.py:290-292`）。`gemm_dg`（`common.py:49-66`）里 `nn` 分支把 b 二元组转成 [K, N] 视图后调 `deep_gemm.fp8_fp4_gemm_nn(a, b, d, recipe_a=…, recipe_b=…)`——同一份物理数据、四种逻辑视角，**全链路（含 SF 转换、配置选择、Kernel）都被反复验证**。alpha/bias 的四种组合也在此穷举（alpha+bias 断言被拒，`:191-198`）。
2. **参考结果比较**：`d = torch.randn_like(d_ref)`（垃圾初始化，检验整块覆写）→ `gemm_dg(...)` → `torch.npu.synchronize()` → `assert_equal(d, d_ref, threshold=1e-5)`（`:199-201`）。阈值语义在 `tests/utils.py:46-58`：`threshold > 0` 时用 `calc_diff` 的余弦相似度 `1 − 2Σxy/Σ(x²+y²) < 1e-5`。参考结果 `d_ref` 由 `generate_random_quantized_with_major`（`generators.py:124-166`）用**精确解码值**（码 × SF 的 fp32 展开，`_scale_quantized_reference`，`:105-121`）构造——比的是"解码 + fp32 乘加"的忠实度，不是量化误差。
3. **ACLNN 交叉验证**（`:208-213` → `gemm_aclnn`，`:87-128`）：换用 `aclnn_fp8_fp4_gemm_{trans_ab}` 再算一遍并同样比较。四类跳过条件（`:98-122`）：K 不是 64 的倍数；recipe 的 `gran_mn != 1`（ACLNN 的 MX 路径只支持 1×32 逐行缩放）；`k ≤ 64` 且某侧 MN-major（ACLNN 对退化 scale 形状的 permute 处理有误）；fp8×fp4 只支持双 K-major 且 bf16/fp16 输出。`normal` 的 42 个用例 K ∈ {7168, 1536, 512, 16384, 2048} 均为 64 的倍数、recipe 为 (1,1,32)，因此**全部能做交叉验证**（B 组练习验证这一点）。
4. **性能行**（`--skip-prof` 时跳过）：`bench_msprof(kernel_names='gemm_impl')` 计时，`print_perf` 打印 `dur_us / TFLOPS / GB/s / mad% / mte2%`（`:225-236`、`utils.py:106-115`）。

### 3.8 通过标准解答：SF 在哪里转换、在哪里参与计算（含当天产出·参考）

把六站里 SF 的遭遇浓缩成一张生命周期表：

| 阶段 | 发生地（代码站点） | SF 的形态 |
|---|---|---|
| 输入 | 用户张量 `(A, sfA), (B, sfB)` | fp32 `[ceil(MN/gran), ceil(K/32)]` 或 int16，任意 major，可 strided |
| recipe 检查 | `get_recipe`（`layout.hpp:113-128`） | 粒度归一为 `(gran_m, gran_n, gran_k)`，`gran_k` 钉死 32 |
| **转换** | `get_gemm_desc`（`gemm.hpp:128-154`）→ `transform_sf_into_required_layout`（`layout.hpp:68-92`）→ 命中快速路径则零拷贝；否则 `launch_transform_sf`（`transform_sf.hpp:15-105`）在 **AIV** 上执行 `transform_sf` Kernel | **int16 ue8m0x2**，逻辑 `[MN, ceil(K/64)]`，物理 MN-major，已按 `gran_mn` 广播 |
| 传递 | `GemmDesc.sfa_ptr/sfb_ptr`（+ `sfa/sfb` 持有防释放）→ `jit->launch`（`fp8_gemm.hpp:73-78`） | 裸指针 + Kernel 侧 `gm_ptr<int16_t, MN>` 包装（`fp8_gemm.hpp:65-66`） |
| 搬运 | `copy_gm_to_l1_mx`（`fp8_gemm.hpp:173-187`） | 每 `kSFKBlocks` 个 K 块一个 chunk 进 L1 NZ 分形 |
| 装载 | `copy_l1_to_l0a/b_mx`（`:221-224`，`copy.hpp:199-229`） | UE8M0 与数据一起进 L0A/L0B |
| **参与计算** | `mad()` → `asc_mmad_mx`（`copy.hpp:370-381`） | **Cube 的 MAD 内部按 64 元素组应用缩放（指数加法），部分和 FP32 累加在 L0C** |
| 消失 | Epilogue（`aic_store/aiv_drain`）之前 | L0C 里已是反量化后的 FP32 结果，后续与 BF16 路径无异 |

一句话回答：**SF 在 Host 侧 `get_gemm_desc` 内部被转换（第 33 天的 AIV Kernel），在设备侧 Cube 的 `asc_mmad_mx` 里参与计算；两者之间它以 int16 MN-major 契约布局流经 GM → L1（chunk 搬运）→ L0（`sf_y` 定位）。**

**当天产出（参考版）**即 3.1 的全链路图 + 上表；自己合上教程重画一版，重点自查：nn 的二元组转置、转换先于配置选择、`GemmDesc` 持有 SF、27 个模板实参里哪两个是 SF 专属、`sf_y` 的计算、`asc_mmad_mx` 的位置。

## 四、源码导读

今天的主线文件与阅读站点（按六站顺序）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `deep_gemm/__init__.py` / `csrc/python_api.cpp` | :47-48 / :35 | `fp8_fp4_gemm_nn` 的导出与注册起点 |
| 2 | `csrc/apis/gemm.hpp` | `dispatch_fp8_fp4_gemm`（:22-33）、`fp8_fp4_gemm`（:225-251）、`get_gemm_desc`（:60-197）、`register_apis`（:562-569） | 六站的前三站在同一个文件里 |
| 3 | `csrc/utils/layout.hpp` | `get_transposed`（:88-99）、`get_major_type_ab`（:44-47）、`get_shape_by_spec`（:140-219）、`gemm_early_return`（:54-86） | 输入归一化与 shape 推断的机制 |
| 4 | `csrc/apis/layout.hpp` | `get_recipe`（:113-128）、`transform_sf_into_required_layout`（:68-92）、`check_sf_layout`（:37-65） | recipe 检查与转换入口（细节见第 33 天） |
| 5 | `csrc/jit_kernels/transform_sf.hpp` | `launch_transform_sf`（:15-105） | 转换的 JIT 启动（输出构造/tiling/L2 hint） |
| 6 | `csrc/jit_kernels/fp8_gemm.hpp` | `launch_fp8_gemm`（:11-79）、`launch_fp4_gemm`（:83-151） | 27 个模板实参的生成、launch 参数表 |
| 7 | `csrc/jit_kernels/gemm_config.hpp` | `GemmDesc`（:39-95）、`GemmConfig`（:98-121）、`get_compiled_dim`（:17-21）、`get_outer_stride`（:27-34） | 契约"成文处"与两份数据结构的分工 |
| 8 | `csrc/jit_kernels/heuristics.hpp` | `select_gemm_config`（:592-755）、`make_gemm_candidate`（:92-204）、`get_gemm_dtype`（:80-90） | 3.5 节表格的逐条出处；:749-753 的打印开关 |
| 9 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` | `fp8_gemm_impl`（:36-287） | 3.6 节逐段导读见下 |
| 10 | `deep_gemm/include/deep_gemm/ascend/copy.hpp` | `copy_gm_to_l1_mx`（:161-197）、`copy_l1_to_l0a/b_mx`（:199-229）、`mad`（:370-381） | SF 搬运三件套与 `asc_mmad_mx` 切换 |
| 11 | `deep_gemm/include/deep_gemm/common.hpp` | `MX_SF_DIVISOR`（:65）、`gm_ptr`（:137-186）、`GemmType`（:87-92） | 常量与地址公式 |
| 12 | `deep_gemm/include/deep_gemm/scheduler.hpp` | 顶部注释（:11-46）、`get_global_idx`（:113）、`get_k_idx_base/get_sf_idx_base`（:124-125） | 四种 GemmType 下 A/B/D/SF 的偏移来源 |
| 13 | `tests/test_fp8_gemm.py` | `test_fp8_gemm`（:8-22）、CLI（:25-55） | 验证入口与场景选择 |
| 14 | `tests/common.py` | `gemm_dg`（:49-84）、`gemm_aclnn`（:87-128）、`run_gemm_test`（:131-237） | 3.7 节两阶段流程的出处 |
| 15 | `tests/utils.py` | `calc_diff`（:36-43）、`assert_equal`（:46-58）、`print_perf`（:106-115） | 阈值语义与性能行格式 |
| 16 | `tests/generators.py` | `GemmTestDesc`（:249-298）、`generate_random_quantized_with_major`（:124-166）、`_enumerate_normal`（:544-576）、`_enumerate_recipe`（:441-462） | 用例空间与精确解码参考 |
| 17 | `deep_gemm/testing/par_compile.py` | `dry_run`、`par_compile`（:1-31） | meta 并行编译的机制 |

**`fp8_gemm_impl` 逐段导读**（行号见 `deep_gemm/include/deep_gemm/fp8_gemm.hpp`）：

- **`:9-17` 头注释**：一句话契约——"SF is always MN-major. D is always N-major"，以及 K/MN major 只影响 nd2nz 方向和 L1→L0 是否转置。
- **`:18-45` 模板与签名**：27 个模板参数（+2 个默认）；`__global__ __mix__(1, kDirectStore ? 0 : 2)` 决定 AIV 数量；运行时参数里 `gmem_sfa/gmem_sfb` 是裸指针、`grouped_layout` 仅 grouped 类型非空。
- **`:60-72` 派生量**：`shape_sf_k`、`gm_sfa/gm_sfb`（MN-major、稠密 batch stride）、`kSFPairsPerKBlock/kSFPairsPerChunk`、`kFracK`（fp8=32）、`kNumCDStages`。
- **`:74-94` Epilogue 对象与片上布局**：`Epilogue<..., kNumEpilogueStages, 2, ...>`（2 个 AIV）；`l1a → l1b → l1_sfa → l1_sfb` 按 offset 串联，末尾四个 `static_assert` 防 L1/L0A/L0B/L0C 溢出。
- **`:96-102` Scheduler 构造**：`kAlignment = 256`（`Runtime::kMkAlignment`，`csrc/jit/jit.hpp:59-62`）。
- **`:104-122` AIC 开场**：`asc_set_mmad_direction_n()`；三组 `set_flags` 预置事件（L1 数据、L1 SF 各一个环）。
- **`:124-155` 块循环头**：reuse 判定（`can_reuse_a/b`、`can_reuse_sf`）与 stage 索引回卷；`actual_m/actual_n` 截断边界。
- **`:157-171` GM→L1**：`copy_gm_to_l1` + `fill_tail_block`（K 尾补零到 `k64`，第 32 天 3.5 节）。
- **`:173-187` SF chunk**：`k_within_chunk == 0` 时一次搬运 `kSFKBlocks` 个 K 块的 SF；`chunk_sf_pairs` 按实际尾长截断。
- **`:193-232` MAD 双层循环**：`:209` 的 `sf_y`；`:219-220` 在第一个 MAD 处等待 SF 到达（"AB can enter L0 while the first SF request is still in flight"）；`:228-230` `mad` 与 `ENABLE_UPDATE/KEEP`。
- **`:234-243` 写出**：最后一个 K 块时 direct store 或 `epilogue.aic_store`。
- **`:247-265` stage 推进**：L1 数据环与 SF 环分别推进（`:253-258`）；`last_a/b/sfa/sfb` 记录供下个块判 reuse。
- **`:267-286` AIV 分支**：direct store 直接 return；k-grouped 先清空组；`aiv_drain` 按 `s_epi` 轮转消费 UB stage 写 GM。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通正确性测试**：

   ```bash
   python tests/test_fp8_gemm.py normal --skip-prof
   ```

   预期输出 `Testing fp8 x fp8 gemm: Normal`、`Num tests:  42`，随后每行 `> Perf (...): OK`。挑一个 fwd 用例（如 `m=4096, n=7168, k=7168`），对照 3.1 节说出这次调用经过的每个函数。

2. **观察 JIT 配置**：

   ```bash
   export DG_PRINT_CONFIGS=1
   python tests/test_fp8_gemm.py normal --skip-prof 2>&1 | grep -m 8 "GemmDesc"
   ```

   找到 `k=7168` 的配置行，记录 `block/mad/l1_stages/l1_sf_stages/sf_k_blocks`，对照 3.5 节手算：`sf_k_blocks = min(4, ceil(7168/BLOCK_K))`、L1 的 AB+SF 字节占用。

3. **Python 实验：手工构造 `nn` 调用并验证**（复刻测试语义的最小版本）：

   ```python
   import torch, torch_npu
   import deep_gemm

   M, N, K, gran_k = 256, 512, 1024, 32

   # nt 形式生成：A [M, K]、B [N, K]，SF 用合法的 2^(e-127)
   a = torch.randn(M, K).npu().to(torch.float8_e4m3fn)
   b = torch.randn(N, K).npu().to(torch.float8_e4m3fn)
   sfa = (torch.randint(118, 124, (M, K // 32), dtype=torch.int32) * (1 << 23)).view(torch.float32).npu()
   sfb = (torch.randint(118, 124, (N, K // 32), dtype=torch.int32) * (1 << 23)).view(torch.float32).npu()

   # 参考：精确解码 + fp32 乘加
   sfa_full = sfa.repeat_interleave(gran_k, dim=1)
   sfb_full = sfb.repeat_interleave(gran_k, dim=1)
   ref = ((a.float() * sfa_full) @ (b.float() * sfb_full).T).to(torch.bfloat16)

   def calc_diff(x, y):
       x, y = x.float(), y.float()
       return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()

   # nn 调用：B、sfB 一起转成 [K, N] 视角
   d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
   deep_gemm.fp8_fp4_gemm_nn((a, sfa), (b.t(), sfb.t()), d, recipe=(1, 1, gran_k))
   torch.npu.synchronize()
   print('nn  diff =', calc_diff(d, ref))     # < 1e-5

   # 对照 nt：同一物理数据、不同 major 视角
   d_nt = torch.empty_like(d)
   deep_gemm.fp8_fp4_gemm_nt((a, sfa), (b, sfb), d_nt, recipe=(1, 1, gran_k))
   torch.npu.synchronize()
   print('nt  diff =', calc_diff(d_nt, ref))
   print('nn == nt?', torch.equal(d, d_nt))   # 通常一致；若不同，用 major_b 差异解释
   ```

4. **观察 SF 转换的发生**：直接调用 `deep_gemm.transform_sf_into_required_layout(sfa, M, K, (1, 1, gran_k), is_sfa=True)`（`layout.hpp:130-137` 的 binding），打印 `dtype/shape/stride/data_ptr`；再把输入换成物理 `[K对, MN]` 的 int16（`torch.randint(..., (K // 64, M), dtype=torch.int16).npu().t()`），用 `data_ptr` 相等验证零拷贝快速路径——这正是 3.4 节"转换先于配置"的可观察证据。
5. **（选做）去掉 `--skip-prof`** 重跑第 1 步，把 `m=4096` 行的 TFLOPS 与 `2*m*n*k` 手算值对照，记录 mad/mte2 占用，为第 51 天建立基线。

### B 组：无 Ascend 环境（静态任务）

1. **形状推演**：`M=4097, N=7168, K=7296, recipe=(32, 1, 32)`，fp32 SF 输入。写出：sfA/sfB 的输入形状（`[129, 228]` / `[7168, 228]`）、转换后 int16 的逻辑形状与字节数（`[4097, 114]` / `[7168, 114]`）、`shape_sf_k = 114`；再取 `BLOCK_K=256`，算 `num_k_blocks = 29`、`kSFPairsPerKBlock = 4`、`sf_k_blocks = min(4, 29) = 4`，以及最后一个 chunk（`k_block_idx=28`）的 `chunk_k_blocks`、`chunk_k_end`、`chunk_sf_pairs`（应为 1、7296、2）。
2. **sf_y 手算**：`BLOCK_K=256, MAD_K=128`。列出 `k_block_idx=0..7`（两个 chunk）中每个 `(k_within_chunk, k_mad_idx)` 的 `sf_y`，验证"每个 MAD 恰好消耗 2 个 SF pair"。
3. **重画调用链**：合上教程，独立画出 3.1 的全链路图（当天产出），标注：nn 的二元组转置、`get_recipe` 与 SF 转换在 `get_gemm_desc` 内的先后、`dispatch_fp8_fp4_gemm` 的三分支、27 个模板实参中 SF 专属的两个、Kernel 内 SF 的三段搬运、`asc_mmad_mx` 与 L0C 的位置。画不出的环节回到第四节对应站点重读。
4. **代码标记**：打开 `csrc/apis/gemm.hpp` 与 `deep_gemm/include/deep_gemm/fp8_gemm.hpp`，用六个标签把六站对应的实际代码行段标出来，检查与 3.1 图一一对应。
5. **测试覆盖分析**：读 `tests/common.py:64-66` 与 `generators.py:290-292`，回答：一个 `b_major=K` 的用例在 `nn` 绑定下，Kernel 看到的 `major_b` 是什么？为什么同一物理数据能覆盖两种 major？
6. **ACLNN 跳过条件清单**：对 `_enumerate_normal` 的 42 个用例，逐条核对 `common.py:98-122` 的四个条件（K%64、gran_mn、k≤64 且 MN-major、fp4 组合），验证"全部可交叉验证"的结论；再看 `unaligned-k` 场景（`generators.py:519-541`，如 K=17、K=234）里哪些用例会被跳过并说明原因。

## 六、当天产出

**`fp8_fp4_gemm_nn` 全链路流程图 + SF 生命周期表**：以 3.1 与 3.8 的参考版为答案，合上教程自己画一版。流程图至少覆盖六站的函数与文件位置；生命周期表至少回答：SF 以什么形态进入（用户张量）、在哪里变成什么形态（转换站点与输出契约）、如何流经 L1/L0（chunk 与 `sf_y`）、在哪里参与计算（`asc_mmad_mx` + L0C FP32）。有环境的同学附上 A 组第 1～2 步的运行记录（含 `DG_PRINT_CONFIGS` 的配置行）。

## 七、自测题

1. `fp8_fp4_gemm_nn` 的两个模板布尔参数是什么？sfB 的用户视角形状如何随绑定变化？
   > `kTransA=false, kTransB=true`（`gemm.hpp:255`）。`get_transposed` 把 (B, sfB) 一起转置（`gemm.hpp:239-240`），所以 sfB 的两个轴始终跟随 B 的轴：`nn` 下用户传 B: [K, N]、sfB: [ceil(K/gran_k), ceil(N/gran_n)]，内部归一化回 [N, K] 与 [ceil(N/gran_n), ceil(K/gran_k)]（文档 `gemm.hpp:212-213` 描述的是归一化框架）。
2. 从 Python 调用到 Kernel 启动，依次经过哪些关键函数？
   > binding 注册（`gemm.hpp:562`）→ `fp8_fp4_gemm<false,true>`（`:225`）→ `get_gemm_desc`（`:60`，内含 `get_recipe` 与两次 `transform_sf_into_required_layout`）→ `dispatch_fp8_fp4_gemm`（`:22`）→ `launch_fp8_gemm`（`csrc/jit_kernels/fp8_gemm.hpp:11`，内含 `select_gemm_config`、`std::format` 代码生成、`jit->compile`、`jit->launch`）→ 设备侧 `fp8_gemm_impl`（`fp8_gemm.hpp:36`）。
3. recipe 检查在哪里、规则是什么？`gran_k` 为什么必须是 32？
   > `layout::get_recipe`（`layout.hpp:113-128`）：三元组与二元组互斥、两侧 `gran_k` 相等、缺省 `(1, 32)`；`check_sf_layout:55` 钉死 `gran_k == 32`——因为一个 int16 SF pair 的两个字节各覆盖 32 个 K（`MX_SF_DIVISOR = 64`，`common.hpp:65`），这是 MX 标准粒度。
4. Scale 布局转换发生在链路的哪一步？为什么必须在那里？
   > 在 `get_gemm_desc` 内部（`gemm.hpp:128-154`）、`select_gemm_config` 与 Kernel launch 之前。因为：(a) Kernel 的契约（int16 MN-major）必须先确定，`GemmDesc.sfa_ptr/sfb_ptr` 才有稳定的指针可传；(b) 转换输出是临时张量，需要 `GemmDesc.sfa/sfb` 持有，防止异步 launch 前被释放（`gemm_config.hpp:77-78`）；(c) 命中快速路径时零拷贝，避免无谓 Kernel。
5. JIT 配置里哪些是低精度（has_sf）专属？`sf_k_blocks` 和 `num_l1_sf_stages` 分别控制什么？
   > `BLOCK_K` 必须是 64 的倍数（`heuristics.hpp:108-109`）；`sf_k_blocks = min(4, ceil(k/BLOCK_K))` 再按 L1 余量收紧（`:156-164, :177-183`），是**一个 SF chunk 覆盖的 K 块数**（决定 `kSFPairsPerChunk`）；`num_l1_sf_stages = 2` 是 L1 中 SF 缓冲的级数；A/B 与 SF 共享 512KB L1 预算且总级数上限 4（`:166-174`）。
6. Kernel 内 SF 如何被消费？`sf_y` 怎么算？
   > `copy_gm_to_l1_mx` 在 `k_within_chunk == 0` 时一次搬 `kSFKBlocks` 个 K 块的 SF 进 `l1_sfa/l1_sfb`（`fp8_gemm.hpp:173-187`）；MAD 循环里 `sf_y = k_within_chunk × kSFPairsPerKBlock + k_mad_idx / 64`（`:209`）定位 chunk 内的 pair 列，`copy_l1_to_l0a/b_mx` 以它为 `y_start` 把 UE8M0 装入 L0（`:221-224`、`copy.hpp:199-229`）；`mad()` 对 8 位操作数调用 `asc_mmad_mx`，缩放以指数加法在 Cube 内部参与计算，FP32 部分和累加在 L0C（`copy.hpp:370-381`、`fp8_gemm.hpp:87`）。
7. 测试如何验证这条链？`threshold=1e-5` 为什么可行？ACLNN 交叉验证何时跳过？
   > 两阶段（`common.py:131-237`）：meta `par_compile` 预编译，然后同一数据在 nt/tn/nn/tt 四种绑定 + alpha/bias 组合下执行，`assert_equal` 用余弦相似度 `calc_diff < 1e-5` 比较（`utils.py:36-58`）。可行是因为参考结果由精确解码值构造（`generators.py:105-166`），比较对象是"解码 + fp32 乘加"的忠实度而非量化误差。ACLNN 跳过：K%64≠0、`gran_mn≠1`、`k≤64` 且 MN-major、fp8×fp4 的不支持的布局组合（`common.py:98-122`）。
8. （通过标准）完整回答：Scaling Factor 在哪里转换、在哪里参与计算？
   > 转换：Host 侧 `get_gemm_desc` 内的 `transform_sf_into_required_layout`（`gemm.hpp:128-154`）——快速路径零拷贝，否则 `launch_transform_sf` 启动 AIV Kernel `transform_sf` 产出 int16 ue8m0x2、`[MN, ceil(K/64)]`、MN-major 契约布局，由 `GemmDesc.sfa/sfb` 持有并经 `jit->launch` 传指针。参与计算：设备侧 Kernel 中 `copy_gm_to_l1_mx`（chunk 搬运）→ `copy_l1_to_l0a/b_mx`（`sf_y` 定位装载）→ `mad()` 调用 `asc_mmad_mx`，**缩放在 Cube 的 MAD 内部按 64 元素组以指数加法应用**，部分和 FP32 累加在 L0C——之后进入 Epilogue 时 SF 已经"消失"（L0C 中是反量化的 FP32 结果）。参见 3.8 的生命周期表。

---

明天（第 35 天）是机动日：补充低精度数值基础，或复盘 BF16 Kernel 中仍不清楚的部分；第 6 周将进入 Grouped GEMM 与 MQA Logits。


