# 第 7 周 · 第 4 天（总第 46 天）：学习 MegaMoE 测试与性能入口

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 25 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前三天我们读完了 MegaMoE 的接口（第 43 天）、计算主线（第 44 天）和通信融合（第 45 天），但一直把两件事当成"已给定"：一是 `tests/test_mega_moe.py` 这个多 rank 测试到底怎么组织、怎么判对错；二是 README 性能表里 `228.5 TFLOPS / 1850.0 GB/s / 37.5 GB/s` 这些数字是从哪来的、口径是什么。今天补上这两个入口，只读两个"测试侧"文件——`tests/test_mega_moe.py`（322 行，多 rank 测试 + 正确性 + benchmark）和 `scripts/profiling/msprof_mega_moe.py`（199 行，PIPE 级 trace 采集导出），再带上看进 `deep_gemm/testing/bench.py` 里被它们复用的 profiling 设施。今天的四个核心问题：**多 rank 测试如何组织与判对错；benchmark 的计算/内存/通信三个带宽口径公式是什么、各自排除了什么；profiling 基础设施如何保证冷 L2 与跨 rank 对齐；PIPE 级 trace 能回答什么、不能回答什么**。当天产出是**MegaMoE 测试依赖和性能指标清单**（参考版本见 3.7 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 讲清 `tests/test_mega_moe.py` 的多 rank 组织方式（`test_mega_moe.py:285-322`）：`torch.multiprocessing.spawn` 按 `--num-processes`（默认 8）拉起每 rank 一个进程，`init_dist`（`deep_gemm/testing/envs.py:28-52`）用 MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK 环境变量以 hccl 后端建组，命令行列表经笛卡尔积展开成用例集（`:316-319`）；
2. 解释"最大容量注册、逐用例复用"的缓冲区策略（`:249-258`）：先用 `_C.get_symm_buffer_size_for_mega_moe` 选出需求最大的配置注册一次 `base`，后续每个 case 以 `base=base` 借用同一段 HCCL 注册内存切自己的视图——并联系第 45 天"注册无法注销、只能池化复用"的结论；
3. 描述 `MegaMoECase`（`:50-163`）的生命周期与**依赖分叉**：fused 路径只需要 `deep_gemm` 本身；baseline 路径额外依赖 `deep_ep`、`tilelang` 和 `third-party/tilelang_ops`（`:77-100`），这就是 `--skip-baseline` 与 profiling 脚本 `with_baseline=False`（`msprof_mega_moe.py:35`）存在的原因；
4. 拆解 `check_correctness`（`:166-200`）的四层检查：与 baseline 的整体余弦差（阈值 `5e-6`，`:19`）、逐 token 相对差（`:184-187`）、跨 rank 独立重算的 expert 接收计数（`:189-195`）、以及同一输入重复运行的**位级可复现**与累计统计精确累加（`:197-200`），并解释阈值不为零的原因（`:18` 注释：DeepEP 把各 rank 部分和舍入到 BF16，MegaMoE 全程 FP32）；
5. 默写 `benchmark`（`:203-243`）的三个口径公式——`flops = 6·H·I·(recv + T·S)`（`:221`）、逻辑 HBM 字节（`:222-228`）、远端通信字节（`:229-231`）——说明各自的排除项，解释"时延取最慢 rank（amax）、吞吐用平均工作量"（`:239` 注释）的聚合语义，并手工复现 README 数据表（`README.md:164-173`）中的任意一行；
6. 说明 `bench_msprof`（`deep_gemm/testing/bench.py:284-400`）的测量纪律：warmup、冷 L2（`_make_l2_flush_buffer` 分配 2×L2 大小的缓冲并每次迭代清零，`:263-268, :344-346`）、`barrier_comm_profiling` 的 `npu_sleep(20ms) + all_reduce` 对齐（`:335-350`）、三种 backend（`persistent`/`fast`/`slow`）的区别，以及 `KernelProfile`（`:27-71`）中各 pipe 利用率字段对应的硬件管道；
7. 讲清 `scripts/profiling/msprof_mega_moe.py` 的两阶段设计：`main`（`:152-195`）用 `msprof --instr-profiling=on` 包住 `--launch-only` 子进程完成采集，`export`（`:60-149`）把 msprof 的 JSON 时间线与 SQLite `biu_perf.db` 逐事件对齐、按 tick 频率换算成微秒、只导出**最后一次冷 L2 launch** 的按 (core, subcore, pipe) 分轨的 Chrome/Perfetto trace，并说明其边界——"BIU timings explain pipeline overlap, not unprofiled end-to-end performance"（`:8`）；
8. 列出 MegaMoE 测试与 profiling 的完整依赖清单（运行时/基线/性能/trace 四层），并在没有多卡环境时选择正确的静态替代任务。

## 二、预备知识

- **第 29～31 天**：测试数据生成与正确性验证——`calc_diff`（`tests/utils.py:36-43`，基于余弦相似度的差异度量，DeepGEMM 惯例）、参考结果交叉验证的思路，今天在 MegaMoE 上以"DeepEP 基线"的形式重现。
- **第 33 天**：SF 布局——测试里 `make_npu_fp8`/`make_npu_fp4` 生成的 scale factor 是每 32 个元素一个 uint8 指数、再 view 成 int16 的打包布局；FP4 数据两个元素共用一个字节。
- **第 43 天**：MegaMoE 对外接口——`SymmBuffer`（`deep_gemm/mega/__init__.py:9-81`）的 13 个视图、`transform_weights_for_mega_moe` 的权重变换（gate/up 交错、NZ 序、FP4 指数偏置 +6）、`fp8_fp4_mega_moe` 的参数表。
- **第 44 天**：计算主线——`mega_moe_impl` 是融合 Kernel 的名字（benchmark 按 `kernel_names='mega_moe_impl'` 子串匹配它）；`expert_recv_count` 视图由 Kernel 每次 launch 写入本 rank 各 expert 的接收计数。
- **第 45 天**：通信与融合——六类通信对象、拉取式 Dispatch / 推回式 Combine、小 T 时延受限与大 T 带宽/计算受限的初步判断；今天 3.4 的三个口径公式正是那节手算的"官方版本"，3.6 的 PIPE trace 则是重叠关系的可视化证据。
- **分布式基础**：rank、world size、进程组、`torch.multiprocessing.spawn`；知道 HCCL 是昇腾的集合通信库即可。
- **性能基础**：TFLOPS = 浮点次数 / 时间；带宽 = 字节数 / 时间；"冷缓存"指测量前清空 L2，避免上一次迭代的残留数据抬高带宽。第 50 天会系统展开，今天只用到这两条。

硬件常量回顾（第 5/45 天）：AIC 上有 Scalar/CUBE(MAD)/MTE1(L1→L0)/MTE2(GM→L1)/FIX(L0C→出) 管道，AIV 上有 Vector/Scalar/MTE2(GM→UB)/MTE3(UB→GM) 管道——`KernelProfile` 的字段名与它们一一对应。

## 三、详细讲解

### 3.1 多 rank 测试的组织：spawn、init_dist 与用例展开

MegaMoE 是 EP 算子，测试天然是多进程的。入口在文件末尾（`test_mega_moe.py:285-322`）：

```text
python tests/test_mega_moe.py [选项]
  └─ torch.multiprocessing.spawn(test, nprocs=--num-processes)   # :322，默认 8
       └─ 每个进程执行 test(local_rank, args)                      # :246
            ├─ init_dist(local_rank, num_processes)                # envs.py:28-52
            ├─ 选最大容量配置，注册一次 base SymmBuffer             # :255-258
            └─ for 每个 config:                                     # :260-280
                 ├─ MegaMoECase(...)  → create_inputs(num_tokens)
                 ├─ check_correctness(...)        # 除非 --skip-correctness
                 ├─ dist_print(benchmark(case))   # 除非 --skip-prof
                 └─ case.destroy() + torch.npu.empty_cache()
```

**（1）`init_dist` 怎么把 8 个进程连成一组**（`envs.py:28-52`）：读四个环境变量——`MASTER_ADDR`（默认 `127.0.0.1`）、`MASTER_PORT`（默认 `8361`）、`WORLD_SIZE`（节点数，默认 1）、`RANK`（节点序号，默认 0），然后 `dist.init_process_group(backend='hccl', init_method='tcp://ip:port', world_size=节点数×本地进程数, rank=节点序号×本地进程数+local_rank)`。也就是说单机 8 进程就是 world 8；多机只需在启动外层设好这四个变量。初始化后它还做了三件影响全局的事：`torch.set_default_dtype(torch.bfloat16)`、`torch.set_default_device('npu')`（所以后面所有 `torch.empty` 默认落在 NPU 上且是 BF16）、`torch.npu.set_device(local_rank)`，最后按 seed 初始化随机数并返回默认进程组。

**（2）命令行如何展开成用例集**（`:291-319`）：`--hidden`、`--intermediate-hidden`、`--num-experts`、`--num-topk`、`--num-tokens`、`--num-shared-experts`、`--num-max-tokens-per-rank`、`--activation-clamp` 都是列表参数。H 和 I 的规则是"单元素广播或等长配对"（`:312-313` 断言）：`--hidden 5120 7168 --intermediate-hidden 2304 3072` 配成 `(5120,2304)`、`(7168,3072)` 两个 shape；`--hidden 5120` 配任意长的 I 列表则广播。其余列表直接做笛卡尔积（`:316-319`）。默认值下（H={5120,7168}、I={2304,3072}、E=384、k=6、T={64,256,4096,16384}、S=1）共 **2×4 = 8 个用例**，启动时打印 `Initializing 8 ranks; 8 cases`。两个细节：`--num-tokens 0` 专门测空输入（`:296` help 文本 "0 tests an empty input"）；`--num-max-tokens-per-rank` 缺省为"实际 token 数、至少 1"（`:298-299`），即 SymmBuffer 容量按需分配而非 EP 最坏情形。

**（3）最大容量注册与复用**（`:249-258`）：`buffer_size(config)` 调 `_C.get_symm_buffer_size_for_mega_moe` 算出每个配置需要的对称缓冲区字节数；测试用 `max(args.configs, key=buffer_size)` 选出**需求最大**的配置，先注册一次 `base = deep_gemm.SymmBuffer(...)`，之后每个 case 都以 `base=base` 构造。为什么这样做？回忆第 45 天：`HcclCommMemReg` 没有注销 API，注册段进程终身持有、只能池化复用（`csrc/backends/hccl.hpp:166-171`）；与其每个 case 注册一段，不如一次注册最大的、之后借视图。Python 侧的对应检查在 `mega/__init__.py:51-52`：复用时断言 `num_bytes <= base.buffer.nbytes`，且"共享 scratch 必须按顺序使用、每种布局重新填充输入"（`:53` 注释）。

**（4）用例循环与收尾**（`:260-281`）：每个用例先打印表头（`[i/N] H/I=… | E=… | topk=… | shared=… | tokens=…`，仅 local rank 0，`:264-265`），然后构造 `MegaMoECase`，其中 `with_baseline=not args.skip_baseline and num_ranks > 1`（`:267`）——**单 rank 时 baseline 自动关闭**（没有 EP 通信可言），且 `skip_correctness |= … or num_processes == 1`（`:309`）。跑完 `finally` 里 `case.destroy()` + `torch.npu.empty_cache()`，后者有一条值得注意的注释（`:279-280`）："DeepEP allocates via ACL, which cannot reclaim PyTorch's cached memory"——DeepEP 走 ACL 分配绕过了 PyTorch 的缓存分配器，不清缓存的话，下一个用例的 ACL 大分配可能被 PyTorch 缓存的空闲块挤住。全部用例结束后 `base.destroy()`（只弃引用，第 43 天）并销毁进程组。

**（5）输出纪律**：`benchmark` 的结果经 `dist_print(..., once_in_node=True)`（`:273`）打印——`dist_print`（`envs.py:21-25`）只在每个节点的 local rank 0 打印一行，随后 `dist.barrier()` 保证各 rank 输出不交错。

### 3.2 MegaMoECase：一个用例的生命周期与依赖分叉

`MegaMoECase`（`:50-163`）的类注释一句话概括了它的角色："Own one model's inputs and the fused/unfused runners used by tests and profiling."——它同时被正确性测试、benchmark 和 msprof 采集脚本（`msprof_mega_moe.py:28` 导入）复用。

**构造（`:53-104`）分四步：**

1. **SymmBuffer**（`:58-60`）：按 `(group, E, capacity, k, H, I, S)` 构造，拿到 13 个视图（第 43 天）。
2. **激活 clamp 的 BF16 化**（`:61-62`）：`activation_clamp` 先转成 BF16 再取回 Python float，注释 "Match the BF16 clamp consumed by the fused kernel"——融合 Kernel 里 SwiGLU 的 clamp 在 BF16 域生效，参考实现必须用同一个舍入后的值，否则正确性比较会混入精度差。
3. **路由权重（FP4）**（`:63-66`）：每个 rank 持有 `E/R` 个 expert 的 `l1 (2I, H)`、`l2 (H, I)` 权重（`make_npu_fp4`，`:30-36`：int8 数据每字节两个 FP4、SF 每 32 元素一个 uint8 指数再 view int16），随即 `deep_gemm.transform_weights_for_mega_moe` 变换成 Kernel 消费的布局（第 43 天：gate/up 交错进同一 BLOCK_N、NZ 序、FP4 SF 指数 +6 补偿内部反量化的 ÷64）。
4. **共享专家权重（FP8）**（`:69-74`）：`S>0` 时额外构造 `(2·S·I, H)` 与 `(H, S·I)` 的 FP8 权重对，同样变换。

**然后是本类最关键的分叉**（`:77-104`）：

```python
self.ep_buffer = None
if with_baseline:
    import deep_ep          # :78
    import tilelang         # :79
    ...                     # 加载 third-party/tilelang_ops 的 swiglu_apply_weight_to_fp8
else:
    # Fused-only profiling needs only the transformed weight storage.  # :102
    self.l1_weights = self.l2_weights = None                          # :103-104
```

baseline 分支做了四件事：把 `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` 按文件路径动态加载成模块（`:84-89`）；取 `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()` 作为 expert 对齐量，注释 "Dispatch, grouped GEMMs and SwiGLU must agree on expert padding"（`:91-92`）；用 `functools.partial` 固化 swiglu 调用（`:93-94`）；给 DeepEP 单独建一个 hccl 通信组并构造 `deep_ep.EPBuffer(..., use_fp8_dispatch=True, allow_multiple_reduction=True, explicitly_destroy=True)`（`:95-100`，注释 "Give the DeepEP buffer its own communicator"）。**这条分叉就是依赖清单的来源**：fused 路径只要 `deep_gemm` 自身；要跑 baseline（正确性交叉验证 + speedup）就必须装 deep_ep、tilelang 并带上 third-party/tilelang_ops。profiling 脚本构造 case 时传 `with_baseline=False`（`msprof_mega_moe.py:35`），所以采集 PIPE trace **不需要**任何第三方依赖，还会把原始权重置 None 省内存。

**`create_inputs(num_tokens, seed_offset, invalid_routes)`（`:106-120`）**：生成一轮输入并填进 SymmBuffer 视图。三组数据：

- `x`：`make_npu_fp8((T, H), seed)`（`:22-27`）——`randn·32` 转 FP8 e4m3，SF 指数字节在 123–131 间随机（即缩放约 2⁻⁴～2⁴），形状 `(T, H//32)` 再 view int16（第 33 天的打包布局）；
- `topk_idx / topk_weights`：`make_topk`（`:39-47`）——`(T, E)` 的 fp32 随机分数取 top-k（不排序），对选中的 k 个分数做 softmax 得权重（每 token 权重和为 1）；
- 拷贝进 `buffer.x / x_sf / topk_idx / topk_weights` 的前 T 行（`:116-119`），再分配输出 `y = (T, H)` BF16（`:120`）。

`invalid_routes=True`（`:112-115`）会人为制造两种脏路由：每第 11 个 token 把最后一个路由改成与第一个相同（重复路由），每第 13 个 token 置 `-1` 且权重清零（无效路由）。注意：**当前测试主流程并未启用它**（`check_correctness` 和 `test` 都用默认 False），它是留给手工实验的开关——你可以自己改脚本验证 Kernel 对脏路由的容错。

**两个 runner**：`run_fused`（`:122-128`）一行调 `deep_gemm.fp8_fp4_mega_moe(y, 变换后权重, buffer, shared…, cumulative stats, activation_clamp)`；`run_baseline`（`:130-156`）是第 44/45 天见过的七段未融合参考——DeepEP `dispatch` → `m_grouped_fp8_fp4_gemm_nt_contiguous`（Linear1）→ tilelang `swiglu` → 第二个 M-Grouped GEMM（Linear2）→（可选共享专家：两次 `fp8_gemm_nt` + swiglu）→ DeepEP `combine(bias=shared_y)`，行首注释 "without communication overlap"（`:132`），`:155` 注释说明共享专家输出在 DeepEP 的 FP32 combine 累加中加入。

### 3.3 正确性验证 `check_correctness`：四层检查

`check_correctness(case, num_runs)`（`:166-200`）的设计比"跑一次比一次"严谨得多，可以拆成四层：

**第 0 步：埋一个"必须累加"的哨兵**（`:169`）：`stats = torch.arange(E/R) + 17`——把累计统计的初值设成 17、18、19…（而不是 0）。`run_fused(stats)` 把它作为 `cumulative_local_expert_recv_stats` 传入，Kernel 每次运行把本 rank 各 expert 的接收计数**累加**进去。初值非零意味着：如果 Kernel 错误地"覆盖"而非"累加"，断言立刻失败。

**第 1 层：与 baseline 的整体数值差**（`:179-183`）：每个新 seed 第一次出现时跑 `run_baseline`，用 `calc_diff`（`tests/utils.py:36-43`，余弦相似度差异）比较，阈值 `REFERENCE_DIFF_THRESHOLD = 5e-6`（`:19`）。为什么不是 0？文件第 18 行的注释给了答案：

> DeepEP rounds rank partials to BF16; MegaMoE keeps the full sum in FP32

baseline 的 combine 把每个 rank 的部分和舍入到 BF16 再做 FP32 累加，融合 Kernel 全程 FP32——两者本来就是"近似等价"而非"位等价"。

**第 2 层：逐 token 相对差**（`:184-187`）：`token_diff = Σ(actual−ref)² / Σ(actual²+ref²)`（分母 clamp 防零），断言 max < 5e-6。整体余弦差可能把"少数 token 错得离谱"平均掉，这层专抓这种情况。

**第 3 层：跨 rank 独立重算的接收计数**（`:189-195`）：在 CPU 上把 `topk_idx` 展平、滤掉越界值（保留重复！注释 "Count routed selections independently, including duplicates"）、`bincount` 出每 expert 的路由数，`dist.all_reduce` 汇总所有 rank 后 reshape 成 `(R, E/R)` 取本 rank 那一行——这就是"本 rank 各 expert 应接收多少 token"的独立答案，与 Kernel 写进 `expert_recv_count` 的计数对账。注意这条路完全不经过被测代码，是真正的交叉验证。

**第 4 层：位级可复现与累计精确**（`:197-200`）：同一 seed 的后续运行断言 `torch.equal(output.view(torch.int16), expected_bits)`——**输出位级一致**（确定性检查，MoE 融合 Kernel 里有大量原子和进度竞争，这一条并不平凡）；同时 `expected_stats += recv_counts` 后断言 `torch.equal(stats, expected_stats)`——累计统计一次不差。

**运行矩阵**（`:172-175`）：循环 `num_runs + 2` 次（默认 `--num-correctness-tests 10` 即 12 次），seed 交替 `0,1,0,1,…`，**最后一次回到 seed 0**——注释 "Alternate inputs, ending on the original seed for benchmarking"：正确性跑完恰好停在原始输入上，随后的 benchmark（共享同一个 case）用的就是第一组输入。每个 seed 只在首次出现时跑一次 baseline，所以 12 次运行只有 2 次 baseline 评估。

### 3.4 benchmark 的三个口径：TFLOPS、HBM GB/s、Comm GB/s（当天核心）

`benchmark(case)`（`:203-243`）是 README 性能表（`README.md:160-173`，"Benchmarked over EP8 with top-k=6 … one shared expert; all values are averaged across 8 ranks"）的直接来源。分五步：

**第 1 步：测融合 Kernel**（`:204`）：

```python
prof = bench_msprof(case.run_fused, kernel_names='mega_moe_impl',
                    barrier_comm_profiling=True, backend='fast')
```

`bench_msprof` 的细节见 3.5；这里只要知道它返回 `KernelProfile`，`prof.dur_us` 是 30 次冷 L2 迭代的平均时长（微秒），并且因为传的是字符串，要求 timeline 里**恰好只有一个**名字含 `mega_moe_impl` 的 kernel。

**第 2 步：测 baseline（可选）**（`:206-214`）：只有 `case.ep_buffer is not None`（即开了 baseline 且多 rank）才做。两个要点：其一，用 `return_all_kernels=True` 拿到窗口内**全部** kernel 的字典，因为 baseline 会为不同 shape 编译出多个 GEMM 特化实例，而"列表查询只取第一个匹配"（`:210` 注释）；其二，只累加七类 kernel 的时长（`:207-209`）：`dispatch_impl`、`dispatch_copy_epilogue_impl`、`fp8_dequant_gemm_impl`、`fp8_gemm_impl`、`swiglu_forward_kernel`、`combine_impl`、`combine_reduce_epilogue_impl`——正好是 3.2 里 baseline 七段的 device 侧名字。swiglu 要按调用次数加权（`:212-214`）：有共享专家且 token 非零时是 2 次（路由 1 次 + 共享 1 次）。

**第 3 步：从 Kernel 侧取真实计数**（`:216-220`）：`recv_counts = buffer.expert_recv_count.cpu()`——第 44 天见过，这是 SymmBuffer 里由融合 Kernel 每次-launch 写入的"本 rank 各 expert 接收 token 数"视图；由此得 `num_recv_tokens = Σ recv_counts`、`num_touched_experts`（总数 > 0 的 expert 个数）、`num_tokens = y.size(0)`。**三个口径的全部"工作量"都以 Kernel 实际处理的路由为准，而不是均匀路由假设。**

**第 4 步：三个口径的公式**（`:221-231`）：

```python
flops = 6 * hidden * intermediate * (num_recv_tokens + num_tokens * shared)          # :221
# CUDA's logical HBM model: weights once per active expert + activation reads/writes
# + BF16 output; excludes SFs, metadata, reduction traffic and cache/reload effects;
# divide by full kernel time                                                        # :222-223
num_hbm_bytes  = 3 * hidden * intermediate * num_touched_experts / 2                 # :224
num_hbm_bytes += num_recv_tokens * (3 * hidden + 2 * intermediate)                   # :225
if shared and num_tokens:                                                            # :226
    num_hbm_bytes += 3 * hidden * intermediate * shared                              # :227
    num_hbm_bytes += num_tokens * (3 * hidden + 2 * intermediate * shared)           # :228
# Remote FP8 pulls + BF16 pushes; exclude local routes, SFs, metadata, cache effects  # :229
num_remote_tokens = num_recv_tokens - int(recv_counts[本rank].sum())                 # :230
num_comm_bytes = num_remote_tokens * hidden * 3                                      # :231
```

逐项解释：

- **FLOPS**：每条被路由的 token 走两层 GEMM——Linear1 `(2I×H)` + Linear2 `(H×I)` 共 `3·H·I` 个 MAC，即 `6·H·I` flops；共享专家给每个本地 token 再加 `S` 份，故 `recv + T·S` 个"等效 token"。
- **HBM 字节**：源码自称"**CUDA 的逻辑 HBM 模型**"——沿用 DeepGEMM 在 CUDA 上的汇报惯例，是一个刻意简化的模型，不是硬件计数器。权重项 `3·H·I·touched/2`：两层权重共 `3·H·I` 个 FP4 元素 × 0.5 字节 = 1.5·H·I，**每个活跃 expert 只记一次**（"weights once per active expert"）；激活项 `3H + 2I`/token 与共享专家的 `3·H·I·S + T·(3H+2I·S)` 是相应的逻辑读写量。注释明确列了排除项：**SF、元数据、归约流量、缓存重载效应**，且按整段 kernel 时间平均。
- **通信字节**：只算**远端** token（`recv − 本 rank 自发且自收`），每个远端 token `H×3` 字节 = FP8 拉取（1 B/元素）+ BF16 推回（2 B/元素）——正好是第 45 天六类通信对象中的第 3、4 类（数据本体），同样排除本地路由、SF、元数据与缓存效应。

**第 5 步：聚合与输出**（`:232-243`）：每 rank 把 `(fused_us, baseline_us, flops, hbm, comm)` 打包成张量 `dist.all_gather` 到所有 rank，堆叠后：

- `fused_us, baseline_us = rank_stats[:, :2].amax(dim=0)`——**时延取最慢的 rank**。MegaMoE 是全体 rank 同步推进的流水线，任何一个 rank 慢，端到端就是它慢；
- `averages = rank_stats.mean(dim=0)`——**工作量取平均**。`:239` 的注释一锤定音："Latency is the slowest rank; throughput uses mean work/bytes per rank over that time."；
- 输出行（`:241-243`）：

```text
{fused_us:8.2f} us | speedup {baseline/fused} | {avg_flops/fused_us/1e6:5.1f} TFLOPS/rank |
HBM {avg_bytes/fused_us/1e3:6.1f} GB/s/rank | Comm {avg_bytes/fused_us/1e3:5.1f} GB/s/rank
```

单位换算值得手动过一遍：`fused_us` 是微秒，所以 `flops / fused_us / 1e6 = flops / (fused_us×10⁻⁶ s) / 10¹²` 恰是 TFLOPS；`bytes / fused_us / 1e3` 恰是 GB/s。**三个口径共用同一个分母（最慢 rank 的时延），分子是三个不同的"逻辑工作量"**——所以它们是利用率/强度指标，不是硬件实测带宽。

**手工验证 README 表**（B 组核心练习的答案，先遮住）：取第 1 行 `E=96, H=5120, I=2304, T=64`（R=8、k=6、S=1），均匀路由假设下：

| 量 | 公式 | 值 |
|---|---|---|
| 每_rank_接收 token | `T·k = 64·6` | 384 |
| 远端 token | `384·(8−1)/8` | 336 |
| 活跃 expert | `E/R = 96/8` | 12 |
| flops | `6·5120·2304·(384+64)` | 31.71 GFLOP |
| HBM 字节 | `3·5120·2304·12/2 + 384·(3·5120+2·2304) + 3·5120·2304 + 64·(3·5120+2·2304)` | 256.7 MB |
| Comm 字节 | `336·5120·3` | 5.16 MB |
| **TFLOPS** | `31.71e9 / 138.7µs` | **228.6**（表值 228.5） |
| **HBM GB/s** | `256.7e6 / 138.7µs` | **1850.6**（表值 1850.0） |
| **Comm GB/s** | `5.16e6 / 138.7µs` | **37.2**（表值 37.5） |

再验一行大 token 的：`E=384, H=7168, I=3072, T=16384` → recv=98304、remote=86016、touched=48，flops=`6·7168·3072·114688`=15.15 TFLOP、HBM=4.82 GB、comm=1.85 GB，除以 17904.2 µs 得 **846.3 TFLOPS / 269.4 GB/s / 103.3 GB/s**，与表值 846.3/269.3/103.3 吻合。微小偏差（如 37.2 vs 37.5）来自：实际随机路由并非严格均匀、时延取 amax 而工作量取 mean、活跃 expert 数可能有出入。

这两个极端行也定量印证了第 45 天 3.7 的判断：T=64 时通信带宽只有 37.5 GB/s——不是链路慢，而是元数据握手、两次 rank_barrier 等固定成本把总时延撑大，`bytes/时延` 被摊薄；T=16384 时 TFLOPS 升到 846、Comm 升到 103 GB/s，进入带宽/计算主导区。

### 3.5 `bench_msprof` 与 `KernelProfile`：测量纪律

`bench_msprof`（`bench.py:284-400`）是全仓库通用的 kernel 级 profiling 入口（第 31、50 天也会用到），今天重点看它为 MegaMoE benchmark 服务的四个机制：

**（1）`KernelProfile` 的字段就是流水线利用率**（`:27-71`）：`dur_ns` 之外，AIC 侧有 `mad / scalar / mte1 / mte2 / mte3 / fixpipe` 六个 [0,1] 比率（相对各自 `total_cycles`），AIV 侧有 `vec / scalar / mte2 / mte3`。字段注释直接对应第 5/23 天的管道语义：`aic_mte1 = "L1 -> L0"`、`aic_mte2 = "GM -> L1"`、`aic_fixpipe = "L0C -> out"`、`aiv_vec = "Vector compute"`、`aiv_mte2 = "GM -> UB"`、`aiv_mte3 = "UB -> GM"`。辅助方法 `tflops(flops)`（`:51-52`）与 `gbps(bytes)`（`:54-55`）就是 3.4 的单位换算的函数化。对 MegaMoE，`aiv_mte2/mte3` 高意味着通信搬运繁忙、`aic_mad` 高意味着 GEMM 占比大——融合是否生效，先看这两组比率是否同时不低。

**（2）冷 L2**（`:263-268, :334, :344-346`）：默认 `flush_l2=True`——`_make_l2_flush_buffer()` 分配 `2 × properties.L2_cache_size` 字节的 int 缓冲，每次迭代前 `flush_buffer.zero_()` 把 L2 旧数据冲掉，保证每次测量的 launch 都从冷缓存开始。这与 README 性能表、以及第 50 天要系统讲的"冷/热缓存"口径一致。

**（3）跨 rank 对齐：`barrier_comm_profiling`**（`:302, :324, :335-350`）：MegaMoE benchmark 传了 `barrier_comm_profiling=True`。机制：若调用方没给 `barrier`，就用一个 dummy 张量的 `dist.all_reduce` 当屏障（`:336-340`）；每次迭代前先 `npu_sleep(int(2e7))`——睡 20 ms，注释解释 "keep the device busy so the barrier dominates instead of kernel launch gaps"——再执行屏障。没有这个对齐，各 rank 的 Kernel 启动时刻错开，EP 通信在时间上对不齐，单 rank 测出的"通信时间"里混满等待。环境变量 `EP_DISABLE_BARRIER_PROFILING=1` 可以关掉它（`:324`）。

**（4）三种 backend**（`:310-311, :355-393`）：

| backend | 机制 | 代码 |
|---|---|---|
| `persistent`（默认） | 进程常驻的原始 PMU 会话，ctypes 直接绑 `libmsprofiler.so`/`libprofimpl.so`，用 `npu_sleep` 打 begin/end 标记圈出窗口 | `_Session`（`:627-883`）、`_profile_persistent`（`:939-962`） |
| `fast` | `torch_npu.profiler` 一次性采集（`ProfilerLevel.Level1` + `AiCMetrics.PipeUtilization`），直接解析原始 `ffts_profile` 数据 | `:367-393`、`_parse_ffts_profile`（`:403-446`） |
| `slow` | 经 `tensorboard_trace_handler` 导出，再解析 msprof analyse 产出的 `kernel_details.csv` | `:371-372`、`_parse_kernel_details_csv`（`:449-508`） |

`test_mega_moe.py` 的 benchmark 用 `backend='fast'`。最后注意 **kernel 选择规则**（`_select_profiles`，`:271-281`）：字符串要求子串唯一匹配（不唯一就报错并列出候选）；列表每个查询只取第一个匹配（这正是 baseline 改用 `return_all_kernels=True` 的原因）；`return_all_kernels=True` 返回窗口内一切 kernel，docstring（`:304-309`）特别警告其中混着辅助 kernel（L2 flush、`npu_sleep`、屏障用的 all_reduce），**必须按名字挑**，否则测量被污染。

### 3.6 `msprof_mega_moe.py`：PIPE 级 trace 的采集与导出

benchmark 给出"多快"，但要回答第 45 天的"计算与通信到底怎么重叠"，需要看**单次 launch 内部每条管道在什么时候忙**。这就是 `scripts/profiling/msprof_mega_moe.py` 的用途。文件 docstring（`:1-9`）开头即定位："Collect uninstrumented MegaMoE msprof PIPE traces as JSON (no HTML)"，并给出两种用法：

```bash
# 采集 + 导出
python scripts/profiling/msprof_mega_moe.py --case 7168,3072,64,6,128,1 --output profiling/pipes
# 只重新导出已有采集（离线，不需要 NPU/PyTorch，见 :24 注释）
python scripts/profiling/msprof_mega_moe.py --capture profiling/pipes/raw --output profiling/reexport
```

**两阶段设计**（`main`，`:152-195`）：

1. **采集阶段**：解析参数（`--ranks` 默认 8；`--case H,I,E,K,T,S` 默认 `(4096, 2048, 64, 6, 8192, 1)`，按 `MegaMoECase` 形参顺序即 hidden、intermediate、experts、topk、max_tokens_per_rank、shared；`--warmups` 3；`--iters` 1，help 写明 "cold-L2 launches; export the last launch"）。若非 `--launch-only`，先做三个防呆：`--output` 必填（`:172-173`）、已有 `summary.json` 拒绝覆盖（`:175-177`）、`raw/` 采集目录用 `mkdir(exist_ok=False)` 创建——注释 "Refuse mixed captures: a previous run must never silently replace a rank's trace"（`:180-181`）。然后 subprocess 执行：

   ```bash
   msprof --instr-profiling=on --output=<raw>/ python <本脚本> --launch-only ...
   ```

   即用华为 MindStudio Profiler 的 CLI 以**指令级 profiling** 包住一次多进程运行。`--launch-only` 分支（`:165-171`）按 `--ranks` spawn worker。

2. **导出阶段**：在 `raw/` 下找 `PROF_*` 目录（每个 rank 一个），逐个调 `export`，最后汇总写 `summary.json`（`:187-195`）。

**worker（`:23-52`）**——被 msprof 包住的被测负载：`sys.path` 挂上仓库根和 tests（`:25`）后导入 `MegaMoECase`、`init_dist`、`_make_l2_flush_buffer`；`MegaMoECase(group, *args.case, with_baseline=False)`（`:35`）——**采集只用融合路径，不需要 deep_ep/tilelang**；`case.create_inputs(args.case[4])`（`:36`）注意 case 第 5 个元素是 `num_max_tokens_per_rank`，所以 token 数 = 容量，采集总是跑满容量。热身 `--warmups` 次后（`:37-38`），每次受测迭代：清 2×L2 缓冲 → `torch.npu.synchronize()` → `dist.barrier(group)` → `case.run_fused()` → 同步（`:41-46`）——和 3.5 一样是**冷 L2 + 跨 rank 对齐**的纪律。`finally` 里销毁 case 与进程组（`:49-52`）。

**export（`:60-149`）**——把 msprof 的原始产物变成可读 trace，九步：

1. 在 `mindstudio_profiler_output/` 下找到**唯一**的 `msprof_*.json` 时间线（`:61-63`），读出事件；
2. 挑出名字含 `mega_moe_impl` 的任务，按时间戳排序**取最后一个**（`:70-72`，注释 "Warmup calls are also captured; export the last cold-L2 launch"——热身也被采集了，只有最后一次是干净的冷 L2 受测 launch）；
3. 以它的时间戳为原点 `origin`、时长为窗口 `duration`；从 `device_*` 目录名解析 rank，从 `info.json.*` 读 `hwts_frequency`（`:75-79`）；
4. 从 trace 的 `thread_name` 元数据里按正则 `Group(\d+)-(aic|aiv0|aiv1)` 建立 (pid, tid) → (core 编号, 子核) 的 lane 表（`:80-85`）；
5. 打开该 device 的 `sqlite/biu_perf.db`，按 rowid 顺序读 `BiuInstrStatus` 表（`:87-91`）——msprof 指令级采集的逐指令记录；
6. 逐事件对齐（`:92-114`）：对 lane 上每个带 `Core Type` 的 X 事件，弹出该 (group, core) 的下一条 DB 记录，校验 `(name, block)` 一致，不一致即报错 "msprof JSON and BIU DB event ordering differ"（`:100-101`）——**JSON 时间线与 SQLite 记录互为对方的完整性校验**；带 checkpoint 的记录跳过（`:103-104`）；DB 里的时长是**硬件 tick**，除以 `hwts_frequency` 换算成微秒（`:106-107`，注释 "BIU DB durations are hardware ticks; JSON timestamps are microseconds"）；与窗口求交，越界的裁剪并计入 `boundary_clipped`（`:108-113`）；
7. 按 `(block, 子核, 管道)` 分轨，管道名经 `PIPE` 表映射（`:55-56`）：`SU→S`（标量）、`CUBE→M`（矩阵计算）、`MTE1/MTE2/MTE3` 原名、`FIXP→FIX`、`VEC→V`——**正是 3.5 里 `KernelProfile` 那组管道的 BIU 指令名**；子核经 `SUBCORE` 映射为 `AIC / AIV0 / AIV1`（`:57`）；
8. 写出 `rank_<rank>.json`（Chrome/Perfetto 格式）：一条 `mega_moe_impl` 总览条（ts=0）+ 每轨一条 `core NN / AIC / PIPE_M` 式的时间线（`:118-135`），可直接拖进 Perfetto UI 查看；
9. 写出 `rank_<rank>_summary.json`：`task_duration_us`、`matched_biu_events`、`exported_pipe_events`、`sampled_cores`、每轨的首末时刻与事件数等（`:144-149`）。

**怎么用、边界在哪**：打开 `rank_0.json`，AIC 核上的 `PIPE_M`（CUBE）跨度对应 GEMM 计算、`PIPE_MTE1/MTE2` 对应数据搬运；AIV1 上的 `PIPE_MTE2/MTE3` 跨度对应第 45 天的远端 pull/push。**计算轨与通信轨在时间上的交叠**，就是"融合"的直接视觉证据。但 docstring 第 8 行划了边界：

> BIU timings explain pipeline overlap, not unprofiled end-to-end performance.

指令级采集本身有开销，且 trace 只含**采样到的核**（`sampled_cores`）、只含**最后一次 launch**——它回答"重叠结构"，不回答"最终多快"；端到端数字永远以 3.4 的 benchmark（未插桩）为准。另一个实用细节：`--capture` 重导出模式不碰 NPU（`:24` 注释 "Offline export and --help do not need PyTorch or an NPU environment"）——别人拷给你一个 `PROF_*` 目录，你在笔记本上就能出图。

### 3.7 当天产出·参考：测试依赖与性能指标清单

**表一：MegaMoE 测试与 profiling 依赖清单**

| 层 | 依赖 | 代码证据 | 缺失时的行为/替代 |
|---|---|---|---|
| 运行时（fused 路径） | torch + torch_npu、`deep_gemm._C`（含 `SymmBuffer`/`npu_sleep`）、HCCL 注册（`HcclCommMemReg`） | `test_mega_moe.py:12`、`bench.py:327`、第 45 天 `hccl.hpp:89-164` | 无法运行；改为本文 B 组静态任务 |
| 多 rank 组织 | `torch.distributed`（hccl 后端）、MASTER_ADDR/PORT 等环境变量 | `envs.py:28-52`、`test_mega_moe.py:322` | 单进程 `--num-processes 1` 可跑但自动关 baseline 与正确性（`:309`） |
| baseline（正确性 + speedup） | `deep_ep`（EPBuffer）、`tilelang`、`third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py`、独立 hccl 组 | `test_mega_moe.py:77-100` | `--skip-baseline` 连带跳过正确性（`:309`）；只剩 `PASS` 与 fused 性能 |
| benchmark | `torch_npu.profiler`（fast backend）、`deep_gemm._C.npu_sleep`、2×L2 flush 缓冲 | `bench.py:284-400`、`test_mega_moe.py:204` | 无硬件时按 3.4 公式手算，对照 README 表 |
| PIPE trace | `msprof` CLI（`--instr-profiling=on`）、SQLite（读 `biu_perf.db`）、Perfetto/Chrome trace 查看器 | `msprof_mega_moe.py:182-186, :87-91` | 采集必须多卡真机；**重导出**（`--capture`）无需 NPU（`:24`） |

**表二：性能指标清单**

| 指标 | 公式（每 rank 工作量 / fused_us） | 代码位置 | 口径说明 |
|---|---|---|---|
| Latency | `amax` over ranks 的 `mega_moe_impl` 时长（30 次冷 L2 均值） | `:204, :237` | 最慢 rank 决定端到端 |
| Speedup | `baseline_us / fused_us`（七类 kernel 时长之和 ÷ 融合时长） | `:206-214, :240` | baseline 关闭时显示 `N/A` |
| TFLOPS/rank | `6·H·I·(recv+T·S) / fused_us / 1e6` | `:221, :242` | 按实际路由计数，非均匀假设 |
| HBM GB/s/rank | `[3HI·touched/2 + recv·(3H+2I) + S·(3HI + T·(3H+2IS))] / fused_us / 1e3` | `:222-228, :242` | "逻辑 HBM 模型"：排除 SF/元数据/归约/缓存重载；权重每活跃 expert 一次 |
| Comm GB/s/rank | `(recv−本地)·H·3 / fused_us / 1e3` | `:229-231, :243` | 仅远端 FP8 pull + BF16 push；排除本地路由/SF/元数据 |
| 管道利用率 | `KernelProfile` 的 mad/mte1/mte2/fixpipe/vec… 比率 | `bench.py:27-71` | 各管道相对自身 total_cycles 的占用 |
| PIPE 重叠结构 | 每 (core, subcore, pipe) 的时间跨度 | `msprof_mega_moe.py:122-135` | 只看结构与重叠，不代表无插桩端到端性能（`:8`） |

## 四、源码导读

今天的主线文件与阅读站点（建议按顺序读，前两个建议逐行）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `tests/test_mega_moe.py` | `MegaMoECase` `:50-163`（`__init__` `:53-104`、`create_inputs` `:106-120`、`run_fused` `:122-128`、`run_baseline` `:130-156`、`destroy` `:158-163`）、`make_npu_fp8/fp4` `:22-36`、`make_topk` `:39-47`、`check_correctness` `:166-200`、`benchmark` `:203-243`、`test` `:246-282`、`__main__` `:285-322` | 依赖分叉（`:77-104`）、阈值注释（`:18`）、三口径公式（`:221-231`）与聚合注释（`:239`）、最大容量注册（`:255-258`）、empty_cache 注释（`:279-280`） |
| 2 | `scripts/profiling/msprof_mega_moe.py` | `worker` `:23-52`、`export` `:60-149`、`main` `:152-195`、`PIPE`/`SUBCORE` `:55-57` | docstring 边界声明（`:8`）、冷 L2 + barrier（`:41-46`）、取最后 launch（`:70-72`）、tick→µs（`:106-107`）、JSON/DB 对齐断言（`:100-101`）、防覆盖与防混采（`:175-181`） |
| 3 | `deep_gemm/testing/bench.py` | `KernelProfile` `:27-71`、`_make_l2_flush_buffer` `:263-268`、`_select_profiles` `:271-281`、`bench_msprof` `:284-400` | 管道字段注释（`:33-45`）、backend 说明（docstring `:296-317`）、`npu_sleep(2e7)` 注释（`:347-349`）、`EP_DISABLE_BARRIER_PROFILING`（`:324`）；三种解析器 `_parse_ffts_profile` `:403-446`、`_parse_kernel_details_csv` `:449-508`、`_Session` `:627-883` 选读 |
| 4 | `deep_gemm/testing/envs.py` | `init_dist` `:28-52`、`dist_print` `:21-25` | 环境变量与 world/rank 换算、默认 BF16/npu 设备、once_in_node 语义 |
| 5 | `tests/utils.py` | `calc_diff` `:36-43` | 余弦相似度差异的定义（第 30 天已读，今天确认 MegaMoE 复用它） |
| 6 | `deep_gemm/mega/__init__.py` | `SymmBuffer` base 复用 `:50-55`、`expert_recv_count` `:75-77`、`fp8_fp4_mega_moe` `:159-184` | `num_bytes ≤ base` 断言（`:52`）、"按顺序使用共享 scratch"注释（`:53`）——3.1 注册复用的 Python 侧 |
| 7 | `README.md` | MegaMoE 节 `:160-173` | EP8/top-k=6/1 shared expert 的数据表——3.4 手算的验证对象 |

**两个主文件的分块导读**：

- **`test_mega_moe.py` 按执行顺序**：参数与展开（`:285-322`，注意 H/I 广播断言 `:312-313` 与笛卡尔积顺序 `:316-319`）→ `test` 骨架（`:246-282`，重点 `:255-258` 的 `max(…, key=buffer_size)` 与 `:279-280` 的 empty_cache）→ `MegaMoECase`（先读 `:77-104` 的分叉，再回看 `:53-75` 的权重准备）→ `check_correctness`（`:172-175` 的 seed 矩阵是理解全文的钥匙）→ `benchmark`（`:221-231` 逐行抄公式，`:237-243` 逐行换单位）。
- **`msprof_mega_moe.py` 按数据流**：先 `main`（`:152-195`）看两种模式如何分流；再 `worker`（`:23-52`）确认被测负载与 benchmark 同纪律；最后 `export`（`:60-149`）按 3.6 的九步走，重点体会"JSON 给时间戳、SQLite 给时长、`hwts_frequency` 做桥梁"（`:106-107`）这一设计。

## 五、实践步骤

### A 组：有 Ascend 多卡环境（8 卡可对照 README 表；2 卡可跑通但数值不可对照）

1. **复现 README 的两个极端行**（正确性 + 性能一次看全）：

   ```bash
   python tests/test_mega_moe.py --num-processes 8 --num-experts 96 --hidden 5120 \
       --intermediate-hidden 2304 --num-topk 6 --num-tokens 64 --num-shared-experts 1
   python tests/test_mega_moe.py --num-processes 8 --num-experts 384 --hidden 7168 \
       --intermediate-hidden 3072 --num-topk 6 --num-tokens 16384 --num-shared-experts 1
   ```

   记录两行输出的 `us / speedup / TFLOPS/rank / HBM / Comm`，与 README 表第 1、8 行及 3.4 的手算值三方对照。再用 `case.buffer.expert_recv_count.cpu()` 的思路验证：`recv_counts.sum(dim=0) > 0` 的个数是否等于你手算的 `touched`（12 / 48）。

2. **无第三方依赖的快速通道**：给上面任一命令加 `--skip-baseline --num-correctness-tests 0`——不装 deep_ep/tilelang 也能得到 fused 性能行（`test_mega_moe.py:309` 的连带逻辑）。这同时验证了 3.2 的依赖分叉。

3. **PIPE trace 采集与阅读**：

   ```bash
   python scripts/profiling/msprof_mega_moe.py --case 7168,3072,64,6,128,1 --output profiling/pipes
   ```

   把 `profiling/pipes/rank_0.json` 拖进 Perfetto UI（ui.perfetto.dev）：找到 ts=0 的 `mega_moe_impl` 总条；在 AIC 轨上看 `PIPE_M`（计算）与 `PIPE_MTE1/MTE2`（搬运）的分布；在 AIV1 轨上看 `PIPE_MTE2/MTE3`（通信 pull/push）跨度，检验它们与 AIC `PIPE_M` 是否交叠——第 45 天的重叠论断的视觉证据。对照 `rank_0_summary.json` 的 `sampled_cores` 与 `boundary_clipped`。最后练习离线重导出：`python scripts/profiling/msprof_mega_moe.py --capture profiling/pipes/raw --output profiling/reexport`。

4. **（进阶）屏障对通信测量的影响**：设 `EP_DISABLE_BARRIER_PROFILING=1` 重跑第 1 步小 token 用例，比较 Comm GB/s 变化，联系 3.5（3）解释原因。

### B 组：无 Ascend 环境（静态任务）

1. **三口径手算（当天核心练习）**：遮住 3.4 的答案表，独立复算 README 第 1 行（E=96, H=5120, I=2304, T=64）与第 5 行（E=384, H=5120, I=2304, T=4096）的三个指标，与表值对照。再做两个敏感性推演：k 从 6 改 8，T=16384/H=7168 行的 comm bytes 变多少？（`16384·8·(7/8)·7168·3 ≈ 2.47 GB`，Comm GB/s 约 `2.47e9/17904µs ≈ 138 GB/s`。）若只有 2 卡（R=2），EP 系数 `(R−1)/R` 变为多少，对 comm 口径有何影响？
2. **用例展开推演**：命令 `python tests/test_mega_moe.py --hidden 5120 7168 --num-tokens 64 4096`（其余默认）会展开成哪几个 case？写出 4 行表头（预期：`(5120,2304)×{64,4096}` 与 `(7168,3072)×{64,4096}`，E=384、topk=6、shared=1），并说明若写 `--hidden 5120 7168 --intermediate-hidden 2304` 会怎样（H 广播成 `(5120,2304)`、`(7168,2304)`，合法；`--hidden 5120 7168 --intermediate-hidden 2304 3072 2048` 则触发 `:312-313` 断言）。
3. **`check_correctness` 运行矩阵**：`--num-correctness-tests 10` 时共几轮、seed 序列是什么、baseline 评估几次、为什么最后一轮必须是 seed 0？（预期：12 轮，`0,1,0,1,0,1,0,1,0,1,0,0`，baseline 2 次，为 benchmark 收敛回原始输入。）
4. **接收计数对账演算**：设 R=2、E=4、T=3，rank0 的 `topk_idx = [[0,1],[2,3],[0,2]]`，rank1 的 `topk_idx = [[1,2],[3,0],[2,3]]`（k=2）。按 `:189-194` 的逻辑手算：各 rank `bincount` 结果、all_reduce 后的 `(2,2)` 矩阵、每个 rank 的 expected 行，以及 `expert_recv_count` 应写入的值（注意重复路由要计数两次）。
5. **`export` 静态分析**：合上教程回答四个问题——为什么导出只取最后一个 `mega_moe_impl` 任务？`(name, block)` 校验失败说明什么？`checkpoint is not None` 的记录为什么跳过？`boundary_clipped` 计数在什么情况下增长？再解释"JSON 时间戳 + SQLite tick 时长 + `hwts_frequency`"三者如何拼出一条完整的时间跨度。
6. **写依赖与指标清单（当天产出）**：合上教程，凭记忆默写 3.7 的两张表，标注每一条的代码证据行号，再对照修正。

## 六、当天产出

**MegaMoE 测试依赖和性能指标清单**——以 3.7 两张表为参考答案，合上教程独立完成。依赖表至少分清四层（运行时/多 rank 组织/baseline/benchmark+trace），每层写明组件、代码证据和缺失时的降级路径；特别要能回答"哪条路径完全不需要 deep_ep 和 tilelang"（fused-only 与 PIPE 采集）以及"哪些工作可以完全离线做"（`--capture` 重导出、三口径手算）。指标表至少包含 Latency、Speedup、TFLOPS、HBM、Comm 五项的公式、单位换算（除以 `fused_us/1e6`、`/1e3` 的由来）、聚合规则（amax vs mean）与排除项，并附上你亲手复算的一行 README 数据作为验证。有环境的同学附上 A 组第 1、3 步的输出记录。

## 七、自测题

1. `tests/test_mega_moe.py` 如何把 8 个进程组织成一个 EP 测试？
   > `torch.multiprocessing.spawn(test, nprocs=args.num_processes)`（`:322`）每 rank 一进程；`init_dist`（`envs.py:28-52`）按 MASTER_ADDR/PORT/WORLD_SIZE/RANK 以 hccl 后端建组，world_size = 节点数×本地进程数，并把默认设备/ dtype 设为 npu/BF16。
2. 为什么要"最大容量注册一次、逐用例复用"，而不是每个用例各注册一个 SymmBuffer？
   > HCCL 注册没有注销 API，注册段进程终身持有（第 45 天 `hccl.hpp:166-171`）；测试用 `max(configs, key=buffer_size)` 选最大需求注册一次 `base`（`:255-258`），后续 `base=base` 复用同一注册段切视图，Python 侧以 `num_bytes <= base.nbytes` 断言（`mega/__init__.py:51-52`）。
3. 什么情况下 baseline 会被关闭？此时测试还能做什么？
   > `--skip-baseline` 或单 rank（`with_baseline = not skip_baseline and num_ranks > 1`，`:267`）；`skip_correctness` 随之置位（`:309`）。仍可跑 fused 路径的性能（`--skip-prof` 不加时输出 us/TFLOPS/HBM/Comm，speedup 显示 N/A）。
4. `REFERENCE_DIFF_THRESHOLD` 为什么是 5e-6 而不是 0？
   > baseline 的 DeepEP combine 把各 rank 部分和舍入到 BF16 再 FP32 累加，融合 Kernel 全程 FP32（`:18` 注释），两者只能近似相等；`calc_diff` 是余弦差（`tests/utils.py:36-43`）。
5. `check_correctness` 有哪四层检查？各自防什么错？
   > 整体余弦差（防整体偏差）、逐 token 相对差（防少数 token 严重出错被平均掉，`:184-187`）、跨 rank bincount+all_reduce 的独立接收计数（防路由/元数据错，`:189-195`）、同输入位级可复现 + 累计统计精确累加（防非确定性与覆盖而非累加，`:197-200`，配合初值 17+ 的哨兵 `:169`）。
6. 写出三个口径的公式，并各列两个排除项。
   > flops = `6HI(recv+T·S)`；HBM = `3HI·touched/2 + recv·(3H+2I) + S·3HI + T·(3H+2IS)`，排除 SF/元数据/归约流量/缓存重载，权重每活跃 expert 只记一次；Comm = `(recv−本地)·H·3`（FP8 pull 1B + BF16 push 2B），排除本地路由/SF/元数据（`:221-231`）。
7. 为什么时延取 amax 而吞吐用 mean？
   > `:239` 注释："Latency is the slowest rank; throughput uses mean work/bytes per rank over that time."——同步流水线的端到端由最慢 rank 决定，而各 rank 工作量按平均计才是强度口径。
8. `bench_msprof` 靠什么保证每次测量是冷 L2 且各 rank 对齐？
   > 2×L2 大小缓冲每次迭代前 `zero_()`（`:263-268, :344-346`）；`barrier_comm_profiling=True` 时先 `npu_sleep(2e7)`（睡 20 ms 占住设备，避免启动空隙主导）再 all_reduce 屏障（`:335-350`）；可用 `EP_DISABLE_BARRIER_PROFILING=1` 关闭（`:324`）。
9. `KernelProfile` 的 `aic_mte1`、`aic_mte2`、`aic_fixpipe`、`aiv_mte3` 各对应什么数据通路？
   > L1→L0、GM→L1、L0C→出（AIC Fixpipe）、UB→GM（`bench.py:33-45` 字段注释）——即第 5/23 天的管道模型在 profiling 数据上的投影。
10. `msprof_mega_moe.py` 的两种运行模式分别做什么？export 为什么只取最后一个 `mega_moe_impl`？
    > 采集模式用 `msprof --instr-profiling=on` 包住 `--launch-only` 多进程负载（`:182-186`）；`--capture` 模式离线重导出已有 `PROF_*`（不需要 NPU，`:24`）。热身调用也被采集，只有最后一次是冷 L2 的受测 launch（`:70-72` 注释），故取最后。
11. 为什么说 PIPE trace "explain pipeline overlap, not unprofiled end-to-end performance"？
    > 指令级采集本身有开销，且 trace 只含采样到的核（`sampled_cores`）和单次 launch；它展示的是计算/搬运/通信管道在时间上的交叠结构（`:8` docstring），端到端性能必须以未插桩的 benchmark（3.4）为准。

---

明天（第 47 天）离开手写 Kernel 的世界一天：阅读 `deep_gemm/tilelang_ops/hc_prenorm.py` 和 `tests/test_hyperconnection.py`，看 TileLang 生成的 HC Prenorm 融合算子如何与 GEMM 衔接，并比较手写 Ascend Kernel 与 TileLang 两种开发方式在代码结构和工作流上的差异——今天读过的 `bench_msprof` 与 `KernelProfile` 届时还会作为统一的测量底座再次出场。
