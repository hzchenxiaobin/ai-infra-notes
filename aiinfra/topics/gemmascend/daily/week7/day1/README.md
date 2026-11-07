# 第 7 周 · 第 1 天（总第 43 天）：理解 MegaMoE 对外接口

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前六周我们打通了 BF16/低精度 GEMM 的调度、搬运、MAD、流水线与 Epilogue，也见过了 Grouped GEMM 和 MQA 两种"变形"。今天开始进入本仓库最复杂的算子——**MegaMoE：把一整个 MoE 前馈层（EP Dispatch → Grouped GEMM 1 → SwiGLU → Grouped GEMM 2 → Combine）连同多卡通信一起融合进一个 Kernel**。今天只看**对外接口**：Python 侧公开的符号只有三个（`SymmBuffer`、`transform_weights_for_mega_moe`、`fp8_fp4_mega_moe`，`deep_gemm/__init__.py:137-141`），但每一个背后都牵出一条完整的数据准备链——**对称缓冲区**是 EP 通信的载体，**权重转换**负责把用户布局变成 Kernel 直接消费的布局，**输入填充**决定了 token、top-k 路由和 shared expert 数据各就各位。当天产出是**MegaMoE API 与数据准备流程图**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 MegaMoE 融合的五段计算（Dispatch → Grouped GEMM 1 → SwiGLU → Grouped GEMM 2 → Combine），并指出 baseline 是如何用现成组件逐段拼装的（`tests/test_mega_moe.py:130-156`：deep_ep 的 dispatch/combine + `m_grouped_fp8_fp4_gemm_nt_contiguous` × 2 + TileLang SwiGLU）；
2. 列出三个公开符号的来源与职责（`deep_gemm/__init__.py:137-141` ← `deep_gemm/mega/__init__.py`），并画出从 Python 包装 → pybind 绑定（`csrc/apis/mega_moe.hpp:206-225`）→ Host 检查（`:128-204`）→ `launch_mega_moe`（`csrc/jit_kernels/mega_moe.hpp:24-51`）→ JIT 实例化 `mega_moe_impl` 的完整调用链；
3. 解释**为什么需要 SymmBuffer**：HCCL 把各 rank 的同段内存注册为 UBMEM 直访区，Device Kernel 凭 `peer_ptrs` 直接读写远端；并说明"进程生命周期的注册池"（`csrc/backends/hccl.hpp:166-171`）、`destroy()` 为何不释放 HBM、以及多个 case 如何用 `base` 复用同一块注册内存（`tests/test_mega_moe.py:249-258`）；
4. 用 `layout::Workspace`（`deep_gemm/include/deep_gemm/layout/mega_moe.hpp:85-250`）的公式推导 Python 拿到的 **13 个视图**（x、x_sf、topk_idx、topk_weights、shared 四件套、routed 四件套、expert_recv_count）的形状、dtype 与用途，并手算一个小配置下的容量；
5. 手工执行 `transform_weights_for_mega_moe` 的三个变换：l1 的 gate/up 交织（`_interleave_weights`，`mega/__init__.py:91-105`）、l2 的 BLOCK_N 分块（`_reshape_weights`，`:108-113`）、FP8 的 NZ 重排（`_transform_fp8_nz`，`:84-88`）与 FP4 权重 SF 的指数 +6 补偿（`_scale_fp4_weight_sf`，`:116-126`），并对照 C++ `check_weights`（`csrc/apis/mega_moe.hpp:112-126`）验证最终形状；
6. 在测试代码中指出 **EP、top-k、expert、shared expert、无效路由** 等概念各自的位置（3.7 节的概念定位表）；
7. 独立画出 MegaMoE API 与数据准备流程图（当天产出），标注每一步消费和产出的数据。

## 二、预备知识

- **第 4 天**：FP8/FP4 表示与 Scaling Factor——今天的 `recipe=(1, 1, 32)` 即量化粒度 `(gran_m, gran_n, gran_k)`（`csrc/apis/gemm.hpp:218-221`）：激活按 1×32、权重按 1×32 共享一个 UE8M0 指数。
- **第 12 天**：JIT 配置、编译与启动——今天 `launch_mega_moe` 走的是同一条 `jit->compile()` / `jit->launch()` 链，但没有启发式选配置：7 个配置参数全部直接成为模板实参。
- **第 29、30 天**：测试数据生成与正确性验证——今天的 `make_npu_fp8/fp4`、`calc_diff`、参考结果对比沿用同一套方法。
- **第 32 天**：FP8/FP4 Kernel 路径——MegaMoE 目前**只支持 fp8 激活 × fp4 路由权重**（`mma_type == 'fp8xfp4'`，`mega/__init__.py:28`），shared 权重是 FP8。
- **第 33 天**：Scale 布局与 UE8M0 打包——`MX_SF_DIVISOR = 64`（`deep_gemm/include/deep_gemm/common.hpp:65`，"K-elements per SF pair"）：每 64 个 K 元素的两个 SF 指数打包进一个 `int16`，与 README "each pair of UE8M0 scaling factors along the K dimension is packed into an int16"（`README.md:56`）一致。
- **第 36 天**：M-Grouped GEMM——MegaMoE 内部的两个 GEMM 正是 grouped 形态，baseline 直接调 `m_grouped_fp8_fp4_gemm_nt_contiguous` + psum layout（`test_mega_moe.py:140-145`）；今天复习"token 按 expert 排序拼接"的图景即可。
- **分布式基础**：会用 `torch.distributed` 建立 HCCL 进程组（测试通过 `init_dist`，`deep_gemm/testing/envs.py:28-52`）；不需要了解 HCCL 内部，第 45 天再展开通信细节。

另注：第五节 A 组任务需要 Ascend 多卡环境（至少 2 卡）以及 `deep_ep`、`tilelang` 两个第三方依赖（baseline 用）；无环境时完成 B 组静态任务（容量手算、权重变换 CPU 复现、调用链标注、概念定位、画图）即可达到当天目标。

## 三、详细讲解

### 3.1 MegaMoE 要解决什么问题

MoE 前馈层的完整计算是（设 expert 数 E、top-k=k、hidden=H、intermediate=I）：

```text
每个 token：router 选出 top-k 个 routed expert（可能还有 1 个全量计算的 shared expert）
Dispatch：把 token 按 expert 归属发到持有该 expert 的 rank（EP 下每个 rank 只持有 E/R 个 expert）
GEMM 1：  [token] × W1[e]ᵀ → [2I]（gate 与 up 拼接）
SwiGLU：  up × silu(gate) → [I]，可带 clamp
GEMM 2：  [I] × W2[e]ᵀ → [H]
Combine：把各 expert 的输出按 topk_weights 加权送回 token 的源 rank 并求和（shared expert 的输出也在这里合并）
```

不融合时，这五段要靠 5 类独立 kernel 拼装——`tests/test_mega_moe.py` 的 `run_baseline`（`:130-156`）就是这种"积木版"：

| 段 | baseline 组件 | 位置 |
|---|---|---|
| Dispatch | `deep_ep.EPBuffer.dispatch`（返回按 expert 排列的 `recv_x` 与 psum `layout`） | `test_mega_moe.py:134-137` |
| GEMM 1 | `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`（psum layout，`use_psum_layout=True`） | `:140-141` |
| SwiGLU | `tilelang_ops.swiglu_apply_weight_to_fp8`（第三方 TileLang 算子，clamp 生效于此） | `:93-94, :142` |
| GEMM 2 | 同 GEMM 1，换 `l2_weights` | `:144-145` |
| Combine | `deep_ep.EPBuffer.combine`（shared 输出经 `bias=shared_y` 参数一并累加） | `:156` |

积木版的代价：每段之间中间张量都要落回 HBM，通信 kernel 与计算 kernel 无法重叠。MegaMoE 把五段一起融进一个 `mega_moe_impl`，README 对它的定位是"Mega MoE fuses EP dispatch, two grouped GEMMs, SwiGLU, and combine"（`README.md:162`），基准口径为 EP8、top-k=6、外加 1 个 shared expert（`README.md:160-173` 的性能表）。**为什么融合能更快、通信与计算如何重叠，属于第 45 天的主题**；今天先记住这条计算链和它的两端数据即可。

### 3.2 API 总览：三个公开符号与完整调用链

MegaMoE 在 Python 侧的公开面非常小（`deep_gemm/__init__.py:137-141`）：

| 符号 | 定义处 | 职责 |
|---|---|---|
| `deep_gemm.SymmBuffer` | `mega/__init__.py:9-81` | 注册 HCCL 对称缓冲区并切出输入/中间张量视图 |
| `deep_gemm.transform_weights_for_mega_moe` | `:129-156` | 把用户侧 `(weight, scale)` 对转成 Kernel 直接消费的布局 |
| `deep_gemm.fp8_fp4_mega_moe` | `:159-184` | 执行融合算子，把结果写进 `y` |

另有一个内部常量 `BLOCK_N = _C._mega_moe_block_n = 256`（`:6`；由 `csrc/apis/mega_moe.hpp:207` 从 `layout::BLOCK_N`（`layout/mega_moe.hpp:12`）导出）——Python 的权重变换要用它来分块，**用户不能改**。

三个符号的调用链（今天的主线图）：

```text
deep_gemm.fp8_fp4_mega_moe(y, l1', l2', sym_buffer, ...)          mega/__init__.py:159（Python 断言 handle 存活 :170）
  └─ _C.fp8_fp4_mega_moe(...)                                      pybind 绑定 csrc/apis/mega_moe.hpp:217-224
       └─ fp8_fp4_mega_moe(...)                                    Host 检查与布局重建 :128-204
            ├─ get_workspace(...)                                  用 7 个标量参数重建 layout::Workspace :153-154
            ├─ check_weights(...) × 4                              校验 l1/l2/shared 权重形状 :180-188
            └─ launch_mega_moe(...)                                csrc/jit_kernels/mega_moe.hpp:24-51
                 ├─ 生成 mega_moe_impl<R,E,S,T,k,H,I> 实例化代码   :27-36（std::format 拼模板实参）
                 ├─ jit->compile("mega_moe", code)                 :41
                 ├─ aclrtMemsetAsync(progress 区清零)               :44-46
                 └─ jit->launch(..., num_blocks=kNumAICores)       :48-50（32 个 AIC）
```

与第 13 天串联的 GEMM 调用链对比，有两个显著差异：

1. **没有 `GemmDesc` / `GemmConfig` / 启发式**。MegaMoE 的 7 个配置参数（num_ranks、num_experts、num_shared_experts、num_max_tokens_per_rank、num_topk、hidden、intermediate_hidden）直接成为 Kernel 的**模板实参**（`jit_kernels/mega_moe.hpp:31`），JIT 为每种组合生成一份专属代码——因为缓冲区布局、ring 大小、进度数组长度等全是这 7 个数的编译期函数（`layout::Workspace` 里大量 `constexpr`）。
2. **`num_ranks` 不由用户显式传入**，而是 Host 从 `sym_buffer_ptrs.size()` 推出（`csrc/apis/mega_moe.hpp:151`）——对称缓冲区注册了几个 rank，EP 世界就是多大。

### 3.3 SymmBuffer：HCCL 对称缓冲区

**为什么输入和中间张量必须放在"特殊"缓冲区里？** 因为 EP 的 Dispatch/Combine 由 Device Kernel 自己完成：rank r 的 Kernel 要直接读 rank s 的 token、也要把 expert 输出直接写进 rank s 的接收区。这要求所有参与跨 rank 数据搬运的区域位于**各 rank 互相可见的注册内存**里，并且每个 rank 都知道所有伙伴的地址。权重和输出 `y` 则相反——它们只被本 rank 访问（每个 rank 只算自己的 E/R 个 expert），所以是普通 NPU 张量。

Python `SymmBuffer.__init__`（`mega/__init__.py:20-60`）做三件事：

1. **问尺寸**：`_C.get_symm_buffer_size_for_mega_moe(group.size(), E, T_max, k, H, I, mma_type, activation, shared)`（`:40-42`）返回 `(num_bytes, slice_input_buffers)`——C++ 按配置重建 `layout::Workspace` 算出需要的字节数，同时返回一个**切片闭包**（`csrc/apis/mega_moe.hpp:79-108`），布局知识完全留在 C++ 侧，Python 只负责持有视图。
2. **注册**：`_C.SymmBuffer(rank_idx, num_ranks, hccl_group_name, num_bytes)`（`:45-46`）构造 C++ `SymmBuffer`（`csrc/apis/mega_moe.hpp:45-62`），内部调用 `HCCLSymmetricBuffer::acquire`；组名来自 `group._get_backend(torch.device('npu')).get_hccl_comm_name(rank_idx)`。
3. **切片**：`torch.npu.synchronize()` + `dist.barrier()`（`:48-49`，等所有 rank 注册完），然后执行切片闭包得到 13 个视图（`:57-60`）。

C++ 侧 `HCCLSymmetricBuffer`（`csrc/backends/hccl.hpp:29-208`）的注册流程，注释一句话概括："Leases one process-lifetime HCCL-registered segment and resolves every peer's UBMEM VA"（`:28`）：

| 步骤 | 代码 | 说明 |
|---|---|---|
| ① 分配 | `aclrtMalloc(&buffer, num_bytes, ...)` + 清零 | 普通设备内存（`:101-102`） |
| ② 注册 | `HcclCommMemReg(comm, memory_tag, &memory, &memory_handle)`，tag 为 `"deep_gemm_mega_moe_" + 序号`（`:110-112`） | 把本段内存登记到 HCCL；注册顺序各 rank 必须一致（`:109` 注释） |
| ③ 建通道 | 取 rank graph 的第 1 网络层（UBMEM 要求，`:118-122`），为每个伙伴建 `COMM_PROTOCOL_UB_MEM` 描述子并以 `COMM_ENGINE_AIV` 引擎 `HcclChannelAcquire`（`:124-139`） | 通信由**向量核 AIV** 承担 |
| ④ 换地址 | `HcclChannelGetRemoteMems` 在每个通道里找同名 tag 的远端内存，把地址写进 `peer_ptrs[远端 rank]`（`:141-163`） | 至此本 rank 知道所有伙伴缓冲区的虚拟地址 |

随后 `torch::from_blob` 把裸内存包成 `int8` 张量，deleter 里先 `aclrtSynchronizeDevice` 再归还租约（`csrc/apis/mega_moe.hpp:55-59`）；`buffer_ptrs`（全体 rank 的地址表）随对象一起暴露给 Python（`:60`）。Device Kernel 拿到的是 `layout::SymBuffer`（`layout/sym_buffer.hpp:12-46`）：`base` + 每 rank 的偏移表，`map(ptr, dst_rank_idx)` 就是 `ptr + offsets[dst_rank_idx]`——把本 rank 的一个 GM 指针**平移**成目标 rank 的对应指针。

三条重要的工程约束（都出自源码注释，面试级考点）：

1. **注册池永不释放**："`HcclCommMemReg` has no matching HCCL unregister API"（`hccl.hpp:166-167`），所以已注册的段保活到进程退出、进池复用。这也解释了 Python `destroy()` 的 docstring："Registered allocations stay in the pool until process exit; destroy() does not free HBM"（`mega/__init__.py:13`）——它只是解引用 owner。
2. **租约复用**：`acquire` 在池里找"未租出 + 组名/comm/rank/世界大小一致 + 容量够"的最小段复用，复用前清零（`hccl.hpp:59-75`）。
3. **多 case 顺序共享**：注册段不能无限多，测试先算出所有用例里最大的 `buffer_size`，**注册一次** base，之后每个 case 以 `base=base` 复用并重新切视图（`test_mega_moe.py:249-258`，Python 断言 `num_bytes <= base.buffer.nbytes`，`mega/__init__.py:51-55`）。docstring 同时警告："Allocation order and buffer/handle owner lifetimes must match across ranks"（`:14`）、视图是借来的内存（"Data views borrow memory"，`:16`）。

### 3.4 Workspace：缓冲区里的布局与 13 个视图

`layout::Workspace`（`deep_gemm/include/deep_gemm/layout/mega_moe.hpp:85-250`）用一个 `next_offset` 游标按声明顺序排布所有区域，每段由 `Buffer<dtype>` 记录"偏移 + 步长"（`:50-69`），最后 `get_num_bytes()` 按 **2 MiB 对齐**（HCCL 注册粒度，`:246-249`）。区域从上到下：`sync`（grid 同步与 rank barrier，`:134-140`）→ `metadata`（专家计数、接收计数、token 元数据等，`:142-163`）→ `input`（本 rank 的输入四件套，`:165-173`）→ `progress`（四阶段进度计数，`kDispatched/kLinear1Ready/kLinear2Ready/kCombineSent`，`:27-33, :176-185`）→ `routed`（接收池与中间/输出缓冲，`:187-203`）→ `combine_input`、`shared_l2_acts`、`route_m_blocks`（`:206-215`）。前四类是 Kernel 内部机制（第 44/45 天），今天只关注**暴露给 Python 的 13 个视图**（`csrc/apis/mega_moe.hpp:86-107`）：

| 视图 | 形状 / dtype | 源区域 | 用途 |
|---|---|---|---|
| `x` | `[T_max, H]` fp8 | `input.acts` | 本 rank 的输入 token 激活 |
| `x_sf` | `[T_max, H/64]` int16 | `input.acts.sf` | token 的 SF（每 64 个 K 元素一对指数打包 int16） |
| `topk_idx` | `[T_max, k]` int64 | `input.topk_idx` | 路由表：token t 的第 j 个 expert 编号；`-1` 为无效路由 |
| `topk_weights` | `[T_max, k]` fp32 | `input.topk_weights` | Combine 时的加权系数 |
| `shared_l1_acts` | **= x 的别名** | 同上 | shared expert 的第一层直接读原始 token（`:91`） |
| `shared_l1_acts_sf` | `= x_sf`（shared>0 时） | 同上 | 同上 |
| `shared_l2_acts` | `[T_max, S·I]` fp8 | `shared_l2_acts` | shared 第一层经 SwiGLU 后的第二层输入 |
| `shared_l2_acts_sf` | `[pad(T,256)·S·I/64]` int16 | 同上 | 物理布局 `[M-block, K-pair, M]`，固定 BLOCK_M（`:95-97`） |
| `l1_acts` | `[ring_rows, H]` fp8 | `routed.l1_acts` | Dispatch 的**接收池**：按 expert 排好序的 token 激活 |
| `l1_acts_sf` | `[ring_rows, H/64]` int16 | `routed.l1_acts.sf` | 接收池的 SF（token-major） |
| `l2_acts` | `[ring_rows, I]` fp8 | `routed.l2_acts` | SwiGLU 之后、量化过的第二层输入 |
| `l2_acts_sf` | `[ring_rows·I/64]` int16 | `routed.l2_acts.sf` | 第二层 SF（M-block 内 K-major，`:194`） |
| `expert_recv_count` | `[R, E/R]` int32 | `metadata` | 本 rank 各本地 expert 从各源 rank 接收了多少 token（Python 属性 `mega/__init__.py:75-77`，benchmark 消费） |

三个容易误读的点：

1. **`x` 的行数是注册容量 `num_max_tokens_per_rank`，不是本次实际的 token 数**。实际 token 数在调用 `fp8_fp4_mega_moe` 时由 `y.size(0)` 给出（Host 断言 `y.size(0) <= num_max_tokens_per_rank`，`csrc/apis/mega_moe.hpp:155`）；测试也是只填前 `num_tokens` 行（`buffer.x[:num_tokens].copy_(...)`，`test_mega_moe.py:116-119`）。容量按最坏情况注册一次，多 case 复用。
2. **接收池比"全体路由"小**。全部路由的最坏行数是 `num_full_pool_rows = get_num_max_m_blocks() × BLOCK_M`，其中

   ```cpp
   get_num_max_m_blocks() = ceil_div(R · T_max · k, BLOCK_M) + E / R     // layout/mega_moe.hpp:217-219
   ```

   （第一项：所有 rank 的路由总数按 256 行一块；第二项：给每个本地 expert 预留的余量）。但 `l1_acts` 实际只开 `num_ring_rows = get_num_ring_blocks() × BLOCK_M` 行（`:130, :221-226`）：窗口数取 `H/256 ≤ 16 ? 4 : 1`，环形大小为 `min(num_max_m_blocks, 窗口数 × kNumAICores)`——**接收池是一个环形缓冲，计算消费掉一块后 Dispatch 才能复用这块**，这是计算与通信重叠的关键，细节留到第 45 天。
3. **`y` 不在缓冲区里**。`y [T, H]` bf16 由调用者 `torch.empty` 分配（`test_mega_moe.py:120`），是纯本 rank 的输出；跨 rank 归约发生在 `combine_input`（每条路由一个 BF16 slot，shared 再加一个合并 slot，`:206-208`），最后由 Kernel 汇总写进 `y`。

**手算一个贯穿全天的例子**：`R=2, E=8, T_max=128, k=2, H=I=512, S=0`（满足全部断言：`H % 256 == 0`、`I ≥ 512`、`E % R == 0`、`R·T·k = 512 < 2^24`）。先算两个核心派生量：

```text
num_max_m_blocks = ceil_div(2·128·2, 256) + 8/2 = 2 + 4 = 6
get_num_ring_blocks: H/256 = 2 ≤ 16 → 4 个窗口 → 4·32 = 128 块；min(6, 128) = 6
num_ring_rows = 6 × 256 = 1536
```

于是各视图形状：`x [128,512]`、`x_sf [128,8]`、`topk_idx [128,2]`、`topk_weights [128,2]`、`l1_acts [1536,512]`（768 KiB）、`l1_acts_sf [1536,8]`、`l2_acts [1536,512]`（768 KiB）、`l2_acts_sf [12288]`、`expert_recv_count [2,4]`。再加内部区域：`routed.output [1536·512]` bf16 ≈ 1.5 MiB、`combine_input (2+0)·128·512` bf16 = 256 KiB 等，逐字段累加（含各段间 512/128 字节对齐）得到净占用约 3.41 MiB，`get_num_bytes()` 按 2 MiB 向上对齐后**注册 4 MiB**。第六节的 B 组任务会让你完整重算一遍。

### 3.5 权重转换：interleave、NZ 与 FP4 指数补偿

`transform_weights_for_mega_moe(l1_weights, l2_weights)`（`mega/__init__.py:129-156`）处理两类输入，**routed 权重是 int8（两个 FP4 打包一字节）+ int16 SF，shared 权重是 float8_e4m3fn + int16 SF**（`:143-144`；C++ 侧对应断言 `weights.scalar_type() == (is_shared ? kFloat8_e4m3fn : kInt8)`，`csrc/apis/mega_moe.hpp:116`）。入口先校验配对关系：`w.size(-1) × (2 if int8 else 1) == sf.size(-1) × 64`（`:145-146`）——每个 int16 SF 对覆盖 64 个 K 元素（`MX_SF_DIVISOR`）。

三个变换的动机与做法：

1. **`_interleave_weights`（`:91-105`）——只有 l1 需要**。用户的 l1 权重逻辑形状是 `[·, 2I, H]`：前 I 行是 gate、后 I 行是 up（测试就是这样生成的，`test_mega_moe.py:63`）。SwiGLU 要对同一 token 的 `gate[i]` 与 `up[i]` 做逐元素运算，若两者相距 I 行，Kernel 读起来跨度巨大。变换把 gate/up **交织**进同一个 BLOCK_N 块：每块 256 行 = 前 128 行 gate + 后 128 行 up（`gran = BLOCK_N // 2`，`:99`），块数 `num_n_blocks = I / 128`。SF 随权重同步交织，再 `transpose(-1, -2)`（`:148`）得到 C++ 期望的 `[·, num_n_blocks, K/64, BLOCK_N]`。
2. **`_reshape_weights`（`:108-113`）——l2 用**。l2 权重 `[·, H, I]` 没有 gate/up 概念，只需按 BLOCK_N=256 分块为 `[·, H/256, 256, I]`，SF 同样分块 + 转置。
3. **`_transform_fp8_nz`（`:84-88`）——仅 FP8（shared）权重**。把每行的 K 维按 32 字节分组后做块内转置交织，注释："Shared FP8 weights use the NZ order consumed directly by AIC"——AIC 的 MAD 原语直接消费这种 NZ 顺序，免去搬运时的重排。FP4 权重不走这一步（int8 分支直接返回，`:105`）。
4. **`_scale_fp4_weight_sf`（`:116-126`）——仅 FP4（routed）SF**。注释："Internal dequant emits FP4 values divided by 64, so compensate once in the weight layout"——Kernel 反量化时会把 FP4 值除以 64（2⁻⁶），于是**一次性**给权重 SF 的指数加 `exponent_bias = 6` 补偿。合法性检查：指数 ≤ 248 或恰为 255（特殊值，原样保留；"Exponent 0 represents 2^-127"，`:124`）。

转换结果由 C++ `check_weights`（`csrc/apis/mega_moe.hpp:112-126`）收货并二次校验：

```text
routed (fp4)：weight [E/R, num_n_blocks, BLOCK_N, K/2]，sf [E/R, num_n_blocks, K/64, BLOCK_N]
shared (fp8)：weight [num_n_blocks, BLOCK_N, K]（无 expert 维），sf [num_n_blocks, K/64, BLOCK_N]
```

其中 l1 的 `num_n_blocks = I / (BLOCK_N/2) = I/128`、l2 的 `num_n_blocks = H / BLOCK_N = H/256`，K 分别是 H 和 I（`csrc/apis/mega_moe.hpp:180-187`）。shared 权重"omit the expert dimension and store unpacked FP8 data"（`:120` 注释）——shared expert 只有一份，没必要（也不该）挂在 routed 的 expert 维下。

**容量对比一下就有感觉**：README 基准里 `E=384、H=7168、I=3072` 的 routed l1 权重是 `[48, 3072/128, 256, 7168/2]` int8 = 48×24×256×3584 ≈ 1.06 GB（fp4 打包省一半），shared l1 是 `[24, 256, 7168]` fp8 ≈ 44 MB——**权重转换是纯 Host 端的内存重排，只在初始化时做一次**，这就是它值得专门提供一个公开 API 的原因。

### 3.6 执行接口 `fp8_fp4_mega_moe`：检查与启动

Python 包装（`mega/__init__.py:159-184`）几乎只做搬运：断言 `sym_buffer.handle` 存活（`:170`），把 `sym_buffer` 里保存的配置（`num_max_tokens_per_rank/num_experts/num_topk/hidden/intermediate_hidden/num_shared_experts`）连同 `handle.buffer_ptrs`、`group.rank()` 一起传给 `_C`（`:171-183`）——**SymmBuffer 对象是配置的唯一权威来源**，避免调用参数与缓冲区布局不一致。

C++ Host（`csrc/apis/mega_moe.hpp:128-204`）的检查清单，按代码顺序：

1. **硬约束**：`recipe == (1, 1, 32)` 且 `activation == "swiglu"` 且 `fast_math == true`（`:146-148`）——当前唯一支持的组合（Python 侧同样断言，`mega/__init__.py:28-29`）。
2. **`num_ranks` 从 `sym_buffer_ptrs.size()` 推出**（`:151`），重建 `workspace`（`:153-154`）。
3. **`y`**：二维、`y.size(0) ≤ T_max`、`y.size(1) == H`、bf16、contiguous、NPU（`:155, :167-168`）。
4. **`sym_buffer`**：int8 一维 contiguous、字节数 ≥ `workspace.get_num_bytes()`、与 `y` 同设备，并且 `sym_buffer_ptrs[rank_idx] == sym_buffer.data_ptr()`（`:163-166`）——**传进来的必须就是当初注册、被伙伴们记录了地址的那块缓冲**，换一块逻辑拷贝都不行，否则远端按旧地址读写。
5. **shared 权重的有无必须与 `num_shared_experts > 0` 一致**（`:156-157`）；`cumulative_local_expert_recv_stats`（可选）：int32、长度 `E/R`、contiguous、同设备（`:172-177`），用于**跨多次 launch 累计**每个本地 expert 接收的 token 数。
6. **四组权重**过 `check_weights`（`:180-188`）；`activation_clamp` 缺省 +∞、要求 ≥ 0，并**转成 BF16 再传**（`:190-191, :202`）——Kernel 里的 clamp 是 BF16 精度，测试特意先把 clamp 转成 BF16 标量再传入以对齐语义（`test_mega_moe.py:61-62`）。
7. `runtime->get_num_sms() == kNumAICores`（`:159`）——MegaMoE 要求启用全部 32 个 AI Core。

`launch_mega_moe`（`csrc/jit_kernels/mega_moe.hpp:24-51`）把 `MegaMoEArgs`（`:11-22`：workspace、buffer 指针、`sym_buffer_ptrs`、y、stats、四组权重、rank_idx、num_tokens、clamp）变成一次启动：

- 用 `std::format` 生成一段实例化代码，把 7 个配置写进 `mega_moe_impl<R, E, S, T_max, k, H, I>` 的模板实参（`:27-36`）——**每个配置组合一个专属 Kernel**，布局常量全部编译期展开；
- `jit->compile` 后，先在当前流上把 `workspace.progress` 区异步清零（`:44-46`，每次 launch 重置进度计数），再以 `num_blocks = kNumAICores` 启动（`:48-50`），Kernel 参数里带一个构造好的 `layout::SymBuffer`（地址表，见 3.3 节）。

### 3.7 测试视角：`MegaMoECase` 与概念定位

`tests/test_mega_moe.py` 用一个 `MegaMoECase` 类同时持有"fused"和"baseline"两套跑法（`:50-163`）：

- **数据准备**（`:58-74`）：先建 `deep_gemm.SymmBuffer`；routed 权重用 `make_npu_fp4((E/R, 2I, H))` 与 `(E/R, H, I)` 生成（int8 打包 + int16 SF，`:63-64`），立刻 `transform_weights_for_mega_moe` 转换（`:65-66`）；`num_shared_experts > 0` 时 shared 权重用 `make_npu_fp8((2SI, H))` 与 `(H, SI)`（`S` 个 shared expert 的中间维拼在一起，`:69-74`）。
- **baseline 依赖**（`:76-100`）：`import deep_ep`、`import tilelang`，并把 expert 对齐量取 `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()`（`:92`，保证 Dispatch、grouped GEMM、SwiGLU 对 expert padding 口径一致）；`EPBuffer` 用独立的 `dist.new_group(backend='hccl')` 通信组（`:96-100`）。
- **输入填充**（`create_inputs`，`:106-120`）：`make_npu_fp8((T, H))` 生成激活 + SF，`make_topk` 用 `torch.topk` + `softmax` 生成路由（`:39-47`），然后 `buffer.x[:T].copy_(...)` 等逐个填进对称缓冲区前 T 行；`invalid_routes=True` 时人为制造坏路由——`topk_idx[::11, -1]` 复制成第 0 个专家（重复路由），`topk_idx[::13, -1] = -1` 且权重置 0（无效路由，`:112-115`）。
- **`run_fused`（`:122-128`）**：一行调 `deep_gemm.fp8_fp4_mega_moe(y, l1', l2', buffer, shared..., stats, activation_clamp=...)`。
- **`check_correctness`（`:166-200`）**：两组输入交替跑 `num_runs + 2` 次；对每组输入先用 `run_baseline` 求参考并比对（整体 diff 与逐 token diff，阈值 `REFERENCE_DIFF_THRESHOLD = 5e-6`——注释解释了为什么这么严：DeepEP 把各 rank 的部分和按 BF16 舍入，MegaMoE 的 Combine 全程 FP32，`:18-19`）；之后重复运行要求**逐位相等**（`torch.equal(output.view(torch.int16), expected_bits)`，`:199`，确定性验证），并核对 `stats` 的**跨调用累加**等于独立统计的接收计数（含重复路由、剔除无效路由，`:189-200`）。
- **多 rank 组织**（`test`，`:246-282`）：`torch.multiprocessing.spawn` 起 `--num-processes`（默认 8）个进程，`init_dist` 建组；先对全部配置算 `buffer_size` 注册一个最大容量的 base，再逐 case 以 `base` 复用（`:249-258`）；CLI 默认形状 `H=[5120,7168]、I=[2304,3072]、E=384、k=6、T=[64,256,4096,16384]、shared=1、clamp=7.0`（`:292-300`），`--num-tokens 0` 专门测试空输入（`:296`）。注意 `--skip-baseline`、`--num-processes 1` 或 `--num-correctness-tests 0` 都会**连带跳过正确性**（`:309`）——没有 baseline 就没有参考答案。

学习计划要求"找到 EP、top-k、expert、shared expert 的位置"，汇总成一张定位表：

| 概念 | 关键代码位置 | 一句话说明 |
|---|---|---|
| EP（Expert Parallel） | `num_ranks`/`rank_idx`（`mega/__init__.py:20-21`）；`num_experts % num_ranks == 0`（`csrc/apis/mega_moe.hpp:31`）；`expert_recv_count [R, E/R]`（`layout/mega_moe.hpp:238-240`）；`peer_ptrs`（`hccl.hpp:141-163`） | expert 均分到各 rank；每 rank 只持有并计算 E/R 个本地 expert，token 按需跨 rank 搬运 |
| top-k | `num_topk ≤ 32`（`:33`）；`topk_idx [T,k] int64`、`topk_weights [T,k] fp32`（`layout/mega_moe.hpp:167-172`）；`make_topk`（`test:39-47`） | 每 token 路由到 k 个 expert，Combine 按 softmax 权重加权求和 |
| expert | `l1/l2_weights [E/R, ...]`（`test:63-64`）；`check_weights`（`csrc/apis/mega_moe.hpp:112-126`）；`worker/total_expert_count`、`expert_recv_count`（`layout:155-159`） | routed expert 的两层 FFN 权重，int8 打包的 FP4 |
| shared expert | `num_shared_experts ≤ 2`（`:34`）；`shared_l1/l2_weights`（`test:69-74`）；`shared_l1_acts = x`（`csrc/apis/mega_moe.hpp:91`）；`combine_input` 多出的一个 slot（`layout:206-208`） | 不经路由、对全部 token 计算；权重 FP8、无 expert 维；输出在 Combine 一并合并 |
| 无效路由 | `topk_idx = -1`（`test:113-115`）；参考统计里 `topk_idx[(topk_idx >= 0) & (topk_idx < E)]`（`:191`） | `-1` 表示该路由不参与计算与计数 |

### 3.8 当天产出·参考图：MegaMoE API 与数据准备流程图

```text
用户侧                          deep_gemm 内部                              Kernel / 对等 rank
──────                          ────────────                              ─────────────────
torch.distributed（HCCL 进程组，R 个 rank）
      │
      ▼
① SymmBuffer(group, E, T_max, k, H, I, S[, base])                mega/__init__.py:20
      ├─ _C.get_symm_buffer_size_for_mega_moe ─► layout::Workspace 算 num_bytes（2 MiB 对齐）
      │                                           并返回切片闭包                    csrc/apis/mega_moe.hpp:72-110
      ├─ _C.SymmBuffer(rank, R, hccl_comm_name, num_bytes)
      │     └─ HCCLSymmetricBuffer::acquire（池中租约或新建）       csrc/apis/mega_moe.hpp:45-62
      │          ├─ aclrtMalloc + HcclCommMemReg（tag: deep_gemm_mega_moe_N）
      │          ├─ UBMEM 通道（AIV 引擎，rank graph 第 1 层）      csrc/backends/hccl.hpp:101-139
      │          └─ HcclChannelGetRemoteMems → buffer_ptrs[所有 rank]  ──► 伙伴 rank 互换地址
      ├─ synchronize + barrier
      └─ 切出 13 个视图（x/x_sf/topk_idx/topk_weights/shared×4/routed×4/recv_count）
      │
      ▼
② transform_weights_for_mega_moe((w1,sf1),(w2,sf2))               mega/__init__.py:129
      ├─ l1（routed fp4）：gate/up 交织 → [E/R, I/128, 256, H/2]，SF 同步交织 + 转置
      ├─ l2（routed fp4）：BLOCK_N 分块 → [E/R, H/256, 256, I/2]
      ├─ shared（fp8）：分块 + NZ 重排，无 expert 维
      └─ fp4 SF：指数 +6（补偿 Kernel 反量化的 ÷64）
      │
      ▼
③ 填输入：buffer.x[:T]、x_sf[:T]、topk_idx[:T]、topk_weights[:T]   test_mega_moe.py:106-120
      │   （y = torch.empty([T, H], bf16)——本 rank 私有，不在对称区）
      ▼
④ fp8_fp4_mega_moe(y, l1', l2', buffer, shared_l1', shared_l2', stats?, clamp?)
      ├─ Host 检查：recipe=(1,1,32)/swiglu/fast_math、y 与 buffer 一致性（ptr 相等）、
      │             check_weights × 4、stats [E/R] int32                csrc/apis/mega_moe.hpp:128-204
      └─ launch_mega_moe：生成 mega_moe_impl<R,E,S,T,k,H,I> → jit 编译
                          → progress 区清零 → 32 AIC 启动             csrc/jit_kernels/mega_moe.hpp:24-51
                                         │
                                         ▼
                Dispatch ─► GEMM1 ─► SwiGLU ─► GEMM2 ─► Combine ─► y
              （Kernel 内部经 SymBuffer.map 直接读写远端缓冲；第 44/45 天深入）
```

画自己的版本时重点自查：缓冲区注册发生在哪个时机、为什么权重和 `y` 不需要对称性、13 个视图各自写给谁读、以及配置参数如何一路传到模板实参。

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `deep_gemm/mega/__init__.py` | `SymmBuffer` `:9-81`、`_transform_fp8_nz` `:84-88`、`_interleave_weights` `:91-105`、`_reshape_weights` `:108-113`、`_scale_fp4_weight_sf` `:116-126`、`transform_weights_for_mega_moe` `:129-156`、`fp8_fp4_mega_moe` `:159-184` | 今天的主战场：Python 侧全部对外逻辑 |
| 2 | `csrc/apis/mega_moe.hpp` | `get_workspace` `:27-42`、`SymmBuffer` `:44-62`、`slice_buffer` `:64-70`、`get_symm_buffer_size_for_mega_moe` `:72-110`、`check_weights` `:112-126`、`fp8_fp4_mega_moe` `:128-204`、`register_apis` `:206-225` | Host 检查清单、视图切法、pybind 签名 |
| 3 | `deep_gemm/include/deep_gemm/layout/mega_moe.hpp` | `BLOCK_M/BLOCK_N` `:12`、`kNumBufferAlignmentBytes` `:17`、`kCountArrivalBit` `:23-24`、`ProgressStage` `:27-33`、`Config` `:36-47`、`Buffer` `:50-69`、`Workspace` `:85-250` | 缓冲区布局的权威定义；今天重点 `Config`、`input/routed` 区域、`get_num_max_m_blocks` `:217-219`、`get_num_ring_blocks` `:221-226`、`get_num_bytes` `:246-249` |
| 4 | `csrc/backends/hccl.hpp` | `HCCLSymmetricBuffer::acquire` `:49-83`、构造函数 `:89-164`、`make_peer_descriptor` `:173-208` | 注册四步、租约池、UBMEM 通道；`:166-171` 的池生命周期注释必读 |
| 5 | `deep_gemm/include/deep_gemm/layout/sym_buffer.hpp` | `SymBuffer` `:12-46`（`map` `:36-46`） | Device 侧如何把本 rank 指针平移到远端 |
| 6 | `csrc/jit_kernels/mega_moe.hpp` | `MegaMoEArgs` `:11-22`、`launch_mega_moe` `:24-51` | 模板实参生成、progress 清零、32 AIC 启动 |
| 7 | `tests/test_mega_moe.py` | `MegaMoECase` `:50-163`、`create_inputs` `:106-120`、`run_fused` `:122-128`、`run_baseline` `:130-156`、`check_correctness` `:166-200`、`test` `:246-282`、CLI `:285-322` | 测试如何当"接口用法手册"读 |
| 8 | `deep_gemm/__init__.py` | mega 导入 `:137-141` | 公开符号的唯一出口 |
| 9 | `csrc/python_api.cpp` | `mega_moe::register_apis` `:14, :40` | 模块注册位置 |
| 10 | `README.md` | MegaMoE 性能节 `:160-173`、SF 格式说明 `:56` | 基准口径（EP8、top-k=6、1 shared expert） |
| 11 | `deep_gemm/testing/envs.py` | `init_dist` `:28-52` | 测试的分布式初始化（hccl backend、tcp init） |

**`deep_gemm/mega/__init__.py` 逐段导读**：

- **`:10-18`**：`SymmBuffer` 的 docstring 本身就是接口契约——"Shape and capacity attributes describe the C++ layout and must not be modified"（配置属性只是 C++ 布局的描述，改了无效）、注册段进程存活、destroy 不释放、各 rank 分配顺序必须一致、`base` 用于跨配置共享、视图借用内存。
- **`:40-49`**：`get_symm_buffer_size_for_mega_moe` 返回 `(num_bytes, 切片闭包)`；新建路径 `_C.SymmBuffer(...)` 之后 `synchronize` + `barrier`——注册是集体操作，任何 rank 未完成前别人不能开始切片。
- **`:57-60`**：13 个视图的解包顺序，与 3.4 节表格一一对应。
- **`:84-126`**：三个纯 torch 的布局工具函数，见 3.5 节；注意 `_interleave_weights` 对 fp4（int8）与 fp8 的分支（fp8 视图成 uint8 处理并走 NZ，fp4 直接返回，`:105`）。
- **`:159-184`**：执行包装。参数默认值即当前唯一合法组合：`recipe=(1,1,32)`、`activation='swiglu'`、`fast_math=True`。

**`csrc/apis/mega_moe.hpp` 逐段导读**：

- **`:27-42`（`get_workspace`）**：所有容量断言集中地——`num_ranks < 256`（通信 Kernel 用 8 bit 编码 rank，`layout/sym_buffer.hpp:9-10`）、`E % R == 0`、`num_topk ≤ 32`、`S ≤ 2`、`H % 256 == 0`、`I ≥ 512 且 % 256 == 0`、`R·T·k < 2^24`（打包计数器 24 bit 数值 + 8 bit 到达数，`layout/mega_moe.hpp:22-24`）。
- **`:44-70`**：C++ `SymmBuffer` 持有"根张量"（拥有 HCCL 租约），视图全部借用其内存；`slice_buffer` 用 `torch::from_blob` 在 `buffer + offset` 处按给定 dtype 造视图，不发生任何拷贝。
- **`:79-108`**：切片闭包。对照 `layout/mega_moe.hpp` 的区域声明逐个核形状；注意 `shared_l1_acts = x`（别名，`:91`）与 `shared_l2_acts_sf` 的物理布局注释（`:95`）。
- **`:128-204`**：Host 检查清单，见 3.6 节；`:192-203` 是唯一的启动调用点，`MegaMoEArgs` 聚合了 Kernel 需要的一切。
- **`:206-225`**：pybind 绑定。`_mega_moe_block_n`、`SymmBuffer` 类（`buffer`/`buffer_ptrs` 只读）、`get_symm_buffer_size_for_mega_moe`、`fp8_fp4_mega_moe` 四个名字构成 C++ 侧全部对外面。

## 五、实践步骤

### A 组：有 Ascend 环境（至少 2 卡；baseline 另需 `deep_ep` 与 `tilelang`）

1. **小配置跑通正确性**（2 卡即可，形状须满足 3.6 节断言：`H`、`I` 是 256 的倍数、`I ≥ 512`、`E % 卡数 == 0`）：

   ```bash
   python tests/test_mega_moe.py --num-processes 2 \
       --hidden 512 --intermediate-hidden 512 --num-experts 8 \
       --num-topk 2 --num-tokens 64 --num-shared-experts 1 --skip-prof
   ```

   对照输出里的每个 case 打印（`H/I | E | topk | shared | tokens`，`test_mega_moe.py:262-263`）确认参数生效。若没有 `deep_ep`，加 `--skip-baseline` 能跑通 fused 路径，但注意此时正确性也会被跳过（`:309`）——没有 baseline 就没有参考。

2. **观察 JIT 编译与模板实参**：

   ```bash
   export DG_JIT_PRINT_COMPILER_COMMAND=1
   python tests/test_mega_moe.py --num-processes 2 --hidden 512 --intermediate-hidden 512 \
       --num-experts 8 --num-tokens 64 --skip-prof --skip-baseline
   ```

   在打印的编译命令里找到生成的源文件，确认 `mega_moe_impl<2, 8, 1, 64, 2, 512, 512>` 这串模板实参与命令行参数一一对应；再换 `--num-tokens 128` 重跑，观察新配置触发一次新的编译（缓存命中除外）。

3. **视图与容量核对**：在第 1 步跑通的前提下，写一个小脚本（模仿 `test_mega_moe.py` 的 spawn + `init_dist`）打印 `buffer.num_bytes` 与 13 个视图的 `shape/dtype`，与 3.4 节手算表核对；再打印 `buffer.handle.buffer_ptrs`，确认本 rank 地址 == `buffer.data_ptr()`（Host 断言 `:166` 的直观版）。

4. **无效路由与确定性**：把 `MegaMoECase.create_inputs` 里的 `invalid_routes=False` 临时改为 `True`（跑完记得还原，勿提交），确认测试仍然通过——无效路由与重复路由是被接口明确支持的场景；同时观察 `check_correctness` 的逐位相等断言（`:199`）没有触发。

5. **（进阶，8 卡）复现 README 口径**：

   ```bash
   python tests/test_mega_moe.py --num-processes 8 \
       --hidden 5120 --intermediate-hidden 2304 --num-experts 96 \
       --num-topk 6 --num-tokens 64 --num-shared-experts 1 --num-correctness-tests 1
   ```

   对照 `README.md:166` 的第一行（96 experts、H=5120、I=2304、64 tokens、138.7 us）感受基准规模。

### B 组：无 Ascend 环境（静态任务）

1. **容量手算**（对应 3.4 节）：配置 `R=2, E=8, T_max=128, k=2, H=I=512, S=0`。依次计算 `num_max_m_blocks`、`get_num_ring_blocks`、`num_ring_rows`，写出 13 个视图的形状与 dtype；再算数据大头——input 区（约 69 KB）、routed 的 l1/l2 激活（各约 792 KiB）、`routed.output`（1.5 MiB）与 `combine_input`（256 KiB），合计约 3.38 MiB，补上 `route_m_blocks` 及同步/元数据/进度区（约 39 KB，含段间对齐）后净占用约 3.41 MiB，`get_num_bytes()` 按 2 MiB 向上对齐后注册 4 MiB。然后换 `H=7168` 重算 `get_num_ring_blocks`（提示：`H/256 = 28 > 16`，窗口数变为 1，ring 块数 = `min(num_max_m_blocks, 32)`）。
2. **权重变换 CPU 复现**：`_interleave_weights`、`_reshape_weights`、`_scale_fp4_weight_sf` 三个函数只依赖 torch 原生算子。把它们（连同 `BLOCK_N = 256`）抄进一个独立脚本，在 CPU 上构造小张量验证：
   - `t = torch.arange(4*512*8).float().reshape(4, 512, 8)` 走 `_interleave_weights` 后形状应为 `[4, 2, 256, 8]`，并检查 `result[0, 0, 0, 0] == t[0, 0, 0]`（gate 半区）与 `result[0, 0, 128, 0] == t[0, 256, 0]`（up 半区挪进了同一块）；
   - 对 int8 的 fp4 模拟张量走 `_reshape_weights`，对照 `check_weights` 的期望形状；
   - `_scale_fp4_weight_sf`：输入指数 100 → 输出 106；构造 249 应抛 `ValueError`（>248 且 ≠255）。
3. **调用链标注**：合上教程，从 `deep_gemm.fp8_fp4_mega_moe` 出发，在纸上写出到达 `jit->launch` 经过的每个文件与函数（共 5 站），标注每站的输入、检查内容与产出；再对照 3.2 节的链图修正。
4. **概念定位**：不看 3.7 节的表，独立在 `tests/test_mega_moe.py`、`mega/__init__.py`、`csrc/apis/mega_moe.hpp` 中找到 EP、top-k、expert、shared expert、无效路由各至少两处代码位置，注明它们分别出现在"接口、布局、测试"中的哪一层。
5. **画流程图**（当天产出）：不看 3.8 节，独立画出 MegaMoE API 与数据准备流程图，再对照修正；重点自查注册四步、视图归属、权重两套分支（fp4 routed / fp8 shared）以及配置参数到模板实参的流向。

## 六、当天产出

**MegaMoE API 与数据准备流程图**——以 3.8 节的参考版为答案，自己合上教程画一版，至少覆盖：三个公开符号及调用链五站、SymmBuffer 注册四步与"租约池不释放 HBM"的约束、13 个视图的归属区域与谁读谁写（尤其 `shared_l1_acts = x` 别名与 routed 接收池的环形含义）、权重转换的 gate/up 交织 + FP4 指数 +6 补偿、Host 检查清单要点（recipe/ptr 相等/stats 累计）、以及 EP/top-k/expert/shared expert 在代码中的落点。写完后把画不出的环节标出来，回到第四节对应站点重读。有环境的同学附上 A 组第 2、3 步的编译输出与视图核对记录。

## 七、自测题

1. MegaMoE 融合了哪五段计算？baseline 分别用什么组件拼装？
   > Dispatch → Grouped GEMM 1 → SwiGLU → Grouped GEMM 2 → Combine。baseline 用 deep_ep 的 dispatch/combine、`m_grouped_fp8_fp4_gemm_nt_contiguous` × 2（psum layout）、TileLang `swiglu_apply_weight_to_fp8`；shared expert 走 `fp8_gemm_nt` × 2 并经 combine 的 `bias` 参数合并（`tests/test_mega_moe.py:130-156`）。
2. Python 侧有哪三个公开符号？分别定义在哪里？
   > `SymmBuffer`、`transform_weights_for_mega_moe`、`fp8_fp4_mega_moe`，均在 `deep_gemm/mega/__init__.py`，经 `deep_gemm/__init__.py:137-141` 导出。
3. 为什么输入和中间张量要放进 SymmBuffer，而权重和 `y` 不用？
   > Dispatch/Combine 由 Device Kernel 直接跨 rank 读写，要求参与搬运的区域位于 HCCL 注册的 UBMEM 直访区且各 rank 互知地址（`peer_ptrs`）；权重每个 rank 只持有本地 E/R 个 expert、`y` 是本 rank 输出，均为 rank 私有数据（3.3 节）。
4. 描述 HCCL 注册的四步。为什么 `destroy()` 之后 HBM 没有被释放？
   > `aclrtMalloc` → `HcclCommMemReg`（tag `deep_gemm_mega_moe_N`）→ 为每个伙伴建 UBMEM 通道（AIV 引擎，rank graph 第 1 层）→ `HcclChannelGetRemoteMems` 填 `peer_ptrs`（`csrc/backends/hccl.hpp:101-163`）。`HcclCommMemReg` 没有反注册 API，已注册段进池保活到进程退出（`:166-171`）；`destroy()` 只是解引用 owner（`mega/__init__.py:13, :62-73`）。
5. `num_max_m_blocks` 的公式是什么？`l1_acts` 为什么可以比最大池小？
   > `ceil_div(R·T_max·k, 256) + E/R`（`layout/mega_moe.hpp:217-219`）。`l1_acts` 是环形接收池，大小 `min(num_max_m_blocks, 窗口数·32)` 块：计算消费与 Dispatch 接收重叠，消费过的块可复用（`:221-226`；窗口数 `H/256 ≤ 16 ? 4 : 1`）。
6. l1 权重为什么要做 gate/up 交织？变换前后形状是什么？
   > SwiGLU 需要同 token 的 gate/up 相邻读取；交织把 `[·, 2I, K]` 的前半 gate 与后半 up 交错进同一 BLOCK_N 块（每块 128+128 行），得到 `[E/R, I/128, 256, K/2]`（fp4 打包；fp8 为 K 不折半），SF 同步交织并转置成 `[·, I/128, K/64, 256]`（`mega/__init__.py:91-105, :147-148`）。
7. FP4 权重的 SF 指数为什么要加 6？合法范围是什么？
   > Kernel 内部反量化把 FP4 值除以 64（2⁻⁶），在权重布局上一次性给指数 +6 补偿（`_scale_fp4_weight_sf`，`:116-126`）；指数须 ≤ 248 或恰为 255（特殊值原样保留）。
8. `topk_idx` 中的 `-1` 表示什么？`cumulative_local_expert_recv_stats` 统计什么、什么时候用？
   > 无效路由（不参与计算与计数），测试用 `topk_idx[::13, -1] = -1` 且权重置 0 构造（`test_mega_moe.py:113-115`），参考统计用 `(topk_idx >= 0) & (topk_idx < E)` 过滤（`:191`）。stats 是 int32 `[E/R]`，**跨多次 launch 累计**每个本地 expert 接收的 token 数（含重复路由），测试验证其累积等于独立统计（`:166-200`）；上游调度可据此做负载均衡。

---

明天（第 44 天）进入 MegaMoE 计算主线：阅读 `deep_gemm/include/deep_gemm/mega_moe.hpp`、`scheduler/mega_moe.hpp` 与 `layout/mega_moe.hpp` 的执行细节，看两个 Grouped GEMM 与 SwiGLU 如何在一个 Kernel 里衔接、中间张量形状如何流转，把今天"只看外壳"的接口图补上内核。
