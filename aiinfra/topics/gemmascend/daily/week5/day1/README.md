# 第 5 周 · 第 1 天（总第 29 天）：理解测试数据生成

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

前四周我们读完了一条完整的执行链：Python API → C++ Host → JIT → BF16 Kernel 的调度、搬运、MAD、流水线和 Epilogue。本周换一个视角——**怎么证明这条链是对的**。今天从 `tests/generators.py`（825 行）入手，它是所有 GEMM 测试用例的出生地：`tests/test_bf16_gemm.py` 等 CLI 脚本只是选择器，真正决定"测什么形状、数据长什么样、参考结果怎么算"的全部在这一个文件里。要回答四个问题：**一个测试用例由哪些字段组合而成？**（`GemmTestDesc` 的 shape × major × recipe × grouped layout）**major、recipe、组布局如何变成具体的张量字节？**（两条量化路径与 psum 前缀和）**非对齐和 strided 场景在哪里定义？**（五个枚举器）**`GemmTestDesc.generate()` 的六元组输出被谁消费？**（`common.py` 的"先编译后验证"两阶段）。当天产出是一张**测试用例字段说明表**（参考答案见 3.11 节）。它也是本周后续的地基：第 30 天的正确性验证流程、第 32～34 天的低精度路径，全都以今天生成的数据为输入。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出"CLI 选择器 → `run_gemm_test` → `enumerate_gemm_tests` → `GemmTestDesc` 列表 → `generate()` → 六元组"的完整流水线，并说出 `tests/test_bf16_gemm.py:25-47` 的每个选择器分别路由到 `tests/generators.py` 的哪个枚举器（`generators.py:652-679`）；
2. 对 `GemmTestDesc`（`generators.py:249-269`）的 17 个字段逐一说出作用与消费位置，并解释 `m/n/k`、`a_recipe`、`b_recipe`、`has_sf`、`get_available_trans_ab()` 五个 property 的派生规则（`:271-298`）；
3. 解释 **major 的存储语义**——major 维即"单位步长的内层维"：K-major 的 A `(m, k)` 是行主序，MN-major 的 A 的 stride 是 `(1, m)`；并读懂"先生成 major 内层连续的存储，再转成逻辑视图"这一贯穿全部生成代码的技巧（`storage_shape` 交换、`transpose().contiguous().transpose()` 三连）；
4. 解释 `generate_operand_with_major()`（`:169-191`）的三路分派：`device='meta'`（只建 shape/stride，供并行编译）、bf16（`quantize_with_major` 的转换分支）、fp8/fp4（`generate_random_quantized_with_major`）；
5. 手算 `ceil_to_ue8m0()`（`:27-31`）——对 5.0、4.0、2/448 写出位运算过程与结果；解释 `sf_div` 为何取 448.0（E4M3FN 最大值）或 6.0（FP4 最大值）、granularity padding、`sf_full` 的 `repeat_interleave` 广播，以及 FP4 双 nibble 沿 major 维打包的字节格式（`:34-50`）；
6. 解释 GEMM 主测试为什么用"**随机码字 + 精确解码**"而不是"随机浮点 + 量化"来造 FP8/FP4 数据：码字覆盖全部 254 个非 NaN E4M3 编码（`:148-153`）、FP4 用 256 项字节查找表（`:98-102, :159-164`）、SF 只从 2 的幂小集合抽取（`:138-144`），参考结果由 `_scale_quantized_reference`（`:105-121`）精确重建，无舍入误差；
7. 用 `generate_psum_layout()`（`:212-220`）手工推演分组布局：每组大小 ~ U(0.7, 1.3) × expected、组起点按 256 对齐；区分 `iter_psum_slice`（有效区间）与 `iter_psum_sep`（padding 区间，可带 gran 换算，`:223-235`）；解释 `zero_operand_slices_`（数据清零，`:194-205`）与"SF padding 填随机垃圾"（`:342-346, :384-389`）各自在测什么；
8. 找到非对齐场景的 shape 列表（`unaligned-mn`：`:491-516`；`unaligned-k`：`:519-541`）与 FP4 的偶数维约束；说明 strided 场景的现状——`make_strided`（`:238-246`）与 `strided` 字段当前无调用方/消费方，实际的非连续输入来自 `gemm_dg` 的转置视图（`common.py:42-46, 64-65`）；
9. 跟踪 `generate()` 六元组 `(shape, a, b, c, d_ref, (g_layout, cmp_spans))` 在 `common.py:131-237` 的两阶段消费：meta 预编译（`:153-178`）与真实运行验证（`:180-236`），并解释 `d` 的 NaN/随机数预填充策略（`:190`）与 `cmp_spans` 如何收紧比较范围（`utils.py:46-58`）。

**当天产出：测试用例字段说明表**（模板见第六节，参考答案见 3.11 节）。

## 二、预备知识

- **第 2 天**：Row-major / Column-major、stride、contiguous——今天的 major 语义是它的直接延伸。
- **第 4 天**：FP8/FP4 的表示范围（E4M3FN 最大 ±448、E2M1 值集 {0, ±0.5, …, ±6}）、Scaling Factor 与量化粒度。
- **第 8～9 天**：仓库地图与 `tests/test_bf16_gemm.py` 的命令行参数（`normal`/`unaligned-k`/`--skip-prof`）。
- **第 11 天**：`GemmDesc` 的字段来源——今天生成的张量（shape/stride/dtype/grouped_layout）正是 `get_gemm_desc()`（`csrc/apis/gemm.hpp:60-193`）的输入；`get_major_type_ab` 从 stride 推断 major（`gemm.hpp:79-80`）。
- **第 17/19/22 天**：`actual_m/n/k` 与边界 Tile——非对齐 shape 测试的就是这些边界处理代码。
- **第 26 天**：m-grouped 组尾 padding（`get_actual_m` 恒返回 BLOCK_M）、k-grouped 空组由 `aiv_zero_empty_groups` 写零、组起点 256 对齐——今天的 `ensure_zero_padding`、SF 垃圾填充、`d` 的 NaN 预填充都是与它们配套的测试手段。
- **Python 语法**：`dataclass`（字段与 `replace()`）、`enum`、生成器函数（`yield`）、`itertools.product`、PyTorch 的 `transpose`/`as_strided`/`view(dtype)`。

另注：本文只讲**生成侧**。参考结果如何被比较（余弦相似度、ACLNN 交叉验证、随机舍入）是第 30 天的内容；SF 布局转换（`transform_sf_into_required_layout`）是第 33 天的内容，今天只讲到"生成的 SF 是 fp32、逻辑形状为块索引"为止。

## 三、详细讲解

### 3.1 全景：一条测试用例的流水线

先建立整体图景。以 `python tests/test_bf16_gemm.py unaligned-k --skip-prof` 为例，完整数据流是：

```text
tests/test_bf16_gemm.py（命令行选择器，:25-47）
   │  test_bf16_gemm(GemmType.Normal, unaligned_k=True, skip_prof=True)
   ▼
common.run_gemm_test(bf16, bf16, …)（common.py:131）
   │  tests = list(enumerate_gemm_tests(…))（:149 → generators.py:652）
   ▼
[GemmTestDesc × 20] ←────────── 场景枚举层：shape × major × recipe × grouped layout
   │
   ├─ 阶段 1 · 预编译（common.py:153-178）
   │     desc.generate(device='meta')（:157）──► 只含 shape/stride/dtype 的元张量
   │     par_compile(kernels)（:175）──► dry_run 模式下多线程触发 JIT 编译，不启动内核
   │
   └─ 阶段 2 · 运行与验证（common.py:180-236）
         desc.generate()（:189）──► 真实 NPU 数据 + 参考结果
              │  (shape, a, b, c, d_ref, (g_layout, cmp_spans))
              ├─► gemm_dg()（:199）──► deep_gemm.bf16_gemm_nt(a, b, d, …) 等绑定
              ├─► assert_equal(d, d_ref, cmp_spans)（:201，utils.py:46-58）
              ├─► gemm_aclnn()（:208-213，仅 Normal 的 ACLNN 交叉验证）
              └─► print_perf()（:230-236）
```

`generators.py` 内部相应地分三层：

| 层 | 代码位置 | 职责 |
|---|---|---|
| 枚举层 | `_enumerate_normal`（`:544`）、`_enumerate_normal_unaligned_mn/k`（`:491/:519`）、`_enumerate_normal_strided`（`:466`）、`_enumerate_recipe`（`:441`）、`_enumerate_m/k_grouped_contiguous`（`:579/:616`） | 把一个"场景"展开成一批 `GemmTestDesc` |
| 描述层 | `GemmTestDesc`（`:249-438`） | 一个用例的全部参数 + `generate()` |
| 数据层 | `generate_operand_with_major`（`:169`）、`quantize_with_major`（`:61`）、`generate_random_quantized_with_major`（`:124`）、`generate_psum_layout`（`:212`）等 | 把参数变成具体张量与参考结果 |

两个贯穿全天的设计事实：其一，**确定性**——`generate()` 每次调用都先 `reset_seed()`（`:429-430`），把 Python `random`、`torch` CPU 和 NPU 生成器都设回种子 0，因此同一个 desc 反复 generate 得到完全相同的数据（阶段 2 每个 desc 最多调用 20 次 generate，数值不变，只有 alpha/bias 在变）。其二，**meta 与真实两条腿**——同一份 `_generate_*` 代码靠 `device` 参数分叉：`'meta'` 只构造 shape/stride（供编译），`'npu'` 生成真数据并算参考结果。

### 3.2 组合的起点：GemmTestDesc 的字段

`GemmTestDesc` 是一个 `@dataclass`（`:249-269`），一个实例回答四个问题：**算什么**（gemm_type、shape、accumulate/bias/alpha）、**数据长什么样**（dtype、major、recipe、grouped）、**参考结果怎么算**（`_generate_*` 内的公式）、**怎么比**（cmp_spans、ensure_zero_padding）。字段总表见 3.11，这里先看"组合"是怎么发生的。以非对齐场景为例：

```python
for shape, a_major, b_major in product(shapes, [Major.K, Major.MN], [Major.K, Major.MN]):   # :501
    yield GemmTestDesc(GemmType.Normal, a_dtype, b_dtype, torch.float,
                       shape, a_major, b_major, accumulate=True, tag='unaligned-mn')
```

**shape × a_major × b_major** 的笛卡尔积：7 个 shape × 4 种 major 组合 = 28 个用例。同理 `recipe` 场景是 2 个 shape × 3 种 recipe = 6 个（`:444-453`：`(1,32,32)`、`(32,1,32)`、`(32,32,32)`——分别对应"只按 B 的 N 分块""只按 A 的 M 分块""双侧 32×32 块"的 SF 粒度）。grouped 场景的组合更复杂，除了 shape 和 major，还有组数、组大小随机、空组变体和 host 提示变体（3.8 节展开）。

几个容易忽略的派生规则：

- `recipe` 是三元组 `(gran_m, gran_n, gran_k)`，但操作数是二维的，所以拆成 `a_recipe = (recipe[0], recipe[2])`（A 是 `(m, k)`，粒度 `(gran_m, gran_k)`）和 `b_recipe = (recipe[1], recipe[2])`（B 是 `(n, k)`）——`:278-284`。
- `has_sf`（`:287-288`）：任一操作数是 fp8/fp4 就为真，决定 `gemm_dg` 调 `fp8_fp4_gemm_*` 还是 `bf16_gemm_*`（`common.py:51`），也决定操作数是"裸张量"还是"(data, sf) 二元组"。
- `get_available_trans_ab()`（`:290-298`）：Normal 四种 `nt/tn/nn/tt` 全测；M-grouped 只测 `nt`；K-grouped 只测 `tn`。这与 host 侧断言互为镜像——`get_gemm_desc` 里 M-grouped 断言 `major_a == Major::K`（`gemm.hpp:90`），K-grouped 断言双 MN-major（`gemm.hpp:94-95`）；生成端也有同样的断言（`generators.py:332, :372-373`）。

### 3.3 major 的存储语义：生成即布局

第 2 天讲过"逻辑转置不搬数据"，今天要落到字节层面。**major 维 = 单位步长（内层连续）的那个维度**。以 A `(m=2, k=3)`、元素 1..6 为例：

```text
K-major（Major.K，major_dim=1）            MN-major（Major.MN，major_dim=0）
logical (2, 3)，stride (3, 1)              logical (2, 3)，stride (1, 2)
内存顺序：1 2 3 4 5 6                       内存顺序：1 4 2 5 3 6
（先存第 0 行，再第 1 行）                    （先存第 0 列 (1,4)，再第 1 列 (2,5)，…）
k 维单位步长 → "K 连续"                      m 维单位步长 → "MN 连续"
```

为什么生成器要管物理布局？因为 `get_major_type_ab` 在 API 边界**从 stride 推断** major（`gemm.hpp:79-80`），而第 19 天读过的 `copy_gm_to_l1` 系列假设 major 维内层连续。所以生成代码里反复出现同一个技巧——**先按"major 维放在最后"的顺序物化存储，再转置回逻辑形状**：

```python
fp8_repr = x.to(torch.float8_e4m3fn)
fp8_repr = fp8_repr.transpose(major_dim, -1).contiguous()   # major 维换到最后，物化连续存储
fp8_repr = fp8_repr.transpose(major_dim, -1)                # 转回逻辑形状（成为跨步视图）
```

（`_quantize_to_fp8_with_major`，`:53-58`；bf16 分支的三连在 `:64`。）二维时若 major_dim 本来就是最后一维（如 K-major 的 `(m, k)`），`transpose(1, -1)` 是恒等——这正是"K-major 就是普通行主序"的含义。随机码路径用的是等价的另一种写法：交换 `storage_shape` 后直接按存储顺序生成，再 `transpose` 一次（`:146-147, :152`）。

**深入一步（易错点）**：观察 `gemm_dg` 对 Normal 的处理（`common.py:64-65`）——`'n'` 开头时 a 原样传入，否则转成 `(k, m)` 视图；而 API 侧 `bf16_gemm<kTransA, kTransB>` 又做一次方向相反的逻辑转置（`gemm.hpp:434-435`，`get_transposed` 是纯视图转置，`csrc/utils/layout.hpp:88-99`）。两次转置恰好抵消——四种 trans_ab 最终传给 `get_gemm_desc` 的是同一个张量视图，推断出的 `major_a/major_b` 也相同。因此对 dg 内核而言，四种组合检验的是四个 Python 绑定别名（`bf16_gemm_nt/nn/tn/tt`，各自约定 a/b 的逻辑形状为 `[M,K]`/`[K,M]`/`[N,K]`/`[K,N]`，见 `gemm.hpp:415-419` 注释）是否被正确规范化；**真正改变内核所见物理布局的是 `a_major/b_major`（枚举器的 2×2 组合），不是 trans_ab**。`major_opt()`（`:750-752`）给出的 NT/NN/TN/TT 标签，恰好是"输入以连续视图传入"的那个后缀（如双 K-major → 'nt'，双 MN-major → 'tn'）。ACLNN 交叉验证则不同：它 B 侧的规范化方向是反的（`get_transposed<!kTransB>`，`csrc/apis/aclnn.hpp:216`），四种组合确实走不同路径——留到第 30 天。

### 3.4 操作数生成入口：generate_operand_with_major

`generate_operand_with_major(shape, gran, dtype, major_dim, device)`（`:169-191`）是所有操作数生成的统一入口，三路分派：

```python
if device == 'meta':                                   # :176  编译用：只有形状与步长
    storage_shape = list(shape)
    storage_shape[major_dim], storage_shape[-1] = storage_shape[-1], storage_shape[major_dim]
    if dtype == torch.float4_e2m1fn_x2:
        storage_shape[-1] //= 2                        # fp4 沿 major 维两值一字节
    data_dtype = torch.int8 if dtype == torch.float4_e2m1fn_x2 else dtype
    data = torch.empty(storage_shape, dtype=data_dtype, device=device).transpose(major_dim, -1)
    if dtype == torch.bfloat16:
        return data, None
    sf_shape = tuple(aligned(size, block) // block for size, block in zip(shape, gran))
    return (data, torch.empty(sf_shape, dtype=torch.float32, device=device)), None
assert device == 'npu'
if dtype == torch.bfloat16:                            # :188  路径一：bf16
    tensor = torch.randn(shape, dtype=torch.float32, device='npu')
    return quantize_with_major(tensor, gran, dtype, major_dim)
return generate_random_quantized_with_major(shape, gran, dtype, major_dim)   # :191  路径二
```

meta 分支的关键：**用与真实路径完全相同的 storage_shape 交换和转置技巧构造元张量**，保证编译阶段看到的 shape/stride/dtype 与真实运行一致——JIT 配置只依赖这些元信息（第 12 天）。`par_compile`（`deep_gemm/testing/par_compile.py`）在 `dry_run()` 上下文里用线程池并发执行这些 launch，C++ 侧只编译不启动。注意 meta 分支返回的参考值是 `None`——编译不需要数值。

### 3.5 BF16：参考值就是 bf16 本身

bf16 走 `quantize_with_major` 的第一个分支（`:62-65`）：

```python
if quant_dtype == torch.bfloat16:
    tensor_bf16 = tensor_in.to(torch.bfloat16)
    tensor_bf16 = tensor_bf16.transpose(major_dim, -1).contiguous().transpose(major_dim, -1)
    return tensor_bf16, tensor_bf16          # 数据与参考是同一个对象
```

要点有三：随机源是 fp32 `randn`，先舍入到 bf16——**参考结果用的就是舍入后的 bf16 值**（后续 `a_repr.float() @ b_repr.float().T` 在 fp32 里精确定义），这样比较时不会混入"输入量化误差"，只剩内核累加顺序差（所以 `assert_equal` 的阈值是 1e-5 的余弦相似度而非逐位相等，`utils.py:46-58` 注释）；三连转置把存储物化成 major 内层连续（K-major 下是恒等操作）；gran 参数在此分支被完全忽略——**recipe 对 bf16 无意义**，这也是 `_enumerate_normal` 里 bf16 多出两个 fp32 输出 shape（`:559, :563-564`）而 fp8/fp4 没有的原因之一（fp32 输出仅 bf16 内核路径支持）。

### 3.6 低精度路径一：quantize_with_major——从浮点量化

这条路径把一个 fp32 张量按粒度量化，返回 `((量化数据, sf), 精确参考)`（`:61-95`）。GEMM 主测试不用它（用 3.7 的随机码路径），但 bf16 分支（3.5）和 **einsum 测试**（`generate_einsum`，`:789-825`，对收缩轴取 `gran_k=32` 的 MX 风格）都经过它。流程分五步：

```python
shape = list(tensor_in.size())                                  # :67  例如 (m, k)
padded_shape = [aligned(s, g) for s, g in zip(shape, gran)]     # :70  每维补到粒度整倍数
tensor = torch.zeros(padded_shape, ...); tensor[:...].copy_(tensor_in)
# :74-78  reshape + permute 把块拎出来：[..., 块数..., 块内元素...]，块内取 amax
sf_view = tensor.reshape(*view_shape).permute(perm_order).reshape(*sf_shape, -1)
sf_div = 448.0 if quant_dtype == torch.float8_e4m3fn else 6.0   # :80  两种格式的最大可表示值
sf_float = ceil_to_ue8m0(sf_view.abs().amax(dim=-1).clamp(1e-4) / sf_div)   # :81
sf_full = sf_float.repeat_interleave(g, dim=...)  # :84-87  把 SF 广播回每个元素
x_quant, f32_repr = _quantize_to_fp8/fp4_with_major(tensor / sf_full, major_dim)   # :89-94
return (x_quant, sf), (f32_repr * sf_full).contiguous()         # :95  精确参考 = 解码值 × SF
```

**`ceil_to_ue8m0`（`:27-31`）是 UE8M0 标度格式的核心**——"向上取整到 2 的幂"，三行位运算：

```python
bits = x.abs().float().view(torch.int)                # 按 int 读 float32 位模式
exp = ((bits // 2**23) % 256) + (bits % 2**23).bool().int()   # 指数域 + (尾数非零 → 进 1)
return (exp.clamp(1, 254) << 23).view(torch.float)    # 重组回 2^exp
```

注释说明用整除代替右移是因为 torch-npu 不支持右移。手算两个例子（3.10 有练习）：`5.0 = 1.25×2²`，位模式指数域 129、尾数非零 → exp=130 → 8.0；`4.0` 指数域 129、尾数为 0 → 4.0（恰为 2 的幂则不变）。`sf_div` 的含义：`sf ≥ amax/448` 保证 `x/sf ≤ 448` 落在 E4M3FN 可表示范围内（FP4 同理用 6.0），再向上取 2 的幂得到 UE8M0。`clamp(1e-4)` 兜住全零块的下界。

**FP4 量化与打包（`:34-50`）**：值集 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，用 7 个中点阈值（0.25…5.0）做"数边界"式的最近邻舍入——`code += (ax > boundary)` 累计超过的边界数即得码字；符号位放 bit 3（`code | (sign << 3)`，且 code 为 0 时保留 +0）。打包发生在 major 维上：转置到最后一维后，相邻两个 4 bit 拼成一个字节（偶序 nibble 在低位、奇序在高位，`:47`），`view(torch.int8)` 使该维长度减半，再转回逻辑形状。**这就是 FP4 要求 major 维为偶数的来源**——MN-major 的 FP4 要求 m 为偶数、K-major 要求 k 为偶数，3.9 的枚举器里有对应的处理。

### 3.7 低精度路径二：generate_random_quantized_with_major——随机码 + 精确解码

GEMM 主测试的 fp8/fp4 数据不走"量化"而走**反向构造**（`:124-166`）：先随机生成合法码字和 SF，再**精确解码**出参考值。动机：量化随机浮点只能覆盖部分码字、且参考值带舍入误差；反向构造让每个元素独立均匀覆盖**全部**码字，参考值是"码字值 × SF"的精确 fp32 乘积（SF 全取 2 的幂，乘法无舍入）——这让严格比较成为可能。三段关键代码：

```python
sf_values = (2.0**-9, 2.0**-8, 2.0**-7) if fp8 else (0.25, 0.5, 1.0)     # :138-142  2 的幂小集合
sf = sf_table[torch.randint(0, len(sf_values), sf_shape, device='npu')]  # :144  SF 形状 = padded/gran

if quant_dtype == torch.float8_e4m3fn:
    # Cover all non-NaN E4M3 codes independently at every tensor element.
    raw = torch.randint(0, 254, storage_shape, dtype=torch.uint8, device='npu')   # :150
    raw += (raw >= 0x7f).to(torch.uint8)                                          # :151  跳过 0x7f（NaN）
    data = raw.view(quant_dtype).transpose(major_dim, -1)
    f32_repr = data.float()
else:  # fp4
    assert shape[major_dim] % 2 == 0                                              # :155-156
    storage_shape[-1] //= 2
    packed = torch.randint(-128, 128, storage_shape, dtype=torch.int8, device='npu')   # :159
    data = packed.transpose(major_dim, -1)
    decoded = byte_lut[packed.view(torch.uint8).long()]                           # :162  256 项查找表
    ...
```

FP8 的 `randint(0, 254) + (raw >= 0x7f)`：先取 [0, 253]，再把 ≥127 的整体加一，得到 [0,126] ∪ [128,254]——恰好跳过 0x7f 与 0xff 两个 NaN 编码，覆盖其余全部 254 个码字（含 ±0 与 ±448）。FP4 的一个随机字节含两个独立均匀 nibble，`_FP4_BYTE_LUT`（`:98-102`）为每个字节预计算 (低 nibble 值, 高 nibble 值)，解码与 3.6 的打包格式严格互逆。布局上仍是 storage_shape 交换技巧（`:146-147`）。最后 `_scale_quantized_reference`（`:105-121`）把解码值按粒度块乘上 SF（reshape 成 `[块数…, 块长…]`、SF reshape 成 `[块数…, 1…]` 广播相乘），切回真实形状——参考结果完成。

SF 形状值得记一个具体例子：A `(m, k)`、`a_recipe=(gran_m, gran_k)=(1, 32)` 时 `sfa` 是 `(m, ceil(k/32))`——每行每 32 个 K 一个标度，即 MX 风格；`(32, 32)` 时是 `(ceil(m/32), ceil(k/32))` 的块标度。生成的 SF 是 **fp32、按块索引的逻辑形状、连续存储**；内核要求的内部布局由 `transform_sf_into_required_layout` 转换（第 33 天）。

### 3.8 grouped layout：generate_psum_layout 与两个迭代器

分组测试的核心是**前缀和布局**（psum layout）。`generate_psum_layout(expected_size, num_groups, alignment, device, sizes=None)`（`:212-220`）：

```python
for group_idx in range(num_groups):
    size = sizes[group_idx] if sizes is not None else max(int(random.uniform(0.7, 1.3) * expected_size), 0)
    psum_layout.append(psum + size)          # 本组有效数据的结束位置（前缀和）
    psum = aligned(psum + size, alignment)   # 下一组起点按 alignment（默认 256）对齐
return psum, psum_layout, torch.tensor(psum_layout, dtype=torch.int32, device=device)
```

三个要点：未指定 `sizes` 时每组大小在期望值的 0.7～1.3 倍间随机（**可以为 0，即空组**）；组**起点**对齐 256 但**结束位置**（前缀和值）任意——这正是第 26 天读过的调度器语义（`last_psum_k = aligned(current_psum_k, kAlignment)`）；返回的第三项就是传给 API 的 `grouped_layout` int32 张量。配对的两个迭代器：

- `iter_psum_slice(layout, alignment)`（`:223-227`）：产出各组**有效**区间 `[aligned(前缀), ptr)`；
- `iter_psum_sep(layout, alignment, gran=1)`（`:230-235`）：产出**组间 padding** 区间 `[ceil(ptr/gran), aligned(ptr,alignment)/gran)`——带 `gran` 参数时以 SF 行为单位换算（每 `gran_k=32` 个 K 对应一行 SF）。

**手工例子一（m-grouped）**：`sizes=[3, 5]`、alignment=256 → `total_m=512, layout=[3, 261]`：

```text
A 的行号：0 ── 3 ─────────────── 256 ── 261 ─────────────── 512
          ├─ 组 0 有效 ─┤─  padding  ─┼─ 组 1 有效 ─┤─  padding  ─┤
slice               [0, 3)                        [256, 261)
sep（数据）                  [3, 256)                       [261, 512)   → A 数据清零
sep（SF）                    同上（gran=1）                              → SF 填随机垃圾
d_ref              组0: A[0:3]·Bᵀ[0]              组1: A[256:261]·Bᵀ[1]；其余行保持 0
```

`_generate_m_grouped_contiguous`（`:326-359`）据此工作：A 是 `(total_m, k)` 且**必须 K-major**（`:332` 断言，镜像 `gemm.hpp:90`）；B 是 `(G, n, k)` 的逐组权重；`zero_operand_slices_`（`:194-205`）把 padding 行的**数据**清零（对 FP8 用 `view(torch.uint8)` 绕开 torch-npu 没有 FP8 fill 核的问题——全零位模式即 FP8 的 0），参考值同步清零；而 padding 行的 **SF 故意填 `randn` 垃圾**（`:342-346`）——第 26 天已知 m-grouped 的 `get_actual_m` 恒返回 BLOCK_M、padding 行会被照常计算，这个设计强制"padding 行输出精确为 0（0 × 任意有限 SF）且有效行的 SF 寻址不受垃圾影响"。`d_ref` 先 `zeros` 再逐有效区间覆写；`ensure_zero_padding=True`（默认）时全量比较（padding 行期望 0），否则 `cmp_spans` 只比有效行（`:357`）。另注意 `:330` 断言 m-grouped 不支持累加。

**手工例子二（k-grouped）**：`ks_cpu=[64, 0, 300]`、alignment=256 → `total_k=768, layout=[64, 256, 556]`：有效区间 `[0,64)`、`[256,256)`（**组 1 是空组**）、`[256,556)`；数据 padding `[64,256)` 与 `[556,768)` 清零（k-grouped 恒清零，`:379-382`，不受 `ensure_zero_padding` 控制）；SF 垃圾区间按 gran=32 换算为行 `[2,8)` 与 `[18,24)`（`:387-389`）。`_generate_k_grouped_contiguous`（`:361-399`）还有三个专属细节：A/B 是 `(total_k, m)` / `(total_k, n)` 且**双 MN-major**（`:372-373`，镜像 `gemm.hpp:94-95`）；粒度元组要**反转**——`self.a_recipe[::-1]`（`:375`）——因为张量维度顺序变成了 `(K, M)`，`(gran_m, gran_k)` 必须换成 `(gran_k, gran_m)`；参考结果 `d[i] = (c[i] if accumulate else 0) + (A[ks]ᵀ·B[ks]).to(cd_dtype)`（`:396-397`），注意乘积先转 cd_dtype 再加 c——"直接输出 vs fp32 累加"两种语义都会被阶段 2 验证（`common.py:202-206`）。

`ks_cpu` 与 `host_ks_cpu` 是两回事：前者是**真实组大小**，喂给 `generate_psum_layout(sizes=...)` 和 FLOPs 统计（`common.py:223-224`）；后者是**传给 API 的提示**（`common.py:81-82`），供 host 估计平均每组 K（`expected_k_from_ks`，`gemm.hpp:36-43`，空列表或缺省时回落 4096）。`enumerate_k_grouped_contiguous_test_variants`（`:606-613`）造三种变体：`normal`（全实）、`empty`（约 25% 随机置空且首组必空、末组必实）、`single`（仅中间一组非空）——空组正是第 26 天 `aiv_zero_empty_groups` 的测试输入。

### 3.9 非对齐与 strided 场景

**非对齐 MN**（`_enumerate_normal_unaligned_mn`，`:491-516`）的 shape 列表刻意挑选不被块大小整除的值：`(4,4,64)`、`(17,4096,4096)`、`(4096,17,4096)`、`(15,4096,4096)`、`(4096,15,4096)`、`(130,257,2048)`、`(513,257,4096)`，全部 × 4 种 major 组合、`accumulate=True`。FP4 有额外约束（`:505-508`）：MN-major 的 FP4 要求该维为偶数，**奇数则翻倍**（17 → 34，保持不被 64 整除故仍是非对齐用例）。**非对齐 K**（`:519-541`）用 `(64,64,4)`、`(1024,2048,17)`、`(8192,4096,15)`、`(128,128,234)`、`(2048,8192,4095)`；FP4 要求 K 为偶数，代码对奇数 K 做 `k *= 2`（`:532-533`）——注意该处注释写的是"round odd K **down** to the nearest even value"，与代码的翻倍行为不一致，阅读时**以代码为准**（两种做法都保持 K 不被 64 整除）。这些 shape 直接命中第 17/19/22/26 天读过的 `actual_m/n/k` → `eff_mad_*` → AIV 真值写回的整条边界链路。

**strided 场景**（`_enumerate_normal_strided`，`:466-488`）用极端长宽比 shape：`(4,4,16384)`、`(16384,4,4)`、`(16384,4,16384)`、`(4,16384,16384)` 等 10 个，× 4 major、`accumulate=True`，并设 `strided=True, tag='strided'`。

> **现状核对（以代码为准，不猜测）**：`make_strided`（`:238-246`，对一个连续张量在指定维上放大 shape 与 stride 后 `as_strided` + 拷回）在当前代码树中**没有任何调用方**；`GemmTestDesc.strided` 字段（`:261`，类型 `list[int] | None`）被设为 `True`（`:486`）之后，`common.py` 的执行路径中**没有任何地方读取它**。可用 `grep -rn "\.strided\|make_strided" tests/` 验证——只会命中定义与赋值。因此 `strided` 选择器今天的实际作用是换一组极端 shape 并打上 `[strided]` 标签；真正被执行的非连续输入来自 3.3 分析过的另一条路——`gemm_dg` 里 `get_transposed`（`common.py:42-46, 64-65`）产生的转置视图（凡与生成布局自然方向不同的 trans_ab 组合，传入的就是跨步张量），以及 MN-major 生成本身（stride `(1, m)` 非连续）。两者更像"显式注入任意 stride"的预留机制，尚未接线。

顺带一提枚举器家族里还有一个 `_generate_batched`（`:401-427`，三维逐 batch 生成）：`enumerate_gemm_tests` 不支持 Batched（`:677-678` 抛 `ValueError`），标准测试脚本不会走到它；`GemmType.Batched` 目前出现在 `enumerate_sf_layout`（`:689`）等 SF 布局枚举里，供 `test_layout.py` 使用（第 33 天）。

### 3.10 generate() 的输出如何被消费：common.py 两阶段走读

`generate(device='npu')`（`:429-438`）先 `reset_seed()`，再按 gemm_type 分派到四个 `_generate_*`，统一返回六元组：

```python
(m, n, k 实际形状, a, b, c, d_ref, (g_layout, cmp_spans))
```

注意**第一个元素可能不等于 `shape_mnk`**：m-grouped 返回 `(total_m, n, k)`，k-grouped 返回 `(m, n, total_k)`——total 由随机组大小决定。`a`/`b` 对 bf16 是裸张量、对 fp8/fp4 是 `(data, sf)` 二元组。`c` 有三种形态：None、累加用的 `(m, n)`（k-grouped 为 `(G, m, n)`）、bias 用的 `(1, n)`。`d_ref` 的公式按类型：

| 类型 | d_ref 公式 | 位置 |
|---|---|---|
| Normal | `alpha * (a_repr.float() @ b_repr.float().T) + c` → `.to(cd_dtype)` | `:320-322` |
| M-grouped | `d` 先 zeros，逐有效区间 `d[ms,:] = a_repr[ms,:] @ b_repr[i].T` | `:351-353` |
| K-grouped | 逐组 `d[i] = (c[i] if acc else 0) + (a_repr[ks].T @ b_repr[ks]).to(cd_dtype)` | `:396-397` |
| Batched | 逐 batch `d[i] = a_repr[i] @ b_repr[i].T (+ c[i])` | `:419-425` |

消费方 `run_gemm_test`（`common.py:131-237`）分两阶段：

**阶段 1（预编译，`:153-178`）**：对每个 test（Normal 额外叉上 `with_bias ∈ {False, True}`）调 `compile_test.generate(device='meta')`（`:157`），对每种 trans_ab 收集一个 `partial(gemm_dg, ...)`，Normal 无 bias 时再加一个随机 alpha 变体（`:161-164`），k-grouped 非累加时加 fp32 累加变体（`:165-168`），同 dtype bf16 输出时加随机舍入与 fp32 直出变体（`:169-174`）；最后 `par_compile(kernels)`（`:175`）在 dry_run 下并发触发 JIT。**编译只依赖元信息**，alpha 是运行时标量。

**阶段 2（运行验证，`:180-236`）**：`alpha_rng.seed(0)` 重置后，对每个 test × 每个 trans_ab × `(use_alpha, with_bias)` 四组合：`generate()`（`:189`，每次重新播种故数据恒同）→ `d` 预填充（`:190`）——**k-grouped 非累加时填 NaN**（内核漏写任何区域，尤其是空组的 `aiv_zero_empty_groups` 写零，会立刻暴露），其余填 `randn` 垃圾（内核必须完整覆写）→ `gemm_dg()` 启动 → `assert_equal(d, d_ref, cmp_spans)`（`:201`）。alpha+bias 组合预期被 API 拒绝（`:191-197`，断言报错信息含 `alpha and bias cannot be used together`，对应 `gemm.hpp:115-116`）。之后每个 trans_ab 再 generate 一次做 ACLNN 交叉验证（`:208-213`，仅 Normal，fp8 有若干跳过条件）与随机舍入验证（`:215-221`）。`print_perf`（`:230-236`）打印 `major_opt` 布局标签、acc、tag、FLOPs（k-grouped 用 `sum(ks_cpu)`）与 `count_bytes(a, b, c, d)`（`:755-757`，会展开 (data, sf) 二元组计字节）。

### 3.11 参考答案：测试用例字段说明表（当天产出）

**表一：GemmTestDesc 字段（`generators.py:251-269`）。**

| 字段（默认值） | 作用 | 生成侧消费位置 | 备注 |
|---|---|---|---|
| `gemm_type` | GEMM 形态 | `generate()` 分派 `:431-438`；`get_available_trans_ab` `:290-298` | Normal / MGrouped / KGrouped / Batched |
| `a_dtype` / `b_dtype` | 操作数精度 | `generate_operand_with_major` `:188-191`；`has_sf` `:287-288` | bf16 / fp8_e4m3fn / fp4_e2m1fn_x2，可混合（fp8×fp4 走 dequant 路径） |
| `cd_dtype` | C/D 输出精度 | `c`、`d_ref` 的 dtype `:308, :321` | bf16 / fp32 |
| `shape_mnk` | (m, n, k) | `m/n/k` property `:271-276` | grouped 下是**期望**值，真实 total 由 psum 决定 |
| `a_major` / `b_major` | 操作数物理布局 | 换算成 `major_dim` 实参 `:312-315, :334-337` 等 | 决定哪个维是单位步长 |
| `accumulate=False` | `D = C + A·Bᵀ` | `c` 形状与参考公式 `:304-308, :321` | m-grouped 不支持（`:330`） |
| `num_groups=None` | 组数 | `generate_psum_layout` `:328, :363` | m/k-grouped 必填 |
| `alignment=256` | 组起点对齐 | `generate_psum_layout`、`iter_psum_*` | 与内核 kAlignment=256 同源（第 26 天） |
| `strided=None` | （未接线） | **无消费方** | 见 3.9 现状核对 |
| `ensure_zero_padding=True` | padding 清零并全量比较 | m-grouped `:338-357` | k-grouped 恒清零，不受此开关控制 `:379-382` |
| `recipe=(1,1,32)` | SF 粒度 (gran_m, gran_n, gran_k) | `a/b_recipe` `:278-284` → gran 实参 | 决定 SF 形状；bf16 忽略 |
| `alpha=None` | 输出缩放 | 参考公式 `:320-322` | 与 bias 互斥 |
| `tag=None` | 打印标签 | `print_perf`（common.py:232） | ' fwd'/'dgrad'/'wgrad'/'strided' 等 |
| `with_bias=False` | bias | `c` 形状 (1, n) `:304-308` | 仅 Normal（common.py:154） |
| `ks_cpu=None` | k-grouped 每组真实 K | `generate_psum_layout(sizes=)` `:363`；FLOPs common.py:223-224 | 变体见 `:606-613` |
| `host_ks_cpu=None` | 传 API 的 ks 提示 | `gemm_dg` common.py:81-82 → `expected_k_from_ks` gemm.hpp:36-43 | 'list'/None/[] 三种 |
| `profile=True` | 是否测性能 | common.py:226 | k-grouped 仅 normal 变体开 `:648` |

**表二：property（`:271-298`）。**

| 名称 | 定义 | 含义 |
|---|---|---|
| `m/n/k` | `shape_mnk[i]` | 期望形状分量 |
| `a_recipe` | `(recipe[0], recipe[2])` | A `(m,k)` 的 `(gran_m, gran_k)` |
| `b_recipe` | `(recipe[1], recipe[2])` | B `(n,k)` 的 `(gran_n, gran_k)` |
| `has_sf` | 任一操作数为 fp8/fp4 | 选 `fp8_fp4_gemm_*` 还是 `bf16_gemm_*`，操作数是否带 SF |
| `get_available_trans_ab()` | 按 gemm_type | Normal 4 种；M-grouped `['nt']`；K-grouped `['tn']`；Batched 4 种 |

**表三：generate() 六元组按类型的差异。**

| 槽位 | Normal | M-grouped | K-grouped | Batched |
|---|---|---|---|---|
| shape | `(m, n, k)` | `(total_m, n, k)` | `(m, n, total_k)` | `(m, n, k)` |
| a | `(m,k)` | `(total_m,k)`，K-major | `(total_k,m)`，MN-major | `(G,m,k)` |
| b | `(n,k)` | `(G,n,k)` | `(total_k,n)`，MN-major | `(G,n,k)` |
| c | None / `(m,n)` / `(1,n)` | None | None / `(G,m,n)` | None / `(G,m,n)` |
| d_ref | 见 3.10 表 | zeros + 逐区间覆写 | 逐组（含空组=0） | 逐 batch |
| (g_layout, cmp_spans) | `(None, None)` | `(前缀和张量, None 或有效行)` | `(前缀和张量, None)` | `(None, None)` |

（fp8/fp4 时 a、b 换成 `(data, sf)` 二元组，且 k-grouped 的 gran 按 `recipe[::-1]` 反转以匹配 `(K, M/N)` 维度顺序。）

**表四：选择器 → 枚举器 → 用例数**（bf16 口径；已按 3.4/A-4 的方法机器验证）。

| 选择器 | 枚举器 | 组合 | 用例数 |
|---|---|---|---|
| `normal` | `_enumerate_normal` | 3 m_fwd × 9 (n,k) + 1 m_bwd × 9 × 3 | 54 |
| `normal`（fp8/fp4，无 fp32 输出对） | 同上 | 3×7 + 1×7×3 | 42 |
| `unaligned-mn` | `_enumerate_normal_unaligned_mn` | 7 shape × 4 major | 28 |
| `unaligned-k` | `_enumerate_normal_unaligned_k` | 5 shape × 4 major | 20 |
| `strided` | `_enumerate_normal_strided` | 10 shape × 4 major | 40 |
| `mgrouped` | `_enumerate_m_grouped_contiguous` | 1 + 3 组配置 × 4 (n,k) × 2 major_b | 25 |
| `kgrouped` | `_enumerate_k_grouped_contiguous` | 9 组配置 × 3 变体 × 3 host_ks × 4 acc/dtype | 324 |
| `test_recipes`（fp8 脚本的 `all` 才跑） | `_enumerate_recipe` | 2 shape × 3 recipe | 6 |

`_enumerate_normal` 的三组 shape 值得认出来源（`:559-576`）：`m_fwd ∈ {1, 128, 4096}` 叉上 `(n, k)` 列表得到前向形状（tag `' fwd'`）；`m_bwd = {4096}` 叉出 dgrad（`K,K→K,MN`，tag `'dgrad'`）与 wgrad（双 MN-major、带/不带 fp32 累加，tag `'wgrad'`）——正是 Transformer 训练三件套 `Y=X·Wᵀ / dX=dY·W / dW=dYᵀ·X` 的形状镜像。

## 四、源码导读

建议按站点顺序阅读，总计约 40 分钟。所有行号以仓库当前版本为准。

### 站点 1：种子与标度（`tests/generators.py:9-31`）

`reset_seed`（三个生成器统一设种子 0）、`Major`/`GemmType` 两个枚举、`ceil_to_ue8m0` 的三行位运算——对着 3.6 的手算例子逐位走一遍 `5.0` 与 `2/448`。

### 站点 2：两条量化路径（`tests/generators.py:34-102`）

`_quantize_to_fp4_with_major`（阈值→码字→符号→打包，四段各一句话）、`_quantize_to_fp8_with_major`（三连转置）、`quantize_with_major` 全文（重点 `:70-87` 的 padding/reshape/permute/amax/SF 广播五步）、`_FP4_VALUES` 与 `_FP4_BYTE_LUT`（核对 LUT 的 `(byte & 0xf, byte >> 4)` 与打包的"偶低奇高"互逆）。

### 站点 3：随机码生成与操作数入口（`tests/generators.py:105-209`）

`_scale_quantized_reference`（块广播乘法）、`generate_random_quantized_with_major`（`:148-153` 的 NaN 跳过、`:155-164` 的 fp4 打包解码、`:138-144` 的 SF 小集合）、`generate_operand_with_major`（三路分派）、`zero_operand_slices_`（FP8 的 uint8 view 绕路）、`aligned`。

### 站点 4：psum 三兄弟（`tests/generators.py:212-246`）

`generate_psum_layout`（对照 3.8 手工例子一）、`iter_psum_slice`/`iter_psum_sep`（gran 参数何时为 32）、`make_strided`（读完后用 grep 确认它没有调用方——3.9 的现状核对）。

### 站点 5：GemmTestDesc 全文（`tests/generators.py:249-438`，今天的主战场）

按 3.11 表一逐字段核对；重点精读四个 `_generate_*`：Normal 的 `:304-322`（c 的三种形态与参考公式）、m-grouped 的 `:328-357`（zero/garbage/cmp_spans 三件套）、k-grouped 的 `:361-399`（`recipe[::-1]`、空组语义、乘积先转 dtype 再加 c）、`generate` 的 `:429-438`（reset_seed + 分派）。

### 站点 6：七个枚举器与路由（`tests/generators.py:441-679`）

对照 3.11 表四数用例数；`_enumerate_normal` 的 `:559-576` 认出 fwd/dgrad/wgrad；`_enumerate_k_grouped_contiguous` 的 `:620-649` 看 9 个组配置（含 EP64/EP32/EP16 注释的 MoE 形状）、三种空组变体、四种 acc/dtype 组合与 profile 条件；`enumerate_gemm_tests` 的路由与 `:677-678` 对 Batched 的拒绝。

### 站点 7：SF 布局与 einsum 枚举（`tests/generators.py:682-825`，浏览）

`enumerate_sf_layout` / `enumerate_k_grouped_sf_layout` / `enumerate_k_grouped_psum_sf_layout` / `enumerate_sf_layout_performance_tests`（供 `test_layout.py`，第 33 天素材）；`generate_einsum`（`:789-825`，einsum 表达式 → 三维形状，fp8 时对收缩轴取 gran_k=32 走 `quantize_with_major`——3.6 路径的真实消费者）。

### 站点 8：消费方与交叉佐证

`tests/common.py:42-46`（`get_transposed` 对二元组的处理）→ `:49-84`（`gemm_dg` 三种类型的 API 选择）→ `:131-237`（`run_gemm_test` 两阶段，重点 `:157/:175`、`:189-201`、`:208-213`）；`tests/utils.py:36-58`（`calc_diff` 与 `assert_equal` 的 spans/threshold 语义）；`tests/test_bf16_gemm.py:25-47`（选择器）。交叉佐证：`csrc/apis/gemm.hpp:36-43`（`expected_k_from_ks`）、`:60-80`（`get_gemm_desc` 从 stride 推断 major）、`:415-453`（后缀语义与四个别名）、`csrc/utils/layout.hpp:88-99`（纯视图转置）、`deep_gemm/testing/par_compile.py`（dry_run 并行编译）。

## 五、实践步骤

### A. 无 Ascend 环境：静态阅读与手工演算（主任务，约 55 分钟）

**A-1 字段说明表（必做，当天产出的核心）。** 合上本文，凭源码印象默写 3.11 的表一～表三（字段、property、六元组差异），标注行号；写完对照订正。重点自查五处最易出错的细节：`shape_mnk` 在 grouped 下是不是真实形状；`a_recipe` 的两个分量分别来自 recipe 的哪两位；`strided` 字段有没有人读；k-grouped 的 `c` 是什么形状；`cmp_spans` 什么时候是 None。

**A-2 SF 手算（必做）。** 设块 `x = [[1.0, 2.0], [3.0, -5.0]]`、`gran=(1, 2)`：

1. 对 5.0、4.0、2/448、0.0 四个值手算 `ceil_to_ue8m0`（提示：5.0 的位模式指数域 129、尾数 0.25 → 进 1；0.0 会落到 clamp 下界，这正是 `amax.clamp(1e-4)` 存在的原因之一）；
2. FP8 路径：每行 amax = (2, 5)，算出 `sf = (ceil2(2/448), ceil2(5/448))` 与量化后的 `x/sf`（应全部 ≤ 448）；
3. FP4 路径：`sf_div=6.0`，算出 sf 与 `x/sf`，再用 3.6 的阈值表确定 5.0 落到哪个码字（注意边界 5.0 用严格大于比较）。

**A-3 psum 推演（必做）。** (a) m-grouped：`expected_m=4, num_groups=2`，设随机得到 `sizes=[3, 5]`，写出 `total_m`、`layout`、两组有效区间、两个 padding 区间、A 数据与 SF 各自的填充值、`d_ref` 形状与哪些行非零、`ensure_zero_padding` 取 True/False 时各自的比较范围。(b) k-grouped：`ks_cpu=[64, 0, 300]`，写出 `total_k`、`layout`、有效区间（注意组 1 为空组）、数据清零区间、**SF 垃圾区间（gran=32 换算）**、`d_ref` 形状与 `d[1]` 的值。(c) 思考：如果内核把组 2 的 K 循环多读了一个 64 块（读到 `[556, 620)`），哪段 padding 保护了结果不被污染？

**A-4 用例数统计（必做）。** 先手工数 3.11 表四的八个数字，再用下面的方法机器验证（不改仓库文件，绕开 `torch.npu` 依赖；本机 torch 版本若无 `float4_e2m1fn_x2` 属性，补一个占位即可，bf16 用例不会命中 fp4 分支）：

```bash
cd DeepGEMM-Ascend/tests
python - <<'EOF'
import generators
generators.reset_seed = lambda *a, **k: None   # 绕开 torch.npu.manual_seed
import torch
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = torch.float8_e4m3fn   # 占位属性，bf16 不受影响
G, T, B = generators.GemmType, generators.GemmType.Normal, torch.bfloat16
for name, kw in [('normal', {}), ('unaligned-mn', {'unaligned_mn': True}),
                 ('unaligned-k', {'unaligned_k': True}), ('strided', {'strided': True}),
                 ('mgrouped', {'gt': G.MGroupedContiguousWithPsumLayout}),
                 ('kgrouped', {'gt': G.KGroupedContiguousWithPsumLayout})]:
    gt = kw.pop('gt', T)
    print(name, len(list(generators.enumerate_gemm_tests(gt, B, B, **kw))))
EOF
```

再补一个 fp8 口径（`torch.float8_e4m3fn`）验证 `normal` 是 42。对照手工数是否一致，不一致时回到枚举器找漏数的那层循环。

**A-5 静态追踪（必做）。** 对 `python tests/test_bf16_gemm.py unaligned-k --skip-prof`，不看本文写出：选择器行号 → `run_gemm_test` 实参 → 枚举器名与用例数（20）→ 阶段 1 会收集多少个编译内核（提示：`with_bias=False` 时 4 trans × (1 + 1 alpha) = 8，`with_bias=True` 时 4 trans × 1 = 4——alpha 变体只在无 bias 时追加，`common.py:160-164`；每个 desc 共 12，20 个 desc 共 240）→ 阶段 2 每个 desc 执行多少次 `generate()`（4 trans × 4 组合 + 4 次 aclnn 复核 = 20）→ 这些用例有没有随机舍入验证（查 `common.py:215` 的三个条件：`a_dtype == b_dtype` ✓、`not test.accumulate` ✗——`_enumerate_normal_unaligned_k` 全部 `accumulate=True`，所以没有）。

**A-6 判定题自验（必做）。** 判断真假并给出行号依据（答案见第七节末尾）：

1. 同一个 `GemmTestDesc` 两次调用 `generate()`，得到的张量数值完全相同。（  ）
2. bf16 测试的参考结果是 fp32 原始随机数，因此必须容忍 bf16 输入舍入误差。（  ）
3. FP8 随机码生成会跳过两个 NaN 码字，其余 254 个码字（含 ±0 与 ±448）每个元素都可能取到。（  ）
4. m-grouped 生成中，组间 padding 的 A 数据被清零，而 padding 的 SF 被填随机垃圾。（  ）
5. `_generate_batched` 会在 `test_bf16_gemm.py` 的某个选择器下被执行。（  ）
6. Normal 的四种 trans_ab 最终传给 `get_gemm_desc` 的张量视图完全相同；被测差异在绑定别名（以及 aclnn 的不同规范化路径）。（  ）

### B. 有 Ascend 环境：运行与观察（可选，约 25 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
python tests/test_bf16_gemm.py normal --skip-prof
python tests/test_bf16_gemm.py unaligned-k --skip-prof
python tests/test_bf16_gemm.py kgrouped --skip-prof
```

1. 核对每组的 `Num tests:` 输出与 A-4 的统计（54 / 20 / 324）；
2. 观察 ` > Perf (m=..., layout=NT, fp32, acc=1, unaligned-k)` 行：layout 标签来自 `major_opt`，acc 与 tag 来自 desc 字段——把 20 行 unaligned-k 输出与 `_enumerate_normal_unaligned_k` 的 5 个 shape × 4 major 一一对上；
3. 直接生成一个六元组检查（体会"生成即布局"）：

```bash
python - <<'EOF'
import torch, torch_npu
import sys; sys.path.insert(0, 'tests')
from generators import GemmTestDesc, GemmType, Major

t = GemmTestDesc(GemmType.Normal, torch.bfloat16, torch.bfloat16, torch.float,
                 (64, 32, 48), Major.MN, Major.K, accumulate=True, tag='demo')
shape, a, b, c, d_ref, (g, spans) = t.generate()
print('shape:', shape, '| a:', a.shape, a.stride(), '| b:', b.shape, b.stride())
# 期望：a (64,48) stride (1,64) —— MN-major；b (32,48) stride (48,1) —— K-major
print('c:', None if c is None else c.shape, '| d_ref:', d_ref.shape)

t2 = GemmTestDesc(GemmType.KGroupedContiguousWithPsumLayout, torch.bfloat16, torch.bfloat16,
                  torch.float, (4, 4, 8), Major.MN, Major.MN, num_groups=3, ks_cpu=[6, 0, 10])
shape, a, b, c, d_ref, (g, spans) = t2.generate()
print('k-grouped shape:', shape, '| a:', a.shape, a.stride(), '| g_layout:', g.tolist())
# 期望：shape (4,4,512)，a (512,4) stride (4,1)，g_layout [6, 256, 266]
EOF
```

4. `DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof`，把若干 shape 的 `GemmConfig` 打印与 desc 字段对上（shape→BLOCK/MAD、acc→直写资格；连接第 12/26/51 天）。

## 六、当天产出

**测试用例字段说明表**。把 A-1～A-4 的结果整理成一页笔记，至少包含：

1. **表一～表四**（3.11 的四张表，自己默写后订正）；
2. **SF 手算记录**：A-2 的四个 ceil 值、两条路径的 sf 与量化结果；
3. **两张 psum 布局图**：m-grouped（sizes=[3,5]）与 k-grouped（ks=[64,0,300]）的区间条图，标注有效/数据清零/SF 垃圾三色区域与 `d_ref` 形状；
4. **用例数统计表**：A-4 的机器验证输出贴在手工数旁边；
5. **调用链卡片**：A-5 的静态追踪结果（选择器 → 枚举器 → 两阶段数量）；
6. **遗留问题清单**（给第 30/33 天）：至少记两条，例如"`assert_equal` 的余弦相似度阈值 1e-5 为什么对 bf16 合理、对 fp8 却可以逐位相等（提示：`utils.py:46-58` 注释里的 cube 路径）""ACLNN 交叉验证在 fp8 下有哪些跳过条件、为什么（`common.py:97-122`）""生成的 fp32 SF 如何变成内核要求的 packed UE8M0 MN-major 布局（`transform_sf_into_required_layout`）"。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. `GemmTestDesc` 的字段里，哪些决定"算什么形状"、哪些决定"数据长什么样"、哪些决定"和谁比、怎么比"？各举三例。
2. `a_major=Major.MN` 时生成的 A 张量 logical shape 与 stride 各是什么？`_quantize_to_fp8_with_major` 的三行 `transpose().contiguous().transpose()` 在 K-major 与 MN-major 下分别做了什么？
3. `ceil_to_ue8m0(5.0)` 等于多少？写出位运算过程。`sf_div` 为什么对 FP8 取 448、对 FP4 取 6.0？`amax.clamp(1e-4)` 在防什么？
4. GEMM 主测试的 FP8 数据为什么用"随机码 + 精确解码"而不是"随机浮点 + 量化"？SF 为什么只从 `{2^-9, 2^-8, 2^-7}` 抽？哪条路径才是"从浮点量化"（它的真实消费者是谁）？
5. `iter_psum_slice` 与 `iter_psum_sep` 各产出什么区间？m-grouped 生成中 A 的数据与 SF 在分隔区间里分别被填成什么？这两个设计各自在测内核的什么行为？
6. k-grouped 的 `ks_cpu` 与 `host_ks_cpu` 分别流向哪里？`empty` 变体把哪些组置空，为什么首组必空、末组必实？
7. `strided` 选择器今天实际测了什么？`make_strided` 与 `strided` 字段的现状如何用一条 grep 验证？测试中真正被执行的非连续输入来自哪里？

### 参考答案

1. 形状：`shape_mnk`、`num_groups`、`alignment`、`ks_cpu`；数据：`a/b_dtype`、`a/b_major`、`recipe`、（未接线的）`strided`；比较：`accumulate`、`with_bias`、`alpha`、`ensure_zero_padding`（决定 `cmp_spans`）、`cd_dtype`（参考公式 cast 目标）。参见 3.11 表一。
2. `(m, k)`、stride `(1, m)`。K-major：`transpose(1,-1)` 恒等，`contiguous` 无操作——行主序本就 K 连续；MN-major：先转到 `(k, m)` 物化成 m 内层连续的存储，再转回 `(m, k)` 成为跨步视图。见 3.3。
3. 8.0（指数域 129 + 尾数非零进 1 = 130 → 2³）。448 与 6.0 分别是 E4M3FN 与 E2M1 的最大可表示值，`sf ≥ amax/maxval` 保证 `x/sf` 不越界。clamp 防全零块退化到下界（`ceil_to_ue8m0(0)` 会落到 clamp(1,254) 的 2^-126）。见 3.6。
4. 随机浮点量化只覆盖部分码字且参考带舍入误差；随机码让每个元素独立覆盖全部 254 个非 NaN 码字（`:148-153`），解码值 × 2 的幂 SF 在 fp32 中精确（`:138-144, :105-121`），比较可以更严格。"从浮点量化"是 `quantize_with_major`（`:61-95`），其消费者是 bf16 分支和 `generate_einsum`（`:819-820`）。见 3.6/3.7。
5. slice 产出各组有效区间 `[aligned(前缀), ptr)`；sep 产出组间 padding `[ceil(ptr/gran), aligned(ptr,alignment)/gran)`（可按 SF 行换算）。数据清零：配合"m-grouped 的 padding 行被照常计算"（第 26 天 `get_actual_m` 恒 BLOCK_M），保证 padding 输出为 0、全量比较成立；SF 填垃圾：强制内核的 SF 寻址不依赖 padding 内容、垃圾不影响有效行。见 3.8 与 `:338-346`。
6. `ks_cpu` → `generate_psum_layout(sizes=...)`（生成布局与参考结果）及 FLOPs 统计；`host_ks_cpu` → `gemm_dg` 传给 API，`expected_k_from_ks`（`gemm.hpp:36-43`）用它估平均每组 K，空列表/缺省回落 4096。`empty` 变体随机置空约 25% 的组，首组必空覆盖"第 0 组为空"、末组必实保证至少有一组正常计算（`:609-611`）。见 3.8。
7. 实际测的是 10 个极端长宽比 shape × 4 major、`accumulate=True`，标签 `[strided]`；`grep -rn "\.strided\|make_strided" tests/` 只命中定义（`:238-246, :261`）与赋值（`:486`），无读取。真正的非连续输入：MN-major 生成（stride `(1, m)`）与 `gemm_dg` 的 `get_transposed` 转置视图（`common.py:64-65`）。见 3.9。

**A-6 判定题答案**：

1. **真**。`generate()` 首行 `reset_seed()`（`:429-430`），三个随机生成器都设回种子 0。
2. **假**。参考值是**舍入到 bf16 后**的值在 fp32 里的精确矩阵乘（`:63-65` 返回同一个 bf16 张量作参考）；1e-5 阈值容的是内核与参考的 K 维累加顺序差（`utils.py:46-58` 注释），不是输入舍入。
3. **真**。`randint(0, 254) + (raw >= 0x7f)`（`:150-151`）得到 [0,126]∪[128,254]，跳过 0x7f/0xff 两个 NaN 编码。
4. **真**。`zero_operand_slices_`（`:338-340`）清零数据；`:342-346` 对 SF 的分隔区间 `randn_like` 填垃圾。
5. **假**。`enumerate_gemm_tests` 对 Batched 抛 `ValueError`（`:677-678`），标准测试脚本不产生 Batched 的 desc。
6. **真**。测试侧转置（`common.py:64-65`）与 API 侧转置（`gemm.hpp:434-435`，纯视图，`csrc/utils/layout.hpp:88-99`）方向相反恰好抵消；aclnn 的 B 侧用 `!kTransB`（`aclnn.hpp:216`）故其路径确实不同。见 3.3"深入一步"。

---

**明日预告（第 30 天）**：理解正确性验证流程。今天造好了数据，明天读它如何被判定——`tests/common.py` 与 `tests/utils.py` 的参考结果比较（`calc_diff` 的余弦相似度为何对 bf16 用阈值、对 fp8 MX 可逐位相等）、ACLNN 交叉验证（`gemm_aclnn` 的全部跳过条件）、accumulation/bias/alpha 的语义核对、随机舍入为什么需要 `assert_stochastic_bf16_matches_fp32` 这样的确定性哈希比较，以及 meta 张量并行编译的完整原理。当天产出是一条完整测试流程图。
