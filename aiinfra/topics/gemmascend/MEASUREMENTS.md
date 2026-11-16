# DeepGEMM-Ascend 实测记录（Ascend 950PR 单卡）

本文件记录 2026-11-16 在真实昇腾 NPU 上运行本专题第 5–8 周 A 组测试的结果，用于替换教程中"有环境时……"的可选项。每篇教程末尾的"实测记录"小节引用本文件。

**一句话结论**：`DeepGEMM-Ascend` 在 Ascend 950PR 单卡上构建成功，BF16 / FP8 / FP4 GEMM、Grouped GEMM、Scale-Factor 布局转换、MQA 与 Paged MQA 共 **14 个测试套件、1217 个检查点全部通过**；MegaMoE 因只有 1 张卡未测，HC Prenorm 因 TileLang 与本文 CANN 版本不匹配未通过。

## 1. 实测环境

| 项 | 实测值 | 教程要求 | 是否满足 |
|---|---|---|---|
| 服务器 / 容器 | `123.60.114.41:40015`，docker 容器 `c00628023` | — | — |
| 镜像 | `cann:9.0.0-beta.2-a3-ubuntu22.04-py3.11` | — | — |
| NPU | `Ascend950PR_9589`，`device_count=1` | Ascend 950 系列 | ✅（单卡） |
| 驱动 | `Version=7.0.t9.0.B798` | — | — |
| CANN | 容器内 `cann-9.0.0-beta.2` / `cann-9.1.0`；实测改用 **`cann-9.3.0`**（overlay，见 §2） | 9.20 | ⚠️ 版本不同 |
| bisheng | clang 15.0.5（CANN 9.3.0 内置） | `bin/bisheng` + `bin/ld.lld` | ✅ |
| Python | 3.10.12（venv `~/venvs/va`） | ≥ 3.10 | ✅ |
| PyTorch | 2.10.0+cpu | — | ✅ |
| torch_npu | 2.10.0.post4 | — | ✅ |
| 主机编译器 | g++-13.4.0（apt PPA 安装） | 需 C++20 `<format>` | ✅ |
| 源码 | `DeepGEMM-Ascend` @ `8491bbb`（OSS release） | 与教程引用的行号一致 | ✅ |
| 机器 | 40 核 / 125 GiB | — | — |

`8491bbb` 与教程引用行号一致（例如 `bf16_gemm.hpp:27` 正是 `__global__ __mix__(1, kDirectStore ? 0 : 2)`），因此教程里的 `文件:行号` 可直接对照。

## 2. 为跑通所做的适配（6 处，全部可复现）

1. **g++-13**：`setup.py` 要求 `-std=c++20` 且 `csrc/utils/format.hpp` 用了 `<format>`，容器自带 g++ 11 不支持。通过 `ppa:ubuntu-toolchain-r/test` 安装 `g++-13`。
2. **elfutils 头文件**：`third-party/deep-jit/include/deep_jit/utils/exception.hpp` 依赖 `elfutils/libdwfl.h`，需 `libdw-dev libelf-dev zlib1g-dev`。
3. **Python 头文件**：venv 基于系统 python3.10，需 `python3.10-dev` 提供 `Python.h`。
4. **tree-sitter**：`setup.py` 生成 `.pyi` 桩文件需要 `tree_sitter` + `tree_sitter_cpp`（本机 PyPI 不可达，从同机其它 venv 复制 cp310 版本）。
5. **CANN 版本**：
   - 仓库代码用到 `asc_quant_mode`，容器的 CANN 9.1.0 头文件里没有 → 改用同机已有的 **CANN 9.3.0**（`/home/m30064046/20261104000324968/cann-9.3.0`），以符号链接 overlay 到 `~/cann-9.3.0`，并补 `aarch64-linux -> x86_64-linux`（`deep_jit` 硬编码了 `aarch64-linux/asc/include/adv_api`，而本机是 x86_64）。
   - CANN 9.3.0 的 `c_api/asc_simd.h` 不再传递 `ascendc_assert` 宏 → 在 `deep_gemm/include/deep_gemm/transform_sf.hpp` 与 `mqa_logits.hpp` 各补一行 `#include <utils/debug/asc_assert.h>`。
6. **tilelang**：`deep_gemm/__init__.py` 在导入期即 `from .tilelang_ops import ...`。装 `tilelang==0.1.15` + `apache-tvm-ffi==0.1.12`（tilelang 要求 `<0.1.13`，预装的 `0.1.14.post1` 会触发 `Type 'ffi.Tensor' already has a registered class`），并把该导入改为 `try/except`，避免缺 TileLang 时整个包不可用。

> 以上改动都只影响**构建/导入**，没有改动任何 kernel 或测试逻辑；唯一"行为性"改动是 `__init__.py` 的 try/except（缺 TileLang 时 HC Prenorm 不可用，其它全部正常）。

环境脚本（容器内）：

```bash
export PATH=/home/c00628023/venvs/va/bin:$PATH
export ASCEND_HOME_PATH=/home/c00628023/cann-9.3.0
export LD_LIBRARY_PATH=/home/c00628023/venvs/va/lib/python3.10/site-packages/z3/lib:$LD_LIBRARY_PATH
export DG_JIT_CACHE_DIR=/home/c00628023/.dj
```

构建：`CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 ./develop.sh`（约 70 秒，产出 `deep_gemm/_C.cpython-310-x86_64-linux-gnu.so`）。

## 3. 测试结果总表

全部命令在 `DeepGEMM-Ascend/` 根目录执行；"耗时"含首次 JIT 编译（kernel 缓存在 `DG_JIT_CACHE_DIR`）。

| # | 测试套件 | 命令 | 用例数 | 结果 | 耗时 |
|---|---|---|---|---|---|
| 1 | BF16 Normal | `python tests/test_bf16_gemm.py normal --skip-prof` | 54 | 54/54 OK | 4 分 35 秒 |
| 2 | BF16 unaligned-k | `... unaligned-k --skip-prof` | 20 | 20/20 OK | 2 分 21 秒 |
| 3 | BF16 unaligned-mn | `... unaligned-mn --skip-prof` | 28 | 28/28 OK | 3 分 08 秒 |
| 4 | BF16 strided | `... strided --skip-prof` | 40 | 40/40 OK | 3 分 37 秒 |
| 5 | BF16 M-Grouped | `... mgrouped --skip-prof` | 25 | 25/25 OK | 45 秒 |
| 6 | BF16 K-Grouped | `... kgrouped --skip-prof` | 9 | 9/9 OK | 20 秒（缓存已热） |
| 7 | FP8 Normal | `python tests/test_fp8_gemm.py normal --skip-prof` | 42 | 42/42 OK | 约 8 分钟 |
| 8 | FP4 Normal | `python tests/test_fp4_gemm.py normal --skip-prof` | 42 | 42/42 OK | 4 分 32 秒 |
| 9 | FP8×FP4 Normal | `python tests/test_fp8_dequant_gemm.py normal --skip-prof` | 42 | 42/42 OK | 4 分 33 秒 |
| 10 | SF 布局转换 | `python tests/test_layout.py correctness` | 678 | 678/678 OK | 约 10 分钟 |
| 11 | MQA Logits | `python tests/test_attention.py prefill --skip-prof` | 56 | 56/56 OK | 1 分 11 秒 |
| 12 | Paged MQA | `python tests/test_attention.py paged --skip-prof` | 28 | 28/28 OK | 2 分 24 秒 |
| 13 | MQA 边界 | `python tests/test_attention.py boundary --skip-prof` | 21 | 21/21 OK | 20 秒 |
| 14 | Paged MQA 边界 | `python tests/test_attention.py paged_boundary --skip-prof` | 132 | 132/132 OK | 1 分 43 秒 |
| — | **合计** | | **1217** | **全部通过** | — |

日志首行示例（可对照教程里的预期输出）：

```text
Testing bf16 x bf16 gemm: Normal
Num tests:  54
Testing bf16 x bf16 gemm: Normal [unaligned-k]
Num tests:  20
Testing bf16 x bf16 gemm: KGroupedContiguousWithPsumLayout
Num tests:  9
Testing MQA logits (56 cases):
Testing MX Paged MQA Logits (28 cases):
Testing MX Paged MQA Logits boundaries:
```

## 4. 性能实测（`bench_msprof`，冷 L2，默认 30 次）

同一进程内 `deep_gemm.bf16_gemm_nt(a, b, d)`，`flush_l2=True`，`num_sms = 32`：

| Shape（M×N×K） | dur | TFLOPS | GB/s | mad% | mte2% | 判定 |
|---|---|---|---|---|---|---|
| 4096×4096×7168 | 582.2 µs | 413.1 | 259.3 | 96.2 | 93.7 | 计算受限 |
| 4096×7168×16384 | 2264.9 µs | 424.8 | 188.9 | 98.5 | 90.5 | 计算受限 |
| 128×7168×2048 | 25.5 µs | 147.2 | 1242.7 | 39.2 | 87.7 | 访存受限 |

`deep_gemm.get_num_sms()` 返回 **32**，与教程里 `kNumAICores = 32` 一致。BF16 大 shape 达到 ~425 TFLOPS，接近项目 README 给出的 950 系列 BF16 峰值（~432 TFLOPS @1.65GHz，即 99.8% 利用率口径）的 98%，mad 利用率 96–98%，与 README 的"接近硬件极限"一致。

### 真实 `GemmConfig`（`DG_PRINT_CONFIGS=1`）

```text
# 4096×4096×7168
GemmDesc(m=4096, n=4096, k=7168, cd_dtype=bfloat16_t, major_a=Major::K, major_b=Major::K,
         acc=false, epilogue_type=epilogue::operators::Identity, compiled_dims='nk',
         gemm_type=GemmType::Normal)
GemmConfig(block=256x256x128, mad=256x128x64, l1_stages=4, l0_stages=2, epi_stages=4,
           l1_sf_stages=0, sf_k_blocks=0, direct_store=false, resident=false/false/false,
           launch_options=LaunchOptions(num_blocks=32))

# 1024×2048×17（K 非对齐，k < BLOCK_K）
GemmConfig(block=256x256x128, mad=256x128x64, l1_stages=4, l0_stages=2, epi_stages=2,
           direct_store=true, launch_options=LaunchOptions(num_blocks=32))

# 128×7168×2048（小 M，第 52 天的实验 shape）
GemmConfig(block=128x224x128, mad=128x224x64, l1_stages=5, l0_stages=2, epi_stages=1,
           direct_store=true, launch_options=LaunchOptions(num_blocks=32))
```

三条配置分别验证了教程中的三处推断：
- 大 shape 走 `BLOCK 256×256×128`、`MAD 256×128×64`、4 级 L1 stage（第 16、23 天的容量账本）；
- **K=17 < BLOCK_K** 时仍选 `block_k=128`、`epi_stages` 由 4 降到 2、并开启 `direct_store`（第 26、31 天的边界处理）；
- **128×7168×2048** 实测选中 `BLOCK_N=224` —— 与第 52 天 A/B 组的预测表（"`BLOCK_N=224` 附近最优"）一致，可作为该实验的真实基线。

## 5. 第 5–8 周逐日实测对照

| 天 | 教程 A 组要求 | 实测结果 |
|---|---|---|
| 29 | 构建 + `normal`/`unaligned-k`/`kgrouped` | `./develop.sh` 成功；54 / 20 / 9 全部 OK |
| 30 | `normal`、`unaligned-k`，去掉 `--skip-prof` 看 perf | 20/20 OK；perf 见 §4 |
| 31 | `normal`/`unaligned-k` + `DG_PRINT_CONFIGS` | 通过；`(1024,2048,17)` 实测 `block=256x256x128, epi_stages=2, direct_store=true` |
| 32 | `test_fp8_gemm` / `test_fp4_gemm` / `test_fp8_dequant_gemm` | 42 / 42 / 42 全部 OK |
| 33 | `test_layout.py correctness`（678 例） | 678/678 OK |
| 34 | `test_fp8_gemm.py normal` + 配置打印 | 42/42 OK |
| 35 | `mgrouped` | 25/25 OK |
| 36 | `kgrouped` | 9/9 OK（含空组变体） |
| 37 | `mgrouped` + `kgrouped` 连跑 | 通过 |
| 38 | `test_attention.py prefill` | 56/56 OK（1 分 11 秒） |
| 39 | 稀疏/分页 MQA | `paged` 28/28、`boundary` 21/21、`paged_boundary` 132/132 OK |
| 40 | 复盘（可选复跑） | — |
| 41–44 | MegaMoE（≥2 卡，8 卡对照 README） | ❌ 只有 1 张卡，未测 |
| 45 | `test_hyperconnection.py default` + `peak` | ❌ TileLang 代码生成需 CANN 新符号（`float8_e6m2_t`、`CACHELINE_UB`、`__nop`），CANN 9.3.0 上没有，编译失败 |
| 46 | 复盘 | — |
| 50 | 基线性能（`bench_msprof`） | §4 三个 shape |
| 51 | `DG_PRINT_CONFIGS` / `DG_JIT_PRINT_COMPILER_COMMAND` | 已采集（§4、§7） |
| 52 | `set_num_sms` + BLOCK_N 单变量实验 | 未做完整实验；已确认 `128×7168×2048` 实测选中 `BLOCK_N=224` |
| 53 | 综合项目（A/B/C 运行验证） | 未做 |
| 54 | 回归 + 故障注入 | 未做 |
| 55 | 复盘分享 | 可用 §3/§4 数据 |

## 6. 未能完成的部分

1. **MegaMoE（第 41–44 天）**：容器只暴露 `/dev/davinci0`（`device_count=1`），而 MegaMoE 需要 ≥2 卡（8 卡才能对齐 README 的 138.7 µs / 2567.7 µs）。跨 rank 的 `expert_recv_count`、rank barrier、HBM 直访均无法验证。
2. **HC Prenorm（第 45 天）**：`test_hyperconnection.py` 走 TileLang 代码生成，`tilelang 0.1.15` 生成的 kernel 引用 `float8_e6m2_t` / `vector_f8e6m2` / `CACHELINE_UB` / `__nop`，这些符号在 CANN 9.3.0 的 `asc/impl` 头文件里都不存在（需与仓库匹配的私有 TileLang/CANN 组合）。PyPI 上 0.1.9/0.1.11 与预装的 `apache-tvm-ffi` ABI 冲突，亦不可用。
3. **稳定性观察（重要）**：把 14 个套件放在**同一个容器会话里连续跑**时，`kgrouped` 之后出现过一次设备级故障：

   ```text
   LazySetDevice: NPU function error, error code is 507033
   Inner_Error_Device_Subprocess_Startup_Timeout(E39007)
   TsdOpen failed. devId=0, tdt error=31
   ```

   故障后同一容器的后续进程全部 `double free or corruption` / core dump；**单独重跑**（新进程）则 100% 通过。建议：套件之间串行、每个套件独立进程、失败后先确认设备恢复（跑一次 `smoke`）再继续。

## 7. 实测发现的教程勘误

1. **`test_attention.py` 子命令名不对**：第 39/40 天教程写 `sparse` / `paged_sparse` / `sparse_boundary` / `paged_sparse_boundary`，本版本实际可选值为 `all|prefill|paged|boundary|paged_boundary`（`python tests/test_attention.py -h` 可查）。照教程原样执行会得到 `invalid choice` 并 `exit 2`。
2. **`kgrouped` 的用例数**：第 5 周教程提到 324，实测打印 `Num tests: 9`（9 个用例描述符，其下再展开 kernel 变体）。
3. **`test_layout.py` 需要 `DG_JIT` 缓存可写**：678 个用例会编译 48 个 kernel，首次运行约 10 分钟；若 `DG_JIT_CACHE_DIR` 不可写会反复编译。
4. **`ascendc_assert` 在 CANN 9.3.0 上不再被 `asc_simd.h` 传递**：第 33 天（SF 布局）和第 38 天（MQA）的 kernel 会编译失败，需要显式 `#include <utils/debug/asc_assert.h>`（见 §2.5）。
5. **`DG_JIT_PRINT_COMPILER_COMMAND=1` 的真实输出**（第 51 天可用）：

   ```text
   /home/c00628023/cann-9.3.0/bin/bisheng -O2 --npu-arch=dav-3510 --cce-aicore-only \
     -I<repo>/deep_gemm/include \
     -I<CANN>/aarch64-linux/asc/include/adv_api \
     kernel.asc -o kernel.rel.o
   ```

## 8. 复现步骤

```bash
# 0) 容器内
docker exec -it c00628023 bash
source ~/npu-tests/env.sh        # 见 §2 的环境变量
cd ~/DeepGEMM-Ascend

# 1) 构建（首次约 70 秒）
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 ./develop.sh

# 2) 单卡冒烟
python3 - <<'PY'
import torch, torch_npu, deep_gemm
a = torch.randn(128, 512, dtype=torch.bfloat16, device='npu')
b = torch.randn(256, 512, dtype=torch.bfloat16, device='npu')
d = torch.empty(128, 256, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d); torch.npu.synchronize()
ref = (a.float() @ b.float().t()).to(torch.bfloat16)
print('max diff', (d.float() - ref.float()).abs().max().item())
PY

# 3) 逐套件执行（每个套件独立进程）
python tests/test_bf16_gemm.py normal --skip-prof
python tests/test_bf16_gemm.py unaligned-k --skip-prof
python tests/test_bf16_gemm.py unaligned-mn --skip-prof
python tests/test_bf16_gemm.py strided --skip-prof
python tests/test_bf16_gemm.py mgrouped --skip-prof
python tests/test_bf16_gemm.py kgrouped --skip-prof
python tests/test_fp8_gemm.py normal --skip-prof
python tests/test_fp4_gemm.py normal --skip-prof
python tests/test_fp8_dequant_gemm.py normal --skip-prof
python tests/test_layout.py correctness
python tests/test_attention.py prefill --skip-prof
python tests/test_attention.py paged --skip-prof
python tests/test_attention.py boundary --skip-prof
python tests/test_attention.py paged_boundary --skip-prof

# 4) 性能与配置
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal   # 含 perf 行
```

> 说明：本记录由 agent 在 `123.60.114.41` 的 `c00628023` 容器内完成，容器内的 apt/pip 安装与符号链接（`g++-13`、`python3.10-dev`、`libdw-dev`、`~/cann-9.3.0`、`asc_assert` 补丁、`__init__.py` try/except）在容器重启后不会保留，需要按 §2 重做。
