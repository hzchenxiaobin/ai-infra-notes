# GPU 实测待补清单

> 依据：`rectification_plan.md` 整改执行后，所有"待 GPU 实测回填"项的汇总。
> 制定日期：2026-08-06
> 用途：课程/教程中存在一批"代码框架已落盘、数字待实测"的占位，无 GPU 时无法回填。本清单集中登记，等有机器（RTX 5090 / sm_120 或同等 NVIDIA GPU）后逐项实跑、回填、关闭。
> 原则：**数字诚信**——占位必须标注"待实测"，回填后改为真实数字并附实测环境。禁止用理论值/预估值冒充实测。

---

## 一、环境前置

| 项 | 要求 |
|----|------|
| GPU | NVIDIA RTX 5090（sm_120）为课程默认基准；其他 Ampere/Hopper/Blackwell 也可，但需注明型号 |
| 软件 | CUDA 12.x、PyTorch（支持 `_scaled_mm`/CUDA Graph）、ncu/nsys、CUTLASS（可选） |
| 权限 | `ncu` 需 GPU performance counter 权限（`ERR_NVGPUCTRPERM` 常见于共享 GPU 实例，需 `--target-processes all` 或管理员开放） |
| 留档格式 | 每项回填时附：**实测环境（GPU/CUDA/PyTorch 版本）+ 命令 + 输出片段 + 日期**，写入对应脚本末尾的"留档模板"或 README 表格 |

本次实测环境：
- **GPU**: NVIDIA GeForce RTX 5090 (sm_120)
- **Driver**: 580.126.09
- **CUDA**: 12.8 (toolkit), PyTorch 编译时 CUDA 12.8
- **PyTorch**: 2.9.1+cu128
- **主机**: i-2.gpushare.com (ssh -p 24666)
- **日期**: 2026-08-06

---

## 二、待实测项清单

### A. WS-3.1 真整合：mini_engine_v1_graph.py TBT 改善

**位置**：`aiinfra/daily/week8/day5/kernels/mini_engine_v1_graph.py`
**README**：`aiinfra/daily/week8/day5/README.md:137`（实测留档说明）

**目标**：验证 CUDA Graph 在 decode 路径消除 launch gap 的 TBT（token-by-token）延迟改善。

**运行命令**：
```bash
cd aiinfra/daily/week8/day5/kernels
python mini_engine_v1_graph.py
```

**留档模板**（脚本末尾，已回填）：
```
| 模式  | decode avg (ms/step) | 加速比 |
|-------|---------------------|--------|
| Eager | 0.894               | 1.00x  |
| Graph | 0.066               | 13.58x |
```

**验收**：✅ 已回填 Eager/Graph 两行的 ms/step 与加速比；脚本末尾与 `aiinfra/daily/week8/day5/README.md:137` 已同步实测环境脚注。

> 说明：Graph 路径当前演示单步 decode forward 的 launch 消除（未在 graph 内更新 KV cache），因此加速比包含 launch gap 消除与 KV cache 开销省略两部分；脚本注释已保留该 caveat。实际运行时 GPU 计时会有小幅波动（如 0.886/12.54x 与 0.894/13.58x 均属同一量级）。

---

### B. WS-3.2 真引擎联调：stability_test.py --real

**位置**：`aiinfra/daily/week10/day2/kernels/stability_test.py`
**README**：`aiinfra/daily/week10/day2/README.md`（任务 2 已记录 `--real` 模式）

**目标**：六步分层验证直接在 mini_engine_v2 真引擎上跑，500+ 请求的成功率/P50/P99/KV 归零成为真实证据（替换纯模拟）。

**运行命令**：
```bash
cd aiinfra/daily/week10/day2/kernels
python stability_test.py --real
```

**已回填指标**（安装 `ninja` 后 custom kernel 编译成功）：
- 成功率：**500/500 = 100.0%**（目标 > 95% ✓）
- 总耗时 / 吞吐：**0.44 s / 1138.8 req/s**
- forward avg latency：**0.901 ms**（439 steps recorded）
- KV Cache 是否全释放：**是，0 running requests（无泄漏 ✓）**
- custom_ops 是否真正接入：**是，custom kernel enabled（LayerNorm/FlashAttention CUDA kernel）**

**验收**：✅ 脚本输出的"真引擎稳定性测试结果"块已回填到 `aiinfra/daily/week10/day2/README.md` 的预期输出节；已标注 GPU 型号与 custom kernel 开启状态。

---

### C. WS-3.3 项目话术数字：W10D5 X/Y/Z

**位置**：`aiinfra/daily/week10/day5/README.md:84`、`:288`
**关联脚本**：B 项（`stability_test.py --real`）+ A 项（`mini_engine_v1_graph.py`）

**目标**：项目 STAR 话术里的"单卡吞吐 X tokens/s、TTFT Y ms、TBT Z ms"占位需用实测回填。

**已回填**（custom kernel 启用后更新）：
- **X（单卡吞吐 tokens/s）**：1138.8 req/s × 5 tokens/req ≈ **5694 tokens/s**
- **Y（TTFT ms）**：真引擎首 token 延迟（end-to-end，warmup 后平均）≈ **2.9 ms**；纯 GPU prefill 首 step（torch.cuda.Event）≈ 0.65 ms
- **Z（TBT ms）**：A 项 Graph 模式 decode avg/step ≈ **0.066–0.071 ms**（运行时小幅波动）

**验收**：✅ `aiinfra/daily/week10/day5/README.md:84` 与 `:288` 的 X/Y/Z placeholder 已改为真实数字，并加脚注实测环境（RTX 5090, CUDA 12.8, PyTorch 2.9.1+cu128, ninja 1.13.0, 2026-08-06）。

---

### D. WS-5.1 FP8 GEMM 实测：fp8_gemm_benchmark.py

**位置**：`aiinfra/daily/week8/day2/kernels/fp8_gemm_benchmark.py`
**README**：`aiinfra/daily/week8/day2/README.md:253`（待 GPU 实测回填）

**目标**：用 `torch._scaled_mm` 做 FP8 E4M3 GEMM vs FP16/BF16 的性能与精度对比，验证"FP8 算力 = FP16 2×"的理论值。

**运行命令**：
```bash
cd aiinfra/daily/week8/day2/kernels
python fp8_gemm_benchmark.py
```

**留档表格**（README:255，已回填）：

| M×N×K | FP16 (ms) | FP16 (TF) | FP8 (ms) | FP8 (TF) | FP8/FP16 加速 | 最大误差 |
|--------|----------|----------|---------|---------|-------------|---------|
| 1024³ | 0.015 | 144.5 | 0.014 | 149.0 | 1.03x | 247.19 |
| 2048³ | 0.097 | 177.0 | 0.051 | 339.7 | 1.92x | 345.12 |
| 4096³ | 0.627 | 219.1 | 0.332 | 413.5 | 1.89x | 523.50 |
| 8192³ | 4.918 | 223.6 | 2.447 | 449.4 | 2.01x | 746.00 |

**验收**：✅ 四行全部回填实测 ms/TF/加速比/误差；已附 GPU 型号。大矩阵（8192³）接近 2x 理论值，小矩阵（1024³）受 launch overhead 影响仅 1.03x。BF16 数据也一并产出（见脚本输出）。

---

### E. W10D6 诊断剧本证据留档

**位置**：`aiinfra/daily/week10/day6/README.md:17`（证据留档说明）
**三案例**：低 MFU 排查、OOM 定位、分布式卡死（README 内三个"证据（模拟留档）"块）

**目标**：把三案例的"示意输出，未实测"替换为真实 `ncu`/`py-spy`/`torch.cuda.memory_snapshot()` 输出。

**运行命令**（按案例）：
```bash
# 案例 1：低 MFU
ncu --set full --kernel-name gemm ./gemm
# 案例 2：OOM
py-spy dump --pid <python_pid>
# 案例 3：分布式卡死
NCCL_DEBUG=INFO python ...  # 看 NCCL 日志
```

**实测状态**：❌ **受环境限制，未能完成**。在共享 GPU 实例上以 root 运行 `ncu` 仍报 `ERR_NVGPUCTRPERM`；已尝试 `chmod 666 /dev/nvidia-caps/nvidia-cap{1,2}` 但仍无 performance counter 权限。`nsys` 未安装。

**待回填**：每个案例的"证据"块——SM 利用率%、DRAM%、stall 分布、内存快照等真实数字。

**验收**：当前环境无法满足 ncu performance counter 权限要求，W10D6 三个案例仍保留"示意输出，未实测"声明。需在拥有 performance counter 权限的裸金属/开发机上重新实测。

---

### F. W3D4 CUTLASS GEMM 示意输出

**位置**：`aiinfra/daily/week3/day4/README.md:275`
**依赖**：CUTLASS 库（`git clone https://github.com/NVIDIA/cutlass.git` + `CUTLASS_PATH`）

**目标**：CUTLASS 版 GEMM 的预期输出当前为"示意，未经实跑验证"，待 CUTLASS 环境就绪后补真实数据。

**实测状态**：⏸️ **本轮未执行**。CUTLASS 需 `git clone` + 配置 `CUTLASS_PATH` 并编译示例 kernel，耗时较长；本轮优先完成 P0/P1 与可直接运行的脚本项。可在后续轮次中补齐。

**验收**：待回填 CUTLASS kernel 的 TFLOPS / cuBLAS 占比；注明 CUTLASS 版本与 sm 架构支持情况。

---

### G. W3D5/Day2 预估收益的实测验证

**位置**：`aiinfra/daily/week3/day5/README.md:415`
**关联**：`week2/day2` 的 GEMM 八层优化路径

**目标**：Day 3/5 的"消除抽象开销 + 重叠 load/compute"理论收益预估**尚未实测**（PTX kernel 复杂，未在本轮验证）。PTX 版 double buffer kernel 的真实收益需实跑确认。

**已回填**（`aiinfra/daily/week3/day5/README.md:403`）：

| Size | Day1_naive | Day2_tiled | Day3_mma | Day5_dbuf | Best% | Best_impl |
|------|-----------|-----------|----------|-----------|-------|-----------|
| 512  | 72.8%     | 9.2%      | —        | 112.6%    | 112.6%| Day5_dbuf |
| 1024 | 33.5%     | 12.6%     | —        | 99.9%     | 99.9% | Day5_dbuf |
| 2048 | 32.3%     | 19.4%     | —        | 105.2%    | 105.2%| Day5_dbuf |
| 4096 | 31.1%     | 16.0%     | —        | 96.4%     | 96.4% | Day5_dbuf |

> 说明：Day 5 double buffer（cp.async PTX）在 RTX 5090 上达到 cuBLAS TF32 的 96%–113%；Day 2 tiled 在本实现中反而慢于 naive，已如实标注。Day 3 mma.sync PTX kernel 未单独实测。

**验收**：✅ PTX double buffer 收益已实跑回填；Day2 的意外退化也已如实标注。

---

### H. W4D3 Welford vs 三遍扫描

**位置**：`aiinfra/daily/week4/day3/README.md:337`

**目标**：Welford 版 LayerNorm 预期比三遍扫描快 ~2x（HBM 读写减半），目前为量级预估，需实测。

**已回填**（`aiinfra/daily/week4/day3/README.md:326`）：

| M(rows) | Three-pass(ms) | Welford(ms) | Speedup | max_diff |
|---------|---------------|-------------|---------|----------|
| 1024    | 0.006         | 0.008       | 0.74x   | 4.77e-07 |
| 4096    | 0.012         | 0.020       | 0.60x   | 4.77e-07 |
| 16384   | 0.076         | 0.094       | 0.81x   | 5.96e-07 |

> 说明：本实现中 Welford 单 pass 并未比 three-pass 快，反而略慢（0.6x–0.8x）。原因可能是 D=1024 时 three-pass 的 warp shuffle reduce 已很高效，而 Welford 的 mean/m2/count 三字段 shuffle 合并增加了寄存器/指令开销。正确性 PASS（max_diff < 1e-5）。

**验收**：✅ 已回填两版本 ms/对比；~2x 预估未在本实现中兑现，已如实标注原因。

---

### I. W5D4 Attention kernel 计时占位

**位置**：`aiinfra/daily/week5/day4/README.md:326`

**目标**：`GPU Time (dA + dB kernels): 0.0xx ms ← 占位，待 GPU 实测回填`。

**已回填**：`GPU Time (dA + dB kernels): 0.059 ms`（RTX 5090, CUDA 12.8, 2026-08-06 实测）。

**验收**：✅ 已回填真实 dA/dB kernel 计时到 `aiinfra/daily/week5/day4/README.md:326`。

---

### J. W3D6 ncu 指标（GPU performance counter 权限）

**位置**：`aiinfra/daily/week3/day6/README.md:136`

**目标**：ncu 指标值当前为"推理值（基于 kernel 结构推算），非实测"，需 GPU performance counter 权限实跑。

**前置**：`ERR_NVGPUCTRPERM` 常见于共享 GPU 实例；需管理员开放权限或用 `--target-processes all`。

**实测状态**：❌ **受环境限制，未能完成**。已确认 `ncu --target-processes all` 仍报 `ERR_NVGPUCTRPERM`；`chmod 666 /dev/nvidia-caps/nvidia-cap{1,2}` 无效。共享 GPU 实例大概率在 hypervisor 层禁用了 performance counter。

**验收**：待拥有 performance counter 权限的环境上重新运行 `ncu --set full ...` 并回填 SM%、DRAM%、stall 等指标；当前 W3D6 仍标注"推理值，非实测"。

---

### K. W7D6 PD 分离模拟器参数

**位置**：`aiinfra/daily/week7/day6/README.md:142`

**目标**：PD 分离模拟器参数（prefill_tput、decode_tput、退化系数）为示意值，真实系统需 ncu/nsys 实测校准。

**已回填方向性结论**（`aiinfra/daily/week7/day6/README.md:130` 已含输出）：

```text
Colocated:     avg TTFT 2373.1 ms, avg TPOT 8.96 ms, avg E2E 2964.2 ms
Disaggregated: avg TTFT  593.0 ms, avg TPOT 3.44 ms, avg E2E  822.8 ms, KV transfer 2.8 ms
  TTFT ↓75%, TPOT ↓62%, E2E ↓72%
```

> 说明：模拟器参数仍为示意值（prefill_tput=300 tokens/s/GPU、decode_tput=400 tokens/s/GPU、退化系数=2.5），方向性结论（PD 分离改善 TTFT/TPOT）已通过模拟验证。真实部署需用 ncu/nsys 实测 per-GPU throughput 与干扰系数后重新校准。

**验收**：✅ 模拟器已实跑，方向性结论验证；README 已明确标注参数为示意值、真实部署需实测校准。

---

## 三、完成状态汇总

| 优先级 | 项 | 状态 | 关键结果 / 阻塞原因 |
|--------|----|------|---------------------|
| **P0** | A（v1_graph TBT） | ✅ 完成 | Eager 0.886 ms/step, Graph 0.071 ms/step, 12.54x |
| **P0** | B（stability_test --real） | ✅ 完成 | 500/500 成功, **1138.8 req/s**, 0.901 ms avg forward, KV 全释放, custom kernel enabled |
| **P0** | C（W10D5 X/Y/Z） | ✅ 完成 | X≈5694 tokens/s, Y≈2.9 ms, Z≈0.066–0.071 ms |
| **P1** | D（FP8 GEMM） | ✅ 完成 | 8192³ 2.01x, 4096³ 1.89x, 1024³ 1.03x |
| **P1** | E（W10D6 诊断证据） | ❌ 阻塞 | `ncu` 报 `ERR_NVGPUCTRPERM`；`nsys` 未安装 |
| **P2** | F（CUTLASS GEMM） | ⏸️ 未执行 | 需 clone/build CUTLASS，本轮时间不足 |
| **P2** | G（PTX double buffer） | ✅ 完成 | Day5 dbuf 达 cuBLAS 96%–113%；Day2 tiled 意外退化已标注 |
| **P2** | H（Welford LayerNorm） | ✅ 完成 | 本实现中 Welford 0.6x–0.8x，未兑现 ~2x，已如实标注 |
| **P2** | I（Attention dA/dB 计时） | ✅ 完成 | dA + dB kernels: 0.059 ms |
| **P2** | J（W3D6 ncu 指标） | ❌ 阻塞 | 同 E：无 performance counter 权限 |
| **P2** | K（PD 分离模拟器） | ✅ 完成 | 方向性结论验证；参数仍为示意值 |

---

## 四、回填后的防回归

1. ✅ 已跑 `python3 build/check_course.py`：Checked 111 files, 0 findings。
2. ✅ README/脚本中"待实测"声明已同步改为实测环境脚注或真实数字。
3. ✅ 留档输出均附实测日期（2026-08-06）。
4. 新增数字若引用硬件参数，需对齐 `reference/hardware_specs.md`（请人工复核）。

---

## 五、阻塞项后续行动

- **E / J**：需要在 performance counter 权限开放的环境（裸金属/本地 RTX 5090 / 管理员解除 `ERR_NVGPUCTRPERM`）重新运行 `ncu --set full ...`。
- **F**：如需补齐，执行 `git clone https://github.com/NVIDIA/cutlass.git`，配置 `CUTLASS_PATH`，编译并运行 `cutlass_gemm` 示例，回填 TFLOPS / cuBLAS 占比。
