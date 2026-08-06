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

**留档模板**（脚本末尾，待回填）：
```
| 模式  | decode avg (ms/step) | 加速比 |
|-------|---------------------|--------|
| Eager | <待实测回填>         | 1.00x  |
| Graph | <待实测回填>         | <待实测> |
```

**验收**：回填 Eager/Graph 两行的 ms/step 与加速比；即使 1.2x 也是真数字。附 nsys timeline 截图（eager 有 gap、graph 无 gap）更佳。

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

**待回填指标**：
- 成功率（目标 > 95%）
- 总耗时 / 吞吐（req/s）
- forward avg latency（ms，来自 `engine.forward_times`）
- KV Cache 是否全释放（无泄漏）
- custom_ops 是否真正接入（CUSTOM_OPS_AVAILABLE 状态）

**验收**：脚本输出的"真引擎稳定性测试结果"块回填到 README 的预期输出节；标注 GPU 型号与 custom kernel on/off。

---

### C. WS-3.3 项目话术数字：W10D5 X/Y/Z

**位置**：`aiinfra/daily/week10/day5/README.md:84`、`:288`
**关联脚本**：B 项（`stability_test.py --real`）+ A 项（`mini_engine_v1_graph.py`）

**目标**：项目 STAR 话术里的"单卡吞吐 X tokens/s、TTFT Y ms、TBT Z ms"占位需用实测回填。

**待回填**：
- **X（单卡吞吐 tokens/s）**：从 B 项 `--real` 模式的 throughput 换算（req/s × avg tokens/req）
- **Y（TTFT ms）**：真引擎首 token 延迟（可用 `torch.cuda.Event` 测 prefill 首 step）
- **Z（TBT ms）**：从 A 项 graph 模式的 decode avg/step 取值

**验收**：README 表格与示例话术的 X/Y/Z 由 placeholder 改为真实数字，并加脚注"实测环境：RTX 5090, CUDA x.x, PyTorch x.x, 2026-MM-DD"。

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

**留档表格**（README:255，待回填）：
```
| M×N×K | FP16 (ms) | FP16 (TF) | FP8 (ms) | FP8 (TF) | FP8/FP16 加速 | 最大误差 |
|--------|----------|----------|---------|---------|-------------|---------|
| 1024³ | ~0.5 | ~4.3 | ~0.3 | ~7.2 | ~1.7x | ~0.5 |  ← 全部待实测替换
| 4096³ | ~15 | ~9.2 | ~8 | ~17.2 | ~1.9x | ~2.0 |
| 8192³ | ~110 | ~9.9 | ~60 | ~18.2 | ~1.8x | ~5.0 |
```

**验收**：三行全部回填实测 ms/TF/加速比/误差；附 GPU 型号。大矩阵（8192³）应接近 2x 理论值，小矩阵受 launch overhead 影响偏离。

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

**待回填**：每个案例的"证据"块——SM 利用率%、DRAM%、stall 分布、内存快照等真实数字。

**验收**：三处"证据（模拟留档）"改为"证据（实测）"，删掉 W10D6:17 的"示意输出，未实测"声明；与 D3 数字诚信标准一致。

---

### F. W3D4 CUTLASS GEMM 示意输出

**位置**：`aiinfra/daily/week3/day4/README.md:275`
**依赖**：CUTLASS 库（`git clone https://github.com/NVIDIA/cutlass.git` + `CUTLASS_PATH`）

**目标**：CUTLASS 版 GEMM 的预期输出当前为"示意，未经实跑验证"，待 CUTLASS 环境就绪后补真实数据。

**验收**：回填 CUTLASS kernel 的 TFLOPS / cuBLAS 占比；注明 CUTLASS 版本与 sm 架构支持情况。

---

### G. W3D5/Day2 预估收益的实测验证

**位置**：`aiinfra/daily/week3/day5/README.md:415`
**关联**：`week2/day2` 的 GEMM 八层优化路径

**目标**：Day 3/5 的"消除抽象开销 + 重叠 load/compute"理论收益预估**尚未实测**（PTX kernel 复杂，未在本轮验证）。PTX 版 double buffer kernel 的真实收益需实跑确认。

**验收**：PTX kernel 实测收益表回填；若收益在噪声范围内（如 W2D3 的同步双缓冲 62.9%→63.8%），如实标注。

---

### H. W4D3 Welford vs 三遍扫描

**位置**：`aiinfra/daily/week4/day3/README.md:337`

**目标**：Welford 版 LayerNorm 预期比三遍扫描快 ~2x（HBM 读写减半），目前为量级预估，需实测。

**验收**：回填两版本 ms/TFLOPS 对比；~2x 量级预估的实测验证。

---

### I. W5D4 Attention kernel 计时占位

**位置**：`aiinfra/daily/week5/day4/README.md:326`

**目标**：`GPU Time (dA + dB kernels): 0.0xx ms ← 占位，待 GPU 实测回填`。

**验收**：回填真实 dA/dB kernel 计时。

---

### J. W3D6 ncu 指标（GPU performance counter 权限）

**位置**：`aiinfra/daily/week3/day6/README.md:136`

**目标**：ncu 指标值当前为"推理值（基于 kernel 结构推算），非实测"，需 GPU performance counter 权限实跑。

**前置**：`ERR_NVGPUCTRPERM` 常见于共享 GPU 实例；需管理员开放权限或用 `--target-processes all`。

**验收**：回填真实 ncu 指标（SM%、DRAM%、stall 等）；注明环境是否开放 performance counter。

---

### K. W7D6 PD 分离模拟器参数

**位置**：`aiinfra/daily/week7/day6/README.md:142`

**目标**：PD 分离模拟器参数（prefill_tput、decode_tput、退化系数）为示意值，真实系统需 ncu/nsys 实测校准。

**验收**：模拟器参数用实测值校准（或明确标注"示意值，真实部署需实测"）；方向性结论验证即可。

---

## 三、优先级与建议顺序

| 优先级 | 项 | 价值 | 依赖 |
|--------|----|----|------|
| **P0** | B（stability_test --real） | 真引擎联调是 W10 验收硬指标 | 需 torch + GPU |
| **P0** | A（v1_graph TBT） | W8D5 真整合产出 | 需 torch + GPU |
| **P0** | C（W10D5 X/Y/Z） | 项目话术数字诚信，直接依赖 B+A | 依赖 B、A 完成 |
| **P1** | D（FP8 GEMM） | 填补"FP8 只在嘴上说" | 需 Hopper/Blackwell + `_scaled_mm` |
| **P1** | E（W10D6 诊断证据） | 面试剧本真实性 | 需 ncu 权限 |
| **P2** | F、G、H、I、J、K | 教程数字完整性 | 各自前置（CUTLASS/ncu 权限等） |

**建议**：拿到机器后先做 P0 三项（B→A→C），这三项打通后 W10 项目话术的每个数字都能指向留档输出，满足整改验收标准 §三。P1 两项补强证据链。P2 按时间余量选做。

---

## 四、回填后的防回归

1. 回填后跑 `python3 build/check_course.py`（确保无新增悬空链接/旧口径）
2. README/脚本中"待实测"声明同步删除或改为"实测环境：..."
3. 留档输出附实测日期，便于后续核对
4. 新增数字若引用硬件参数，必须对齐 `reference/hardware_specs.md`
