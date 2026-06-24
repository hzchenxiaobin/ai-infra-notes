# AI Infra 学习笔记

> 从「会写 kernel」进阶到「能做系统优化」—— 8 周 AI Infra 工程实战学习仓库，涵盖 GPU 执行模型、算子优化、推理系统与 Profiling 实践。

## 项目简介

本仓库记录 AI Infra（推理系统 / 分布式 / 内核优化）方向的系统化学习过程，核心是一条 8 周冲刺路线：

| 阶段 | 主题 | 关键产出 |
|------|------|---------|
| Week 1 | GPU 执行本质 + Profiling | SM/Warp/Memory 直觉、Nsight 实战 |
| Week 2 | GEMM & Kernel 优化 | Naive → Tiled → Register Blocking GEMM |
| Week 3 | Transformer 执行本质 | Softmax / LayerNorm kernel、Attention IO 分析 |
| Week 4 | FlashAttention 深挖 | Block-wise attention、IO 优化方法论 |
| Week 5 | 推理系统 | Prefill/Decode、KV Cache、Mini 推理引擎 v0 |
| Week 6 | Batching & 调度 | Dynamic / Continuous Batching、Scheduler |
| Week 7 | 系统整合 | 多请求并发、全链路 Profiling |
| Week 8 | 项目打磨 + 面试准备 | README、架构图、高频面试题 |

详细计划见 [docs/AI_Infra_8_week_plan_detailed.md](docs/AI_Infra_8_week_plan_detailed.md)。

## 仓库结构

```
ai-infra-notes/
├── docs/                          # 8 周学习计划文档
│   ├── AI_Infra_8_week_plan.md           # 计划总览
│   ├── AI_Infra_8_week_plan_detailed.md  # 详细每日任务
│   └── learning_plan_week2_expanded.md   # Week 2 深度展开
├── week1/                         # Week 1 完整学习材料
│   ├── README.md                  # 本周教程（按天拆分）
│   ├── kernels/                   # CUDA kernel 示例
│   ├── exercise/                  # 每日练习（含手算题与验证程序）
│   ├── notes/                     # 学习笔记与延伸阅读
│   ├── profiles/                  # Nsight Profiling 报告
│   ├── tools/                     # 辅助工具（Occupancy Calculator）
│   └── website/                   # 本周静态网站源码
├── leetcode/                      # LeetCode 算法题解
│   ├── 最大总价值题解.md
│   ├── images/
│   └── website/
├── LeetGPU/                       # LeetGPU CUDA 挑战题解
│   └── leetgpu-prefix-sum-solution.md
├── build.py                       # 组合构建 GitHub Pages 网站
├── setup_github_ssh.sh            # 一键配置 GitHub SSH Key
└── .github/workflows/deploy.yml   # GitHub Pages 自动部署
```

## 在线网站

每次推送到 `main` 分支会自动构建并部署到 GitHub Pages，内容包括 Week 1 每日教程、8 周计划总览、LeetCode 题解等。

## 本地预览

### 方式 1：构建组合网站

```bash
python3 build.py
cd public && python3 -m http.server 8080
```

浏览器访问 `http://localhost:8080`。

### 方式 2：单独构建 Week 1 网站

```bash
python3 week1/website/build.py
cd week1/website && python3 -m http.server 8080
```

## 编译运行 Kernel

Week 1 的 CUDA 示例可用 `nvcc` 直接编译：

```bash
cd week1
nvcc -o kernels/hello_gpu kernels/hello_gpu.cu && ./kernels/hello_gpu
nvcc -o kernels/transpose kernels/transpose.cu && ./kernels/transpose
nvcc -o kernels/bank_conflict kernels/bank_conflict.cu && ./kernels/bank_conflict
```

Profiling 示例：

```bash
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./kernels/transpose
```

## 工具链

- CUDA Toolkit 11.8+ / 12.x
- Nsight Compute (`ncu`) / Nsight Systems (`nsys`)
- Python 3.10+（仅用于网站构建）
- cuBLAS（Week 2 起对比基准）

## 学习路线建议

1. 从 [docs/AI_Infra_8_week_plan.md](docs/AI_Infra_8_week_plan.md) 了解整体节奏
2. 进入 [week1/README.md](week1/README.md) 按 Day 1 → Day 7 推进
3. 每个 kernel 都配套 Nsight Profiling 任务，参考 [week1/profiles/](week1/profiles/)
4. Day 3 起配合 [week1/tools/cuda_occupancy_calculator.py](week1/tools/cuda_occupancy_calculator.py) 手算并验证 Occupancy

## 目录约定

- `kernels/`：可直接编译运行的 `.cu` 示例
- `exercise/`：按天组织的练习题与验证程序
- `notes/`：理论笔记与官方文档摘要
- `profiles/`：Profiling 命令模板与报告

> 💡 本计划为理想节奏，实际执行中可根据个人进度调整。建议每周保留 Day 7 作为缓冲，避免进度积压。
