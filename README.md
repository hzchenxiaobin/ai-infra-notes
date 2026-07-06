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
├── docs/                               # 8 周学习计划与教程写作 Skill
│   ├── AI_Infra_8_week_plan.md               # 计划总览
│   ├── AI_Infra_8_week_plan_detailed.md      # 详细每日任务
│   ├── learning_plan_week2_expanded.md       # Week 2 深度展开
│   └── skills/daily-tutorial/SKILL.md        # 写每日教程的通用 Skill
├── aiinfra/                            # 课程教程目录
│   ├── week1/                             # Week 1：GPU 执行本质 + Profiling
│   │   ├── README.md                           # 本周概览 + Day 索引
│   │   ├── day1/~day7/                         # 按天拆分，每天含：
│   │   │   ├── README.md                       #   当日教程（11 段固定骨架）
│   │   │   ├── kernels/                        #   可直接编译的 .cu 示例
│   │   │   ├── exercise/                       #   练习题与验证程序
│   │   │   └── notes/                          #   理论笔记与延伸阅读
│   │   ├── tools/                              # 辅助工具（Occupancy Calculator）
│   │   ├── profiles/                           # Nsight Profiling 报告汇总
│   │   └── website/                            # 本周静态网站源码
│   ├── week2/                             # Week 2：CUDA 进阶优化与性能分析
│   │   ├── README.md                           # 本周概览 + Day 索引
│   │   ├── day1/~day7/                         # 按天拆分（同 Week 1 结构）
│   │   └── website/                            # 本周静态网站源码
│   └── week3/                             # Week 3：Transformer 执行本质
│       ├── README.md                           # 本周概览 + Day 索引
│       ├── day15~day19/                        # 按天拆分
│       └── website/                            # 本周静态网站源码
├── leetgpu/                            # LeetGPU CUDA 挑战题解（12 道）
│   ├── leetgpu-vector-add-solution.md
│   ├── leetgpu-prefix-sum-solution.md
│   ├── leetgpu-gemm-solution.md
│   └── ...
├── leetcode/                           # LeetCode 算法题解
│   └── 最大总价值题解.md
├── build.py                            # 组合构建 GitHub Pages 网站
├── setup_github_ssh.sh                 # 一键配置 GitHub SSH Key
└── .github/workflows/deploy.yml        # GitHub Pages 自动部署
```

## 每日教程结构

每天的学习内容独立存放在 `aiinfra/weekN/dayM/` 目录下，遵循固定的 11 段骨架：

```
## Day N：<主题>
### 🎯 目标              ← 6 条编号目标 + "为什么重要"
### 学前导读              ← 动机铺垫，衔接前一日
### 理论学习              ← 分小节讲解，配 SVG 图表
### 昇腾对照              ← CUDA ↔ 昇腾 CANN 跨平台映射表
### Coding 任务           ← 含完整可编译 kernel + LeetGPU 在线题目
### 扩展实验              ← 3 个递进/对比实验
### 常见错误与调试         ← 三列表格
### 验证 Checklist        ← 7-8 条复选框
### 今日总结              ← 5-7 条加粗编号
### 面试要点              ← 5 题问答
```

详细的写作规范见 [docs/skills/daily-tutorial/SKILL.md](docs/skills/daily-tutorial/SKILL.md)。

## 在线网站

每次推送到 `main` 分支会自动构建并部署到 GitHub Pages，内容包括 Week 1 / Week 2 每日教程、8 周计划总览、LeetCode 题解等。

## 本地预览

### 方式 1：构建组合网站

```bash
python3 build.py
cd public && python3 -m http.server 8080
```

浏览器访问 `http://localhost:8080`。

### 方式 2：单独构建某一周网站

```bash
python3 aiinfra/week1/website/build.py
cd aiinfra/week1/website && python3 -m http.server 8080
```

## 编译运行 Kernel

Kernel 示例按天组织在 `aiinfra/weekN/dayM/kernels/` 下，可用 `nvcc` 直接编译：

```bash
cd aiinfra/week1
nvcc -o day1/kernels/hello_gpu day1/kernels/hello_gpu.cu && ./day1/kernels/hello_gpu
nvcc -o day4/kernels/transpose day4/kernels/transpose.cu && ./day4/kernels/transpose
nvcc -o day5/kernels/bank_conflict day5/kernels/bank_conflict.cu && ./day5/kernels/bank_conflict
```

Profiling 示例：

```bash
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./day4/kernels/transpose
```

## LeetGPU 每日一题

每天 Coding 任务的最后一道是 [LeetGPU](https://leetgpu.com/) 在线题目，与当日主题强相关：

| Day | Week 1 | Week 2 |
|-----|--------|--------|
| 1 | [vector-add](leetgpu/leetgpu-vector-add-solution.md) | [prefix-sum](leetgpu/leetgpu-prefix-sum-solution.md) |
| 2 | [relu](leetgpu/leetgpu-relu-solution.md) | [gemm](leetgpu/leetgpu-gemm-solution.md) |
| 3 | [matrix-addition](leetgpu/leetgpu-matrix-addition-solution.md) | [convolution](leetgpu/leetgpu-convolution-solution.md) |
| 4 | [matrix-transpose](leetgpu/leetgpu-matrix-transpose-solution.md) | [softmax](leetgpu/leetgpu-softmax-solution.md) |
| 5 | [reduction](leetgpu/leetgpu-reduction-solution.md) | [attention](leetgpu/leetgpu-attention-solution.md) |
| 6 | [matrix-multiplication](leetgpu/leetgpu-matrix-multiplication-solution.md) | [histogram](leetgpu/leetgpu-histogram-solution.md) |

每道题的完整题解（含 CPU 基线、GPU 设计、Kernel 实现、性能分析、复杂度）存放在 `leetgpu/` 目录下。

## 工具链

- CUDA Toolkit 11.8+ / 12.x
- Nsight Compute (`ncu`) / Nsight Systems (`nsys`)
- Python 3.10+（仅用于网站构建）
- cuBLAS（Week 2 起对比基准）

## 学习路线建议

1. 从 [docs/AI_Infra_8_week_plan.md](docs/AI_Infra_8_week_plan.md) 了解整体节奏
2. 进入 [aiinfra/week1/README.md](aiinfra/week1/README.md) 按 Day 1 → Day 7 推进
3. 每个 kernel 都配套 Nsight Profiling 任务，参考各 day 的 `notes/` 目录
4. Day 3 起配合 [aiinfra/week1/tools/cuda_occupancy_calculator.py](aiinfra/week1/tools/cuda_occupancy_calculator.py) 手算并验证 Occupancy
5. 每天完成 LeetGPU 在线题目，题解归档到 `leetgpu/`

## 目录约定

- `dayN/`：按天组织，每天一个目录，含 `README.md`（教程）、`kernels/`、`exercise/`、`notes/`
- `kernels/`：可直接编译运行的 `.cu` 示例
- `exercise/`：按天组织的练习题与验证程序
- `notes/`：理论笔记与官方文档摘要
- `website/`：静态网站源码（`build.py` 从 `dayN/README.md` 生成 `dayN.html`）

> 💡 本计划为理想节奏，实际执行中可根据个人进度调整。建议每周保留 Day 7 作为缓冲，避免进度积压。
