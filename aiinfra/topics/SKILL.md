---
name: topics
description: Use when writing or revising a topic-based deep-dive tutorial (topics/<name>/README.md) in this AI Infra learning repo. Triggers on requests like "write a cutlass topic", "add triton deep dive", "写专题教程", "写 cutlass 专题". Produces focused deep-dives on a single technology/library (CUTLASS, Triton, cuDNN, NCCL, etc.) following the repo's Chinese-first style, with compilable code, benchmarks, and interview Q&A. Do NOT use for per-day tutorials (use daily-tutorial skill) or opencode's own config.
---

# 写专题学习 Skill

本工程(`ai-infra-notes`)的专题学习是对 `aiinfra/daily/` 每日教程的**补充与延伸**。每日教程按"周/日"线性推进，覆盖 8 周主干路线；专题学习则针对**单个技术/库/工具**做横向深挖，不受日程约束，可随时开题。

典型专题方向：

| 类别 | 示例 |
|------|------|
| 算子库 | CUTLASS、cuBLAS、XLA |
| 编译器/DSL | Triton、TVM、Mojo |
| 推理框架 | TensorRT、vLLM 内核、Llama.cpp |
| 通信库 | NCCL、RCCL、Gloo |
| 工具链 | Nsight 深度用法、CUDA-GDB、nsys/ncu 进阶 |
| 底层机制 | CUDA Async Memory、TMA、Warp Specialization |

## 1. 与每日教程的区别

| 维度 | 每日教程（daily） | 专题学习（topics） |
|------|-------------------|--------------------|
| 组织方式 | 按 weekN/dayM 线性推进 | 按 `<topic-name>/` 横向开题 |
| 篇幅 | 单日 ~550 行，固定 8 段骨架 | 无固定行数，按主题自由展开 |
| 深度 | 覆盖广度，点到为止 | 单点深挖，含源码级分析 |
| 代码 | 1 个可编译 kernel | 可含多个示例、benchmark、对比实验 |
| 依赖 | 自包含，无需第三方库 | 可依赖第三方库（CUTLASS、Triton 等） |
| 前置 | 对应周的已完成教程 | 标注前置教程（如"先学 week2 GEMM"） |

## 2. 文件落位

```
topics/
├── SKILL.md                      # 本文件（写作规范）
├── images/                       # 专题共享 SVG（所有专题共用，同 leetgpu/images/ 模式）
└── <topic-name>/                 # 每个专题一个目录，如 cutlass/、triton/
    ├── README.md                 # 专题主体
    ├── kernels/                  # 可编译代码示例（.cu / .py / .cpp）
    ├── notes/                    # 源码笔记、论文精读、延伸阅读
    └── benchmark/                # 性能对比脚本与结果
```

- 专题目录名用**全小写 + 连字符**：`cutlass`、`triton-kernel`、`tensorrt-plugin`
- 教程中用相对路径引用本地文件：`[kernels/hello_triton.py](kernels/hello_triton.py)`
- SVG 引用：`![中文alt描述](../images/xxx.svg)`（从 `<topic-name>/` 出发，`../images/` 解析到 `topics/images/`）
- 若一个专题 SVG 较多（>5 张），可在专题目录下建本地 `images/`，引用改为 `(images/xxx.svg)`

## 3. 专题骨架

专题无固定段数，但建议遵循以下顺序（可按主题裁剪）：

```
# <专题名>：<一句话定位>

## 🎯 目标
## 为什么学这个
## 核心概念
## 最小可运行示例
## 深入原理
## 性能对比与 Benchmark
## 常见陷阱与最佳实践
## 面试要点
## 推荐资源
```

### 3.1 `# <专题名>：<一句话定位>`
- 标题用"技术名 + 定位"，如 `# CUTLASS：高性能 GEMM 算子库`、`# Triton：Python 写 GPU Kernel 的 DSL`
- 冒号用中文全角 `：`

### 3.2 `## 🎯 目标`（必有，紧贴标题）
```markdown
## 🎯 目标

通过本专题，你将：

1. <动词开头的目标，如"理解 CUTLASS 的三层抽象">
2. ...
3. ...

> 💡 **前置知识**：<衔接教程，如"建议先完成 week2 GEMM 教程">
> ⚠️ **环境要求**：<如 Triton 0.2.3、CUDA 12.x、cuDNN 9.x>
```
- 3-6 条编号目标（动词开头：理解/掌握/学会/能/实现）
- 末尾固定 `> 💡 **前置知识**：<...>` 和（如有）`> ⚠️ **环境要求**：<...>`

### 3.3 `## 为什么学这个`
- 1-2 段 + 对比表/代码块，回答"这个技术解决了什么痛点"
- 与已有教程/手写方案对比（如"手写 GEMM vs CUTLASS"）
- 常以 `> 💡 **一句话总结**：<...>` 收束

### 3.4 `## 核心概念`
- 用 `### N.1`、`### N.2` 分小节
- 配 SVG 图，插在小节首行：`![<中文alt>](../images/<filename>.svg)`
- 表格化对比（API 分层、内存模型、调度策略等）
- 善用 `#### 深入：<为什么 X?>` 做子解释
- 形象类比（把 CUTLASS 的 threadblock 比作工厂流水线等）

### 3.5 `## 最小可运行示例`（必有）
- **完整可编译/可运行代码**，读者复制即能跑通
- 代码块标注语言：` ```python` / ` ```cuda` / ` ```cpp`
- 代码块首行带注释：`# hello_triton.py —— 最小 Triton kernel` + 运行命令
- 运行命令单独用 ` ```bash ` 代码块
- 预期输出用 ` ```text ` 代码块
- 文件链接：用相对路径 `[kernels/xxx.py](kernels/xxx.py)` 引用真实文件

### 3.6 `## 深入原理`
- 源码级分析：关键数据结构、调用链、优化技巧
- 可引用源码片段（标注仓库版本/commit）
- 配 SVG 架构图/数据流图
- 这是专题与每日教程的核心差异——**深挖到源码级别**

### 3.7 `## 性能对比与 Benchmark`
- 量化对比表：手写 vs 库、naive vs optimized
- 给出可复现的 benchmark 脚本（`benchmark/xxx.py`）
- 必含**量化数据**：带宽、TFLOPS、延迟、加速比
- 分析瓶颈（用 Roofline 模型等）

### 3.8 `## 常见陷阱与最佳实践`
- 列举 3-5 个典型坑（如"Triton 的 `tl.load` 边界处理"）
- 每个坑给**错误写法 → 正确写法**对比
- 工程实践建议（编译选项、调试技巧、版本兼容性）

### 3.9 `## 面试要点`
- 5-8 题问答，覆盖原理 + 工程 + tradeoff
- 格式与每日教程一致：`**Q：问题？**` + 缩进答案

### 3.10 `## 推荐资源`
- 官方文档、论文、源码、优质博客
- 标注优先级（⭐ 必读 / 📌 推荐 / 📎 参考）

## 4. 写作规范

### 语言
- **中文为主**，概念加粗
- 善用 blockquote：`> 💡 **一句话总结**：<...>`、`> ⚠️ **注意**：<...>`
- 代码块标注语言：` ```cuda` / ` ```python` / ` ```cpp` / ` ```bash` / ` ```text`

### 数学公式

- 行内公式用 `$...$`，块级公式用 `$$...$$`
- **禁止**用反引号 `` `...` `` 包裹数学公式，否则会被渲染为等宽代码，KaTeX 不会识别
- 公式内函数/运算符使用 LaTeX 命令：`\exp`、`\log`、`\sum`、`\max`、`\frac`、`\sqrt`，避免直接写 `exp`、`log`、`Σ`、`√`


### 量化指标
- 单专题建议 **400-800 行**（比每日教程长，但不宜超 1000 行）
- 过长则拆分为多个专题（如 `cutlass-gemm`、`cutlass-epilogue`）

### 图片
- SVG 命名：全小写 + 下划线，语义化（如 `cutlass_three_level_abstraction.svg`）
- 每个专题引用 3-6 张 SVG
- alt 文本用中文
- **风格统一为手绘 sketch 风**（Excalidraw-like），与每日教程/题解保持一致：
  - 线条：手绘不均匀、略带抖动
  - 配色：极简，不超过 3-4 种柔和颜色
  - 字体：英文用 Comic Sans MS，CJK 用楷体（Kaiti SC）
  - 滤镜：`feTurbulence` 抖动 + `feDisplacementMap`

### 交叉引用
- 引用本专题文件：相对路径 `(kernels/xxx.py)`
- 引用每日教程：`(../../daily/weekN/dayM/README.md)`
- 引用其他专题：`(../<other-topic>/README.md)`
- 引用根目录资源：`(../../images/xxx.svg)`

## 5. 构建与 GitHub Pages 集成（统一自动发布）

专题已纳入统一的 GitHub Pages 构建流程，**不再需要为每个专题单独写构建脚本**。

### 5.1 自动发现机制

- 统一构建脚本：`aiinfra/topics/build.py`
- 构建入口：`build.py`（根目录）会自动调用 `aiinfra/topics/build.py`
- 发现规则：扫描 `aiinfra/topics/` 下所有包含 `README.md` 的子目录（自动排除 `images/`、`website/`、`__pycache__`）
- 新增专题无需修改 `build.py` 或 `.github/workflows/deploy.yml`，push 后自动部署

### 5.2 每个专题的预期文件结构

```
aiinfra/topics/<topic-name>/
├── README.md              # 专题主体（必需）
├── day1.md ~ day7.md      # 可选：拆分的每日学习文件
├── kernels/               # 可选：可编译代码示例
├── notes/                 # 可选：源码笔记、延伸阅读
├── benchmark/             # 可选：性能对比脚本
└── images/                # 可选：专题私有图片（较少时使用 topics/images/）
```

- `README.md` 中可用 `## Day N（周 x）：标题` 组织大纲；若存在对应的 `dayN.md`，构建时会生成独立页面
- 拆分 `dayN.md` 的好处：便于维护、便于按日更新、自动生成 `dayN.html`
- 拆分方式：把 `README.md` 中每个 `## Day N...` 章节拷贝为 `# Day N...` 开头的 `dayN.md`，`README.md` 保留概览和目录结构

### 5.3 图片资源

- 推荐放在共享目录：`aiinfra/topics/images/<topic-name>_<description>.svg`
- 在 Markdown 中引用：`![中文alt](../images/<topic-name>_xxx.svg)`
- 构建时会自动把匹配 `<topic-name>_*.svg` 的图片复制到 `public/<topic-name>/images/`

### 5.4 交叉引用规范

Markdown 中可写源文件相对路径，构建时会自动重写为部署后的路径：

| 源路径示例 | 部署后路径 |
|---|---|
| `](../cutlass/README.md)` | `](../cutlass/index.html)` |
| `](../cutlass/day2.md)` | `](../cutlass/day2.html)` |
| `](../../paper/flashattention3/README.md)` | `](../paper/flashattention3/index.html)` |
| `](../../daily/week6/day1/README.md)` | `](../daily/week6/day1.html)` |

### 5.5 本地构建与预览

```bash
python3 build.py
# 专题站点生成在 public/<topic-name>/
```

部署地址：`https://<username>.github.io/ai-infra-notes/<topic-name>/`

### 5.6 新增专题 checklist

- [ ] 目录名全小写，如 `my-topic`
- [ ] 放入 `README.md`
- [ ] （可选）按需要拆分 `day1.md` ~ `day7.md`
- [ ] （可选）把 SVG 图片放入 `aiinfra/topics/images/<topic-name>_*.svg`
- [ ] push 到 `main`，GitHub Actions 自动发布

> 历史变更：原先 cutlass/triton/cute/deepgemm/moe 各有独立的 `website/build.py`，现已统一删除，统一由 `aiinfra/topics/build.py` 处理。

## 6. 开题流程

1. **确认前置**：该专题需要哪些每日教程作为基础？在 `🎯 目标` 中标注
2. **搭目录**：`mkdir -p topics/<name>/{kernels,notes,benchmark}`
3. **写最小示例**：先让一个可运行代码跑通，再回头补理论
4. **补理论 + SVG**：核心概念配图，源码分析配数据流图
5. **跑 benchmark**：量化数据是专题的骨架，务必有真实测量
6. **自检**：用下方检查清单

## 7. 检查清单（写完一个专题后自检）

- [ ] 标题是 `# <专题名>：<一句话定位>`（中文全角冒号）
- [ ] `## 🎯 目标` 含 3-6 条编号 + `> 💡 前置知识`
- [ ] `## 为什么学这个` 有与已有方案的对比
- [ ] `## 核心概念` 分小节，配 SVG
- [ ] `## 最小可运行示例` 代码完整可运行，带运行命令 + 预期输出
- [ ] `## 深入原理` 含源码级分析
- [ ] `## 性能对比与 Benchmark` 有量化数据 + 可复现脚本
- [ ] `## 常见陷阱与最佳实践` 有 错误→正确 对比
- [ ] `## 面试要点` 5-8 题问答
- [ ] 所有文件链接用相对路径且指向真实文件
- [ ] SVG 引用格式 `![中文alt](../images/xxx.svg)`，风格为手绘 sketch 风
- [ ] 目录名全小写 + 连字符

## 8. 本对话中的相关操作记录

本次会话中对 `aiinfra/topics/` 及周边仓库结构做了以下调整，作为后续维护的背景参考：

| 操作 | 说明 |
|------|------|
| 下载 `deepseek_v3.pdf` 的 HTML 版本 | 从 arXiv 下载 `DeepSeek-V3 Technical Report` 的 HTML，保存为 `aiinfra/paper/deepseek_v3/deepseek_v3.html`，并设置 `<base href>` 使其本地可正确加载图片与样式。 |
| 部署 DeepGEMM / MoE 专题 | 为 `aiinfra/topics/deepgemm/` 和 `aiinfra/topics/moe/` 增加构建能力，接入根 `build.py`，并推送到 GitHub Pages。 |
| 拆分 Day 1-7 文件 | 将 `deepgemm`、`moe`、`cute` 三个专题的 `README.md` 中的 `## Day N...` 章节分别拆成独立的 `day1.md` ~ `day7.md`，`README.md` 保留概览。 |
| 统一 topics 构建逻辑 | 删除各专题独立的 `website/build.py`，改为 `aiinfra/topics/build.py` 自动发现所有含 `README.md` 的专题目录；新增目录无需再改构建脚本即可自动发布到 GitHub Pages。 |

> 现在新增一个专题只需：创建 `aiinfra/topics/<name>/README.md`（可选 `day1.md`~`day7.md` 和 `images/`），push 到 `main` 即可自动部署。
