---
name: leetcode-daily
description: 用于在 leetcode/daily 下编写每日 LeetCode 题解。规定了目录组织、题解文档结构、手绘 sketch 风 SVG 插图规范与网站构建集成。触发于"写 leetcode 每日题解"、"补全 daily 题解"、"加一道 leetcode 题解"等请求。
---

# 写 LeetCode Daily 题解 Skill

本工程(`ai-infra-notes`)的 LeetCode 每日题解遵循一套固定的目录组织、文档结构和插图规范。本 skill 描述如何产出符合仓库惯例的题解。

## 1. 目录组织

所有每日题解**必须写在 `leetcode/daily/` 下**，按"周 → 日"两级组织：

```
leetcode/daily/
├── week1/
│   ├── day1/
│   │   ├── 接雨水.md              # 题解主体（文件名 = 题目中文名）
│   │   └── images/                # 当日 SVG 插图（手绘 sketch 风）
│   ├── day2/
│   │   ├── 最大子数组和.md
│   │   └── images/
│   ├── day3/
│   │   ├── 无重复字符的最长子串.md
│   │   └── images/
│   ...
│   └── day7/                      # 每周 7 天，day1 ... day7
├── week2/
├── week3/
...
└── SKILL.md                       # 本文件
```

**规则**：

1. **题解根目录**：`leetcode/daily/`，不要写到 `leetcode/` 根下或其它位置。
2. **周级目录**：`weekN`（`week1`、`week2`、`week3` …），编号从 1 开始连续。
3. **日级目录**：每周固定 `day1` 到 `day7` 共 7 个目录，编号从 1 开始连续。
4. **题解文件名**：用题目的中文译名（如 `接雨水.md`、`最大子数组和.md`），与文档一级标题一致。
5. **图片目录**：每个 `dayN/` 下有一个 `images/` 子目录，存放当日 SVG 插图。

## 2. 题解文档结构

每篇题解 `.md` 遵循固定章节顺序（参考 `day1/接雨水.md`、`day2/最大子数组和.md`、`day3/无重复字符的最长子串.md`）：

```markdown
# <题目中文名>

- **题目名称**：<题目中文名>
- **链接**：[<题号>. <题目名>](https://leetcode.cn/problems/<slug>/)
- **难度**：简单 / 中等 / 困难
- **标签**：<算法标签，逗号分隔>

## 1. 题目概述
（题意描述 + 示例 1/2/3 + 约束条件）

## 2. 解题思路
### 2.1 暴力思路
### 2.2 核心观察：<最优解法名>        ← 配 SVG 图
### 2.3 算法流程图                     ← 配 SVG 图
### 2.4 示例演算                       ← 配 SVG 图（可选但推荐）

## 3. 参考代码
### C++
### Python

## 4. 复杂度分析
（三列表格：维度 / 复杂度 / 说明）

## 5. 扩展：<进阶解法或变体>（可选）

## 6. 面试要点
（4-5 题 Q&A，问题加粗，答案缩进）
```

**写作规范**：

- **中文为主**，概念加粗，善用 `> 💡` / `> ⚠️` blockquote。
- 代码块标注语言：` ```cpp` / ` ```python` / ` ```text`。
- 图片引用用相对路径：`![<中文alt>](images/<filename>.svg)`（从 `dayN/` 出发）。
- 每个 Day 平均引用 **2-4 张 SVG**。
- alt 文本用中文。

## 3. 图片风格：手绘 sketch 风（Excalidraw-like）

**所有插图统一为手绘 sketch 风**，追求"轻松白板涂鸦"的直观感，避免冰冷完美的矢量感。具体要求：

### 3.1 视觉特征

| 维度 | 要求 |
|------|------|
| **线条** | 手绘不均匀、略带抖动，避免完美直线或圆润矢量边 |
| **笔触** | 粗糙、类似马克笔/铅笔描边，线宽可略有变化 |
| **配色** | 极简，一般不超过 3-4 种柔和颜色（蓝 `#e8f0fe`/`#446688`、绿 `#e6f4ea`/`#4a7a3a`、橙 `#fff8e1`/`#d6a040`、红 `#fce4ec`/`#b85450`），背景白色或米白 `#fafafa` |
| **形状** | 简单几何块——矩形、网格、箭头、圆角框，不画复杂 3D 或写实元素 |
| **标签** | 手写感字体：英文用 `Comic Sans MS` / `Bradley Hand`，CJK 用 `Kaiti SC` / 楷体 |
| **整体感觉** | 轻松白板涂鸦，标注随意、轻微错位也无妨，优先可读性和直观性 |

### 3.2 SVG 实现技法（仓库统一用法）

用 SVG 滤镜 `feTurbulence` + `feDisplacementMap` 给所有图形叠加轻微抖动，实现手绘效果。**每张 SVG 顶部固定引入以下 `<defs>`**：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 360"
     font-family="'Comic Sans MS', 'Segoe UI', 'Kaiti SC', 楷体, cursive">
  <defs>
    <filter id="rough">
      <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="3"/>
      <feDisplacementMap in="SourceGraphic" scale="2"/>
    </filter>
    <filter id="rough2">
      <feTurbulence type="fractalNoise" baseFrequency="0.025" numOctaves="2" seed="7"/>
      <feDisplacementMap in="SourceGraphic" scale="1.5"/>
    </filter>
  </defs>

  <rect width="520" height="360" fill="#fafafa"/>

  <!-- 所有矩形/路径/文本都加 filter="url(#rough2)" -->
  <rect x="80" y="48" width="50" height="30" fill="#e8f0fe"
        stroke="#446688" stroke-width="1.5" rx="4" filter="url(#rough2)"/>
  <text x="105" y="68" text-anchor="middle" font-size="12" fill="#333">−2</text>
  ...
</svg>
```

**要点**：

- `font-family` 必须包含 `'Comic Sans MS'` 和 `'Kaiti SC', 楷体`，保证中英文都有手写感。
- 每个图形元素（`rect`/`path`/`text`/`ellipse`/`polygon`）都加 `filter="url(#rough2)"`。
- 圆角矩形用 `rx="4"`（普通格子）或 `rx="8"`~`10`（说明框）。
- 强调元素用更粗的 `stroke-width`（2~2.5）和红色/橙色填充。
- 箭头用 `<marker id="arr">` 定义，`<path>` 引用 `marker-end="url(#arr)"`。

### 3.3 SVG 命名

全小写 + 下划线，语义化，建议加题目缩写前缀避免冲突：

- `trap_water_column.svg`、`trap_two_pointers.svg`（接雨水）
- `maxsubarray_kadane.svg`、`maxsubarray_algorithm_flow.svg`（最大子数组和）
- `lswrc_sliding_window.svg`、`lswrc_algorithm_flow.svg`、`lswrc_example_walkthrough.svg`（无重复字符的最长子串）

### 3.4 常见图类型

| 图类型 | 用途 | 示例 |
|--------|------|------|
| **概念图** | 直观展示核心观察（窗口、状态转移、水位） | `lswrc_sliding_window.svg` |
| **流程图** | 决策菱形 + 矩形框 + 箭头，展示算法步骤 | `lswrc_algorithm_flow.svg` |
| **演算表** | 逐步表格展示示例执行过程 | `lswrc_example_walkthrough.svg` |
| **对比图** | 多行格子对比数据变化（如 dp 数组） | `maxsubarray_kadane_v2.svg` |

## 4. 网站构建集成

题解写完后会被 `leetcode/website/build.py` 自动读取并生成网页：

- `build.py` 递归扫描 `leetcode/` 下所有 `.md`（排除 `website/`、`images/`）。
- 解析一级标题 `# <title>` 作为侧边栏与列表页标题。
- 图片路径 `images/xxx.svg` 在题解页被重写为 `../images/xxx.svg`。
- **重要**：`build.py` 只扫描 `leetcode/images/` 作为网站图片目录，因此 `dayN/images/` 下的 SVG **必须同步复制一份到 `leetcode/images/`**，否则网页图片会裂开。

**验证命令**：

```bash
python3 leetcode/website/build.py   # 单独构建 leetcode 网站
python3 build.py                     # 组合构建全站（含 leetcode）
```

**自检清单**：

- [ ] 题解位于 `leetcode/daily/weekN/dayM/<题目名>.md`
- [ ] 一级标题 `# <题目中文名>` 与文件名一致
- [ ] 含 2-4 张 SVG 插图，引用格式 `![中文alt](images/xxx.svg)`
- [ ] SVG 为手绘 sketch 风（含 `feTurbulence` 抖动滤镜 + Comic Sans/Kaiti SC 字体）
- [ ] SVG 同步复制到 `leetcode/images/`
- [ ] `python3 build.py` 成功生成对应 `public/leetcode/problems/<题目名>.html`
