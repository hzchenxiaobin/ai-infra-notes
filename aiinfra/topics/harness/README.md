# Harness Engineering 一周学习计划

> **适用对象**：零基础——无需 CUDA / 编译器 / 算子库前置知识，只要你写过代码、用过 AI 编程助手即可
> **本周目标**：理解 Harness Engineering 的核心范式（"人类掌舵，智能体执行"），掌握 Agent = Model + Harness 的组件清单，能用 AGENTS.md + 自定义 linter + 反馈回路搭建一个最小可用的 harness，让 AI 智能体在约束下可靠地完成编码任务
> **时间投入**：工作日每天 2.5h，周末每天 5h，周计 22.5h
> **周日里程碑**：为一个现有小项目搭建完整 harness（AGENTS.md + 结构化约束 + 反馈回路 + 背压门控），记录智能体在 harness 内外的行为差异

---

## 为什么学 Harness Engineering

传统软件工程是"人类写代码 → 机器执行代码"。Harness Engineering 是 OpenAI 在 2026 年 2 月提出的范式转变：

```
传统工程：          人类写代码 → 机器执行代码
Harness Engineering：人类设计约束 → 智能体写代码 → 机器执行代码
```

核心转变：**工程师的产出从代码变成了约束系统**——AGENTS.md、架构规则、自定义 linter、反馈回路。

| 维度 | 传统工程 | Harness Engineering |
|------|----------|---------------------|
| 工程师产出 | 代码 | 约束系统（AGENTS.md / linter / 反馈回路） |
| 人类角色 | 执行者 | 掌舵者（设计环境、明确意图） |
| 智能体角色 | 辅助 | 执行者（在约束内自主写代码） |
| 质量保障 | Code Review + 测试 | 机械化护栏（lint + 结构测试 + 背压门控） |
| 知识载体 | 散落在文档/Slack/人脑 | 仓库即记录系统 |
| 效率 | 人工编码 | ~1/10 时间（OpenAI 实测） |

> 💡 **一句话总结**：Harness = 模型之外的一切代码、配置和执行逻辑。裸模型只是文本输入/输出引擎，当 harness 给它状态、工具、反馈回路和可执行约束时，它才成为智能体。

---

## 本周学习计划

| 天数 | 主题 | 核心产出 |
|------|------|----------|
| Day 1 | Harness 总览与核心范式 | 理解 Agent = Model + Harness，画出组件清单 |
| Day 2 | 仓库即记录系统 + 地图而非手册 | 为一个项目写出第一版 AGENTS.md |
| Day 3 | 机械化执行 | 实现一个自定义 linter 规则 + 结构测试 |
| Day 4 | 智能体可读性 + 反馈回路 | 搭建背压门控（测试/构建/类型检查 = 自我验证） |
| Day 5 | 熵管理与吞吐量理念 | 编写"黄金规则"文档 + 设计漂移扫描方案 |
| Day 6 | Harness 综合实战 | 为真实项目搭建完整 harness，对比 harness 内外行为 |
| Day 7 | 进阶专题与总结 | Ralph 循环实践 + 面试题复盘 + 知识图谱 |

---

## 前置准备

- 会用任意编程语言写基本程序（Python / TypeScript / Go 均可）
- 用过 AI 编程助手（Cursor / Claude Code / GitHub Copilot 等）
- 有 GitHub 账号，会用 git 基本操作
- 本机装有一个 AI 编程助手（推荐 Claude Code 或 Cursor）

```bash
# 验证 git 可用
git --version

# 验证 AI 编程助手可用（以 Claude Code 为例）
claude --version

# 准备一个练手用的小项目（可以是任意 < 500 行的项目）
# 如果没有，创建一个：
mkdir my-harness-practice && cd my-harness-practice
git init
```

---

## Day 1（周一）：Harness 总览与核心范式

> **今日目标**：理解 Harness Engineering 的来龙去脉，掌握 Agent = Model + Harness 公式，能画出 harness 的完整组件清单
> **面试考察度**：⭐⭐⭐ 了解级，能说清 harness 是什么、为什么需要它

---

### 学习任务 1：Harness Engineering 的起源（45 分钟）

#### 阅读内容

- **OpenAI 原文**：[Harness Engineering: Harnessing Codex in an Agent-First World](https://openai.com/zh-Hans-CN/index/harness-engineering/)
- **命名出处**：[Mitchell Hashimoto: Engineer the Harness](https://mitchellh.com/writing/my-ai-adoption-journey#step-5-engineer-the-harness)
- **概念扩展**：Martin Fowler — "Harness Engineering" 正式版与备忘录

#### 核心要点

Harness Engineering 的背景：OpenAI 的 Ryan Lopopolo 用 3 人团队、5 个月时间，让 Codex 从空仓库产出了 ~100 万行代码、~1500 个 PR，**零手写代码**。

| 指标 | 数据 |
|------|------|
| 团队规模 | 3 人 → 7 人 |
| 时间跨度 | 5 个月 |
| 代码量 | ~100 万行 |
| PR 数量 | ~1,500 个 |
| 人均日 PR | 3.5 个（扩展后仍在增长） |
| 单次运行时长 | 6+ 小时（通常在人类睡眠时间） |
| 效率估算 | 手工编写的 ~1/10 时间 |

> 💡 **一句话总结**：这不是"用 AI 辅助写代码"，而是"工程师完全不写代码，只设计约束系统让 AI 写代码"——工程角色的根本性转变。

### 学习任务 2：Agent = Model + Harness（45 分钟）

#### 公式拆解

```
Agent = Model + Harness
Harness = 模型之外的一切代码、配置和执行逻辑
```

裸模型只是文本输入/输出引擎，它**不能**：
- 维护状态（没有记忆）
- 执行代码（没有工具）
- 访问实时知识（没有文件系统/网络）
- 搭建环境（没有沙箱）

当 harness 给它这些能力时，它才成为智能体。

#### Harness 完整组件清单（三方综合）

| 来源 | 组件 | 说明 |
|------|------|------|
| LangChain | System Prompts | AGENTS.md、CLAUDE.md |
| | Tools & MCP | 扩展智能体能力的工具和协议 |
| | Skills | 渐进式加载的知识包 |
| | 沙箱基础设施 | 文件系统、浏览器、隔离执行环境 |
| | 编排逻辑 | 子智能体生成、handoff、模型路由 |
| | Hooks/中间件 | compaction、续接、lint 检查 |
| HumanLayer | AGENTS.md | ≤60 行，禁止自动生成 |
| | MCP Servers | 信任边界 + 工具数量控制 |
| | Skills | 渐进式披露，按需加载 |
| | Sub-Agents | 上下文防火墙，隔离防 context rot |
| | Hooks | 生命周期脚本，成功静默/失败报错 |
| | Back-Pressure | 测试/构建/类型检查 = 自我验证回路 |
| Martin Fowler | Context Engineering | 知识库 + 动态上下文 |
| | Architectural Constraints | LLM 审查 + linter + 结构测试 |
| | Garbage Collection Agents | 定期扫描 + 修复漂移 |

> ⚠️ **注意**：不要被组件数量吓到。Day 1 只需要理解"有哪些零件"，Day 2-6 会逐一动手搭建。

### 学习任务 3：六大核心概念速览（30 分钟）

| # | 概念 | 一句话 |
|---|------|--------|
| 1 | 仓库即记录系统 | 不在仓库里的东西，对智能体不存在 |
| 2 | 地图而非手册 | AGENTS.md 是目录页，不是百科全书 |
| 3 | 机械化执行 | 文档会腐烂，lint 规则不会 |
| 4 | 智能体可读性 | 优先为智能体的推理能力优化 |
| 5 | 吞吐量改变合并理念 | 纠错成本低，等待成本高 |
| 6 | 熵管理 = 垃圾回收 | 技术债是高息贷款 |

总纲：**人类掌舵，智能体执行**。

### 今日检查清单

- [ ] 能说出 Agent = Model + Harness 并解释每个词
- [ ] 能列出 harness 的至少 6 个组件
- [ ] 能说出六大核心概念的名称和一句话解释
- [ ] 读了 OpenAI 原文和 Hashimoto 博客
- [ ] 在笔记中记录了 OpenAI 的关键数据点

---

## Day 2（周二）：仓库即记录系统 + 地图而非手册

> **今日目标**：理解"仓库即记录系统"和"地图而非手册"两大概念，为一个项目写出第一版 AGENTS.md
> **面试考察度**：⭐⭐⭐⭐ 实践级，能独立写出合格的 AGENTS.md

---

### 学习任务 1：仓库即记录系统（45 分钟）

#### 核心原则

```
Slack 讨论 = 对智能体不可见
Google Docs = 对智能体不可见
脑子里的知识 = 对智能体不可见
```

一切决策、规范、计划都必须以**版本化工件**提交到仓库。仓库是智能体唯一的"现实"。

#### 实践：审计你的项目

打开你的练手项目，检查以下内容是否都在仓库里：

| 内容 | 在仓库里？ | 应该放哪 |
|------|-----------|----------|
| 项目目标 | | README.md |
| 架构决策 | | docs/adr/ 或 README.md |
| 编码规范 | | AGENTS.md 或 .editorconfig |
| 依赖说明 | | package.json / requirements.txt |
| 环境搭建步骤 | | README.md 或 Makefile |
| 已知问题 | | GitHub Issues 或 TODO.md |
| 测试约定 | | AGENTS.md |

### 学习任务 2：地图而非手册（45 分钟）

#### AGENTS.md 的设计哲学

AGENTS.md ≈ 目录页（~100 行），不是百科全书。核心原则：**渐进式披露**——智能体从小入口点开始，被指导下一步该看什么。

巨型指令文件的三个死因：
1. **挤占上下文**：AGENTS.md 太长会挤占智能体的 context window
2. **无法维护**：超过 200 行的文件没人会持续更新
3. **无法机械验证**：自然语言描述的规则无法被 linter 检查

#### AGENTS.md 的结构模板

```markdown
# <项目名>

> 一句话定位

## 仓库结构

| 目录 | 内容 | 说明 |
|------|------|------|
| `src/` | 源码 | 主逻辑 |
| `tests/` | 测试 | 单元测试 |
| `docs/` | 文档 | 架构决策、设计文档 |

## 开发约定

- 语言：Python 3.10+
- 测试：pytest，覆盖率 > 80%
- 提交：conventional commits

## 导航

- 编码规范详见 [docs/coding-standards.md](docs/coding-standards.md)
- 架构决策详见 [docs/adr/](docs/adr/)
- 环境搭建详见 [README.md](README.md)

## 机械化检查

`bash scripts/check.sh` 运行所有检查（lint + 测试 + 类型检查）
```

> ⚠️ **注意**：AGENTS.md ≤ 60 行（HumanLayer 建议）。超过就拆分到子文档，用导航链接指向。

### 学习任务 3：动手写 AGENTS.md（45 分钟）

为你的练手项目创建 `AGENTS.md`：

```bash
# 在你的练手项目根目录
touch AGENTS.md
```

要求：
1. ≤ 60 行
2. 包含仓库结构表
3. 包含开发约定（语言、测试、提交规范）
4. 包含导航（指向更详细的子文档）
5. 包含机械化检查说明
6. **禁止自动生成**——必须手写，因为手写的过程就是梳理项目约束的过程

### 今日检查清单

- [ ] 理解"不在仓库里的东西对智能体不存在"
- [ ] 能说出巨型指令文件的三个死因
- [ ] AGENTS.md ≤ 60 行，手写完成
- [ ] AGENTS.md 包含仓库结构、开发约定、导航、检查说明
- [ ] 审计了练手项目，确保所有关键信息都在仓库里

---

## Day 3（周三）：机械化执行

> **今日目标**：理解"文档会腐烂，lint 规则不会"，实现一个自定义 linter 规则 + 结构测试
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，机械化执行是 harness 的基石

---

### 学习任务 1：为什么文档会腐烂（30 分钟）

#### 文档 vs 机械化规则

| 维度 | 自然语言文档 | Lint 规则 |
|------|-------------|-----------|
| 执行 | 靠人读 | 自动执行 |
| 腐烂 | 随时间过时 | 代码即文档，始终同步 |
| 纠错 | 靠人发现 | 智能体可自我纠正 |
| 强制力 | 软约束（可忽略） | 硬约束（阻断提交） |

> 💡 **一句话总结**：把"你应该做 X"从文档变成 lint 规则——错误信息里内嵌修复指令，智能体看到 lint 报错就能自我纠正。

### 学习任务 2：自定义 Linter 规则（60 分钟）

以 Python 项目为例，用 ruff 自定义规则：

```bash
# 安装 ruff
pip install ruff

# 创建配置
cat > pyproject.toml << 'EOF'
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["N802"]  # 测试函数可以用大写开头
EOF
```

#### 关键技巧：错误信息内嵌修复指令

```python
# scripts/custom_checks.py —— 自定义结构检查
"""项目级结构约束：lint 规则无法覆盖的部分用脚本守护。"""

import sys
from pathlib import Path

def check_agents_md_exists():
    """AGENTS.md 必须存在——智能体的入口文件。"""
    if not Path("AGENTS.md").exists():
        print("ERROR: AGENTS.md not found. "
              "Fix: create AGENTS.md with repo structure, conventions, "
              "and navigation. See template in docs/agents-template.md")
        return False
    return True

def check_agents_md_length():
    """AGENTS.md ≤ 60 行——防止挤占智能体上下文。"""
    lines = Path("AGENTS.md").read_text().splitlines()
    if len(lines) > 60:
        print(f"ERROR: AGENTS.md has {len(lines)} lines (max 60). "
              f"Fix: split content into sub-documents under docs/ "
              f"and add navigation links in AGENTS.md")
        return False
    return True

def check_no_circular_deps():
    """src/ 下的模块不能有循环依赖。"""
    # 简化版：检查 import 图是否有环
    # 完整版用 grimp 或 pydeps
    pass

if __name__ == "__main__":
    checks = [check_agents_md_exists, check_agents_md_length]
    failed = [c.__name__ for c in checks if not c()]
    if failed:
        print(f"\nFailed checks: {', '.join(failed)}")
        sys.exit(1)
    print("All checks passed.")
```

```bash
# 运行
python scripts/custom_checks.py
```

```text
# 预期输出
All checks passed.
```

### 学习任务 3：结构测试（45 分钟）

结构测试 = 验证代码结构的测试，不验证行为，验证"架构规则是否被遵守"。

```python
# tests/test_structure.py —— 结构测试
"""验证项目结构约束。这些测试不测功能，只测结构。"""

from pathlib import Path

def test_agents_md_exists():
    assert Path("AGENTS.md").exists(), "AGENTS.md must exist"

def test_agents_md_under_60_lines():
    lines = Path("AGENTS.md").read_text().splitlines()
    assert len(lines) <= 60, f"AGENTS.md has {len(lines)} lines, max 60"

def test_no_direct_db_access_in_routes():
    """routes/ 不能直接访问 database/——必须通过 service/ 层。"""
    routes_dir = Path("src/routes")
    if not routes_dir.exists():
        return  # 没有这个目录就跳过
    for py_file in routes_dir.glob("*.py"):
        content = py_file.read_text()
        assert "from database" not in content, \
            f"{py_file.name}: routes must not import database directly. " \
            f"Fix: route through service/ layer instead."

def test_all_modules_have_init():
    """src/ 下每个目录都有 __init__.py。"""
    src = Path("src")
    for subdir in src.iterdir():
        if subdir.is_dir():
            assert (subdir / "__init__.py").exists(), \
                f"{subdir}/ is missing __init__.py"
```

```bash
# 运行结构测试
pytest tests/test_structure.py -v
```

### 今日检查清单

- [ ] 理解"文档会腐烂，lint 规则不会"
- [ ] 为项目配置了 ruff（或对应语言的 linter）
- [ ] 实现了至少 2 个自定义结构检查（scripts/custom_checks.py）
- [ ] 实现了至少 2 个结构测试（tests/test_structure.py）
- [ ] 所有错误信息都内嵌了修复指令
- [ ] `python scripts/custom_checks.py` 和 `pytest tests/test_structure.py` 都通过

---

## Day 4（周四）：智能体可读性 + 反馈回路

> **今日目标**：理解"智能体可读性"原则，搭建背压门控让智能体自我验证
> **面试考察度**：⭐⭐⭐⭐ 实践级，背压门控是 harness 的核心反馈机制

---

### 学习任务 1：智能体可读性（45 分钟）

#### 核心原则

智能体不是人类，它的"可读性"标准不同：

| 维度 | 人类可读性 | 智能体可读性 |
|------|-----------|-------------|
| API 选择 | 新潮框架也行 | 选"无聊"技术（API 稳定、训练集覆盖好） |
| 文档形式 | 详细教程 | 结构化、可机器解析 |
| 错误信息 | 人能看懂就行 | 内嵌修复指令 |
| 代码结构 | 灵活 | 清晰的模块边界 |

#### "无聊"技术原则

选择 API 稳定、在训练数据中覆盖好的技术：

| 推荐 | 不推荐 | 原因 |
|------|--------|------|
| Python 3.10+ | 最新 nightly 版 | 稳定，训练集覆盖 |
| FastAPI | 自研框架 | 文档丰富，API 稳定 |
| SQLite | 自研存储引擎 | 智能体见过无数次 |
| pytest | 自研测试框架 | 约定俗成 |

> 💡 **关键洞察**：有时重新实现一个子集比包装不透明的上游行为更划算——因为智能体能理解你重写的代码，但理解不了黑盒包装。

### 学习任务 2：反馈回路设计（30 分钟）

#### Guides × Sensors 矩阵

| | 计算性（确定性） | 推理性（语义） |
|--|---------|---------|
| **引导器/前馈** | bootstrap 脚本、LSP、类型检查 | AGENTS.md、Skills、architecture.md |
| **传感器/反馈** | linter、结构测试、覆盖率 | AI code review、LLM-as-judge |

- **引导器（前馈）**：在智能体行动**之前**引导它，增加首次成功概率
- **传感器（反馈）**：在智能体行动**之后**观察，启用自我纠正

> ⚠️ **注意**：单独使用任一维度都不行——只有反馈 = 反复犯同样错误；只有前馈 = 不知道规则是否生效。

### 学习任务 3：搭建背压门控（60 分钟）

背压（Back-Pressure）= 智能体做完任务后，系统自动验证结果，不通过就拒绝。

#### Ralph 信条映射

| Ralph 信条 | Harness 对应 |
|-----------|-------------|
| Backpressure Over Prescription | 不规定怎么做，但门控拒绝坏结果 |
| Fresh Context Is Reliability | 每次迭代重新读取，防止 context rot |

#### 实现一个完整的检查脚本

```bash
#!/bin/bash
# scripts/check.sh —— 完整的背压门控脚本
# 智能体完成任务后必须跑这个脚本，全绿才算完成

set -euo pipefail

echo "=== 1. Lint 检查 ==="
ruff check src/ tests/ || { echo "FAIL: lint errors"; exit 1; }

echo "=== 2. 类型检查 ==="
mypy src/ --strict || { echo "FAIL: type errors"; exit 1; }

echo "=== 3. 结构测试 ==="
pytest tests/test_structure.py -v || { echo "FAIL: structure tests"; exit 1; }

echo "=== 4. 单元测试 ==="
pytest tests/ --cov=src --cov-report=term-missing || { echo "FAIL: unit tests"; exit 1; }

echo "=== 5. AGENTS.md 检查 ==="
python scripts/custom_checks.py || { echo "FAIL: custom checks"; exit 1; }

echo ""
echo "✅ All checks passed. Safe to proceed."
```

```bash
chmod +x scripts/check.sh
./scripts/check.sh
```

#### 让智能体使用背压

在 AGENTS.md 中加入：

```markdown
## 完成任务后的检查

完成任务后，必须运行 `bash scripts/check.sh` 并确保全绿。
如果有失败，根据错误信息中的 Fix 提示自我纠正，然后重新运行。
```

### 今日检查清单

- [ ] 理解"无聊"技术原则，审计了项目的技术选型
- [ ] 能画出 Guides × Sensors 2×2 矩阵
- [ ] 实现了完整的 `scripts/check.sh` 背压门控脚本
- [ ] 背压脚本覆盖至少 5 项检查（lint / 类型 / 结构测试 / 单元测试 / 自定义检查）
- [ ] 在 AGENTS.md 中加入了"完成任务后必须运行检查"的说明

---

## Day 5（周五）：熵管理与吞吐量理念

> **今日目标**：理解"熵管理 = 垃圾回收"和"吞吐量改变合并理念"，编写黄金规则文档 + 设计漂移扫描方案
> **面试考察度**：⭐⭐⭐⭐ 理解级，能解释为什么智能体会复现坏模式

---

### 学习任务 1：熵管理 = 垃圾回收（45 分钟）

#### 核心问题

智能体会**复现仓库中已有的模式**——包括坏模式。如果你有一个写得很烂的文件，智能体会以它为"范例"生成更多烂代码。

#### 熵管理的三层

```
1. 黄金规则编码：把"应该怎么做"写进仓库
2. 定期扫描偏差：后台任务扫描偏离黄金规则的代码
3. 修复漂移：发起针对性重构 PR
```

技术债 = 高息贷款，需要**小额持续偿还**。

#### 实践：编写黄金规则文档

```markdown
<!-- docs/golden-rules.md -->

# 黄金规则

> 这些规则是项目的"不变量"。智能体生成的代码必须遵守。

## GR-1：函数不超过 50 行

**为什么**：长函数难以测试、难以理解、难以让智能体正确修改。

**检查**：`scripts/check.sh` 中的 `check_function_length`

## GR-2：所有 public 函数必须有类型注解

**为什么**：类型注解是智能体理解函数契约的主要途径。

**检查**：`mypy --strict`

## GR-3：禁止在 routes/ 层直接访问 database/

**为什么**：违反分层架构，导致逻辑耦合。

**检查**：`tests/test_structure.py::test_no_direct_db_access_in_routes`

## GR-4：每个模块必须有对应的测试文件

**为什么**：无测试的代码智能体不敢改、不敢删。

**检查**：`pytest --cov` 覆盖率 > 80%
```

### 学习任务 2：吞吐量改变合并理念（30 分钟）

#### 核心转变

| 传统理念 | Harness 理念 |
|----------|-------------|
| PR 要仔细审查，慢慢合并 | 纠错成本低，等待成本高 |
| 测试必须全绿才合并 | 偶发失败通过后续重跑解决 |
| 人类 Review 是质量门 | 机械化检查是质量门 |

在智能体吞吐量远超人类注意力的系统中，**快速合并 + 快速纠错**比"慢慢审查"更高效。

> ⚠️ **注意**：这不适用于所有场景。安全关键代码、公共 API、不可逆变更仍需人类审查。区分"可快速纠错"和"不可逆"是关键。

### 学习任务 3：设计漂移扫描方案（45 分钟）

```python
# scripts/scan_drift.py —— 漂移扫描脚本
"""定期运行，扫描代码是否偏离黄金规则。"""

import ast
import json
from pathlib import Path

def scan_function_length(max_lines=50):
    """扫描超过 max_lines 行的函数。"""
    violations = []
    for py_file in Path("src").rglob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > max_lines:
                    violations.append({
                        "file": str(py_file),
                        "function": node.name,
                        "line": node.lineno,
                        "length": length,
                        "rule": "GR-1",
                        "fix": f"Split {node.name} into smaller functions (<{max_lines} lines)"
                    })
    return violations

def scan_missing_type_hints():
    """扫描缺少类型注解的 public 函数。"""
    violations = []
    for py_file in Path("src").rglob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    if not node.returns:
                        violations.append({
                            "file": str(py_file),
                            "function": node.name,
                            "line": node.lineno,
                            "rule": "GR-2",
                            "fix": f"Add return type annotation to {node.name}"
                        })
    return violations

if __name__ == "__main__":
    all_violations = []
    all_violations.extend(scan_function_length())
    all_violations.extend(scan_missing_type_hints())

    if all_violations:
        print(json.dumps(all_violations, indent=2, ensure_ascii=False))
        print(f"\nTotal violations: {len(all_violations)}")
    else:
        print("No drift detected. All golden rules satisfied.")
```

```bash
python scripts/scan_drift.py
```

### 今日检查清单

- [ ] 理解"智能体会复现仓库中已有的模式——包括坏模式"
- [ ] 编写了 `docs/golden-rules.md`，至少 4 条黄金规则
- [ ] 每条黄金规则都有对应的机械化检查方式
- [ ] 理解"纠错成本低，等待成本高"的合并理念
- [ ] 实现了 `scripts/scan_drift.py` 漂移扫描脚本
- [ ] 运行漂移扫描，记录了当前项目的偏差

---

## Day 6（周六）：Harness 综合实战

> **今日目标**：为真实项目搭建完整 harness，对比智能体在 harness 内外的行为差异
> **面试考察度**：⭐⭐⭐⭐⭐ 综合应用，能端到端搭建 harness

---

### 学习任务 1：完整 Harness 搭建（90 分钟）

将 Day 2-5 的所有产出组装成一个完整的 harness：

#### 目标文件结构

```
my-harness-practice/
├── AGENTS.md                    # Day 2: 入口文件（≤60 行）
├── docs/
│   ├── golden-rules.md          # Day 5: 黄金规则
│   ├── coding-standards.md      # Day 2: 编码规范细节
│   └── architecture.md          # Day 2: 架构说明
├── scripts/
│   ├── check.sh                 # Day 4: 背压门控
│   ├── custom_checks.py         # Day 3: 自定义结构检查
│   └── scan_drift.py            # Day 5: 漂移扫描
├── tests/
│   └── test_structure.py        # Day 3: 结构测试
├── pyproject.toml               # Day 3: linter 配置
└── src/
    └── (你的项目代码)
```

#### 检查你的 AGENTS.md 是否完整

```markdown
# <项目名>

> 一句话定位

## 仓库结构

| 目录 | 内容 |
|------|------|
| `src/` | 主逻辑 |
| `tests/` | 单元测试 + 结构测试 |
| `docs/` | 黄金规则、编码规范、架构说明 |
| `scripts/` | 检查脚本 |

## 开发约定

- 语言：Python 3.10+
- Lint：ruff（配置见 pyproject.toml）
- 测试：pytest，覆盖率 > 80%
- 提交：conventional commits

## 黄金规则

详见 [docs/golden-rules.md](docs/golden-rules.md)

## 导航

- 编码规范：[docs/coding-standards.md](docs/coding-standards.md)
- 架构说明：[docs/architecture.md](docs/architecture.md)

## 机械化检查

完成任务后必须运行：`bash scripts/check.sh`
漂移扫描：`python scripts/scan_drift.py`
```

### 学习任务 2：对比实验——Harness 内 vs 外（60 分钟）

#### 实验设计

给智能体同一个任务，分别在"有 harness"和"无 harness"的环境下执行，对比结果。

**任务示例**："给项目添加一个用户注册功能"

#### 实验A：无 Harness

```bash
# 创建一个没有 harness 的副本
cp -r my-harness-practice my-no-harness
cd my-no-harness
rm AGENTS.md docs/ scripts/ tests/test_structure.py pyproject.toml

# 让 AI 助手直接写功能
# 记录：代码质量、是否符合项目规范、是否有类型注解、是否可测试
```

#### 实验B：有 Harness

```bash
cd my-harness-practice

# 让 AI 助手在 harness 约束下写功能
# AGENTS.md 告诉它项目结构、约定、检查方式
# 完成后运行 bash scripts/check.sh
# 记录：是否一次通过检查、自我纠正了几次、最终质量
```

#### 记录对比

| 维度 | 无 Harness | 有 Harness |
|------|-----------|-----------|
| 一次通过检查 | | |
| 自我纠正次数 | | |
| 类型注解完整度 | | |
| 测试覆盖率 | | |
| 符合分层架构 | | |
| 函数长度合规 | | |

### 学习任务 3：Harnessability 评估（30 分钟）

不是所有代码库都同样适合被 harness。评估你的项目：

| 维度 | 评估 | 说明 |
|------|------|------|
| 强类型 | ✅/❌ | 类型检查是天然传感器 |
| 清晰模块边界 | ✅/❌ | 支持架构约束规则 |
| 成熟框架 | ✅/❌ | 隐式提高智能体成功概率 |
| 测试覆盖 | ✅/❌ | 背压门控的基础 |
| 文档完备 | ✅/❌ | 仓库即记录系统的基础 |

> 💡 **Ashby 必要多样性定律**：调节器必须至少拥有与被调节系统同等的多样性。LLM 能生成几乎任何东西（高多样性）→ 选定拓扑结构 = 削减多样性 → 全面 harness 变得可行。这就是"约束越严，自主性越强"的控制论根基。

### 今日检查清单

- [ ] 完整的 harness 文件结构搭建完毕
- [ ] AGENTS.md ≤ 60 行，包含所有必要部分
- [ ] `bash scripts/check.sh` 全绿
- [ ] 完成了 Harness 内 vs 外的对比实验
- [ ] 记录了对比数据（至少 5 个维度）
- [ ] 评估了项目的 Harnessability

---

## Day 7（周日）：进阶专题与总结

> **今日目标**：了解 Ralph 循环、模型与 harness 的耦合、Loop Engineering 等进阶方向，完成面试复盘
> **面试考察度**：⭐⭐⭐ 了解级，能说出进阶方向和 tradeoff

---

### 学习任务 1：Ralph 循环——Harness 的实战模式（45 分钟）

#### Ralph Wiggum 循环

Ralph 循环是 Harness Engineering 的核心实现模式：让智能体在循环中自主工作直到任务完成。

```
while PRD not fully satisfied:
    spawn AI agent (fresh context)
    agent reads AGENTS.md + PRD + repo state
    agent writes code
    run backpressure checks (scripts/check.sh)
    if checks pass:
        commit
    else:
        agent self-corrects based on error messages
```

#### Ralph 六条信条

| 信条 | 含义 | Harness 对应 |
|------|------|-------------|
| Fresh Context Is Reliability | 每次迭代清空上下文 | 智能体可读性 |
| Backpressure Over Prescription | 不规定怎么做，门控拒绝坏结果 | 机械化执行 |
| The Plan Is Disposable | 计划可随时重新生成 | 熵管理 |
| Disk Is State, Git Is Memory | 文件是交接机制 | 仓库即记录系统 |
| Steer With Signals, Not Scripts | 加路标，不加脚本 | 人类掌舵 |
| Let Ralph Ralph | 坐在循环上，不坐在循环里 | 智能体执行 |

> 💡 **关键洞察**："坐在循环上，不坐在循环里"——人类监控循环的运行，在关键节点介入掌舵，但不深入到每一步执行中。

### 学习任务 2：模型与 Harness 的耦合（30 分钟）

#### LangChain 的关键发现

- 模型在 post-training 阶段与特定 harness **共同训练**
- 模型可能 **overfit 到特定 harness**，换 harness 后表现暴跌
- Terminal Bench 2.0 数据：**纯 harness 优化**可以把排名从 Top 30 拉到 Top 5
- 推论：最适合你任务的 harness，不一定是模型 post-training 时用的那个

| 发现 | 启示 |
|------|------|
| 模型 overfit 到特定 harness | 换模型时要验证 harness 是否还合适 |
| 纯 harness 优化收益巨大 | 投入 harness 优化的 ROI 可能高于换模型 |
| 最优 harness 因任务而异 | 不要盲目复制别人的 harness |

### 学习任务 3：三个规制维度（30 分钟）

| 维度 | 成熟度 | 说明 |
|------|--------|------|
| 可维护性 Harness | 最成熟 | 内部代码质量，现有工具丰富（lint/格式化/覆盖率） |
| 架构适应度 Harness | 中等 | 本质是 Fitness Functions（结构测试/依赖规则） |
| 行为 Harness | **最弱** | "房间里的大象"——功能正确性验证仍无可靠答案 |

> ⚠️ **注意**：行为 Harness 是当前最薄弱的环节。你的 harness 能保证代码"结构正确"（lint 通过、类型正确、分层合规），但很难保证"行为正确"（功能真的对）。这就是为什么测试覆盖率仍然是最后的防线。

### 学习任务 4：面试题复盘（45 分钟）

#### 高频面试题

1. **什么是 Harness Engineering？它和传统工程有什么区别？**
   - 工程师不再写代码，而是设计环境、明确意图、构建反馈回路，让 AI 智能体可靠地完成工作
   - 核心转变：产出从代码变成约束系统（AGENTS.md / linter / 反馈回路）
   - 公式：Agent = Model + Harness

2. **Agent = Model + Harness 是什么意思？**
   - 裸模型只是文本输入/输出引擎，不能维护状态、执行代码、访问实时知识
   - Harness = 模型之外的一切：System Prompts、Tools、Skills、沙箱、编排逻辑、Hooks、背压
   - 当 harness 给模型状态、工具、反馈回路时，它才成为智能体

3. **什么是"仓库即记录系统"？**
   - 不在仓库里的东西，对智能体不存在
   - Slack 讨论、Google Docs、脑子里的知识 = 对智能体不可见
   - 一切决策、规范、计划必须以版本化工件提交到仓库

4. **为什么 AGENTS.md 要"地图而非手册"？**
   - AGENTS.md 是目录页（~60 行），不是百科全书
   - 巨型指令文件的三个死因：挤占上下文、无法维护、无法机械验证
   - 渐进式披露：智能体从小入口开始，被指导下一步看什么

5. **什么是"机械化执行"？为什么比文档好？**
   - 文档会腐烂，lint 规则不会
   - 自定义 linter + 结构测试 = 不变量的守护者
   - lint 错误信息里内嵌修复指令，智能体可以自我纠正

6. **什么是背压（Back-Pressure）？**
   - 智能体做完任务后，系统自动验证结果，不通过就拒绝
   - 不规定怎么做（给自主权），但门控拒绝坏结果（给约束）
   - 实现：scripts/check.sh = lint + 类型检查 + 结构测试 + 单元测试

7. **什么是熵管理？为什么智能体会复现坏模式？**
   - 智能体以仓库中已有代码为范例，包括坏代码
   - 熵管理三层：黄金规则编码 → 定期扫描偏差 → 修复漂移
   - 技术债 = 高息贷款，需小额持续偿还

8. **"吞吐量改变合并理念"是什么意思？**
   - 纠错成本低，等待成本高
   - 在智能体吞吐量远超人类注意力的系统中，快速合并 + 快速纠错更高效
   - 区分"可快速纠错"和"不可逆"是关键

9. **Guides × Sensors 矩阵是什么？**
   - 引导器（前馈）：行动前引导，增加首次成功率
   - 传感器（反馈）：行动后观察，启用自我纠正
   - 计算性 vs 推理性：确定性/便宜 vs 概率性/贵
   - 单独使用任一维度都不行

10. **模型和 harness 的耦合关系是什么？**
    - 模型在 post-training 阶段与特定 harness 共同训练
    - 模型可能 overfit 到特定 harness
    - 纯 harness 优化可以把排名从 Top 30 拉到 Top 5

### 学习任务 5：总结与知识图谱（30 分钟）

#### 本周知识图谱

```
                Harness Engineering
               /        |          \
          范式转变    组件清单      实战模式
          /    \      / | \ \         |
     人类掌舵  约束系统  AGENTS  Tools  Skills  Hooks  背压    Ralph 循环
          \    /      \  |  / /         |
         智能体执行     六大概念                    Fresh Context
                      /  |  |  |  |  |  \         Backpressure
                     1   2  3  4  5   6            Let Ralph Ralph
                     |   |  |  |  |   |
                  仓库  地图 机械 可读 吞吐 熵管理
                  记录  而非 执行  性  改变 = GC
                  系统  手册            合并
                        |              |
                     ≤60行        黄金规则+扫描
                        |              |
                     导航链接      scan_drift.py
                        |              |
                     渐进式披露     小额持续偿还
                                     |
                              Ashby 必要多样性定律
                              "约束越严，自主性越强"
```

#### 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [OpenAI — Harness Engineering 原文](https://openai.com/zh-Hans-CN/index/harness-engineering/) | 官方 | ⭐ 必读 |
| [Mitchell Hashimoto: Engineer the Harness](https://mitchellh.com/writing/my-ai-adoption-journey) | 博客 | ⭐ 必读 |
| [Martin Fowler — Harness Engineering 正式版](https://martinfowler.com/articles/harness-engineering.html) | 博客 | ⭐ 必读 |
| [snarktank/ralph](https://github.com/snarktank/ralph) | 开源项目 | 📌 推荐 |
| [LangChain — Scaling Managed Agents](https://blog.langchain.dev/scaling-managed-agents/) | 博客 | 📌 推荐 |
| [Anthropic — How We Contain Claude](https://www.anthropic.com/research/containment) | 官方 | 📌 推荐 |
| [harness-engineering 学习档案](https://github.com/deusyu/harness-engineering) | 社区 | 📎 参考 |

---

## 目录结构

```
aiinfra/topics/harness/
├── README.md                # 本文件（一周学习计划）
├── kernels/                 # 可运行代码示例
│   └── (实践产出)
├── notes/                   # 原理笔记
│   └── (阅读笔记)
└── benchmark/               # 对比实验
    └── (harness 内外对比数据)
```
