# 学习进度记录系统设计说明书

> 版本：v1.0（草案） · 日期：2026-09-05 · 适用仓库：ai-infra-notes
>
> 目标：为「10 周每日教程 + 16 个专题 + 25 篇论文 + Profiling 任务 + 题库练习」提供统一的进度记录、可视化与待办跟踪，零后端、与现有 GitHub Pages 静态站无缝集成。

---

## 1. 背景与问题

仓库内容已具相当规模，但**没有任何活的进度机制**：

| 内容板块 | 规模 | 现状 |
|---|---|---|
| 每日教程 `aiinfra/daily/` | 10 周 × 7 天 = 70 个教学日 | 状态靠人脑记，README 表格中的 ✅ 是手写的静态标记 |
| 专题 `aiinfra/topics/` | 16 个专题（12 个 7 天制 + 4 个特殊结构） | 无进度概念，"完成"只靠文件是否存在 |
| 论文精读 `aiinfra/paper/` | 25 个目录，仅 9 篇完成精读 | 有天然两态（PDF=立项 / README=完成），但无中间态与展示 |
| Profiling `profiling/` | week1–3 稀疏分布 + LeetGPU 专项 | 与 daily 对应关系靠人工对照 |
| Markdown checkbox | 全仓库 738 处 `- [ ]` | 无工具消费，勾选后无人聚合 |
| GPU 实测待补 | `plan/gpu_pending_measurements.md` | 最接近进度台账的文件，孤立存在 |
| LeetGPU / LeetCode | 43 道 LeetGPU + 每日一题 | 在外部站点，仓库内无完成记录 |

**核心矛盾**：站点是纯静态构建（push → `build.py` → GitHub Pages），没有后端、没有账号体系，但学习者需要"我今天学到哪了 / 还差什么"的答案。

## 2. 设计目标与原则

**目标**

1. 一眼看清全局：每个 week / topic / paper 的完成度与下一待办。
2. 记录成本低：学习页面上一次点击完成打卡，不用离开上下文。
3. 数据可信：能由仓库证据（文件、报告、checkbox）自动派生的状态，不依赖手填。
4. 零运维：不加后端、不加依赖，纯静态可部署。

**原则**

- **目录即目录（Catalog from tree）**：学习单元由构建期扫描目录自动发现，复用 `build/topics.py` / `build/paper.py` 已有的发现范式，新增 week/topic/paper 无需改进度代码。
- **证据优先，勾选兜底**：状态分两层——仓库客观证据（README/kernel/ncu-rep 存在性、checkbox 聚合）构建期静态计算；个人主观进度（在学/完成/搁置）由学习者在浏览器勾选。
- **本地优先（Local-first）**：进度存 `localStorage`，可导出/导入 JSON 做备份与跨设备同步，不离开浏览器。
- **渐进增强**：JS 失效时所有内容页面照常可用，只是没有勾选框与仪表盘。

**非目标**（本期不做）：多人协作、账号与云同步、服务端统计、移动端 App。

## 3. 总体架构

```
┌──────────────────────── 构建期（CI / 本地 build.py） ────────────────────────┐
│                                                                              │
│  aiinfra/daily/week*/day*/   aiinfra/topics/*/   aiinfra/paper/*/            │
│  profiling/                  plan/gpu_pending_measurements.md                │
│        │                           │                     │                   │
│        └───────────┬───────────────┴───────────┬─────────┘                   │
│                    ▼                           ▼                             │
│         build/progress_catalog.py      build/progress_evidence.py            │
│         （扫描发现学习单元）             （提取客观完成证据）                  │
│                    └───────────┬───────────────┘                             │
│                                ▼                                             │
│                    public/progress/catalog.json  ← 单元树 + 证据状态         │
│                    public/progress/index.html    ← 仪表盘静态壳              │
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────── 运行期（浏览器，纯前端） ────────────────────────────┐
│                                                                              │
│  static/js/vp-progress.js                                                    │
│   ├─ 注入器：在 week/day/topic/paper 页面渲染打卡控件（读 catalog.json）      │
│   ├─ 存储层：localStorage `aiinfra-progress-v1` 读写 + 导出/导入 JSON        │
│   └─ 仪表盘：/progress/ 页聚合渲染（环形图、周进度条、待办清单、连续天数）    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 4. 学习单元模型（Catalog）

### 4.1 单元类型与 ID 规范

所有可跟踪对象统一为 **Unit**，ID 稳定、可读、可定位：

| 类型 | ID 格式 | 示例 | 发现方式 |
|---|---|---|---|
| 周 | `week:N` | `week:4` | 扫 `aiinfra/daily/week*/` |
| 教学日 | `day:N-M` | `day:4-2` | 扫 `weekN/dayM/`（沿用 `load_overview_and_days()`） |
| 专题 | `topic:<dir>` | `topic:cutlass` | 含 README 的子目录（沿用 `topics.py`） |
| 专题天 | `topic:<dir>:dayK` | `topic:triton:day3` | 扫专题内 `dayK.md`；单文件专题无子单元 |
| 论文 | `paper:<dir>` | `paper:flashattention2` | 扫 `paper/*/`（沿用 `paper.py`） |
| Profiling 任务 | `prof:weekN-dayM` / `prof:leetgpu-<slug>` | `prof:week2-day4` | 扫 `profiling/` 任务目录 |
| GPU 实测待办 | `todo:gpu:<行号-hash>` | — | 解析 `gpu_pending_measurements.md` 表格行 |
| 题库 | 外部链接，不入 catalog | — | 在各 day 页面上作为链接展示，完成状态可手勾 |

特殊结构专题的映射规则：

- `interview`：子单元 = `mock_interview/*.md` + `notes/*.md` 每篇一个 `topic:interview:<file>`。
- `misc` / `shengteng`：子单元 = 目录内每篇 `.md`（不含 README）。
- `cpp` / `cute` 等 7 天制：`day1.md`–`day7.md` 天然对齐。

### 4.2 Unit 字段

```json
{
  "id": "day:4-2",
  "type": "day",
  "title": "手写 FlashAttention Kernel",
  "parent": "week:4",
  "order": 2,
  "url": "/week4/day2.html",
  "evidence": {
    "readme": true,
    "kernels": ["flash_attention_v2.cu"],
    "ncu_report": false,
    "checkbox_total": 6,
    "checkbox_done": 0
  },
  "derived": "content_ready"
}
```

- `evidence`：构建期从文件系统派生，**只读**，前端不修改。
- `derived`：证据聚合出的客观态（见 §5.2），与个人勾选态分离展示。

### 4.3 catalog.json 结构

```json
{
  "version": 1,
  "generated_at": "2026-09-05T10:00:00+08:00",
  "stats": { "units": 217, "days": 70, "topics": 16, "papers": 25 },
  "units": [ /* Unit 数组，按 order 排序 */ ]
}
```

单文件承载全量目录（预估 200+ 单元，gzip 后 < 30 KB），一次请求喂饱仪表盘，避免逐页请求。

## 5. 状态模型

### 5.1 个人进度状态（学习者勾选，存 localStorage）

| 状态 | 含义 | 触发 |
|---|---|---|
| `not_started` | 未开始 | 默认 |
| `in_progress` | 在学 | 点"开始"或自动（首次访问该页时提示） |
| `done` | 完成 | 手动打卡（鼓励满足验收条件再勾，见 §5.3） |
| `blocked` | 卡住/搁置 | 手动，附一行备注（如"缺 GPU 实测"） |

存储结构（key：`aiinfra-progress-v1`）：

```json
{
  "version": 1,
  "units": {
    "day:4-2": { "s": "done", "t": 1725480000, "note": "" },
    "paper:mamba": { "s": "in_progress", "t": 1725566400, "note": "读到 §Method" }
  }
}
```

只存非默认状态，70 天全完成也只有几 KB，远低于 localStorage 限额。

### 5.2 证据派生状态（构建期计算，客观）

| 单元 | `derived` 判定规则 |
|---|---|
| day | `content_ready`：README 存在且含 8 段骨架标题；否则 `skeleton` |
| topic | `complete`：所有 dayK.md 存在；`partial`：部分；单文件专题恒 `complete` |
| paper | `planned`（仅 PDF）→ `reading`（README 存在但 17 节骨架不全）→ `done`（17 节齐全，标题模式匹配 `paper/SKILL.md`） |
| profiling 任务 | `done`：目录内存在 `*_full.ncu-rep`；`pending`：只有 Makefile/源码 |
| GPU 待办 | 从 `gpu_pending_measurements.md` 表格解析"待实测/已回填"列 |

### 5.3 双轨展示与验收提示

仪表盘上每个单元显示**两个徽章**：客观证据徽章（文档/代码/报告是否齐）+ 个人状态徽章。打卡 `done` 时若证据不全（如该天有 checkbox 未勾、无 ncu 报告），弹出非阻塞提示："该天的 6 项任务清单尚未勾完，确认完成？"——**提醒但不强制**，尊重学习者的自主判断。

## 6. 功能设计

### 6.1 学习页面打卡控件（注入器）

- `vp-progress.js` 在页面加载后取 `catalog.json`，按当前 URL 反查单元 ID，在标题旁渲染状态切换按钮（未开始 → 在学 → 完成，循环切换；长按/右键菜单选 `blocked`）。
- 教学日页面额外渲染该日 `- [ ]` checkbox 的完成计数条（如 `2/6`），checkbox 点击**双向同步**：页面勾选即写入 localStorage，下次进入自动回填勾选状态（解决 738 处 checkbox 无法持久化的问题）。
- 样式复用现有 VitePress 风（`vp-solution.css` 变量），暗色模式自动适配。

### 6.2 仪表盘 `/progress/`

构建期生成静态壳 + 前端用 catalog.json 与 localStorage 聚合渲染：

1. **总览条**：4 个环形图——教程（x/70 天）、专题（x/16）、论文（x/25，细分 planned/reading/done）、Profiling（x/N）。
2. **周进度矩阵**：10×7 方格热力图（类 GitHub contributions），点击方格直达对应 day 页面。
3. **进行中清单**：所有 `in_progress` / `blocked` 单元，按最近更新时间排序，附备注。
4. **客观待办区**：
   - GPU 实测待补（来自 `gpu_pending_measurements.md`，逐项列出 + 跳转链接）；
   - 论文"已立项未精读"清单（16 篇仅 PDF 的）；
   - Profiling 任务缺报告清单。
5. **连续学习 streak**：按 `done` 时间戳计算当前连续天数与历史最长（纯前端，时区用本地）。
6. **数据管理**：导出 JSON（下载备份）、导入 JSON（恢复/换设备）、清空重置（二次确认）。

### 6.3 命令行工具（可选，P1）

`tools/progress.py`，服务不依赖浏览器的场景：

```bash
python3 tools/progress.py status            # 终端打印完成度摘要（读导出的 JSON）
python3 tools/progress.py mark day:4-2 done # 直接改 progress.json 备份文件
python3 tools/progress.py evidence          # 只跑证据扫描，打印客观待办（等价 §6.2 第 4 区）
```

CLI 与浏览器共享同一份 JSON 格式（导入即可互通），不引入第二份真源。

## 7. 与现有系统的集成点

| 位置 | 改动 |
|---|---|
| `build/progress_catalog.py`（新增） | 单元扫描与 catalog.json 生成，复用 `common.py` 的路径常量与 `DAY_TITLE_PATTERN` |
| `build/progress_evidence.py`（新增） | 证据派生规则（§5.2），被 catalog 与 CLI 共用 |
| `build.py` | 在 paper 构建之后追加一步 `progress` 构建；检查门禁失败行为与现有一致 |
| `build/home.py` | 首页统计条加"我的进度"入口链接（静态文案，不读 localStorage，避免构建期依赖） |
| `static/js/vp-progress.js`（新增） | 注入器 + 存储层 + 仪表盘渲染，< 500 行 vanilla JS，无新依赖 |
| `static/css/` | 追加少量进度组件样式，复用现有 CSS 变量 |
| `.github/workflows/deploy.yml` | **无需改动**（catalog 在 build.py 内生成，随 artifact 一起部署） |

兼容性约束：仪表盘页与注入逻辑必须容忍 catalog.json 缺失（本地只开了部分页面时）——静默降级为"进度不可用"，不报错不阻塞渲染。

## 8. 数据安全与隐私

- 进度数据只存在学习者自己的浏览器 localStorage，不上传任何服务器；GitHub Pages 静态站天然无收集能力。
- 导出 JSON 不含任何凭据；**明确禁止**把 `env.md`（含明文 token）纳入导出/同步范围——该文件本就不应入库，本系统不触碰它。
- 导入时做 schema 版本校验与字段白名单过滤，防止构造的 JSON 注入异常键。

## 9. 边界情况

| 场景 | 处理 |
|---|---|
| 新增 week11 / 新专题 / 新论文 | 构建期自动发现，ID 稳定不变，旧进度不受影响 |
| 目录重命名（如专题改名） | ID 变化导致旧进度成孤儿 → 导出 JSON 提供 `migrations` 映射表手工修正；仪表盘对孤儿 ID 折叠显示"未知单元"可一键清除 |
| localStorage 被清 | 靠导出备份恢复；仪表盘常驻"距上次导出已过 N 天"提醒（>7 天提示） |
| 多设备 | 手动导出/导入；不追求实时同步（超范围） |
| 多人共用同一浏览器 | 不区分身份（单学习者假设，符合仓库性质） |
| paper 17 节判定误报 | 规则保守化：只要缺的节是空标题即判 `reading`，允许 `<!-- TODO -->` 占位不计完成 |

## 10. 实施计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0（核心闭环） | `progress_catalog.py` + `progress_evidence.py` + catalog.json；仪表盘静态页（总览 + 周矩阵 + 客观待办）；打卡控件 + localStorage | `python3 build.py` 产出 `/progress/`；在 day 页面打卡后仪表盘数字实时变化；70 天 + 16 专题 + 25 论文全部被收录 |
| P1（数据流动） | 导出/导入 JSON；`tools/progress.py` CLI；checkbox 双向同步 | 导出后清空 localStorage，导入可完整还原；CLI `evidence` 输出与仪表盘客观待办一致 |
| P2（体验增强） | streak 统计、导出提醒、进行中清单备注编辑、`blocked` 原因聚合视图 | 连续打卡 3 天显示 streak=3；孤儿 ID 可清理 |

工作量预估：P0 约 2–3 天（构建侧 Python 1 天，前端 1–2 天），P1/P2 各约 1 天。

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 前端状态与仓库证据不一致（勾了 done 但无报告） | 双轨徽章并列展示，不一致本身就是提醒，不做强制校验 |
| catalog 扫描规则漏掉特殊结构专题 | §4.1 已为 interview/misc/shengteng 定义映射；P0 验收含单元总数对账 |
| localStorage 5MB 限制 | 实测全量进度 < 10 KB，无风险 |
| 构建时间增加 | 扫描为纯文件系统操作，预估 < 1s，可忽略 |

## 12. 成功度量

- 覆盖率：catalog 收录单元 ≥ 200，覆盖 daily/topics/paper/profiling 四类；
- 可信度：仪表盘"客观待办"与 `gpu_pending_measurements.md`、论文 PDF/README 清单逐项一致；
- 可用性：从任意 day 页面完成一次打卡到仪表盘反映，≤ 2 次点击、0 次刷新。
