# Week 1 学习网站

这是一个按天拆分的多页面静态网站，内容源为 `../README.md`。

## 文件结构

```
week1/website/
├── index.html          # 课程概览页
├── day1.html ~ day7.html   # 每日学习页面
├── build.py            # 从 README.md 重新生成所有页面的脚本
├── css/
│   ├── style.css
│   └── prism-tomorrow.min.css
├── js/
│   ├── main.js
│   ├── marked.min.js
│   ├── prism.min.js
│   ├── prism-c.min.js
│   ├── prism-cpp.min.js
│   ├── prism-bash.min.js
│   └── prism-python.min.js
└── images/             # 图片资源（SVG 图表）
    ├── gpu_memory_hierarchy.svg
    ├── sm_architecture.svg
    ├── coalesced_access.svg
    ├── bank_conflict.svg
    ├── roofline_model.svg
    ├── week1_roadmap.svg
    ├── grid_block_thread.svg
    ├── warp_divergence.svg
    └── simt_vs_simd.svg
```

## 查看网站

### 方式 1：直接打开

双击 `index.html`，用浏览器打开即可。

### 方式 2：使用本地 HTTP 服务器（推荐）

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1/website
python3 -m http.server 8080
```

然后在浏览器访问：`http://localhost:8080`

## 重新生成网站

如果 `../README.md` 内容有更新，可以运行：

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1/website
python3 build.py
```

## 功能特性

- 📑 按天拆分的独立页面
- 🧭 左侧导航栏，快速切换每一天
- 🌙 深色主题，适合长时间阅读
- 💻 代码块语法高亮
- 📋 鼠标悬停代码块显示复制按钮
- 📱 移动端响应式布局
- ⬆️ 返回顶部按钮
- 🏠 概览页显示每日任务卡片
- 🖼️ 配套 SVG 图表：SM 架构、内存层次、Bank Conflict、Roofline 等
- 🧮 Day 3 集成交互式 CUDA Occupancy Calculator
