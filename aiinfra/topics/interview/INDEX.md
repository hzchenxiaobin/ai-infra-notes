# AI Infra 面试专题索引

> 本页面是 `interview` 专题的导航入口，汇总目录下全部页面，新增页面后请同步更新此索引。

## 页面导航

### 专题主页

- [AI Infra 面经与面试题整理](README.md)：面试的一般形式、高频考点分类（CUDA / 推理 / 训练分布式 / C++ / 量化 / RL / K8s）、公司风格差异与准备建议

### notes/ 面经与面试题整理

- [AI Infra 面试题与参考答案（北美面经篇）](notes/us_interview_qa.md)：知乎《AI infra 面试经验贴》全帖面试题整理，逐题附参考答案（GPU 基础 / 手撕 kernel / LeetCode / 系统设计）
- [AI Infra 社招面试实录与参考答案（面试官视角）](notes/social_interview_qa.md)：知乎《记AI-infra/大模型推理社招面试一兄弟的全过程》面试题整理，附参考答案（项目深挖 / 推理 / 量化 / CUDA / C++ 手撕）
- [面经 1](notes/面经%201.md)：无线软开 + AI 算法背景 4 年经验的社招面经实录，覆盖 AI 芯片初创 / 芯片中大厂 / 互联网 Infra 的算子开发岗面试题、手撕题与谈薪情况

### mock_interview/ 模拟面试精讲

- [ldmatrix 读行主序 A 的 shared memory bank conflict](mock_interview/ldmatrix_bank_conflict.md)：Tensor Core GEMM 高频面试题精讲——bank conflict 判定方法、XOR swizzle 消除手段（CUTLASS / CuTe / TMA）

## 目录结构

```text
interview/
├── README.md                       # 面经与面试题整理（专题主页）
├── INDEX.md                        # 本索引页
├── notes/                          # 面经整理与面试题参考答案
│   ├── us_interview_qa.md          # 北美面经篇 QA
│   ├── social_interview_qa.md      # 社招面试实录 QA（面试官视角）
│   └── 面经 1.md                   # 社招面经实录（算子开发岗）
└── mock_interview/                 # 模拟面试 / 题目精讲
    └── ldmatrix_bank_conflict.md   # ldmatrix bank conflict 精讲
```
