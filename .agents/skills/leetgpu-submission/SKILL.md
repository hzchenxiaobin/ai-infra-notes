---
name: leetgpu-submission
description: 为 /mnt/workspace/code/github/ai-infra-notes/leetgpu 目录下的 LeetGPU 题解 Markdown 补充可直接提交到 LeetGPU 平台的 CUDA/Python 提交版本小节。
---

# LeetGPU 提交版本补充

## 适用场景

用户要求为 `leetgpu/` 下的题解 Markdown 文件补充或修正“可直接提交到 LeetGPU 平台的代码”时触发。通常表述如：

- “给所有 LeetGPU 题解加上提交代码”
- “补充 LeetGPU 提交版本”
- “让这个题解能直接提交到 LeetGPU”

## 目标

每个题解 Markdown 都必须包含一个 **“LeetGPU 提交版本”** 小节，内有完整的提交代码：

- **CUDA 题**：`#include`、helper/warp/block 函数、`__global__` kernel、`extern "C" void solve(...)` 入口、同步。
- **Python 题**（如 simple-inference）：`def solve(...)` 与 torch 调用。
- `solve` 签名必须与 LeetGPU 官方 starter 一致。

参考格式见：

```text
leetgpu/week2/day3/leetgpu-2d-convolution-solution.md
→ “### 4.1 LeetGPU 提交版本”
```

## 关键数据源

### 1. 官方 starter 仓库

LeetGPU 官方挑战仓库为 `AlphaGPU/leetgpu-challenges`。每个挑战目录结构为：

```text
challenges/<difficulty>/<number>_<challenge_name>/starter/starter.cu
```

示例：

```text
https://api.github.com/repos/AlphaGPU/leetgpu-challenges/contents/challenges/easy/1_vector_add/starter/starter.cu
```

优先读取 `starter/starter.cu`，提取其中的 `extern "C" void solve(...)` 签名。若目录下无 `starter.cu`（如 simple-inference 只有 Python starter），则以 Markdown 题解中的签名/题意为准。

### 2. URL slug → challenge_name 映射

Markdown 中的 URL 形如 `https://leetgpu.com/challenges/vector-addition`。仓库目录名是把 slug 的连字符换成下划线，但有例外：

| URL slug | challenge_name |
|----------|----------------|
| vector-addition | `vector_add` |
| general-matrix-multiplication-gemm | `gemm` |
| max-subarray-sum | `max_subarray_sum` |
| causal-self-attention | `casual_attention` |
| gpt-2-transformer-block | `gpt2_block` |
| vector-reversal | `reverse_array` |
| element-reversal | `reverse_array` |
| scalar-multiply | —（无官方 CUDA starter，按题解已有签名处理） |
| argmax | —（无官方 starter，按题解已有签名处理） |
| attention | —（无官方 starter，按题解已有签名处理） |

其余 slug 直接 `slug.replace('-', '_')`。

## 工作流

1. **枚举文件**
   - 目标：`leetgpu/weekN/dayM/leetgpu-*.md`
   - 跳过 `leetgpu/website/`。

2. **判断是否需要补充**
   - 若 Markdown 已存在 `^#{2,}\s*.*(?:提交|LeetGPU 提交版本)` 标题，可跳过或仅修正格式。
   - 否则需要新增。

3. **获取 starter 签名**
   - 按上表得到 `challenge_name`。
   - 从 `AlphaGPU/leetgpu-challenges` 拉取 `starter.cu`。
   - 提取 `extern "C" void solve(...)`；若 starter 只给空签名，也按该签名实现。

4. **生成提交代码**
   - 优先复用 Markdown 中已有的 kernel + `solve`（去掉 `int main` 与 CPU 参考代码）。
   - 若 Markdown 没有 `solve`，用 starter 签名写包装函数：
     - 按参数类型（指针/int/float/half 等）和名称把 `solve` 参数传给 kernel。
     - 需要多阶段 kernel 时，`solve` 内完成分配、启动、同步、释放。
   - 若官方签名与题解 kernel 维度不一致（如 RMSNorm 的 `gamma` 标量 vs 向量），按官方签名调整 kernel/调用，不要直接照搬题解本地版。

5. **插入位置**
   - 放在“Kernel 实现”主节后，性能/复杂度分析之前。
   - 小节编号跟随父节，例如父节是 `## 4. Kernel 实现` 时，用 `### 4.1 LeetGPU 提交版本`。

6. **格式避坑**
   - marked.js 对 `**text `code`**` 解析异常。必须写成 `**text `code` 中文**` 形式，确保 `**` 不与反引号相邻。
   - 图片路径保持 `](../../images/xxx.svg`，构建脚本会自动重写为 `./images/xxx.svg`。

7. **构建与验证**
   - 运行 `python3 build.py`。
   - 检查 `public/leetgpu/` 下 HTML 的图片引用是否 broken。
   - 检查 Markdown 代码围栏是否成对。

8. **提交**
   - 只提交 `leetgpu/` 下的 Markdown 变更。
   - 不要提交 `public/`（已在 `.gitignore`）。
   - commit message 示例：`feat(leetgpu): 为全部题解补充 LeetGPU 提交版本代码`。
   - `git push`。

## 常见题型处理要点

| 题型 | 注意点 |
|------|--------|
| elementwise（vector add / relu / silu / swiglu） | grid-stride loop + coalesced，签名通常为 `(const float* in, float* out, int N)`。 |
| 归约（reduction / dot-product / argmax） | 需要 warp shuffle + shared memory 块归约；结果可能需要 `atomicAdd` 或第二阶段 kernel。 |
| softmax / RMSNorm / batch norm | 一个 block 处理一行/一个 channel；注意官方签名的维度与题解本地版可能不同。 |
| GEMM / batched GEMM | 注意 `half`/`float`、WMMA、alpha/beta；签名顺序 `(A,B,C,M,N,K,alpha,beta)`。 |
| convolution | 常用 `__constant__` 放 kernel，动态 shared memory 传第三个尖括号参数。 |
| attention 系列 | 输入维度多，签名易错；确保 `Q/K/V/output` 顺序与维度参数与 starter 一致。 |
| simple-inference | Python 题，保留 `def solve(input, model, output)` 与 `output.copy_(model(input))`。 |

## 质量检查清单

- [ ] 每个 `leetgpu-*.md` 都有“提交版本”小节。
- [ ] 提交代码块包含完整 `solve` 入口且签名与官方 starter 一致。
- [ ] 没有 `**` 紧挨反引号的情况。
- [ ] `python3 build.py` 成功。
- [ ] `public/leetgpu/` 图片引用 0 broken。
- [ ] 已 `git push`。
