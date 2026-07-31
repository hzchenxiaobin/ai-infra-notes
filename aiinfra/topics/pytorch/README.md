# PyTorch：从基础入门

> **适用对象**：有 Python 基础，想系统入门 PyTorch，为后续 CUDA / Triton / 算子开发打底的学习者
> **目标**：掌握 Tensor、autograd、nn.Module、训练循环四大核心，能独立写出一个完整的训练脚本，并理解 PyTorch 底层在做什么
> **时间投入**：每天 2-3h，约一周可过完全部内容

---

## 🎯 目标

通过本专题，你将：

1. 理解 **Tensor** 的本质：内存布局、dtype、device、与 NumPy 的关系
2. 掌握 **autograd** 自动微分机制：计算图、`backward()`、梯度累加
3. 学会用 **nn.Module** 组织模型，用 `Dataset`/`DataLoader` 组织数据
4. 能写出**标准训练循环**：前向、loss、反向、优化器、验证
5. 了解 PyTorch 的**底层栈**（ATen、Dispatcher、CUDA 后端），为后续算子开发铺路

> 💡 **前置知识**：Python 基础语法、NumPy 基本操作、线性代数（矩阵乘法）、一点微积分（链式法则）
> ⚠️ **环境要求**：Python >= 3.9，PyTorch >= 2.0（`pip install torch`），有 GPU 更佳但 CPU 也能跑全部示例

---

## 为什么学 PyTorch

PyTorch 是 AI Infra 工程师的"通用语言"。无论你之后做 CUDA kernel 开发、Triton 算子、推理框架还是分布式训练，上游的模型代码几乎都是 PyTorch 写的。不懂 PyTorch，就看不懂你要优化的对象。

与同类框架的对比：

| 维度 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| 编程范式 | 动态图（eager） | 静态图为主（2.x 也支持 eager） | 函数式 + 变换（jit/grad/vmap） |
| 调试体验 | 就是 Python，pdb 直接调 | 图模式调试困难 | 变换后栈帧晦涩 |
| 社区生态 | 学术界/工业界主流 | 工业部署仍有一席之地 | 研究向，DeepMind 系 |
| 底层 | ATen (C++) + Dispatcher | XLA 编译 | XLA 编译 |
| 对 kernel 开发者友好度 | ⭐⭐⭐ 自定义算子生态成熟 | ⭐⭐ | ⭐ |

> 💡 **一句话总结**：PyTorch = NumPy 风格的 Tensor 计算 + 自动微分 + GPU 加速 + 模块化神经网络组件，学习曲线平缓，是入门 AI 系统的最佳起点。

---

## 核心概念

### 1.1 Tensor：一切的基础

**Tensor（张量）** 是 PyTorch 的核心数据结构，可以理解为"能放 GPU 上、能自动求导的 NumPy 数组"。

```python
import torch

# 创建 Tensor 的常见方式
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # 从数据创建
b = torch.zeros(2, 3)                         # 全 0
c = torch.randn(2, 3)                         # 标准正态分布
d = torch.arange(0, 10, 2)                    # [0, 2, 4, 6, 8]

print(a.shape, a.dtype, a.device)  # torch.Size([2, 2]) torch.float32 cpu
```

每个 Tensor 有三个关键属性：

| 属性 | 含义 | 常见取值 |
|------|------|----------|
| `shape` | 形状 | `torch.Size([batch, channel, H, W])` |
| `dtype` | 数据类型 | `float32`、`float16`、`bfloat16`、`int64` |
| `device` | 所在设备 | `cpu`、`cuda:0`、`cuda:1` |

#### 深入：Tensor 的内存布局

Tensor 的数据存储在一块**连续内存**中，`shape` 只是这块内存的"视图解释"。`stride`（步长）描述了在每个维度上移动一个元素需要跳过多少个内存位置：

```python
x = torch.arange(12).reshape(3, 4)
print(x.stride())   # (4, 1)：行方向跳 4 个元素，列方向跳 1 个
y = x.t()           # 转置，零拷贝！只改了 shape/stride
print(y.stride())   # (1, 4)
print(y.is_contiguous())  # False —— 非连续内存
z = y.contiguous()  # 真正搬移数据，变成连续
```

> ⚠️ **注意**：很多 kernel（包括手写 CUDA）假设输入是 contiguous 的。遇到奇怪的数值错误时，先检查 `.is_contiguous()`。

**视图（view）vs 拷贝（copy）** 是新手最容易踩的坑：

- `reshape` / `t` / `transpose` / `unsqueeze`：尽量返回视图，**共享内存**，改一个会影响另一个
- `clone` / `contiguous` / `to(dtype)`：产生**新内存**

```python
a = torch.zeros(4)
b = a.view(2, 2)
a[0] = 1.0
print(b)  # b 也被改了！b 和 a 共享底层存储
```

### 1.2 dtype 与类型提升

训练大模型时 dtype 的选择直接影响显存和速度：

| dtype | 字节数 | 精度 | 典型用途 |
|-------|--------|------|----------|
| `float32` | 4 | ~7 位有效数字 | 默认，训练/推理基线 |
| `float16` | 2 | ~3 位有效数字，范围 ±65504 | 混合精度训练（容易溢出） |
| `bfloat16` | 2 | ~2 位有效数字，范围与 fp32 相同 | 大模型训练主流（不易溢出） |
| `float64` | 8 | ~16 位有效数字 | 科学计算、梯度检查 |

```python
x = torch.randn(3, dtype=torch.float32)
y = x.to(torch.bfloat16)          # 显式转换
print((x * 1.0).dtype)            # torch.float32：Python 标量不提升类型
```

### 1.3 广播（Broadcasting）

形状不同的 Tensor 运算时，PyTorch 按 NumPy 广播规则自动扩展维度：

```python
a = torch.randn(4, 3)     # (4, 3)
b = torch.randn(3)        # (3,)
c = a + b                 # b 被广播成 (4, 3)

mean = a.mean(dim=0, keepdim=True)  # (1, 3)
c = a - mean              # 按行减均值，常用归一化写法
```

规则：从右往左对齐维度，每个维度要么相等、要么有一方为 1（可扩展）、要么一方不存在。不满足则报错。

### 1.4 autograd：自动微分

PyTorch 会记录 Tensor 上的所有运算，构建**动态计算图**。调用 `backward()` 时，沿图反向传播，用链式法则算出梯度：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1
y.backward()             # dy/dx = 2x + 3
print(x.grad)            # tensor([7.])，x=2 时导数为 7
```

关键概念：

- `requires_grad=True`：声明"我需要对这个 Tensor 求梯度"，叶子节点才需要设置
- `grad_fn`：非叶子 Tensor 记录"我是由什么运算产生的"，形成反向图
- `x.grad`：`backward()` 后梯度**累加**到这里（注意是累加，不是覆盖）
- `torch.no_grad()`：上下文管理器，内部不建图，验证/推理时必用，省显存

```python
# 梯度累加：多次 backward 会累加
x = torch.tensor([1.0], requires_grad=True)
(x * 2).backward()
(x * 2).backward()
print(x.grad)   # tensor([4.]) —— 2 + 2，不是 2！

# 所以训练循环里每步都要清零
x.grad = None   # 或 optimizer.zero_grad()
```

#### 深入：为什么梯度是累加的？

这是**梯度累加（gradient accumulation）** 特性的基础：显存放不下大 batch 时，可以把大 batch 拆成多个小 batch，逐个小 batch `backward()` 累加梯度，最后统一 `step()`，数学上等价于大 batch 训练。

### 1.5 nn.Module：模型的组织方式

所有神经网络模块都继承 `nn.Module`，它帮你管理**参数注册、设备迁移、训练/评估模式切换**：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)   # 自动注册为参数
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = MLP(784, 256, 10)
print(sum(p.numel() for p in model.parameters()))  # 参数总数：784*256+256 + 256*10+10 = 203530
model = model.cuda()        # 一键把所有参数搬到 GPU
model.eval()                # 切换到评估模式（影响 Dropout/BatchNorm）
```

`nn.Module` 的两个魔法：

- `__setattr__` 被重载：赋值 `self.fc1 = nn.Linear(...)` 时，自动把子模块参数登记到 `self._parameters` / `self._modules`，所以 `model.parameters()` 能遍历到
- `__call__` 被重载：`model(x)` 实际走 `Module.__call__` → 前后 hook → `forward(x)`。**永远写 `model(x)`，不要写 `model.forward(x)`**，否则 hook 全部失效

### 1.6 Dataset 与 DataLoader

数据管线的标准两件套：

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, n):
        self.x = torch.randn(n, 784)
        self.y = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

loader = DataLoader(
    MyDataset(1000),
    batch_size=32,       # 每批 32 条
    shuffle=True,        # 每个 epoch 打乱
    num_workers=4,       # 多进程加载（CPU 预处理与 GPU 计算重叠）
    pin_memory=True,     # 锁页内存，加速 CPU→GPU 拷贝
)

for batch_x, batch_y in loader:
    print(batch_x.shape)  # torch.Size([32, 784])
    break
```

### 1.7 优化器

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# 学习率调度：训练大模型几乎必配
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

常见优化器速查：

| 优化器 | 特点 | 适用 |
|--------|------|------|
| `SGD`（+momentum） | 泛化好，需调 lr | CV 经典训练 |
| `Adam` / `AdamW` | 自适应 lr，收敛快，几乎不用调 | NLP / 大模型默认选择 |
| `AdamW` vs `Adam` | weight decay 解耦，更合理 | 现在基本都用 AdamW |

---

## 最小可运行示例

一个完整的 MNIST 风格训练脚本（用随机数据模拟，不依赖下载数据集，复制即跑）：

```python
# train_mlp.py —— 最小完整训练循环
# 运行：python3 train_mlp.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. 数据
class RandomDataset(Dataset):
    """随机生成 10 分类数据，模拟 MNIST 形状"""
    def __init__(self, n=10000):
        self.x = torch.randn(n, 784)
        # 让标签与数据有点关系，模型才学得到东西
        self.y = (self.x[:, :10].argmax(dim=1)).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(RandomDataset(10000), batch_size=64, shuffle=True)
    val_loader = DataLoader(RandomDataset(1000), batch_size=256)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)              # 前向
            loss = loss_fn(logits, y)      # 算 loss
            optimizer.zero_grad()          # 清梯度（必须在 backward 之前）
            loss.backward()                # 反向传播
            optimizer.step()               # 更新参数

            total_loss += loss.item()

        # ---- 验证 ----
        model.eval()
        correct = 0
        with torch.no_grad():              # 验证不建图，省显存
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()

        acc = correct / len(val_loader.dataset)
        print(f"epoch {epoch}: train_loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f}")

if __name__ == "__main__":
    main()
```

```bash
python3 train_mlp.py
```

预期输出（数值会因随机种子略有不同——验证集是独立随机数据，准确率饱和在 ~0.65 属正常）：

```text
epoch 0: train_loss=1.9440 val_acc=0.5040
epoch 1: train_loss=0.7418 val_acc=0.6200
epoch 2: train_loss=0.2209 val_acc=0.6700
epoch 3: train_loss=0.0688 val_acc=0.6740
epoch 4: train_loss=0.0286 val_acc=0.6690
```

**训练循环五步**是 PyTorch 的灵魂，所有复杂训练（GPT、LLaMA）本质都是这五步的扩展：

1. `logits = model(x)` —— 前向
2. `loss = loss_fn(logits, y)` —— 计算损失
3. `optimizer.zero_grad()` —— 清零梯度
4. `loss.backward()` —— 反向传播
5. `optimizer.step()` —— 更新参数

---

## 深入原理

### 3.1 PyTorch 的分层架构

PyTorch 不是铁板一块，从上到下的调用栈大致是：

| 层 | 语言 | 职责 |
|----|------|------|
| Python API（`torch.*`、`nn.*`） | Python | 用户接口 |
| torch._C / autograd 引擎 | C++ | 计算图构建与反向执行 |
| **ATen** | C++ | Tensor 运算的核心库，2000+ 算子 |
| **Dispatcher** | C++ | 按 device/dtype/后端分发到具体 kernel |
| 后端 kernel | C++/CUDA | CPU（Vec256/OpenMP）、CUDA（cuBLAS/cuDNN/手写 kernel） |

你写的 `a + b`，实际路径是：Python 绑定 → Dispatcher 查表（key = device × dtype × layout）→ 命中 `add.Tensor` 的 CUDA kernel → 启动 GPU kernel。理解 Dispatcher 是后续写自定义算子（`torch.library` / C++ extension）的前提。

### 3.2 autograd 引擎如何反向传播

前向时每个运算生成一个 `Node`（如 `MulBackward0`、`AddBackward0`），存进输出 Tensor 的 `grad_fn`。`loss.backward()` 时引擎做拓扑排序，从 loss 反向执行每个 Node 的 `apply()`：

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b          # c.grad_fn = <MulBackward0>
d = c + a          # d.grad_fn = <AddBackward0>
d.backward()
print(a.grad, b.grad)
# tensor(4.) tensor(2.)
# d = a*b + a，∂d/∂a = b + 1 = 3 + 1 = 4，∂d/∂b = a = 2
```

关键实现细节（PyTorch 2.x）：

- 反向图是**动态**的：每次前向都重新建图，所以 PyTorch 天然支持带控制流（`if`/`for`）的模型——这是"动态图"称号的由来
- 中间激活默认**保留**给反向用（这就是显存大头）；`torch.utils.checkpoint` 可以丢弃中间激活、反向时重算，用时间换显存
- `in-place` 运算（`x += 1`、`relu_()`）会改写前向值，可能导致反向需要的值被破坏而报 `RuntimeError: a leaf Variable that requires grad...`

### 3.3 `torch.compile`：eager 之外的性能开关

PyTorch 2.0 引入 `torch.compile`，一行代码把 eager 模型编译成优化后的图：

```python
model = MLP().to(device)
model = torch.compile(model)   # 就这么简单
```

背后的栈：**Dynamo**（捕获 Python 字节码 → FX Graph）→ **AOTAutograd**（前向反向一起图化）→ **Inductor**（代码生成，GPU 上生成 **Triton** kernel）。这也是为什么学完 PyTorch 基础后值得接着学 Triton——`torch.compile` 生成的 kernel 就是 Triton 写的，看懂它就能调试和手写更优的算子。

### 3.4 显存去哪了

训练时显存大致分四块：

| 成分 | 大小估算（参数量 $N$） | 说明 |
|------|------------------------|------|
| 参数 | $4N$ 字节（fp32） | 模型权重 |
| 梯度 | $4N$ 字节 | 与参数同形 |
| 优化器状态 | $8N$ 字节（Adam：m + v） | AdamW 是显存大户 |
| 激活值 | 与 batch、序列长度、层数正相关 | 往往是大头 |

> 💡 这也是为什么混合精度（fp16/bf16 权重 + fp32 master copy）和梯度检查点是省显存的两大杀器——进阶话题，先记住这个表。

---

## 性能对比与 Benchmark

PyTorch 入门阶段最值得量化感知的一件事：**GPU 到底比 CPU 快多少，瓶颈在哪**。

```python
# bench_matmul.py —— CPU vs GPU 矩阵乘法
# 运行：python3 bench_matmul.py
import time
import torch

def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

n = 4096
a_cpu = torch.randn(n, n)
b_cpu = torch.randn(n, n)
t_cpu = bench(lambda: a_cpu @ b_cpu)
flops = 2 * n**3
print(f"CPU  fp32: {t_cpu*1e3:8.1f} ms, {flops/t_cpu/1e12:6.1f} TFLOPS")

if torch.cuda.is_available():
    a_gpu, b_gpu = a_cpu.cuda(), b_cpu.cuda()
    for dtype in (torch.float32, torch.bfloat16):
        ag, bg = a_gpu.to(dtype), b_gpu.to(dtype)
        t = bench(lambda: ag @ bg)
        print(f"GPU  {str(dtype).split('.')[-1]}: {t*1e3:8.1f} ms, {flops/t/1e12:6.1f} TFLOPS")
```

典型结果（RTX 4090，数值仅供量级参考）：

| 设备 / dtype | 耗时 | 算力 |
|---|---|---|
| CPU fp32 | ~800 ms | ~0.17 TFLOPS |
| GPU fp32 | ~15 ms | ~9 TFLOPS |
| GPU bf16 | ~4 ms | ~35 TFLOPS |

要点：

- **GPU 在小矩阵上不一定赢**：kernel 启动、PCIe 传输都有固定开销，矩阵小于几百维时 CPU 可能更快
- `torch.cuda.synchronize()`：GPU 调用是异步的，不计时同步的话测出来的只是"提交 kernel 的时间"
- 大矩阵乘法逼近**算力瓶颈（compute bound）**，小算子往往是**带宽瓶颈（memory bound）**——Roofline 模型的直觉从这里开始建立

---

## 常见陷阱与最佳实践

**坑 1：忘记 `zero_grad()`，梯度越训越大**

```python
# ❌ 错误：梯度每步累加，loss 不降反升
for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

# ✅ 正确
for x, y in loader:
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**坑 2：验证时忘了 `no_grad()` 和 `eval()`**

```python
# ❌ 错误：验证也建图，显存爆炸；Dropout/BN 行为错误
pred = model(x_val)

# ✅ 正确
model.eval()
with torch.no_grad():
    pred = model(x_val)
```

**坑 3：在训练循环里 `print(tensor)` 或 `loss.item()` 过度使用**

每次 `.item()` / `print` / CPU 使用 Tensor 值都会强制 **GPU 同步**，把异步流水线卡成串行。正确做法：累加 loss 到 Python float 的频次降低（如每 50 步记录一次），或用 `torch.cuda.Event` 精确计时。

**坑 4：CPU/GPU 来回搬运藏在循环里**

```python
# ❌ 错误：每步都在循环里反复 .cpu()/.numpy()，PCIe 成为瓶颈
for x, y in loader:
    feat = model(x.cuda())
    result.append(feat.cpu().numpy())   # 每步同步 + 拷贝

# ✅ 正确：尽量留在 GPU，最后一次性搬回
for x, y in loader:
    feat = model(x.cuda(non_blocking=True))
    result.append(feat)
result = torch.cat(result).cpu().numpy()
```

**坑 5：`model.forward(x)` 直接调用**

```python
# ❌ 绕过 hook，torch.compile/DDP 下行为可能出错
out = model.forward(x)
# ✅ 永远这样写
out = model(x)
```

**最佳实践速查**：设随机种子（`torch.manual_seed(42)`）、`pin_memory=True` + `non_blocking=True` 配合、新代码优先试 `torch.compile`、用 `torch.autograd.set_detect_anomaly(True)` 调试 NaN 梯度。

---

## 面试要点

**Q：PyTorch 动态图和 TensorFlow 静态图的本质区别？**
> 动态图：每次前向实时构建计算图，反向用完即弃，支持任意 Python 控制流，调试友好，代价是图优化机会少。静态图：先定义完整图再执行，便于全局优化（算子融合、常量折叠）和部署，但控制流需特殊算子表达。PyTorch 2.x 的 `torch.compile` 是在动态图易用性之上补静态图性能的折中方案。

**Q：`backward()` 为什么默认梯度累加而不是覆盖？**
> 支持梯度累加训练（小显存模拟大 batch）和多 loss 分支共享参数的场景。代价是用户必须手动 `zero_grad()`，这是 PyTorch 刻意选择的"显式优于隐式"。

**Q：Tensor 的 `view` 和 `reshape` 有什么区别？**
> `view` 要求内存连续，只改元数据（shape/stride），零拷贝；`reshape` 在内存连续时等价于 `view`，不连续时会静默拷贝一份。性能敏感场景用 `view` 并处理异常，或先 `contiguous()` 再 `view`。

**Q：训练中显存主要由什么构成？怎么省？**
> 参数、梯度、优化器状态（Adam 两份动量）、激活值。省显存手段：混合精度（bf16）、梯度检查点（重算激活）、梯度累加、ZeRO/FSDP 切分、换 8bit 优化器。

**Q：`model.train()` 和 `model.eval()` 到底切换了什么？**
> 只改一个 `self.training` 布尔标志，影响行为依赖该标志的层：Dropout（训练时随机丢弃，评估时直通）、BatchNorm（训练时用 batch 统计量并更新 running 统计，评估时用 running 统计）。与 `no_grad()` 无关——`eval()` 不阻止建图。

**Q：一个 `a + b` 从 Python 到 GPU 经历了什么？**
> Python 绑定 → autograd 层包一个 Node（如果 requires_grad）→ Dispatcher 根据 device/dtype/layout 计算 dispatch key → 命中 ATen 注册的 CUDA `add` kernel → 参数打包、grid/block 计算 → 启动 kernel（异步）→ 返回 Python 端 Tensor 句柄。

**Q：为什么 `torch.compile` 能加速？**
> 三个层面：① 算子融合，减少 kernel 启动和中间结果的显存往返；② 消除 Python 开销（eager 模式每个算子都有 Python→C++ 边界成本）；③ Inductor 生成的 Triton kernel 可针对具体 shape 做 tiling 和布局优化。小模型/控制流重的模型可能反而变慢或触发大量重编译。

---

## 推荐资源

- ⭐ 必读：[PyTorch 官方教程（60 Minute Blitz）](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)——官方入门最快路径
- ⭐ 必读：[Autograd 官方文档](https://pytorch.org/docs/stable/autograd.html)——理解计算图与 `backward` 语义
- 📌 推荐：[PyTorch Internals 2.0（官方博客）](https://pytorch.org/blog/pytorch-2-paper-tutorial/)——Dispatcher/ATen 架构详解
- 📌 推荐：[《Dive into Deep Learning》PyTorch 版](https://d2l.ai/)——理论与实践结合
- 📎 参考：[PyTorch 源码](https://github.com/pytorch/pytorch)——`aten/src/ATen/native/` 下是算子实现
- 📎 参考：本仓库后续专题 [Triton](../triton/README.md)——`torch.compile` 生成的 kernel 语言

---

> 💡 **一句话总结**：Tensor 是数据，autograd 是求导引擎，nn.Module 是组织结构，训练循环五步是灵魂——掌握这四样，PyTorch 就入门了。
