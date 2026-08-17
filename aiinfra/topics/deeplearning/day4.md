# Day 4｜调参与训练技巧

> 今日目标：学会看训练曲线，亲手做调参实验，建立"实验驱动"的调参直觉
> 时间投入：约 3~5 小时

---

## 今日路线

```
上午：看懂训练曲线 + 关键概念（约 1.5 小时）
下午：调参实验（约 2 小时）
晚上：GPU 加速（约 0.5~1 小时）
```

> 今天全部基于昨天的 MNIST 代码做改动，不换新项目。

---

## 上午｜看懂训练曲线

### 任务

- [ ] 搞懂三个术语：epoch、batch size、iteration
- [ ] 学会画训练曲线，并用它判断过拟合/欠拟合

### 三个必须分清的术语

以 MNIST（6 万张训练图）+ `batch_size=64` 为例：

```
1 个 iteration（迭代）= 喂一批（64 张）数据，更新一次参数
1 个 epoch（轮次）  = 把全部 6 万张数据完整过一遍
                    = 60000 / 64 ≈ 938 个 iteration
batch size         = 每批多少张（这里是 64）
```

昨天训练了 5 个 epoch，就是把 6 万张图完整学了 5 遍。

### 训练曲线：深度学习的"仪表盘"

给昨天的代码加上曲线记录（这是今天的第一个动手任务）：

```python
import matplotlib.pyplot as plt

# 在训练循环前
train_losses = []

# 在每个 epoch 结束时
train_losses.append(total_loss / len(train_loader))

# 训练全部结束后
plt.plot(train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

**更专业的做法**：同时记录训练集和测试集（验证集）的 loss，两条曲线一起看：

```
                loss
                 │      ╭──── 测试 loss（开始上升 = 过拟合）
                 │   ╭──╯
                 │  ╭╯
                 │ ╭╯        ← 理想停止点
                 │╱  ╭─────── 训练 loss（还在降）
                 └────────────── epoch
```

| 曲线形态 | 诊断 | 对策（下午会验证） |
| --- | --- | --- |
| 训练 loss 很高，测试 loss 也很高 | **欠拟合**：模型没学会 | 训练更久 / 加大网络 / 调学习率 |
| 训练 loss 一直降，测试 loss 反而升 | **过拟合**：在背题 | Dropout / 减少参数量 / 更多数据 |
| 两条一起降、趋于平稳 | 健康 | 保持 |

### 自测三问

1. 6 万张图、batch size 128、训练 10 个 epoch，一共更新了多少次参数？
2. 为什么"训练 loss 降、测试 loss 升"说明过拟合了？
3. 只画训练 loss 不画测试 loss，会漏掉什么信息？

---

## 下午｜调参实验（今天的核心产出）

在昨天代码的基础上做以下实验，**每次只改一个变量**，其余保持不变，记录结果。

### 实验记录表（边做边填）

```markdown
| 实验 | 改动 | 最终准确率 | 收敛速度 | 现象/结论 |
| --- | --- | --- | --- | --- |
| 基准 | lr=0.01, 1 隐藏层(128), 无 Dropout | | | |
| A1 | lr=0.5 | | | |
| A2 | lr=0.0001 | | | |
| B1 | 隐藏层 128→1024 | | | |
| B2 | 加一层：784→256→128→10 | | | |
| C1 | 加 Dropout(0.5) | | | |
| D1 | batch_size 64→8 | | | |
```

### 实验 A：改学习率（最重要）

```python
optimizer = optim.SGD(model.parameters(), lr=0.5)     # A1: 太大
optimizer = optim.SGD(model.parameters(), lr=0.0001)  # A2: 太小
```

预期（和 Day 2 手写实验呼应）：A1 震荡甚至发散，A2 慢得着急。验证你 Day 2 的预测。

### 实验 B：改网络结构

```python
# B1: 单层加宽
model = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 10),
)

# B2: 加深一层
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
```

观察：更深/更宽是否一定更准？训练时间变长了多少？

### 实验 C：加 Dropout（对抗过拟合的利器）

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(0.5),          # 训练时随机"关掉" 50% 的神经元
    nn.Linear(256, 128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10),
)
```

**Dropout 的直觉**：训练时随机让一部分神经元"请假"，迫使网络不能把希望寄托在某几个神经元上，学到的特征更稳健。类似考试时抽掉你一半的笔记，逼你真正理解而不是背笔记。

> 注意：`model.eval()` 模式下 Dropout 自动失效（评估时用全部神经元），这就是昨天代码里 `eval()` 的另一个作用。

### 实验 D：改 batch size

```python
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
```

观察：训练变慢了多少？每个 epoch 的更新次数变成多少？（60000/8 = 7500 次）

---

## 晚上｜GPU 加速

### 任务

- [ ] 理解 GPU 为什么快
- [ ] 在 Colab 里开启免费 GPU，对比训练速度

### 为什么 GPU 快

神经网络的计算本质是**海量矩阵乘法**。CPU 像几个数学教授（核心少但每个很强），GPU 像几千个小学生（核心多但每个只会简单运算）。矩阵乘法恰好可以拆成几千个互不相干的简单乘法——小学生的天堂。

```
CPU： 4~16 个核心，适合复杂逻辑
GPU： 数千个核心，适合大规模并行计算（矩阵乘、卷积）
```

### Colab 开启 GPU

1. 菜单：**代码执行程序 → 更改运行时类型 → 硬件加速器选 T4 GPU**
2. 验证：

```python
import torch
print(torch.cuda.is_available())      # 应为 True
print(torch.cuda.get_device_name(0))  # 例如 "Tesla T4"
```

3. **代码不用改**——昨天已经写了 `.to(device)`，这就是好习惯的价值

### 对比实验

- [ ] 同样的代码，CPU 跑 5 个 epoch 计时 vs GPU 跑 5 个 epoch 计时
- [ ] 思考：MNIST 这么小，GPU 可能快不了多少——为什么？（提示：数据搬运有开销，小任务体现不出并行优势；Day 5 的 CNN 差距就明显了）

---

## 今日产出

✅ **一份完整的"调参实验记录"**（上面的实验表填满）+ CPU/GPU 训练时间对比

关键验收：看到任意一条训练曲线，能判断它是欠拟合、过拟合还是健康，并说出至少一种对策。

---

## 今日笔记模板

```markdown
## Day 4 学习笔记

### 调参实验结论（每条一句话）
- 学习率：
- 网络加深/加宽：
- Dropout：
- batch size：

### GPU vs CPU
- CPU 耗时：___  GPU 耗时：___
- 为什么差距不大/很大：

### 训练曲线诊断口诀（自己总结）
-

### 还模糊的地方
-
```

---

## 常见坑提醒

1. **一次改多个变量**——实验就白做了，说不清是哪个改动起的作用。一次只改一个
2. **Dropout 放在最后一层之后**——会把输出也"关掉"，结果没法看。Dropout 只放隐藏层之间
3. **Colab 换 GPU 后运行时会重置**，所有变量丢失，代码要重新跑（Cell → 全部执行）
4. **Colab 免费 GPU 有使用时限**，连续挂机会被断开，今天的内容半小时内能跑完，不用担心

---

*明天预告：Day 5 认识卷积神经网络 CNN，理解为什么它天生适合看图，并用 CNN 重做 MNIST 挑战更高准确率。*
