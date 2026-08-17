# Day 3｜第一个神经网络：手写数字识别（MNIST）

> 今日目标：用 PyTorch 跑通第一个真正的神经网络，逐行理解代码
> 时间投入：约 4~5 小时（全天动手）

---

## 今日路线

```
上午：跟着教程搭网络、跑通训练（约 2 小时）
下午：逐行理解 + 拆解四大步骤（约 2 小时）
收尾：自测 + 笔记（约 30 分钟）
```

---

## 背景知识：MNIST 是什么

MNIST 是深度学习界的"Hello World"：7 万张 28×28 的灰度手写数字图片，任务是识别图片上是 0~9 哪个数字。

```
输入：28×28 = 784 个像素值（拉平成一个向量）
输出：10 个数字各自的概率
```

为什么用全连接网络就够了？因为 MNIST 简单清晰，不需要 CNN 也能达到 97%+ 准确率。Day 5 会再用 CNN 重做一遍，到时可以对比。

---

## 上午｜跟着教程跑通

### 任务

- [ ] 选择一份教程，**亲手敲完**完整代码（二选一）：
  - **PyTorch 官方 60 分钟入门**（[pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，英文，简洁权威）
  - **李沐《动手学深度学习》第 3-4 章**（[zh.d2l.ai](https://zh.d2l.ai)，中文，B 站有配套视频，推荐）
- [ ] 在 Colab（或本地环境）跑通训练，确认准确率达到 **97%+**

### 参考实现（先跟教程敲，卡住再对照）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===== 第 1 步：准备数据 =====
transform = transforms.ToTensor()  # 图片 → 张量，像素值归一化到 [0, 1]

train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)   # 训练数据要打乱
test_loader  = DataLoader(test_data,  batch_size=1000, shuffle=False)

# ===== 第 2 步：定义网络 =====
model = nn.Sequential(
    nn.Flatten(),          # 28×28 → 784
    nn.Linear(784, 128),   # 全连接层：784 个输入 → 128 个输出
    nn.ReLU(),             # 激活函数（Day 2 学过：加非线性）
    nn.Linear(128, 10),    # 全连接层：128 → 10（10 个数字）
)

# ===== 第 3 步：训练循环 =====
loss_fn = nn.CrossEntropyLoss()               # 分类问题 → 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"使用设备: {device}")

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)             # ① 前向传播
        loss = loss_fn(predictions, labels)     # ② 算损失

        optimizer.zero_grad()                   # ③ 清空上一轮的梯度
        loss.backward()                         # ④ 反向传播算梯度
        optimizer.step()                        # ⑤ 梯度下降更新参数

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5 | 平均 loss: {total_loss/len(train_loader):.4f}")

# ===== 第 4 步：评估准确率 =====
model.eval()
correct = 0
total = 0
with torch.no_grad():                          # 评估时不用算梯度，省内存
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        predicted_labels = predictions.argmax(dim=1)   # 取概率最大的那个数字
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

print(f"测试集准确率: {correct/total:.2%}")
```

---

## 下午｜拆解四大步骤，逐行理解

昨天手写的 20 行 NumPy 和今天的 PyTorch 代码是**同一件事**，对照着看：

### 第 1 步：准备数据

- `DataLoader` 把 6 万张图片切成一批批（batch）喂给模型。`batch_size=64` 表示每次拿 64 张图算一次梯度、更新一次参数——这比逐张更新快，也比全量更新省内存
- `shuffle=True`：打乱顺序，防止模型记住数据的排列

> 对应 Day 2 手写版里的 `x, y`，只是那次数据太小，直接全量用了。

### 第 2 步：定义网络

- `nn.Linear(784, 128)` 就是一个全连接层，内部自动创建权重矩阵 `W(128×784)` 和偏置 `b(128)`——不用像昨天那样手动初始化
- `nn.ReLU()` 就是昨天学的 `max(0, x)`
- 网络结构 `784 → 128 → 10` 的含义：784 个像素 → 压缩出 128 个特征 → 映射到 10 个数字的得分

### 第 3 步：训练循环（最重要，必须背下来的五件套）

```python
predictions = model(images)   # ① 前向传播：算预测
loss = loss_fn(...)           # ② 算损失
optimizer.zero_grad()         # ③ 清空旧梯度（PyTorch 默认累加，必须手动清）
loss.backward()               # ④ 反向传播：自动算出所有参数的梯度
optimizer.step()              # ⑤ 梯度下降：w ← w - lr × 梯度
```

和 Day 2 手写版逐行对应：

| Day 2 手写 | Day 3 PyTorch |
| --- | --- |
| `y_pred = w * x + b` | `model(images)` |
| `loss = np.mean(...)` | `loss_fn(predictions, labels)` |
| 手算 `dw`、`db` | `loss.backward()`（自动求导！） |
| `w -= lr * dw` | `optimizer.step()` |
| —（不需要） | `optimizer.zero_grad()`（框架特有） |

**这就是深度学习框架的价值**：昨天你手算的梯度，今天 `loss.backward()` 一行搞定，不管网络有几百层。

### 第 4 步：评估

- `model.eval()` + `torch.no_grad()`：评估模式，关闭梯度计算
- `argmax(dim=1)`：模型输出 10 个得分，取最大的作为预测结果
- 在**测试集**上评估（Day 1 学过：用没见过的数据考试）

### 自测三问

1. `optimizer.zero_grad()` 如果删掉，会发生什么？（提示：PyTorch 的梯度默认是累加的）
2. 为什么训练时要 `shuffle=True`，评估时不用？
3. 网络的最后一层是 `Linear(128, 10)`，为什么是 10？

---

## 进阶小实验（有时间就做）

- [ ] 看看模型长什么样：`print(model)`，数一数总参数量
- [ ] 从测试集抽一张图，用模型预测，打印预测值和真实值对比：

```python
import matplotlib.pyplot as plt
image, label = test_data[0]
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(device)).argmax().item()
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"预测: {pred} | 真实: {label}")
plt.show()
```

- [ ] 故意找几张模型**预测错**的图看看——通常是人也不太好认的字

---

## 今日产出

✅ **跑通 MNIST，准确率达到 97%+，并逐行理解代码**

验收自测：合上书和代码，能否在白纸上默写出训练循环五件套（①前向 ②损失 ③清梯度 ④反向 ⑤更新）？这是深度学习面试和日常开发的基本功。

---

## 今日笔记模板

```markdown
## Day 3 学习笔记

### 四大步骤回顾（用自己的话）
1. 准备数据：
2. 定义网络：
3. 训练循环：
4. 评估：

### 最终准确率：___%

### 踩过的坑
-

### 还模糊的地方
- （明天带着这些问题继续）
```

---

## 常见坑提醒

1. **`download=True` 下载慢/失败**：换网络环境重试，或去镜像站手动下载放到 `./data/MNIST/raw/`
2. **准确率卡在 10% 左右**（相当于瞎猜）：九成是忘了 `optimizer.zero_grad()` 或学习率不合适
3. **报 `Expected object of device type cuda but got cpu`**：图片和模型必须在同一个设备上，检查是不是漏了 `.to(device)`
4. **不要在 Colab 里 `pip install torch`**：Colab 已预装，重装可能搞坏环境

---

*明天预告：Day 4 学会看训练曲线，动手做调参实验，理解过拟合与欠拟合。*
