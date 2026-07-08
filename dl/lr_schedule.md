# 学习率调度：Warmup + Cosine Annealing + OneCycle

学习率（Learning Rate）是深度学习训练中**最重要的超参数**。固定学习率很难兼顾训练初期和末期：**初期需要小学习率避免发散，末期需要小学习率收敛到尖锐极小值**。本文系统讲解三大主流调度策略：**Warmup + Cosine Annealing + OneCycle**，以及它们的组合使用。

---

## 一、为什么需要学习率调度？

考虑一个深层网络的训练：

- **训练初期**：模型参数随机初始化，loss 表面崎岖。**大学习率容易把网络「一脚踢飞」**，梯度爆炸。
- **训练中期**：模型已经学到一些有用特征，可以用**中等学习率**快速逼近最优点。
- **训练末期**：参数在最优解附近，需要**小学习率**做精细调整，找到更尖锐、泛化更好的极小值。

固定的单一学习率难以同时满足这些需求。**学习率调度**就是按时序动态调整 LR 的曲线。

---

## 二、Warmup（学习率预热）

### 2.1 动机

最初的 Transformer 论文 *Attention Is All You Need* 观察到：**直接用大学习率（如 5e-4）训练初期几个 step 就会发散**。解决方案：从一个很小的 LR 线性增加到目标 LR，给优化器「预热」时间。

### 2.2 Linear Warmup

$$
\text{lr}(t) = \text{lr}_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}}
$$

其中 $T_{\text{warmup}}$ 是 warmup 步数（通常是总步数的 1%–10%）。

### 2.3 Warmup 的理论解释

- **自适应优化器的早期偏差**：Adam 的 $m, v$ 初始化为 0，前几步估计不准
- **大 batch 训练的早期稳定性**：batch 越大，warmup 越重要
- **梯度量级在训练初期变化大**：warmup 让 LR 适应梯度变化

### 2.4 PyTorch 实现

```python
import torch
import math


def linear_warmup_lr(step, warmup_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    return peak_lr


# 配合 LambdaLR 使用
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: linear_warmup_lr(step, warmup_steps=1000, peak_lr=1.0),
)
```

---

## 三、Step Decay（阶梯衰减）

最经典也最简单的策略：每隔若干 epoch 把 LR 降低为原来的 $\gamma$ 倍（通常 $\gamma = 0.1$）。

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1  # 每 30 个 epoch 衰减 10 倍
)
```

**ResNet 训练的标准做法**：120 epoch 训练中，在 epoch 30 和 60 各衰减 10 倍。

**优点**：简单有效。**缺点**：LR 突变可能让训练不稳定。

---

## 四、Exponential Decay（指数衰减）

$$
\text{lr}(t) = \text{lr}_0 \cdot \gamma^t
$$

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

比 Step Decay 平滑，但衰减速度需要手动调。

---

## 五、Cosine Annealing（余弦退火）

### 5.1 基础公式

$$
\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

LR 从 $\text{lr}_{\max}$ 平滑下降到 $\text{lr}_{\min}$，曲线形状像半个余弦波。

### 5.2 Cosine Annealing with Warm Restarts (SGDR)

$$
\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\frac{t \mod T_i}{T_i}\pi\right)\right)
$$

每隔 $T_i$ 步把 LR 重置到 $\text{lr}_{\max}$（warm restart）。直觉：周期性跳出当前极小值，寻找更优解。

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

### 5.3 Cosine Annealing 变体

PyTorch 提供的 `CosineAnnealingLR` 在达到 `T_max` 步后保持 `eta_min`，不再重启。要实现 restart 行为，需要手动循环或用上文的 SGDR 版本。

---

## 六、OneCycle Policy（Smith, 2018）

### 6.1 核心思想

**OneCycle** 是一种「**单周期、退火 + 动量反向**」的调度策略：

1. **第一阶段（前 45%）**：LR 从 `max_lr/div_factor` 升到 `max_lr`，动量从 `mom_max` 降到 `mom_min`
2. **第二阶段（45%–90%）**：LR 从 `max_lr` 降到 `max_lr/final_div_factor`，动量反向
3. **第三阶段（最后 10%）**：LR 在更小范围内再退火

### 6.2 为什么 OneCycle 效果好？

- **超级收敛（Super-Convergence）**：用 OneCycle 可以用比常规大 10 倍的学习率，训练速度提升 5–10 倍
- **隐式正则化**：高 LR 阶段相当于在平坦的极小值之间跳跃，反而起到正则化效果
- **简单**：只需指定 `max_lr` 和 `total_steps`

### 6.3 PyTorch 实现

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(dataloader),
    pct_start=0.3,         # 升 LR 阶段占比
    anneal_strategy='cos', # 退火方式
    div_factor=25.0,      # 起始 LR = max_lr / 25
    final_div_factor=1e4, # 终止 LR = max_lr / 25 / 1e4
)
```

### 6.4 LR Range Test（找最优 max_lr）

**Leslie Smith** 同时提出了 LR Range Test：用一个 epoch，从极小 LR 线性增长到极大 LR，观察 loss 变化：

- **LR 太小**：loss 几乎不动
- **LR 适中**：loss 快速下降
- **LR 太大**：loss 发散或震荡

**最佳 max_lr ≈ loss 开始发散前的 1/3 处**。

```python
import matplotlib.pyplot as plt

def lr_range_test(model, train_loader, criterion, optimizer,
                  start_lr=1e-7, end_lr=1e-1, num_steps=200):
    lrs, losses = [], []
    lr_lambda = lambda step: (start_lr + (end_lr - start_lr) * step / num_steps) / start_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    iterator = iter(train_loader)
    for step in range(num_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

    # 绘图找最优 LR
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.show()
```

---

## 七、Warmup + Cosine Annealing（Transformer 默认）

现代 Transformer/LLM 训练的事实标准：**线性 warmup + 余弦退火**。

### 7.1 公式

设总步数为 $T$，warmup 步数为 $T_w$，warmup 阶段结束后进入 cosine 退火：

$$
\text{lr}(t) = \begin{cases}
\text{lr}_{\text{peak}} \cdot \dfrac{t}{T_w} & 0 \le t < T_w \\[6pt]
\text{lr}_{\min} + \dfrac{1}{2}(\text{lr}_{\text{peak}} - \text{lr}_{\min})\left(1 + \cos\left(\dfrac{t - T_w}{T - T_w}\pi\right)\right) & T_w \le t \le T
\end{cases}
$$

### 7.2 PyTorch 实现（无内置版本，手写）

```python
import math
import torch


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Transformers 库内置
# from transformers import get_cosine_schedule_with_warmup
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=100000)
```

### 7.3 在 LLaMA/BERT 中

| 模型   | Peak LR | Warmup | Min LR Ratio | 总步数 |
| ------ | ------- | ------ | ------------ | ------ |
| BERT-base | 1e-4 | 10000 (1%) | 0 | 1M |
| GPT-2 (small) | 2.5e-4 | 2000 | 0.1 | ~600K |
| LLaMA-7B | 3e-4 | 2000 | 0.1 | 1T tokens |

---

## 八、各调度策略对比

| 策略       | 曲线形状       | 主要优点       | 主要缺点       | 典型场景     |
| ---------- | -------------- | -------------- | -------------- | ------------ |
| 固定 LR    | 水平线         | 简单           | 训练慢 / 不收敛 | 实验基线     |
| Step       | 阶梯下降       | 简单           | 突变可能不稳定   | ResNet 训练  |
| Exponential | 指数下降       | 平滑           | 衰减速度难调     | RNN         |
| Cosine     | 半余弦下降     | 平滑、性能稳定  | 需要预设总步数   | 通用         |
| Cosine WR  | 多次余弦重启   | 跳出局部最优    | 重启时机难调     | 通用         |
| OneCycle   | 单峰 + 反向动量 | 训练快、隐式正则 | 需要 LR Range Test | fastai 框架  |
| Linear Decay | 线性下降到 0 | 平滑           | 无重启能力       | 微调         |

可视化：

```
Fixed:        ───────────────────────
Step:         ─────┐    ┌────┐
                       ┘    └─────
Exponential:  ─────╲
                  ─────╲
Cosine:       ╭───────────╮
              ────────────────
OneCycle:     ╭╮
              ─╯╰─────────
```

---

## 九、最佳实践

### 9.1 通用模板（Transformer / LLM）

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = int(0.03 * num_training_steps)  # 3% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    min_lr_ratio=0.1,
)
```

### 9.2 找最优学习率

**先跑 LR Range Test** 确定 max_lr，再选择调度策略：

- **小 batch + CNN**：OneCycle
- **大模型 / LLM**：Linear Warmup + Cosine Annealing
- **CNN 微调**：Step Decay

### 9.3 训练失败时排查

| 现象 | 可能原因 | 解决方案 |
| ---- | -------- | -------- |
| Loss 初期 NaN/Inf | LR 太大 | 加 warmup、降低 peak LR |
| Loss 震荡不收敛 | LR 太大 | 降低 LR、加 gradient clip |
| Loss 收敛但 plateau | LR 衰减不够 | 加 cosine / OneCycle |
| 训练后期 loss 不再下降 | LR 太小 | 检查 eta_min 是否过小 |

---

## 十、一句话总结

> **训练初期小 LR 预热，训练中期峰值加速，训练末期余弦退火**——这套「Warmup + Cosine Annealing」组合是现代深度学习（尤其是 Transformer）的事实标准。OneCycle 则在 CNN/小模型场景下能加速 5–10 倍，是 fastai 框架的默认选择。