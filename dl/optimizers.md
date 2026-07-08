# 优化器演进：SGD → Adam → AdamW → LAMB

优化器是深度学习训练的核心组件。从最简单的随机梯度下降（SGD），到带动量的 SGD，到 Adam、AdamW，再到 LAMB，每一次演进都围绕三个核心问题展开：**怎么估计梯度方向？怎么自适应学习率？怎么处理大规模 batch？** 本文系统梳理主流优化器的原理、代码实现和适用场景。

---

## 一、SGD 家族

### 1.1 朴素 SGD

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
$$

朴素 SGD 收敛慢，且容易在高曲率方向震荡、在低曲率方向爬行。

### 1.2 SGD with Momentum

引入动量项，把过去梯度的指数移动平均作为更新方向：

$$
v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot v_t
$$

其中 $\beta \in [0, 1)$ 通常取 0.9。**Nesterov 动量**进一步做前瞻修正：

$$
v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t - \eta \beta v_{t-1})
$$

直观上，动量让优化器像一颗有惯性的球，在平坦方向加速、在陡峭方向减速。

### 1.3 PyTorch 实现

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,    # L2 正则化
    nesterov=True,
)
```

---

## 二、AdaGrad（Adaptive Gradient, 2011）

AdaGrad 让**每个参数用自适应学习率**：对更新频繁的参数降低学习率，对稀疏更新的参数保持较大学习率。

$$
g_t = \nabla_\theta L(\theta_t)
$$

$$
s_t = s_{t-1} + g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot g_t
$$

**问题**：$s_t$ 单调递增，学习率会越来越小，最终无法继续学习。**适合稀疏数据**（NLP 词嵌入），但不适合深度网络。

---

## 三、RMSProp（Geoff Hinton, 未发表）

RMSProp 解决了 AdaGrad 学习率单调下降的问题，用**指数移动平均**代替累积和：

$$
s_t = \beta s_{t-1} + (1 - \beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot g_t
$$

通常 $\beta = 0.99$。

---

## 四、Adam（Kingma & Ba, 2015）

Adam = **Momentum（动量项）+ RMSProp（自适应学习率）**。它同时维护两个移动平均：

**一阶矩（梯度的均值）**：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**二阶矩（梯度的方差）**：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**偏差修正**（因为 $m_0 = v_0 = 0$）：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**更新参数**：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

默认超参：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

### PyTorch 使用

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### Adam 的优势与问题

**优势**：
- 对学习率不敏感（默认 1e-3 在大多数任务上都能用）
- 收敛快
- 适合非平稳目标

**问题**：
- **泛化性差**：在很多任务上 Adam 训练 loss 很低，但测试精度不如 SGD+Momentum
- 原因是自适应学习率倾向于找到「尖锐极小值」，泛化不好

---

## 五、AdamW（Loshchilov & Hutter, 2019）

### 5.1 Adam + L2 正则化的陷阱

在 Adam 中直接加 L2 正则化（`weight_decay`）会让**自适应学习率缩放权重衰减**，导致正则化效果不均匀。

### 5.2 解耦权重衰减

AdamW 把权重衰减从梯度中**解耦**出来，直接作用于参数：

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

其中 $\lambda$ 是权重衰减系数。注意**这里 $\lambda$ 通常比 Adam 的 `weight_decay` 大 10–100 倍**（典型值 0.01–0.1）。

### 5.3 PyTorch 使用

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 5.4 现代默认选择

**Transformer / ViT / LLaMA / Stable Diffusion 等几乎都用 AdamW**。它解决了 Adam 的两个问题：
- 更好的泛化
- 训练更稳定

---

## 六、LAMB（You et al., 2020）

### 6.1 动机

BERT 预训练中常用 **batch size 几万**（如 LAMB 用 65536）。Adam 直接用大学习率会发散；SGD 用大学习率太慢。LAMB 通过**层级自适应学习率**让大 batch 训练稳定。

### 6.2 算法

对每一层 $\theta^l$，先计算标准的 Adam 更新方向 $u^l$，再做**层级归一化**：

$$
g^l = \frac{\hat{m}_t^l}{\sqrt{\hat{v}_t^l} + \epsilon}
$$

$$
\phi^l = \frac{\|\theta^l\|}{\|g^l\| + \epsilon}
$$

$$
\theta^l \leftarrow \theta^l - \eta \cdot \phi^l \cdot (g^l + \lambda \theta^l)
$$

直觉：**每层的更新幅度被自动缩放到与该层参数范数相当**，避免深层网络累积过大更新。

### 6.3 PyTorch 实现

```python
from torch.optim import Optimizer

class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']

                # Adam update direction
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                update = m_hat / (v_hat.sqrt() + group['eps'])

                # Weight decay
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p

                # Layer-wise adaptive scaling
                weight_norm = p.norm().clamp(min=0)
                update_norm = update.norm()
                if weight_norm == 0 or update_norm == 0:
                    continue
                trust_ratio = weight_norm / update_norm

                p.add_(update, alpha=-group['lr'] * trust_ratio.item())
        return loss
```

### 6.4 适用场景

- **BERT/RoBERTa 预训练**：batch ≥ 8K 时基本必用
- **大模型分布式训练**

---

## 七、Lion（Chen et al., 2023）

### 7.1 动机

Google 在 2023 年提出的新型优化器，**只用动量（一阶矩）**，不维护二阶矩，**显存占用减半**。

### 7.2 算法

$$
c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \left( \text{sign}(c_t) + \lambda \theta_t \right)
$$

$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$

注意 `sign(c)` 把动量量化到 $\{-1, +1\}$，所有维度用相同步长。

### 7.3 优势与局限

- **优势**：训练速度快、显存占用小、对小 batch 友好
- **局限**：泛化性需要仔细调参，目前主要在 LLM 上验证

---

## 八、八大优化器对比

| 优化器    | 自适应 LR | 动量 | 权重衰减 | 显存/参数 | 主要场景 |
| --------- | --------- | ---- | -------- | --------- | -------- |
| SGD       | ✗         | 可选 | L2       | 1×        | CNN 分类（需要更好泛化） |
| SGD+M     | ✗         | ✓    | L2       | 1×        | ResNet 系列 |
| AdaGrad   | ✓         | ✗    | ✗        | 2×        | 稀疏特征 |
| RMSProp   | ✓         | ✗    | ✗        | 2×        | RNN |
| Adam      | ✓         | ✓    | 弱       | 3×        | 通用、GAN |
| AdamW     | ✓         | ✓    | 强（解耦）| 3×        | **Transformer 默认** |
| LAMB      | ✓         | ✓    | 强       | 3×        | 大 batch 训练 |
| Lion      | ✗         | ✓    | 强       | 2×        | LLM 显存优化 |

---

## 九、工程实践指南

| 任务 | 推荐优化器 | 典型学习率 | 关键超参 |
| ---- | --------- | --------- | -------- |
| ResNet 图像分类     | SGD + Momentum + nesterov | 0.1 | momentum=0.9, wd=1e-4 |
| Transformer 预训练  | AdamW                  | 1e-4 ~ 3e-4 | betas=(0.9, 0.95), wd=0.1 |
| BERT 大 batch 预训练 | LAMB                    | 1e-3 ~ 2e-3 | warmup + linear decay |
| GAN               | Adam (不带 WD)           | 1e-4 ~ 2e-4 | betas=(0.0, 0.9) |
| 扩散模型（Stable Diffusion） | AdamW                  | 1e-5 ~ 1e-4 | wd=0.01 |
| LLaMA 微调        | AdamW                   | 2e-5 ~ 2e-4 | wd=0.1, betas=(0.9, 0.95) |

---

## 十、AdamW 完整训练循环示例

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 注意:AdamW 的 weight_decay 通常 0.01 ~ 0.1
# 不要和 Adam 的 weight_decay 数值混淆

for epoch in range(num_epochs):
    for x, y in dataloader:
        # 1. 前向传播
        pred = model(x)
        loss = criterion(pred, y)

        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 3. 梯度裁剪(Transformer 必加)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. 优化器更新
        optimizer.step()
```

---

## 十一、一句话总结

> **SGD 适合泛化，Adam/AdamW 适合快速收敛，LAMB 适合大 batch 训练，Lion 适合显存受限的 LLM**。**Transformer 时代 AdamW 是默认选择**，但别忘了：在图像分类任务上仔细调过的 SGD+Momentum 仍然可能打败 Adam。