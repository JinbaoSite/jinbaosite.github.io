# 归一化层：BatchNorm / LayerNorm / GroupNorm

深度神经网络的训练困难，很大程度源于**内部协变量偏移（Internal Covariate Shift）**：每一层的输入分布在训练过程中不断变化，让梯度难以稳定。**归一化层**是直接针对这一问题的标准武器，从 2015 年的 **BatchNorm** 开始，衍生出 LayerNorm、GroupNorm、InstanceNorm 四大变体，分别面向不同任务。本文系统梳理它们的原理、对比和工程选择。

---

## 一、为什么需要归一化？

考虑一个深度网络第 $l$ 层，输入 $x^{(l)}$ 的分布取决于前面所有层的参数。当这些参数被更新，$x^{(l)}$ 的分布就跟着变化。这种「**每一层都必须在变化的输入分布上学习**」的现象就是内部协变量偏移。

归一化通过**强制把每一层的输入拉回到标准分布**，让优化景观（loss landscape）更平滑。实验上能带来：

- 允许更大的学习率
- 训练更快收敛
- 对初始化不那么敏感
- 隐式正则化效果

---

## 二、BatchNorm（Ioffe & Szegedy, 2015）

### 2.1 算法

对一个 mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$，在每个通道独立计算均值和方差：

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

归一化：

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

可学习的仿射变换：

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\gamma, \beta$ 是可学习参数，初始化为 $(\gamma=1, \beta=0)$。

### 2.2 训练 vs 推理

**训练时**：用当前 mini-batch 的均值和方差。
**推理时**：用**滑动平均**的 running mean 和 running variance（训练时累积）。

```python
bn = nn.BatchNorm2d(num_features=64)  # 输入 (B, 64, H, W)

# 训练时
bn.train()
y = bn(x)  # 用 batch 统计

# 推理时
bn.eval()
y = bn(x)  # 用 running 统计
```

### 2.3 BatchNorm 的三大问题

| 问题 | 说明 |
| ---- | ---- |
| **依赖 batch size** | batch=1 时方差为 0，归一化无意义；小 batch 时估计噪声大 |
| **RNN/Transformer 不友好** | 时序数据中每个时间步的统计量不一致 |
| **分布式训练同步问题** | 多卡训练时 BN 统计量需要跨卡同步（SyncBN） |

### 2.4 BatchNorm 的位置：Conv-BN vs BN-Conv

现代框架默认**Conv → BN → ReLU**（post-activation）。Pre-activation ResNet 改为 **BN → ReLU → Conv**，效果更好。

### 2.5 PyTorch 实现

```python
import torch
import torch.nn as nn

# 2D BatchNorm: (B, C, H, W) 在 (B, H, W) 维上求均值
bn = nn.BatchNorm2d(num_features=64)
x = torch.randn(8, 64, 32, 32, requires_grad=True)
y = bn(x)
print(f"输入均值/方差: {x.mean():.4f} / {x.var():.4f}")
print(f"输出均值/方差: {y.mean():.4f} / {y.var():.4f}")
```

---

## 三、LayerNorm（Ba et al., 2016）

### 3.1 算法

LayerNorm **不依赖 batch 维度**，对每个样本**自身**的所有特征做归一化：

$$
\mu_l = \frac{1}{H} \sum_{i=1}^{H} x_i
$$

$$
\sigma_l^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu_l)^2
$$

$$
y = \gamma \cdot \frac{x - \mu_l}{\sqrt{\sigma_l^2 + \epsilon}} + \beta
$$

其中 $H$ 是单个样本的特征总数。对 NLP 任务就是 `seq_len × hidden_dim` 的全部元素；对 CNN 应用较少（因为会把空间信息也归一化掉）。

### 3.2 与 BatchNorm 的对比

| 维度       | BatchNorm                  | LayerNorm                        |
| ---------- | -------------------------- | -------------------------------- |
| 归一化维度 | batch + 空间                | 单个样本的特征                   |
| 是否依赖 batch | 是                      | 否                                |
| RNN/Transformer | 需要 padding mask 处理 | 直接用                           |
| 可学习参数 | $\gamma, \beta \in \mathbb{R}^C$ | $\gamma, \beta \in \mathbb{R}^H$ |
| 主流场景   | CNN                        | Transformer / RNN                |

### 3.3 现代 LLM 标准配置

现代 Transformer 几乎都用 **Pre-Norm**（LN 在注意力/FFN 之前）：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        # Pre-Norm
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x
```

**Pre-Norm vs Post-Norm**：Pre-Norm 训练更稳定（梯度直通残差连接），但 Post-Norm 表达力更强（GPT-2/3 早期用 Post-Norm）。

### 3.4 PyTorch 实现

```python
# 1D LayerNorm: (B, L, C) 在 (L, C) 维归一化
ln = nn.LayerNorm(normalized_shape=512)
x = torch.randn(8, 128, 512)
y = ln(x)
```

---

## 四、GroupNorm（Wu & He, 2018）

### 4.1 动机

目标检测和分割任务中，受显存限制 batch size 往往只有 1–2 张图，**BatchNorm 几乎失效**。GroupNorm 介于 BatchNorm 和 LayerNorm 之间：

- 把通道分成 $g$ 组
- 每组内对 `(B, C/g, H, W)` 计算均值和方差

### 4.2 算法

设 $C$ 个通道分成 $G$ 组，每组 $C/G$ 个通道。在每个样本上，按组内 $(C/G, H, W)$ 的元素计算均值和方差：

$$
\mu_{ng} = \frac{G}{C \cdot H \cdot W} \sum_{c \in \text{group}_g} \sum_{h, w} x_{nchw}
$$

$$
\sigma_{ng}^2 = \frac{G}{C \cdot H \cdot W} \sum_{c \in \text{group}_g} \sum_{h, w} (x_{nchw} - \mu_{ng})^2
$$

### 4.3 PyTorch 实现

```python
# 2D GroupNorm: 把 C 个通道分成 G 组,每组独立归一化
gn = nn.GroupNorm(num_groups=32, num_channels=64)  # 要求 C 能被 G 整除
x = torch.randn(2, 64, 32, 32)
y = gn(x)  # batch=2 也能稳定工作
```

### 4.4 特殊情况

| 分组数 $G$ | 等价于 |
| ---------- | ------ |
| $G = 1$    | LayerNorm |
| $G = C$    | InstanceNorm |
| $G = 32$（常用） | GroupNorm |

### 4.5 实验效果

论文中在 COCO 检测 + ImageNet 分类上，**GroupNorm 在 batch=2 时明显优于 BatchNorm**，对显存受限场景特别友好。**YOLOv5/v8 等检测框架默认用 GroupNorm 替代 BatchNorm**。

---

## 五、InstanceNorm（Ulyanov et al., 2016）

### 5.1 算法

对每个样本、每个通道单独归一化：

$$
\mu_{nc} = \frac{1}{HW} \sum_{h,w} x_{nchw}
$$

$$
\sigma_{nc}^2 = \frac{1}{HW} \sum_{h,w} (x_{nchw} - \mu_{nc})^2
$$

### 5.2 适用场景

**风格迁移**：每个样本（图像）的对比度/亮度需要独立归一化，去掉风格信息。

### 5.3 PyTorch 实现

```python
in_norm = nn.InstanceNorm2d(num_features=64, affine=True)
```

---

## 六、四大归一化方法对比

| 方法     | 归一化范围               | 依赖 batch | 主要场景     |
| -------- | ----------------------- | --------- | ---------- |
| BatchNorm | $(N, H, W)$ 每个通道     | 是        | CNN 分类   |
| LayerNorm | 单样本所有特征            | 否        | Transformer / RNN |
| GroupNorm | $(C/G, H, W)$ 每组       | 否        | 检测/分割小 batch |
| InstanceNorm | $(H, W)$ 每样本每通道 | 否        | 风格迁移   |

可视化（$\text{N=batch, C=channel, H/W=空间}$）：

```
BatchNorm        LayerNorm       GroupNorm       InstanceNorm
[ ■■■■■ ]        [ ■■■■■ ]       [ ■■□□□ ]       [ □□□□□ ]
[ ■■■■■ ]        [ ■■■■■ ]       [ ■■□□□ ]       [ □□□□□ ]
[ ■■■■■ ]        [ ■■■■■ ]       [ ■■□□□ ]       [ □□□□□ ]
 同一C跨B         同一样本全部    同一组跨B        单个C单样本
```

---

## 七、现代实践指南

| 任务                          | 推荐归一化       | 理由 |
| ----------------------------- | ---------------- | ---- |
| 图像分类 (ResNet 等)          | BatchNorm        | batch 通常较大，效果最佳 |
| 目标检测/分割 (YOLO/Faster R-CNN) | GroupNorm / SyncBN | 显存受限，batch 较小 |
| Transformer / LLM             | LayerNorm (RMSNorm) | 与序列无关，训练稳定 |
| 风格迁移                      | InstanceNorm     | 去除图像风格信息 |
| GAN (DCGAN/StyleGAN)          | 不归一化或 PixelNorm | BN 不稳定 |
| 小 batch size（医疗/遥感）     | GroupNorm / LayerNorm | BN 估计噪声大 |

---

## 八、RMSNorm：LayerNorm 的简化版

**RMSNorm**（Zhang & Sennrich, 2019）发现 LayerNorm 中的均值平移 $\mu$ 对结果影响很小，可以直接去掉：

$$
y = \gamma \cdot \frac{x}{\sqrt{\frac{1}{H}\sum x_i^2 + \epsilon}}
$$

计算量减少约 7%–64%，效果几乎不变。**LLaMA、Falcon 等现代 LLM 都用 RMSNorm**：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
```

---

## 九、一句话总结

> **BatchNorm 看 batch，LayerNorm 看样本，GroupNorm 看组，InstanceNorm 看通道**。选哪一种，**取决于任务的数据形态和 batch size 限制**。现代 LLM 用 RMSNorm 简化版，是 LayerNorm 的极致精简。