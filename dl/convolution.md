# 卷积运算详解：从 2D 卷积到深度可分离卷积

卷积是 CNN 的核心算子。从 LeNet 到 MobileNet、ConvNeXt，每一代网络的进化都伴随着卷积形式的演进。本文系统梳理卷积的几种核心变体：标准 2D 卷积、Padding/Stride/Dilation、1×1 卷积、转置卷积、深度可分离卷积，并给出 PyTorch 实现与计算量对比。

---

## 一、卷积的数学定义

对一个二维输入 $X \in \mathbb{R}^{H \times W}$，用核 $K \in \mathbb{R}^{k \times k}$ 做卷积，输出 $Y$ 在 $(i, j)$ 处为：

$$
Y(i, j) = \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} K(u, v) \cdot X(i + u, j + v)
$$

在深度学习语境中通常用**互相关**代替严格数学意义的卷积（不翻转核），但习惯上仍称其为卷积。多通道情况下核扩展到 $(C_{in}, k, k)$，输出是每个通道卷积结果的和。

---

## 二、标准 2D 卷积

### 2.1 输出尺寸公式

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

其中 $p$ 是 padding，$k$ 是 kernel size，$s$ 是 stride。

### 2.2 参数量与计算量

$$
\text{Params} = k^2 \cdot C_{in} \cdot C_{out}
$$

$$
\text{FLOPs} = 2 \cdot k^2 \cdot C_{in} \cdot C_{out} \cdot H_{out} \cdot W_{out}
$$

乘 2 是因为每个输出元素要做 $k^2 C_{in}$ 次乘法和加法。

### 2.3 Padding 的三种选择

| Padding | 输出尺寸 | 适用场景 |
| ------- | -------- | -------- |
| `valid` (p=0)    | 缩小 | 需要逐步降采样 |
| `same` (p=k/2)   | 不变 | 保持分辨率 |
| `full`  (p=k-1)  | 扩大 | 转置卷积 |

### 2.4 Stride 的两个作用

- **降采样**：替代池化层。
- **扩大感受野**：用更大的 stride 快速增加单层视野。

### 2.5 PyTorch 基本用法

```python
import torch
import torch.nn as nn

# 标准 2D 卷积
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=1, bias=False)
x = torch.randn(1, 3, 224, 224)
print(conv(x).shape)  # torch.Size([1, 64, 224, 224])

# Conv vs Linear
# Conv: 共享权重 + 局部连接 (translation equivariance)
# Linear: 全连接,无结构假设
```

---

## 三、1×1 卷积：通道维上的线性变换

1×1 卷积在空间维度不做任何混合，只在**通道维**做线性变换 + 非线性激活。它有三个核心用途：

### 3.1 用途 1：跨通道信息融合

把 $C_{in}$ 个通道线性组合成 $C_{out}$ 个通道，类似在每个像素位置上跑一个 MLP。

### 3.2 用途 2：升降维（GoogLeNet 风格）

在 3×3/5×5 卷积之前用 1×1 卷积降维，大幅减少计算量。

### 3.3 用途 3：瓶颈结构（ResNet 风格）

Bottleneck 中先 1×1 降维 → 3×3 → 1×1 升维，计算量减为原来的 1/4。

```python
# 1x1 卷积等价于在每个像素位置跑一个全连接
bottleneck = nn.Sequential(
    nn.Conv2d(256, 64, 1),    # 降维
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 256, 1),    # 升维
)
```

---

## 四、空洞卷积（Dilated / Atrous Convolution）

空洞卷积在卷积核元素之间插入「空洞」，**在不增加参数的情况下扩大感受野**。

### 4.1 数学形式

设 dilation 为 $d$，核元素 $K(u, v)$ 在输入 $X$ 上的采样位置变为：

$$
Y(i, j) = \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} K(u, v) \cdot X(i + d \cdot u, j + d \cdot v)
$$

3×3 卷积、dilation=2 等价于 5×5 卷积的**感受野**，但只有 9 个参数。

### 4.2 主要应用

- **语义分割**（DeepLab 系列）：在保持分辨率的同时扩大感受野，避免下采样损失信息。
- **音频/时序**：TCN（时间卷积网络）。

```python
dilated = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
```

### 4.3 网格效应

堆叠相同 dilation 的空洞卷积会导致**采样点不连续**（gridding artifact）。实践上常用 **Hybrid Dilated Convolution（HDC）** 混合不同 dilation（如 1, 2, 5）来缓解。

---

## 五、转置卷积（Transposed Convolution）

转置卷积是卷积的**梯度算子**的对应前向操作，常用于上采样（如分割、生成）。它不是简单地把像素插值，而是**可学习的上采样**。

### 5.1 工作原理

普通卷积把 $H_{in} \times W_{in}$ 映射到 $H_{out} \times H_{out}$（通常更小）；转置卷积把这种映射反转回来。注意**这不是卷积的逆运算**，只是形状翻转。

### 5.2 输出尺寸公式

$$
H_{out} = (H_{in} - 1) \cdot s - 2p + k
$$

### 5.3 PyTorch 实现

```python
# 转置卷积: stride=2 把 14x14 -> 28x28
upconv = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                            kernel_size=4, stride=2, padding=1)
x = torch.randn(1, 64, 14, 14)
print(upconv(x).shape)  # torch.Size([1, 32, 28, 28])
```

### 5.4 棋盘格伪影

转置卷积在 kernel size 不能被 stride 整除时，会出现**棋盘格伪影**。常见缓解方法：

- 使用 `kernel_size = 2 * stride`
- 改用 **双线性插值 + 普通卷积**（许多现代分割模型用这种）

---

## 六、分组卷积与深度可分离卷积

### 6.1 分组卷积（Grouped Convolution）

把输入和输出通道都分成 $g$ 组，每组内部独立做卷积：

$$
\text{Params} = k^2 \cdot \frac{C_{in}}{g} \cdot \frac{C_{out}}{g} \cdot g = \frac{k^2 C_{in} C_{out}}{g}
$$

**计算量也减少为 $1/g$**。最早出现在 AlexNet（为解决单卡显存不足），后来成为 ResNeXt 的核心思想。

```python
grouped = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4)
```

### 6.2 深度可分离卷积（Depthwise Separable Convolution）

**MobileNet (Howard et al., 2017)** 的核心结构，由两步组成：

**Step 1 — Depthwise Convolution**：每个输入通道用一个独立的 $k \times k$ 卷积核处理。

```python
depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False)
```

**Step 2 — Pointwise Convolution**：用 1×1 卷积把通道混合起来。

```python
pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
```

### 6.3 计算量对比

标准卷积：

$$
\text{FLOPs} = 2 \cdot k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W
$$

深度可分离卷积：

$$
\text{FLOPs} = 2 \cdot k^2 \cdot C_{in} \cdot H \cdot W + 2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W
$$

压缩比：

$$
\frac{\text{DSConv}}{\text{Conv}} = \frac{1}{C_{out}} + \frac{1}{k^2}
$$

对 $k=3, C_{out}=256$，压缩比约 $\frac{1}{256} + \frac{1}{9} \approx 0.115$，**计算量减少约 88.5%**。

### 6.4 PyTorch 完整实现

```python
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积:Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# MobileNet Block (带残差)
class MobileNetBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_c, out_c, stride=stride)
        self.use_residual = (stride == 1 and in_c == out_c)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

---

## 七、各种卷积变体对比

| 算子                 | 参数量             | 计算量            | 主要用途            |
| -------------------- | ------------------ | ----------------- | ------------------- |
| 标准卷积             | $k^2 C_{in} C_{out}$ | $k^2 C_{in} C_{out} H W$ | 一般特征提取 |
| 1×1 卷积             | $C_{in} C_{out}$    | $C_{in} C_{out} H W$ | 通道混合/升降维 |
| 空洞卷积             | $k^2 C_{in} C_{out}$ | 同上（更稀疏采样） | 扩大感受野 |
| 分组卷积 ($g$ 组)     | $\frac{k^2 C_{in} C_{out}}{g}$ | $\frac{1}{g}$ 倍 | 减少计算 |
| 转置卷积             | $k^2 C_{in} C_{out}$ | 类似反向卷积 | 上采样 |
| 深度可分离卷积       | $k^2 C_{in} + C_{in} C_{out}$ | 约 $1/C_{out} + 1/k^2$ | 移动端 |

---

## 八、典型应用场景

- **图像分类 backbone**：标准卷积 + 瓶颈结构（ResNet、ConvNeXt）。
- **目标检测**：1×1 卷积做分类/回归头（Faster R-CNN、YOLO）。
- **语义分割**：空洞卷积（DeepLab）+ 转置卷积或双线性上采样。
- **移动端/边缘端**：深度可分离卷积（MobileNet、EfficientNet）。
- **超分辨率 / 生成模型**：转置卷积 + PixelShuffle。

---

## 九、一句话总结

> **标准卷积负责混合空间和通道信息；1×1 卷积混合通道、3×3 卷积混合空间**。**深度可分离卷积把这两件事拆开**，让 MobileNet 这种轻量网络在手机端实时运行成为可能。理解每种卷积的「工作分工」，是设计高效网络的前提。