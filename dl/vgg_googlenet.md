# VGG 与 GoogLeNet：更深更宽的探索

AlexNet 之后，研究者开始系统性地探索「**网络深度和宽度对性能的影响**」。2014 年 ImageNet 竞赛诞生了两大经典：**VGG**（Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*）和 **GoogLeNet**（Szegedy et al., *Going Deeper with Convolutions*）。两者从不同方向回答了同一个问题：如何让 CNN 更深、更准？

---

## 一、VGG（Visual Geometry Group, 2014）

### 1.1 核心思想

VGG 的核心贡献非常简洁——**全部使用 3×3 小卷积核，通过反复堆叠把网络做到 16–19 层**。

为什么选择 3×3？两个 3×3 卷积堆叠的感受野等价于一个 5×5 卷积，三个 3×3 卷积等价于一个 7×7 卷积，但：

- **参数更少**：$3 \times (3^2 C^2) = 27C^2$ vs $7^2 C^2 = 49C^2$，减少 45%。
- **非线性更多**：每层一个 ReLU，3 层比 1 层多 2 个非线性变换。
- **隐式正则化更强**：更深的网络相当于更强的特征分解。

### 1.2 网络配置

VGG 提供了 6 种配置（A、A-LRN、B、C、D、E），常用的是 VGG-16（C）和 VGG-19（E）：

| 层级         | VGG-16（C）                       | 输出尺寸          |
| ------------ | -------------------------------- | ----------------- |
| Block 1      | [Conv 3×3 → ReLU] × 2, 64       | 224×224×64        |
| Pool         | MaxPool 2×2                       | 112×112×64        |
| Block 2      | [Conv 3×3 → ReLU] × 2, 128      | 112×112×128       |
| Pool         | MaxPool 2×2                       | 56×56×128         |
| Block 3      | [Conv 3×3 → ReLU] × 3, 256      | 56×56×256         |
| Pool         | MaxPool 2×2                       | 28×28×256         |
| Block 4      | [Conv 3×3 → ReLU] × 3, 512      | 28×28×512         |
| Pool         | MaxPool 2×2                       | 14×14×512         |
| Block 5      | [Conv 3×3 → ReLU] × 3, 512      | 14×14×512         |
| Pool         | MaxPool 2×2                       | 7×7×512           |
| Classifier   | FC 4096 → FC 4096 → FC 1000      | 1000              |

参数量约 **1.38 亿**，其中全连接层就占了 1.19 亿。

### 1.3 PyTorch 实现

```python
import torch
import torch.nn as nn

VGG16_CFG = [64, 64, 'M',
             128, 128, 'M',
             256, 256, 256, 'M',
             512, 512, 512, 'M',
             512, 512, 512, 'M']


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = make_layers(VGG16_CFG)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = VGG16()
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)  # torch.Size([2, 1000])
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
```

### 1.4 VGG 的优点与局限

**优点**：
- 结构简单、规整、易理解、易迁移。
- 3×3 卷积成为后续网络的默认选项。
- 在迁移学习中是经典 backbone（如 VGG-Face、VGG-Fish）。

**局限**：
- 参数量巨大（1.38 亿），推理慢。
- 全连接层占了大半参数，造成冗余。
- 19 层之后再加深就难以训练（梯度消失）。

---

## 二、GoogLeNet / Inception-v1（Szegedy et al., 2014）

### 2.1 核心思想

GoogLeNet 与 VGG 同期发表，但走了完全不同的路：**用稀疏连接替代密集连接**。

论文的核心洞见来自 Arora 等人的理论工作：一个**最优的神经网络拓扑结构应该是逐层稀疏的**，但当前的硬件（GPU）更适合密集计算。GoogLeNet 通过 **Inception 模块**在密集计算中近似这种稀疏性。

### 2.2 Inception 模块（Naive 版本）

最朴素的 Inception 模块在同一层并行使用 1×1、3×3、5×5 卷积和 3×3 池化，最后拼接：

```
        Input
       /  |  \  \
     1x1 3x3 5x5 3x3 pool
       \  |  /  /
        Concat
```

**问题**：5×5 卷积在高层计算量爆炸。例如一个 5×5 卷积，输入通道 256，输出 256：

$$
\text{FLOPs} = 5^2 \times 256 \times 256 \times W \times H = 1.6 \times 10^6 \times W \times H
$$

### 2.3 降维 Inception 模块

GoogLeNet 的关键创新是**在 3×3 和 5×5 卷积之前用 1×1 卷积降维**：

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_red, ch3x3, ch5x5_red, ch5x5, pool_proj):
        super().__init__()
        # 1x1 conv branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 3x3 conv branch (with 1x1 reduction)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 5x5 conv branch (with 1x1 reduction)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # 3x3 pool branch (with 1x1 projection)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x),
                          self.branch2(x),
                          self.branch3(x),
                          self.branch4(x)], dim=1)
```

1×1 卷积的作用：
- **降维**：减少 3×3/5×5 输入的通道数。
- **跨通道信息融合**：在保持空间分辨率的同时混合通道。
- **增加非线性**：每个 1×1 卷积后接 ReLU。

降维后，5×5 卷积的计算量减少到原来的 1/10，而性能几乎不受影响。

### 2.4 完整 GoogLeNet 结构

GoogLeNet 由 9 个 Inception 模块堆叠而成，共 **22 层**（含参数层），但参数量只有 VGG 的 1/12（约 **500 万**）。

```
Input (224x224x3)
  → Conv 7x7 stride 2  → 112x112x64
  → MaxPool 3x3 stride 2 → 56x56x64
  → Conv 1x1 → Conv 3x3 → 28x28x192
  → MaxPool 3x3 stride 2 → 14x14x192
  → Inception(3a) → Inception(3b) → 28x28x256
  → MaxPool → 14x14x480
  → Inception(4a) → Inception(4b) → Inception(4c) → Inception(4d) → Inception(4e) → 14x14x832
  → MaxPool → 7x7x832
  → Inception(5a) → Inception(5b) → 7x7x1024
  → AvgPool → 1x1x1024
  → Dropout(0.4)
  → FC 1000 → Softmax
```

### 2.5 辅助分类器

GoogLeNet 在中间层（4a、4d）额外接了两个 softmax 分类器，训练时以 0.3 权重加到总损失上。这是为了：
- 把梯度直接注入中间层，缓解梯度消失。
- 提供额外的正则化。

推理时这两个分支被丢弃。

### 2.6 PyTorch 完整实现

```python
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        if aux_logits:
            self.aux2 = AuxClassifier(528, num_classes)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x), aux1, aux2
```

> 注：实际工程中可用 `torchvision.models.googlenet(weights=...)` 直接调用预训练版本。

---

## 三、VGG vs GoogLeNet 的对比

| 维度              | VGG-16                       | GoogLeNet                    |
| ----------------- | ---------------------------- | ---------------------------- |
| 核心思路          | 更深 + 小卷积                 | 更宽 + 多尺度并行             |
| 深度              | 16 层                        | 22 层（9 个 Inception）       |
| 参数量            | 1.38 亿                      | ~500 万                       |
| Top-5 错误率       | 7.3%                         | 6.7%                          |
| 关键创新          | 3×3 卷积堆叠                  | 1×1 降维 + Inception 多分支    |
| 训练技巧          | 无特殊                       | 辅助分类器                    |
| 主要问题          | 参数量大、推理慢               | 结构复杂、定制化困难            |

两者在 2014 年的 ImageNet 上几乎打平（GoogLeNet 略胜），但**GoogLeNet 用 1/12 的参数达到了更好的精度**，开启了「高效 CNN」的研究方向。

---

## 四、对后续工作的影响

- **VGG** → 几乎所有后续 backbone 都用 3×3 卷积；FCN（全卷积网络）、Faster R-CNN 等经典检测/分割模型都用 VGG 作为初始 backbone。
- **GoogLeNet** → 催生了 Inception v2/v3/v4（加入 BatchNorm、Factorization），并与 ResNet 思想结合产生了 Inception-ResNet。

---

## 五、一句话总结

> **VGG 用深度证明「更深的简单网络可以更准」，GoogLeNet 用宽度证明「并行多尺度结构能用更少参数达到同样的精度」**。两者殊途同归，共同把 CNN 推向「更深更宽」的下一站——**ResNet**。