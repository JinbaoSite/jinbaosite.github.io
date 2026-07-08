# ResNet 与 DenseNet：残差连接的胜利

VGG 和 GoogLeNet 把网络推到了 20 层左右，**再深就训不动了**——不是因为过拟合，而是因为梯度消失/爆炸以及优化难度陡然上升。2015 年，何恺明等人提出的 **ResNet**（*Deep Residual Learning for Image Recognition*）用「残差连接」一举把网络深度推到 152 层，并在 ImageNet 上把 top-5 错误率降到 3.57%。随后 **DenseNet** 进一步把「跨层连接」推到极致。本文拆解这两篇里程碑论文。

---

## 一、问题：为什么「更深」反而变差？

直觉上，更深的网络至少应该和浅层网络一样好——把后面的层学成恒等映射即可。但实验表明：

- **网络越深，训练误差越高**（不是验证误差）。
- 这说明 **优化器根本找不到那个恒等映射的解**。

这不是过拟合，而是**退化（degradation）问题**。它说明深度网络的解空间比浅层网络更复杂，solver 难以驾驭。

残差学习的动机：**让网络更容易学到恒等映射**。

---

## 二、ResNet（He et al., 2015）

### 2.1 残差学习

经典 CNN 让堆叠的非线性层 $H(x)$ 拟合一个底层映射 $H(x)$。ResNet 让这些层拟合**残差** $F(x) = H(x) - x$，于是输出为：

$$
H(x) = F(x) + x
$$

网络结构上体现为一条**捷径连接（shortcut connection）**：

```
        ┌──────────────┐
        │              │
   x →  │  Conv → BN → ReLU → Conv → BN  │  → (+) → ReLU → out
        │              │
        └────── + ─────┘
            (identity)
```

如果某一层是冗余的，最优解就是让 $F(x) \to 0$，这时 $H(x) = x$ 即恒等映射。**把「拟合零」比「拟合恒等映射」对优化器友好得多**。

### 2.2 两种残差结构

**Basic Block（ResNet-18/34）**：两层 3×3 卷积。

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
```

**Bottleneck Block（ResNet-50/101/152）**：用 1×1 卷积降维→3×3 卷积→1×1 卷积升维，把计算量降低到原来的约 1/4。

```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)
```

> 当残差块的输入输出维度不一致时（通道数翻倍或 stride=2 减半），需要 `downsample` 模块（1×1 卷积 + BN）把 shortcut 也调整到匹配维度。

### 2.3 ResNet-50 完整实现

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
```

### 2.4 ResNet 为什么能训超深网络？

残差连接让反向传播的梯度有一条**无衰减的直连通道**：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(1 + \frac{\partial F}{\partial x}\right)
$$

无论 $F$ 的梯度多小，恒等项 $+1$ 都能保证信号不会消失。这就是 **「梯度高速公路」**。

### 2.5 ResNet 实验效果

| 模型          | 层数 | Top-5 错误率 | 参数量 |
| ------------- | ---- | ------------ | ------ |
| VGG-19        | 19   | 7.3%         | 144 M  |
| GoogLeNet     | 22   | 6.7%         | 6.8 M  |
| ResNet-50     | 50   | 5.25%        | 25.6 M |
| ResNet-152    | 152  | 4.49%        | 60.2 M |

ResNet-152 相比 VGG-19 错误率几乎减半，参数量反而更少。

---

## 三、DenseNet（Huang et al., 2016）

### 3.1 核心思想

ResNet 是**跨 1 层**的加法捷径，DenseNet 把这个思路推到极致——**每一层都和它前面所有层通过拼接（concat）相连**：

$$
x_l = H_l([x_0, x_1, \dots, x_{l-1}])
$$

其中 $[x_0, \dots, x_{l-1}]$ 表示在通道维上的拼接。**ResNet 是加法，DenseNet 是拼接**。

### 3.2 DenseNet 的优点

- **特征复用**：每一层都能直接访问前面所有层的特征图，最大化信息流。
- **梯度友好**：密集连接让梯度直接流向浅层，进一步缓解梯度消失。
- **参数更高效**：拼接而非相加，意味着后面层不需要重新学一遍前面已经学过的特征。
- **隐式深度监督**：单层可以看做「监督了整个网络的浅层表征」。

### 3.3 DenseNet 的关键设计

**Dense Block**：在 block 内部，每层都接收所有前面层的输出。设每个 Dense Layer 输出 $k$ 个特征图（$k$ 称为 **growth rate**），那么第 $l$ 层输入通道数为 $k_0 + k(l-1)$。

**Bottleneck Layer（BN-ReLU-Conv 1×1 → BN-ReLU-Conv 3×3）**：当 DenseNet-121 中用了 1×1 卷积降维，称为 **DenseNet-B**。

**Transition Layer**：两个 Dense Block 之间用 1×1 卷积 + 2×2 平均池化压缩空间尺寸和通道数。压缩因子 $\theta$（默认 0.5）称为 **compression**。带 transition 的 DenseNet 称为 **DenseNet-C**，同时带 bottleneck 和 compression 的就是 **DenseNet-BC**。

```python
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        # Bottleneck: 1x1 conv to reduce channels
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        # 3x3 conv
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-activation style (BN -> ReLU -> Conv)
        new_features = self.conv1(torch.relu(self.norm1(x)))
        new_features = self.conv2(torch.relu(self.norm2(new_features)))
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], dim=1)  # 关键:拼接


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(torch.relu(self.norm(x))))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, bn_size=4, theta=0.5, dropout=0.0):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Dense Blocks
        in_channels = 64
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = nn.ModuleList()
            for j in range(num_layers):
                block.append(DenseLayer(in_channels, growth_rate, bn_size, dropout))
                in_channels += growth_rate
            self.dense_blocks.append(block)
            if i != len(block_config) - 1:
                out_channels = int(in_channels * theta)
                self.transitions.append(TransitionLayer(in_channels, out_channels))
                in_channels = out_channels
        # Classifier
        self.final_norm = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            for layer in block:
                x = layer(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transitions[i](x)
        x = torch.relu(self.final_norm(x))
        return self.fc(self.avgpool(x).flatten(1))


def densenet121(num_classes=1000):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)
```

### 3.4 DenseNet 的代价

- **显存占用大**：拼接需要保留所有前面层的中间结果，训练时显存占用比 ResNet 高。
- **密集拼接难以并行**：每层都依赖前面所有层，CUDA 难以高效并行。

实践中 DenseNet 的效果往往略优于 ResNet，但训练更慢，工程上 ResNet 系列（ResNet、ResNeXt、RegNet）更受欢迎。

---

## 四、ResNet vs DenseNet 对比

| 维度              | ResNet                | DenseNet                  |
| ----------------- | --------------------- | ------------------------- |
| 连接方式          | 加法（addition）       | 拼接（concatenation）       |
| 信息流            | 单条 shortcut          | 每层连接到所有前面层          |
| 参数量            | 中等                  | 较少                       |
| 显存占用          | 较低                  | 较高                       |
| 训练速度          | 快                    | 较慢                       |
| ImageNet Top-5     | 4.49%（152 层）       | 5.65%（264 层 DenseNet-BC） |
| 现代使用          | 主流 backbone         | 较少                       |

---

## 五、对后续工作的影响

- **ResNet** → 几乎所有后续的视觉 backbone 都用残差连接（ResNeXt、ResNeSt、ConvNeXt、ViT 等）。
- **DenseNet** → 思想被吸收进 **CSPNet**、**VoVNet**、**EfficientDet** 等。

---

## 六、一句话总结

> **ResNet 用「加法捷径」让网络深到 152 层，DenseNet 用「拼接捷径」让每层复用所有历史特征**。两者共同确立了「跨层连接 = 现代 CNN 标配」的设计原则。