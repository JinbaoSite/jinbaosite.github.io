# LeNet-5 与 AlexNet：CNN 的奠基之作

LeNet-5（1998）和 AlexNet（2012）是卷积神经网络发展史上两座最重要的里程碑。LeNet-5 首次确立了「卷积-池化-全连接」的现代 CNN 范式，让手写数字识别达到工业可用水平；AlexNet 则在 ImageNet 竞赛中以压倒性优势夺冠，把深度学习推上了 AI 的中心舞台。本文从两篇论文出发，剖析它们的核心结构、设计动机以及对后续网络的影响。

---

## 一、LeNet-5（LeCun et al., 1998）

### 1.1 历史背景

1998 年 Yann LeCun 等人在论文 *Gradient-Based Learning Applied to Document Recognition* 中提出 LeNet-5，用于美国邮政系统的手写数字识别。这是**第一个被大规模商用的卷积神经网络**，其结构定义了一个沿用至今的 CNN 模板：

```
INPUT → Conv → Pool → Conv → Pool → FC → FC → OUTPUT
```

### 1.2 网络结构详解

LeNet-5 接收 32×32 的灰度图像，输出 10 个类别概率，整体由 7 层组成（C1、C3、C5 为卷积层，S2、S4 为池化层，F6 为全连接层）：

| 层  | 类型   | 输入           | 输出           | 参数量        |
| --- | ------ | -------------- | -------------- | ------------- |
| C1  | Conv   | 1×32×32        | 6×28×28        | 6×(5×5)+6 = 156 |
| S2  | Pool   | 6×28×28        | 6×14×14        | 0 (无参数)    |
| C3  | Conv   | 6×14×14        | 16×10×10       | 16×(5×5)×6+16 = 2416 |
| S4  | Pool   | 16×10×10       | 16×5×5         | 0             |
| C5  | Conv   | 16×5×5         | 120×1×1        | 120×(5×5)×16+120 = 48120 |
| F6  | FC     | 120            | 84             | 84×120+84 = 10164 |
| OUT | FC     | 84             | 10             | 10×84+10 = 850 |

总参数量约 **6 万**，在当时是非常紧凑的设计。

### 1.3 关键设计思想

- **局部感受野**：每个神经元只连接到输入的一小块区域，模拟视觉皮层细胞。
- **权值共享**：同一卷积核在整个输入上滑动，大幅减少参数。
- **空间下采样（S2/S4）**：逐步降低空间分辨率，增大感受野。
- **层级特征提取**：浅层学到边缘、角点；深层学到整体形状。

### 1.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # C1: 1@32x32 -> 6@28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: 6@28x28 -> 6@14x14
        # C3: 6@14x14 -> 16@10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: 16@10x10 -> 16@5x5
        # C5: 16@5x5 -> 120
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # F6: 120 -> 84
        self.fc1 = nn.Linear(120, 84)
        # OUT: 84 -> num_classes
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 32)
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)   # -> (B, 6, 14, 14)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)   # -> (B, 16, 5, 5)
        x = torch.tanh(self.conv3(x))                    # -> (B, 120, 1, 1)
        x = x.flatten(1)                                  # -> (B, 120)
        x = torch.tanh(self.fc1(x))                      # -> (B, 84)
        return self.fc2(x)                                # -> (B, num_classes)


if __name__ == "__main__":
    model = LeNet5()
    x = torch.randn(8, 1, 32, 32)
    print(model(x).shape)  # torch.Size([8, 10])
```

注意原论文用的是 `tanh` 激活和**平均池化**，而不是现代常见的 ReLU 和最大池化。

### 1.5 局限

- 激活函数 `tanh` 容易饱和，梯度消失问题严重。
- 平均池化让信息损失较多。
- 没有用 GPU 训练，只能在小规模数据集上跑。
- 这些缺陷都在 AlexNet 中被针对性解决。

---

## 二、AlexNet（Krizhevsky et al., 2012）

### 2.1 历史背景

2012 年，Hinton 组的学生 Alex Krizhevsky 凭借 AlexNet（*ImageNet Classification with Deep Convolutional Neural Networks*）以 **top-5 错误率 15.3%** 拿下 ImageNet ILSVRC 比赛冠军，第二名的传统方法只有 26.2%。这一差距直接引爆了深度学习浪潮。

AlexNet 的成功可以归结为「**3 个 ReLU + 2 个 Dropout + 1 个 GPU + 1 个大数据库**」：

- ReLU 激活
- Dropout 正则化
- 双 GPU 训练
- ImageNet（120 万张图，1000 类）

### 2.2 网络结构详解

AlexNet 输入是 227×227×3 的彩色图像（论文中写 224 但实际是 227），5 个卷积层 + 3 个全连接层，最后接 1000-way softmax：

| 层 | 类型 | 输出尺寸 | 核/步长 |
| --- | --- | --- | --- |
| Conv1 | Conv + ReLU + Pool + LRN | 27×27×96 | 11×11, stride 4 |
| Conv2 | Conv + ReLU + Pool + LRN | 13×13×256 | 5×5, pad 2 |
| Conv3 | Conv + ReLU | 13×13×384 | 3×3, pad 1 |
| Conv4 | Conv + ReLU | 13×13×384 | 3×3, pad 1 |
| Conv5 | Conv + ReLU + Pool | 6×6×256 | 3×3, pad 1 |
| FC6 | FC + ReLU + Dropout | 4096 | - |
| FC7 | FC + ReLU + Dropout | 4096 | - |
| FC8 | FC + Softmax | 1000 | - |

总参数量约 **6000 万**，是 LeNet-5 的一千倍。

### 2.3 关键技术创新

#### （1）ReLU 激活

解决了 sigmoid/tanh 在深层网络中的饱和问题，**让 6 层的网络在没有无监督预训练的情况下也能收敛**。训练速度比 tanh 快约 6 倍。

#### （2）Dropout

在全连接层以 0.5 的概率随机丢弃神经元，相当于在训练时集成指数级数量的子网络，是 AlexNet 防止过拟合的关键武器。

#### （3）Local Response Normalization（LRN）

对每个位置、每个通道在邻近通道间做归一化，模拟生物神经元的「侧抑制」现象。后来被证明收益不大，被 BatchNorm 取代。

#### （4）Overlapping Max Pooling

步长 2、核大小 3 的最大池化（输出有重叠），错误率下降约 0.4%。

#### （5）数据增强 + Dropout

- 训练时随机裁剪 224×224 区域，并做水平翻转；
- 用 PCA 在 RGB 空间做颜色扰动，减少对颜色和光照的过拟合。

#### （6）双 GPU 训练

把模型按通道切分到两块 GTX 580 3GB 上，单卡显存装不下整个模型。这也是深度学习第一次大规模使用 GPU。

### 2.4 PyTorch 实现

```python
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = AlexNet()
    x = torch.randn(2, 3, 227, 227)
    print(model(x).shape)  # torch.Size([2, 1000])
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
```

### 2.5 AlexNet 的历史意义

- 第一次让 **GPU 成为深度学习的标配**。
- 第一次用 **ImageNet 这种百万级数据集** 训练 CNN。
- 第一次用 **ReLU + Dropout** 这对组合在大模型上稳定收敛。
- 直接催生了 2013 年的 ZFNet、2014 年的 VGG 和 GoogLeNet。

---

## 三、LeNet-5 → AlexNet 的演进逻辑

| 维度         | LeNet-5（1998）        | AlexNet（2012）             |
| ------------ | ---------------------- | --------------------------- |
| 数据规模     | MNIST（6 万）          | ImageNet（120 万）          |
| 输入尺寸     | 32×32 灰度             | 227×227 彩色               |
| 网络深度     | 5 层                   | 8 层                        |
| 参数量       | ~6 万                  | ~6000 万                    |
| 激活函数     | tanh                   | ReLU                        |
| 正则化       | 无                     | Dropout                     |
| 池化         | 平均池化               | 最大池化（overlapping）     |
| 硬件         | CPU                    | 双 GPU                      |
| 归一化       | 无                     | LRN                         |
| 训练时长     | 数天                   | 5–6 天（双 GTX 580）        |

可以清楚地看到，**AlexNet 不是一项孤立发明，而是一组针对性技术的合集**：针对「梯度消失」上 ReLU，针对「过拟合」上 Dropout，针对「算力瓶颈」上 GPU，针对「感受野不足」上大卷积核。

---

## 四、一句话总结

> **LeNet-5 给出模板，AlexNet 把它放大了一千倍并证明了深度 + 数据 + 算力 = 奇迹**。后续所有经典网络（VGG、GoogLeNet、ResNet）都是在 AlexNet 开辟的道路上继续向前。