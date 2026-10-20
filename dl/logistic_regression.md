# 逻辑回归：从概率建模到交叉熵优化

逻辑回归根据输入特征估计类别概率，是二分类任务中的基础模型。点击率预估、垃圾邮件识别和用户流失预测都可以用它建立基线。模型将线性得分映射为概率，再以伯努利分布的负对数似然训练参数。

[线性回归](/dl/linear_regression/)预测连续目标，逻辑回归则在对数几率空间中建立线性关系。在神经网络中，单输出全连接层配合二元交叉熵，就可以实现二分类逻辑回归。

## 1. 背景

设样本特征为 $\mathbf x\in\mathbb R^d$，标签 $y\in\{0,1\}$。目标是估计：

$$
p(\mathbf x)=P(y=1\mid\mathbf x)
$$

线性函数 $\mathbf x^T\mathbf w+b$ 的取值没有上下界，不能直接作为概率。逻辑回归使用逻辑函数将得分转换到 $(0,1)$：

$$
z=\mathbf x^T\mathbf w+b,\qquad
p=\sigma(z)=\frac{1}{1+\exp(-z)}
$$

其中 $z$ 称为对数几率得分，也常称为 Logit。模型输出概率以后，仍需通过阈值或业务决策规则生成最终类别。概率估计和类别决策属于两个不同环节。

## 2. 模型原理

### 2.1 概率、几率与对数几率

正类概率为 $p$ 时，正类相对负类的几率为：

$$
\operatorname{odds}=\frac{p}{1-p}
$$

对几率取自然对数：

$$
\log\frac{p}{1-p}=z=\mathbf x^T\mathbf w+b
$$

因此，逻辑回归对特征建立的是线性的对数几率关系。保持其他特征固定，$x_j$ 增加一个单位，对数几率增加 $w_j$，几率乘以 $\exp(w_j)$。

例如，$w_j=\log2$ 表示对应特征增加一个单位后几率翻倍，而不是概率翻倍。系数解释依赖特征单位、编码方式和变量选择，也不能直接当作因果效应。

逻辑函数满足：

$$
\sigma(0)=\frac12,\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

得分很大或很小时，概率分别接近 1 或 0。偏置控制整体概率水平；只有截距、没有其他特征时，有限最优截距等于训练正类比例的对数几率，前提是两类样本都存在。

### 2.2 决策边界与阈值

给定阈值 $\tau\in(0,1)$，类别决策为：

$$
\widehat y=\mathbb I(p\ge\tau)
$$

由于逻辑函数单调递增，可以直接对得分判断：

$$
\widehat y=\mathbb I\left(z\ge\log\frac{\tau}{1-\tau}\right)
$$

当 $\tau=0.5$ 时，决策边界为：

$$
\mathbf x^T\mathbf w+b=0
$$

它在输入特征空间中是超平面。若输入加入平方项、交叉项等非线性特征，模型仍对参数线性，但映射回原始变量空间后，边界可以变成曲线。

在误报成本为 $C_{FP}$、漏报成本为 $C_{FN}$，正确分类成本为零的设定下，若概率已针对部署分布校准，最小化单次期望成本的阈值为：

$$
\tau^*=\frac{C_{FP}}{C_{FP}+C_{FN}}
$$

漏报成本越高，阈值越低。实际阈值应在验证数据上结合容量、召回率和业务成本选择，不能通过测试集反复调参。

### 2.3 从似然到二元交叉熵

给定特征，标签服从伯努利分布：

$$
P(y_i\mid\mathbf x_i)=p_i^{y_i}(1-p_i)^{1-y_i}
$$

假设样本条件独立，训练集似然为各样本概率的乘积。取负对数并求均值，得到二元交叉熵：

$$
J(\mathbf w,b)=-\frac1n\sum_{i=1}^{n}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

当 $y_i=1$ 时，单样本损失为 $-\log p_i$；当 $y_i=0$ 时，损失为 $-\log(1-p_i)$。给错误类别极高置信度会受到很大惩罚。

以 $z=\log3$、$y=1$ 为例，$p=0.75$，损失为 $-\log0.75\approx0.2877$。若同样的得分对应 $y=0$，损失变为 $-\log0.25\approx1.3863$。

### 2.4 数值稳定的损失计算

把概率代回交叉熵，单样本损失可写为：

$$
\ell(z,y)=\log(1+\exp(z))-yz
$$

直接计算 $\exp(z)$ 在大正数处可能溢出，先算概率再取对数也可能遇到概率舍入为 0 或 1。稳定形式为：

$$
\ell(z,y)=\max(z,0)-yz+\log(1+\exp(-|z|))
$$

PyTorch 的 `BCEWithLogitsLoss` 接收未经过逻辑函数的得分，将概率变换与交叉熵合并计算。训练时直接传入 $z$，推理需要概率时再执行 `torch.sigmoid`。

### 2.5 梯度与优化

对单样本得分求导：

$$
\frac{\partial\ell}{\partial z}=p-y
$$

设 $X\in\mathbb R^{n\times d}$，则：

$$
\nabla_{\mathbf w}J=\frac1nX^T(\mathbf p-\mathbf y),\qquad
\frac{\partial J}{\partial b}=\frac1n\mathbf1^T(\mathbf p-\mathbf y)
$$

梯度下降更新为：

$$
\mathbf w_{t+1}=\mathbf w_t-\eta\nabla_{\mathbf w}J,\qquad
b_{t+1}=b_t-\eta\frac{\partial J}{\partial b}
$$

梯度中的 $p-y$ 表示概率误差。若正样本的预测概率过低，参数会沿提高该样本得分的方向更新。

将偏置并入设计矩阵 $\widetilde X$，交叉熵的海森矩阵为：

$$
H=\frac1n\widetilde X^TD\widetilde X,\qquad
D=\operatorname{diag}(p_i(1-p_i))
$$

由于 $D$ 的对角元素非负，目标关于参数是凸函数。普通逻辑回归没有与最小二乘相同的闭式解，可以使用梯度下降、牛顿法或拟牛顿法求解。

凸性并不保证存在有限的最优参数。完全可分数据上，无正则逻辑回归可以持续增大参数范数，让损失逼近零而不取得有限最小值；准完全分离也可能导致参数发散。

### 2.6 正则化与特征预处理

常见的平方范数正则化目标为：

$$
J_{\lambda}=J+\frac\lambda2\|\mathbf w\|_2^2
$$

对应权重梯度增加 $\lambda\mathbf w$，截距通常不参与惩罚。正则化可以限制系数幅度，缓解共线性和过拟合。若所有样本都属于同一类，仅惩罚权重仍不能阻止未受惩罚的截距发散。

绝对值惩罚 $\lambda\|\mathbf w\|_1$ 可以产生稀疏系数，适合高维特征筛选。不同实现对损失求和或求均值的约定不同，正则系数需要结合目标函数核对。

训练前通常对数值特征标准化，对类别特征进行独热编码。均值、缩放因子和类别字典只能从训练集拟合，验证集与测试集复用同一套变换。正则化下，特征尺度会改变相同系数惩罚对应的实际约束。

### 2.7 类别不平衡与概率校准

正类稀少时，仅预测负类也可能获得很高准确率。可以使用类别加权、重采样或阈值调整，但三者改变的环节不同。

若正类损失乘以 $\alpha>0$，加权交叉熵为：

$$
\ell_{\alpha}=-\alpha y\log q-(1-y)\log(1-q)
$$

在真实正类概率为 $p$、模型表达能力充分的理想条件下，加权目标的逐点最优解为：

$$
q^*=\frac{\alpha p}{1-p+\alpha p}
$$

因此加权训练会改变概率尺度。`pos_weight` 可以增强正类样本对损失的贡献，但输出不能自动解释为部署分布下的真实点击概率。重采样改变类别比例时也需要关注类似问题，必要时使用独立验证集进行校准。

### 2.8 分类评估与多类别扩展

| 指标 | 衡量内容 | 使用要点 |
| --- | --- | --- |
| 准确率 | 分类正确的比例 | 类别不平衡时需要结合其他指标 |
| 精确率 | 预测正类中真实正类的比例 | 关注误报，受阈值影响 |
| 召回率 | 真实正类中被识别的比例 | 关注漏报，受阈值影响 |
| 调和平均分数 | 精确率与召回率的调和平均 | 用于综合观察两者，不衡量校准 |
| 曲线下面积 | 正负样本的相对排序能力 | 需说明使用受试者工作曲线还是精确率召回率曲线 |
| 对数损失 | 概率预测的负对数似然 | 对高置信度错误敏感 |

二分类以外，互斥的多类别任务可用归一化指数函数构造多项逻辑回归：

$$
P(y=k\mid\mathbf x)=\frac{\exp(z_k)}{\sum_{j=1}^{C}\exp(z_j)}
$$

训练时使用多类别交叉熵。一个样本可以同时具有多个标签时，通常使用每个标签独立的二元输出与二元交叉熵，不强制各标签概率之和等于 1。

## 3. 模型实现

### 3.1 数据构造与标准化

实现使用同一份合成二分类数据，比较手动梯度与自动求导。标签根据概率随机采样，因此包含不可完全消除的噪声，不强求训练准确率达到 100%。

```python
import torch
from torch import nn
from torch.nn import functional as F


torch.manual_seed(42)
dtype = torch.float64
X = torch.randn(1200, 2, dtype=dtype)
X = X * torch.tensor([2.0, 0.5], dtype=dtype)
true_w = torch.tensor([1.5, -2.0], dtype=dtype)
true_probability = torch.sigmoid(X @ true_w - 0.4)
y = torch.bernoulli(true_probability)

order = torch.randperm(len(X))
train_index, test_index = order[:900], order[900:]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

mean = X_train.mean(dim=0)
std = X_train.std(dim=0, correction=0)
scale = torch.where(std > 0, std, torch.ones_like(std))
Z_train = (X_train - mean) / scale
Z_test = (X_test - mean) / scale
```

### 3.2 手动梯度下降

优化目标使用均值交叉熵加 $\lambda\|\mathbf w\|_2^2/2$，权重和截距均从零初始化。

```python
regularization = 0.01
learning_rate = 0.2
steps = 2000
w = torch.zeros(Z_train.shape[1], dtype=dtype)
b = torch.zeros((), dtype=dtype)

for step in range(steps):
    logits = Z_train @ w + b
    probability = torch.sigmoid(logits)
    error = probability - y_train
    grad_w = Z_train.T @ error / len(Z_train) + regularization * w
    grad_b = error.mean()
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

manual_logits = Z_test @ w + b
```

### 3.3 自动求导训练

自动求导版本使用相同初始化、学习率、步数和正则化约定，便于直接验证两种实现的一致性。

```python
model = nn.Linear(2, 1).to(dtype=dtype)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

for step in range(steps):
    optimizer.zero_grad()
    logits = model(Z_train).squeeze(-1)
    penalty = 0.5 * regularization * model.weight.square().sum()
    loss = criterion(logits, y_train) + penalty
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    autograd_logits = model(Z_test).squeeze(-1)
    test_probability = torch.sigmoid(autograd_logits)
```

标签采用浮点型的 0 和 1，预测与标签均为 `[样本数]`。模型输出原始得分，损失函数负责稳定计算交叉熵。显式正则项只作用于权重，偏置未受惩罚。

### 3.4 评价与结果核对

```python
def classification_metrics(target, logits, threshold=0.5):
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    probability = torch.sigmoid(logits)
    prediction = probability >= threshold
    positive = target.bool()
    tp = (prediction & positive).sum().item()
    fp = (prediction & ~positive).sum().item()
    fn = (~prediction & positive).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": prediction.eq(positive).double().mean().item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": F.binary_cross_entropy_with_logits(logits, target).item(),
    }


for name, logits in (("manual", manual_logits), ("autograd", autograd_logits)):
    metrics = classification_metrics(y_test, logits)
    print(name, {key: round(value, 6) for key, value in metrics.items()})

print("max_logit_difference:", (manual_logits - autograd_logits).abs().max().item())
```

两种方法的预测得分应在浮点数误差范围内接近。指标函数对没有预测正类或没有真实正类的情况约定对应精确率或召回率为零；报告实际数据结果时，还需同时报告各类别样本数。

测试集只用于最终评价。正则系数、训练步数和分类阈值应通过训练集与验证集确定；合成示例使用固定设置，未执行超参数搜索。

### 3.5 实践要点

1. **检查标签与样本定义**：正负标签必须对应明确业务事件，未曝光样本不能直接视为“曝光后未点击”的负样本。
2. **保持预处理一致**：训练和服务使用相同的特征顺序、缺失值处理、标准化参数和类别编码。
3. **区分得分、概率与类别**：训练接收原始得分，概率由逻辑函数获得，类别由阈值决定。阈值变化不会修改已训练参数。
4. **监控参数范数**：损失持续下降而参数持续增大时，需要检查分离现象、正则化和标签泄漏。
5. **同时观察排序与校准**：排序表现好不意味着概率准确，类别加权、负采样和分布漂移都可能改变校准关系。

## 4. 总结

逻辑回归以线性函数表示对数几率，以逻辑函数得到二分类概率，再通过伯努利负对数似然学习参数。交叉熵的得分梯度为 $p-y$，可以直接使用梯度下降，也可以交给自动求导完成训练。

可靠实现需要统一损失缩放、正则化和预处理规则，并使用数值稳定的得分形式计算损失。评价阶段还要结合类别比例、阈值成本和概率校准，避免仅凭准确率判断模型质量。

## 5. 参考文献

[1] Bishop C M. Pattern Recognition and Machine Learning. Springer, 2006.

[2] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd Edition. Springer, 2009.

[3] Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press, 2016.
