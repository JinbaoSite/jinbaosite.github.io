# 线性回归：从最小二乘到梯度下降

线性回归使用输入特征的线性组合预测连续值，是回归分析和机器学习的基础模型。房价、销量和时长预测都可以用它建立基线。模型形式简单，能够清楚展示预测函数、损失函数、参数估计与泛化评估之间的关系。

在神经网络中，一个带偏置、没有激活函数的单输出全连接层就能表示线性回归。最小二乘求解、手动梯度更新和自动求导训练虽然计算过程不同，优化的可以是同一个目标。

## 1. 背景

假设需要根据房屋面积预测价格。面积相近的房屋价格仍会因地段、房龄等因素变化，因此训练目标是从数据中估计总体趋势，而不是让直线穿过每个样本点。

一元线性回归的预测函数为：

$$
\widehat y=wx+b
$$

其中 $w$ 控制斜率，$b$ 是截距。扩展到 $d$ 个特征后：

$$
\widehat y=\sum_{j=1}^{d}w_jx_j+b=\mathbf x^T\mathbf w+b
$$

在其他特征固定的条件下，$w_j$ 表示特征 $x_j$ 增加一个单位时预测值的变化量。这种系数解释依赖特征单位、特征编码及其他变量的选择，不能直接视为因果效应。

“线性”指模型对待估计参数呈线性关系。使用 $x$ 和 $x^2$ 作为两个输入特征，得到 $\widehat y=w_1x+w_2x^2+b$，仍然可以用线性回归估计参数；预测曲线则可以是非线性的。

## 2. 模型原理

### 2.1 矩阵表示

设训练集有 $n$ 个样本和 $d$ 个特征：

$$
X\in\mathbb R^{n\times d},\qquad
\mathbf y\in\mathbb R^n,\qquad
\mathbf w\in\mathbb R^d
$$

所有样本的预测可以统一写为：

$$
\widehat{\mathbf y}=X\mathbf w+b\mathbf 1
$$

将常数列并入设计矩阵：

$$
\widetilde X=[X,\mathbf 1],\qquad
\boldsymbol\theta=[\mathbf w;b]
$$

则 $\widehat{\mathbf y}=\widetilde X\boldsymbol\theta$。这种写法将权重与截距放入同一个最小二乘问题；施加正则化时则通常需要单独排除截距。

### 2.2 平方损失与最小二乘

定义第 $i$ 个样本的预测残差为 $e_i=\widehat y_i-y_i$，平方损失为：

$$
J(\boldsymbol\theta)=\frac{1}{2n}\sum_{i=1}^{n}e_i^2
=\frac{1}{2n}\|\widetilde X\boldsymbol\theta-\mathbf y\|_2^2
$$

平方会放大较大的残差，因此最小二乘对异常值比较敏感。系数 $1/(2n)$ 便于求导；去掉 $1/2$ 不改变最优解，但会使梯度增大一倍。

从概率角度，若假设：

$$
y_i=\mathbf x_i^T\mathbf w+b+\epsilon_i,\qquad
\epsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2)
$$

则负对数似然中与参数相关的项正比于残差平方和。因此在独立、同方差高斯噪声假设下，最小二乘等价于对权重和截距进行最大似然估计。

计算最小二乘解本身并不要求噪声服从高斯分布。分布假设主要影响概率解释、置信区间和统计检验。

### 2.3 正规方程与数值求解

对参数求导：

$$
\nabla_{\boldsymbol\theta}J
=\frac1n\widetilde X^T(\widetilde X\boldsymbol\theta-\mathbf y)
$$

令梯度为零，得到正规方程：

$$
\widetilde X^T\widetilde X\boldsymbol\theta
=\widetilde X^T\mathbf y
$$

若 $\widetilde X$ 列满秩，则最优参数唯一：

$$
\boldsymbol\theta^*=
(\widetilde X^T\widetilde X)^{-1}\widetilde X^T\mathbf y
$$

该表达式用于理解解析解。实际计算通常使用基于 QR 或奇异值分解的最小二乘求解器，避免显式构造逆矩阵。形成 $\widetilde X^T\widetilde X$ 会放大条件数，对高度相关特征尤其不利。

当设计矩阵秩不足时，参数解可能不唯一，但投影到设计矩阵列空间中的训练预测仍然唯一。伪逆可给出最小欧氏范数解：

$$
\boldsymbol\theta^*=\widetilde X^+\mathbf y
$$

### 2.4 一元回归的手算示例

对于数据点 $(1,2)$、$(2,3)$、$(3,5)$，一元最小二乘解为：

$$
w^*=\frac{\sum_i(x_i-\overline x)(y_i-\overline y)}
{\sum_i(x_i-\overline x)^2},\qquad
b^*=\overline y-w^*\overline x
$$

其中 $\overline x=2$，$\overline y=10/3$，因此：

$$
w^*=\frac32,\qquad b^*=\frac13
$$

三个点的预测分别为 $11/6$、$10/3$、$29/6$，对应残差为 $-1/6$、$1/3$、$-1/6$。残差平方和为 $1/6$，均方误差为 $1/18$。

拟合直线无需穿过每一个点。含截距的最小二乘解满足残差和为零，这也是检查一元回归计算结果的一个方法。

### 2.5 梯度下降

数据规模较大或模型需要与其他模块联合训练时，可以迭代更新参数：

$$
\mathbf w_{t+1}=\mathbf w_t-eta\frac1nX^T(X\mathbf w_t+b_t\mathbf1-\mathbf y)
$$

$$
b_{t+1}=b_t-\eta\frac1n\mathbf1^T(X\mathbf w_t+b_t\mathbf1-\mathbf y)
$$

其中 $\eta$ 是学习率。每轮必须使用同一组旧参数计算权重与截距的梯度，再同步更新。

普通最小二乘目标是凸二次函数，局部最小值也是全局最小值。对于使用全部样本的梯度下降，设：

$$
H=\frac1n\widetilde X^T\widetilde X
$$

在最大特征值大于零时，取 $0<\eta<2/\lambda_{\max}(H)$ 可以保证各正曲率方向收敛。特征尺度差异大或共线性强时，不同方向的曲率差异会使收敛变慢。

全批量梯度每步使用所有样本；小批量梯度每步只用部分样本估计梯度。小批量训练降低单步计算量，但引入梯度噪声，通常需要配合学习率调度。

### 2.6 标准化与数据泄漏

特征标准化为：

$$
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}
$$

$\mu_j$ 和 $s_j$ 必须仅由训练集计算，再应用于验证集、测试集和线上输入。先对全部数据标准化再划分数据集，会把未来或测试信息带入训练过程。

常数特征的标准差为零，可以移除，也可以把缩放因子设为 1。对于未正则化、含截距的最小二乘，非退化的标准化只是重参数化，不改变可表示的预测函数；对于带正则化的模型，特征尺度会直接影响各系数受到的惩罚。

### 2.7 正则化

**（1）岭回归**

在平方损失上增加权重的平方范数：

$$
J_{\mathrm{ridge}}=\frac{1}{2n}\|X\mathbf w+b\mathbf1-\mathbf y\|_2^2
+\frac\lambda2\|\mathbf w\|_2^2
$$

权重梯度变为：

$$
\nabla_{\mathbf w}J_{\mathrm{ridge}}
=\frac1nX^T(X\mathbf w+b\mathbf1-\mathbf y)+\lambda\mathbf w
$$

对已经中心化的 $X$ 和 $\mathbf y$，权重解为：

$$
\mathbf w^*=(X^TX+n\lambda I)^{-1}X^T\mathbf y
$$

$n\lambda$ 来自目标函数中的归一化约定。不同工具可能使用残差平方和而非均值，因此同名正则系数不能直接跨实现照搬。

**（2）稀疏回归**

把惩罚换为 $\lambda\|\mathbf w\|_1$，得到稀疏回归。绝对值惩罚可以使部分系数精确为零，常用于特征选择，但需要近端梯度或坐标下降等适合不可微点的优化方法。截距通常不参与这两种惩罚。

### 2.8 评价指标

| 指标 | 定义或含义 | 使用要点 |
| --- | --- | --- |
| 均方误差 | 残差平方的均值 | 对大误差更敏感，单位是目标单位的平方 |
| 均方根误差 | 均方误差的平方根 | 与目标同量纲，便于解释 |
| 平均绝对误差 | 残差绝对值的均值 | 相对平方误差，对极端残差更不敏感 |
| 决定系数 | 相对当前评估集均值基线的误差改善 | 测试集上可以为负，目标恒定时需单独处理 |

决定系数为：

$$
R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}{\sum_i(y_i-\overline y)^2}
$$

其中 $\overline y$ 是当前评估集的目标均值。目标值全部相同时，分母为零，常规表达式无定义；实现应明确约定处理方式。

## 3. 模型实现

### 3.1 数据与最小二乘求解

实现使用 PyTorch，在同一份合成数据上比较最小二乘、手动梯度下降和自动求导。随机种子固定，训练集与测试集分别包含 400 和 100 个样本，标准化参数仅来自训练集。

```python
import torch
from torch import nn


torch.manual_seed(42)
dtype = torch.float64

# 两个不同尺度的特征，真实参数为 [3, -2]，截距为 5。
X = torch.randn(500, 2, dtype=dtype)
X = X * torch.tensor([10.0, 0.5], dtype=dtype)
true_w = torch.tensor([3.0, -2.0], dtype=dtype)
y = X @ true_w + 5.0 + 0.2 * torch.randn(500, dtype=dtype)

order = torch.randperm(len(X))
train_index, test_index = order[:400], order[400:]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

mean = X_train.mean(dim=0)
std = X_train.std(dim=0, correction=0)
scale = torch.where(std > 0, std, torch.ones_like(std))
Z_train = (X_train - mean) / scale
Z_test = (X_test - mean) / scale

ones = torch.ones(len(Z_train), 1, dtype=dtype)
design = torch.cat([Z_train, ones], dim=1)
theta = torch.linalg.lstsq(design, y_train).solution
prediction_lstsq = Z_test @ theta[:-1] + theta[-1]
```

标准化空间中的权重 $\mathbf w_z$ 和截距 $b_z$ 可以还原为原始特征空间的参数：

$$
w_j=\frac{w_{z,j}}{s_j},\qquad
b=b_z-\sum_j\frac{w_{z,j}\mu_j}{s_j}
$$

### 3.2 手动梯度下降

手动实现与 $1/(2n)$ 归一化的平方损失一致：

```python
w = torch.zeros(Z_train.shape[1], dtype=dtype)
b = torch.zeros((), dtype=dtype)
learning_rate = 0.1

for step in range(1000):
    error = Z_train @ w + b - y_train
    grad_w = Z_train.T @ error / len(Z_train)
    grad_b = error.mean()
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

prediction_manual = Z_test @ w + b
```

### 3.3 自动求导训练

`nn.Linear(2, 1)` 保存两个权重和一个截距。`MSELoss` 默认计算均方误差，因此乘以 $1/2$ 后，与手动梯度的损失缩放保持一致。

```python
model = nn.Linear(2, 1).to(dtype=dtype)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for step in range(1000):
    optimizer.zero_grad()
    prediction = model(Z_train).squeeze(-1)
    loss = 0.5 * criterion(prediction, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    prediction_autograd = model(Z_test).squeeze(-1)
```

预测和目标都保持 `[样本数]` 的形状，避免 `[样本数,1]` 与 `[样本数]` 广播成二维误差矩阵。每轮更新前清空梯度，防止不同轮次的梯度意外累积。

### 3.4 结果评估

```python
def regression_metrics(target, prediction):
    error = prediction - target
    mse = error.square().mean().item()
    mae = error.abs().mean().item()
    total = (target - target.mean()).square().sum().item()
    residual = error.square().sum().item()
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return {"mse": mse, "rmse": mse ** 0.5, "mae": mae, "r2": r2}


predictions = {
    "least_squares": prediction_lstsq,
    "manual_gradient": prediction_manual,
    "autograd": prediction_autograd,
}
for name, prediction in predictions.items():
    metrics = regression_metrics(y_test, prediction)
    print(name, {key: round(value, 6) for key, value in metrics.items()})

original_w = theta[:-1] / scale
original_b = theta[-1] - (original_w * mean).sum()
print("weights:", original_w.tolist())
print("bias:", original_b.item())
```

三个求解方式的预测应在数值误差范围内接近，原始尺度权重接近 $[3,-2]$，截距接近 5。由于数据包含随机噪声，估计参数无需与生成参数完全一致。训练收敛与测试误差都需要检查：参数更新完成并不意味着模型具有良好的泛化能力。

### 3.5 实践要点

1. **先划分数据，再拟合预处理**：有时间依赖的数据按时间划分，同一实体的相关样本可按实体分组，减少跨集合泄漏。
2. **检查残差结构**：残差随预测值呈曲线变化，可能缺少非线性特征；残差方差随预测值扩大，可能存在异方差。异常点需要结合数据来源核查。
3. **控制共线性与模型容量**：高度相关特征会导致系数不稳定。岭回归、特征筛选或降维可缓解问题，高次多项式也需要验证集约束。
4. **谨慎处理外推**：训练范围内的线性拟合不能保证范围外继续成立，房价、需求和成本等任务都可能存在边界或饱和效应。
5. **区分预测与统计推断**：常规系数标准误和显著性检验依赖额外假设。拟合分数高，也不能消除遗漏变量和混杂因素带来的因果解释问题。

## 4. 总结

线性回归通过特征加权求和预测连续目标，以平方损失衡量误差。最小二乘将训练转化为线性代数求解，梯度下降通过反复计算残差与梯度逼近最优解，自动求导则把同一过程接入神经网络训练框架。

模型效果取决于特征表达、数据划分和噪声结构。标准化改善优化条件，正则化控制系数规模，残差分析和测试集指标检验模型适用范围。理解这些环节，有助于进一步学习逻辑回归、多层感知机及更复杂的优化方法。

## 5. 参考文献

[1] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd Edition. Springer, 2009.

[2] Bishop C M. Pattern Recognition and Machine Learning. Springer, 2006.

[3] Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press, 2016.
