# XGBoost：二阶优化与正则化树提升

XGBoost 是面向树提升的机器学习系统，将损失的二阶近似、叶权重正则化、高效分裂搜索和稀疏数据处理结合起来。它在加法模型中逐轮添加回归树，通过梯度和海森统计量评价树结构与叶节点输出。

[梯度提升树](/dl/gradient_boosting/)提供逐轮优化损失的框架，XGBoost 则进一步明确了如何用局部二阶目标计算叶值、分裂增益与复杂度惩罚。分类时，树输出仍是连续得分增量，最终通过对应的概率变换生成预测。

## 1. 背景

单纯拟合负梯度只利用当前下降方向，没有显式考虑不同样本位置的曲率。对同样大小的梯度，曲率较大的方向通常应采用较小更新；树结构越复杂，也需要更强的约束以控制过拟合。

XGBoost 的树模型把训练损失与树复杂度放在同一个目标中，通过局部二阶近似将每个节点压缩为梯度和、海森和，从而快速比较候选划分。

模型可以写为：

$$
F_M(\mathbf x)=F_0(\mathbf x)+\eta\sum_{m=1}^{M}f_m(\mathbf x)
$$

其中 $\eta$ 为学习率，$f_m$ 是新增的回归树。树之间存在轮次依赖，同一轮内部的特征统计、候选搜索等操作可以并行。

## 2. 模型原理

### 2.1 带正则化的目标

先推导未乘学习率的单轮候选树。设已有预测为 $F_{m-1}$，新增树为 $f_m$：

$$
\mathcal L_m=\sum_i\ell(y_i,F_{m-1}(\mathbf x_i)+f_m(\mathbf x_i))+\Omega(f_m)
$$

树把输入映射到叶节点：

$$
f_m(\mathbf x)=w_{q(\mathbf x)}
$$

$q$ 表示树结构，$w_j$ 是叶节点 $j$ 的输出。常用复杂度项为：

$$
\Omega(f)=\gamma T+\frac\lambda2\sum_{j=1}^{T}w_j^2
$$

$T$ 是叶节点数，$\gamma$ 惩罚新增叶子，$\lambda$ 收缩叶权重。实际算法确定树后，通常再将其输出乘以学习率；相应的分裂增益是未收缩局部目标的近似改善，不等于最终全局训练损失的精确下降量。

### 2.2 二阶泰勒展开

在当前预测处定义：

$$
g_i=\left.\frac{\partial\ell(y_i,F)}{\partial F}\right|_{F=F_{m-1}(\mathbf x_i)},\qquad
h_i=\left.\frac{\partial^2\ell(y_i,F)}{\partial F^2}\right|_{F=F_{m-1}(\mathbf x_i)}
$$

去掉与新增树无关的常数项，得到：

$$
\widetilde{\mathcal L}_m=
\sum_i\left[g_if_m(\mathbf x_i)+\frac12h_if_m(\mathbf x_i)^2\right]+\Omega(f_m)
$$

该式是局部二阶近似。平方误差下展开是精确的，其他损失下则需要结合学习率和容量限制保证更新合理。

### 2.3 最优叶权重

设叶节点 $j$ 包含样本集合 $I_j$，定义：

$$
G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i
$$

按叶节点整理目标：

$$
\widetilde{\mathcal L}_m=
\sum_{j=1}^{T}\left[G_jw_j+\frac12(H_j+\lambda)w_j^2\right]+\gamma T
$$

若 $H_j+\lambda>0$，对 $w_j$ 求导得到：

$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

代回目标：

$$
\widetilde{\mathcal L}_m^*=
-\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T
$$

固定结构下无需逐样本搜索叶值，只需累计 $G_j$ 和 $H_j$。分类概率接近 0 或 1 时，海森值可能很小，正则化及最小子节点海森和可以抑制不稳定更新。

### 2.4 分裂增益

将父节点划分为左右子节点，分裂增益为：

$$
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma
$$

二元分裂使叶节点数增加一个，因此只减去一次 $\gamma$。候选划分还需要满足深度、子节点海森和等条件，不能只看公式数值是否为正。

例如，平方损失采用 $\ell=(y-F)^2/2$，目标为 $1,1,5,5$，当前预测均为 3，则：

$$
\mathbf g=(2,2,-2,-2),\qquad\mathbf h=(1,1,1,1)
$$

在第二和第三个样本间划分，取 $\lambda=1$、$\gamma=0$，有：

$$
G_L=4,\quad H_L=2,\quad G_R=-4,\quad H_R=2
$$

$$
w_L=-\frac43,\qquad w_R=\frac43,\qquad
\operatorname{Gain}=\frac{16}{3}
$$

若学习率为 $0.5$，更新后预测为 $7/3,7/3,11/3,11/3$。正则项使叶值幅度小于未正则残差均值，学习率再进一步缩小增量。

### 2.5 常用损失的导数

| 任务 | 当前输出含义 | 一阶导数 | 二阶导数 |
| --- | --- | --- | --- |
| 平方误差回归 | 连续预测值 $F$ | $F-y$ | $1$ |
| 二分类交叉熵 | 对数几率 $F$，$p=\sigma(F)$ | $p-y$ | $p(1-p)$ |

回归初始常数可取标签均值，二分类可取正类比例的对数几率。二分类先累加树得分，再通过逻辑函数转成概率，不直接累加各树概率。

自定义目标需要与预测变换一致。若直接套用要求正曲率的公式处理非凸损失，负海森值可能破坏叶目标的凸性；生产接口的具体约束与近似处理必须单独核对。

### 2.6 绝对值正则与容量参数

加入绝对值惩罚 $\alpha\sum_j|w_j|$ 后，叶权重变为软阈值形式：

$$
w_j^*=-\frac{\operatorname{sign}(G_j)\max(|G_j|-\alpha,0)}{H_j+\lambda}
$$

不同参数约束不同对象：

| 参数 | 控制内容 | 解释 |
| --- | --- | --- |
| `reg_lambda` | 叶权重平方惩罚 | 收缩叶输出 |
| `reg_alpha` | 叶权重绝对值惩罚 | 可令部分叶输出为零 |
| `gamma` | 分裂所需的损失改善 | 增加新叶子的代价 |
| `min_child_weight` | 子节点海森和下限 | 一般不等于最小样本数 |
| `max_depth` | 单树最大深度 | 限制树结构复杂度 |
| `learning_rate` | 新树贡献缩放 | 需要与轮数共同选择 |

平方损失下每个样本海森为 1，海森和才与样本数一致。交叉熵下，概率很极端的样本即使很多，也可能只提供较小的海森和。

### 2.7 精确搜索、近似搜索与直方图

精确搜索对特征排序，沿阈值累计左右两侧的梯度与海森统计。排序确定后，候选分裂不必反复遍历全部样本计算目标。

近似搜索先确定有限候选切分点，直方图方法把特征映射到有限桶，再累计各桶的梯度和海森。减少候选数量可以降低计算与存储开销，但与精确阈值搜索的结果可能不同。

历史上的加权分位数草图与不同版本的近似、直方图实现存在具体差异，不能把所有树方法都描述为同一套海森加权分桶。比较实验应固定树方法、分桶数量和版本。

### 2.8 缺失值与稀疏感知

稀疏感知分裂搜索可以只遍历非缺失项，并比较缺失样本默认进入左侧或右侧时的收益，将更合适的方向保存在节点中。预测时按该默认方向处理缺失特征。

缺失项不等于普通数值零。稀疏矩阵中的未存储项与稠密矩阵中的零值可能具有不同含义；把稀疏矩阵直接转为填零的稠密矩阵，可能改变模型行为。

原生缺失值支持也不能修复标签缺失、训练服务特征错位或未来信息泄漏。数据语义和存储表示需要与模型配置一同检查。

## 3. 模型实现

### 3.1 二阶树桩

标准库实现保留二阶叶权重、分裂增益、海森下限和学习率，用深度为一的树桩演示训练。输入只接受有限数值，不实现直方图、缺失默认方向和并行搜索；这些工程能力与局部目标推导是不同层面的内容。

```python
import math


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class NewtonStump:
    def __init__(self, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def fit(self, X, gradients, hessians):
        total_g, total_h = sum(gradients), sum(hessians)
        self.feature = None
        self.left_value = self.right_value = -total_g / (total_h + self.reg_lambda)
        self.gain = 0.0
        parent_score = total_g ** 2 / (total_h + self.reg_lambda)
        for feature in range(len(X[0])):
            values = sorted(set(row[feature] for row in X))
            for lower, upper in zip(values, values[1:]):
                threshold = lower / 2.0 + upper / 2.0
                if not lower <= threshold < upper:
                    threshold = lower
                left = [i for i, row in enumerate(X) if row[feature] <= threshold]
                right = [i for i, row in enumerate(X) if row[feature] > threshold]
                gl, hl = sum(gradients[i] for i in left), sum(hessians[i] for i in left)
                gr, hr = sum(gradients[i] for i in right), sum(hessians[i] for i in right)
                if min(hl, hr) < self.min_child_weight:
                    continue
                gain = 0.5 * (gl ** 2 / (hl + self.reg_lambda) +
                              gr ** 2 / (hr + self.reg_lambda) - parent_score) - self.gamma
                if gain > self.gain + 1e-12:
                    self.feature, self.threshold, self.gain = feature, threshold, gain
                    self.left_value = -gl / (hl + self.reg_lambda)
                    self.right_value = -gr / (hr + self.reg_lambda)
        return self

    def predict(self, X):
        if self.feature is None:
            return [self.left_value] * len(X)
        return [self.left_value if row[self.feature] <= self.threshold
                else self.right_value for row in X]
```

没有正收益分裂时仍可生成常数树。`gamma` 在分裂比较中惩罚新增叶子，示例不另外实现“整棵树是否加入”的搜索。平方惩罚被限制为正值，以保证示例中的叶值分母安全。

### 3.2 二阶提升训练

```python
class NewtonBoosting:
    def __init__(self, task="regression", n_estimators=30, learning_rate=0.1,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        if task not in {"regression", "classification"}:
            raise ValueError("invalid task")
        if n_estimators < 1 or not 0 < learning_rate <= 1:
            raise ValueError("invalid boosting parameters")
        if not math.isfinite(reg_lambda) or reg_lambda <= 0:
            raise ValueError("reg_lambda must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (gamma, min_child_weight)):
            raise ValueError("invalid tree regularization")
        self.task, self.n_estimators = task, n_estimators
        self.learning_rate, self.reg_lambda = learning_rate, reg_lambda
        self.gamma, self.min_child_weight = gamma, min_child_weight
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        X, y = [list(row) for row in X], list(y)
        if not X or len(X) != len(y) or not X[0]:
            raise ValueError("invalid training shape")
        self.n_features = len(X[0])
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row)
               for row in X) or any(not math.isfinite(v) for v in y):
            raise ValueError("invalid training values")
        mean = sum(y) / len(y)
        if self.task == "classification":
            if any(v not in (0, 1) for v in y) or not 0 < mean < 1:
                raise ValueError("both binary classes are required")
            self.initial_score = math.log(mean / (1.0 - mean))
        else:
            self.initial_score = mean
        scores = [self.initial_score] * len(y)
        self.trees = []
        for _ in range(self.n_estimators):
            if self.task == "regression":
                gradients = [score - label for score, label in zip(scores, y)]
                hessians = [1.0] * len(y)
            else:
                probabilities = [sigmoid(score) for score in scores]
                gradients = [p - label for p, label in zip(probabilities, y)]
                hessians = [p * (1.0 - p) for p in probabilities]
            tree = NewtonStump(self.reg_lambda, self.gamma, self.min_child_weight)
            tree.fit(X, gradients, hessians)
            scores = [score + self.learning_rate * update
                      for score, update in zip(scores, tree.predict(X))]
            self.trees.append(tree)
        self.fitted = True
        return self

    def decision_function(self, X):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        X = [list(row) for row in X]
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row)
               for row in X):
            raise ValueError("invalid prediction features")
        scores = [self.initial_score] * len(X)
        for tree in self.trees:
            scores = [score + self.learning_rate * update
                      for score, update in zip(scores, tree.predict(X))]
        return scores

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("probabilities require classification")
        return [sigmoid(score) for score in self.decision_function(X)]

    def predict(self, X):
        scores = self.decision_function(X)
        return scores if self.task == "regression" else [int(score >= 0) for score in scores]
```

该模型用于复现核心公式，不等同于完整 XGBoost 软件包。树桩限制了交互能力，固定轮数也不包含验证集早停，实际任务应使用经过优化和验证的训练系统。

### 3.3 手算核对与分类示例

```python
X = [[1.0], [2.0], [3.0], [4.0]]
regressor = NewtonBoosting(n_estimators=1, learning_rate=0.5)
regressor.fit(X, [1.0, 1.0, 5.0, 5.0])
print("gain:", round(regressor.trees[0].gain, 6))
print("prediction:", [round(v, 6) for v in regressor.predict(X)])

classifier = NewtonBoosting(task="classification", n_estimators=30,
                             learning_rate=0.3, min_child_weight=0.01)
classifier.fit(X, [0, 0, 1, 1])
print("classification:", classifier.predict(X))
```

输出为：

```text
gain: 5.333333
prediction: [2.333333, 2.333333, 3.666667, 3.666667]
classification: [0, 0, 1, 1]
```

手算数据只验证增益、叶值和更新方向，不作为泛化评价。示例的直接阈值枚举每轮最坏约为 $O(dn^2)$，并不具有完整系统的排序统计或直方图效率。

### 3.4 训练与评估配置

实际训练需要明确目标函数、树方法、评价指标、随机种子和资源配置。回归关注均方根误差或平均绝对误差，二分类关注概率损失、排序质量和阈值下的分类指标。

早停使用独立验证数据，保留最佳轮次对应的模型。类别不平衡时，正类权重可以改变优化重点，但也会改变概率尺度；阈值与校准需要单独验证。

保存模型时，还需保存特征顺序、类别编码和数据处理版本。推理输入的缺失标记、稀疏表示与训练一致，才能复现节点默认方向和最终得分。

## 4. 总结

XGBoost 通过损失二阶近似，将每个叶节点表示为梯度和与海森和，再结合叶权重及叶数量惩罚，推导出叶值和分裂增益。学习率、深度、最小子节点海森和与采样策略共同控制模型复杂度。

二阶目标解释了模型如何学习，高效分裂搜索、稀疏感知和并行执行则使它适用于更大规模数据。评估时需要同时核对目标缩放、训练配置和部署数据表示，避免把局部近似收益直接当作泛化改善。

## 5. 参考文献

[1] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016: 785–794.

[2] Friedman J H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001, 29(5): 1189–1232.
