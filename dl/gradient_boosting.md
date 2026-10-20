# 梯度提升树：从负梯度拟合到加法模型

梯度提升树以回归树为基学习器，逐轮添加新的树来降低整体预测损失。每一轮先计算当前模型在训练样本上的负梯度，再用树拟合这个更新方向，通过加法组合形成更强的模型。

这种框架可以用于回归、分类和具有可优化损失的其他任务。平方误差下的负梯度就是残差；二分类交叉熵下则是标签与当前正类概率之差。理解两者的区别，是从“拟合残差”走向一般梯度提升的关键。

## 1. 背景

单棵[决策树](/dl/decision_tree/)把空间划分为多个区域，在各区域内给出常数预测。浅树表达能力有限，深树又容易过拟合。梯度提升采用多个相对简单的树，逐步修正已有模型的预测。

设第 $m$ 棵树为 $f_m$，学习率为 $\eta$，模型为：

$$
F_M(\mathbf x)=F_0(\mathbf x)+\eta\sum_{m=1}^{M}f_m(\mathbf x)
$$

$F_0$ 通常是最优常数预测。后续树依赖前面所有树形成的当前预测，因此各轮存在顺序依赖；单轮内部的特征统计和节点搜索仍然可以并行。

## 2. 模型原理

### 2.1 从参数更新到函数更新

训练目标为：

$$
\mathcal J(F)=\sum_{i=1}^{n}\ell(y_i,F(\mathbf x_i))
$$

普通梯度下降更新参数向量，梯度提升则在训练样本的预测值上寻找下降方向。第 $m$ 轮的伪残差定义为：

$$
r_{im}=-\left.\frac{\partial\ell(y_i,F)}{\partial F}\right|_{F=F_{m-1}(\mathbf x_i)}
$$

将 $(\mathbf x_i,r_{im})$ 作为回归数据训练树，使树输出近似负梯度：

$$
h_m\approx\arg\min_h\sum_i(r_{im}-h(\mathbf x_i))^2
$$

树只能表达受深度和叶节点约束的分段常数函数，因此是在基学习器允许的函数集合内近似下降方向。

### 2.2 树结构与叶节点更新

设第 $m$ 棵树划分出 $J_m$ 个叶区域 $R_{jm}$。确定结构后，可以按原始损失重新优化各叶输出：

$$
\gamma_{jm}=\arg\min_{\gamma}\sum_{\mathbf x_i\in R_{jm}}
\ell(y_i,F_{m-1}(\mathbf x_i)+\gamma)
$$

随后更新：

$$
F_m(\mathbf x)=F_{m-1}(\mathbf x)+
\eta\sum_{j=1}^{J_m}\gamma_{jm}\mathbb I(\mathbf x\in R_{jm})
$$

拟合负梯度决定树的结构，按原损失求解叶值决定更新幅度。最简单的一阶实现也可以直接使用伪残差均值作为叶值，但对于非平方损失，它与原损失下的最优叶值通常不同。

### 2.3 平方误差回归

使用单样本损失：

$$
\ell(y,F)=\frac12(y-F)^2
$$

最优常数为标签均值：

$$
F_0=\overline y
$$

负梯度为：

$$
r_{im}=y_i-F_{m-1}(\mathbf x_i)
$$

固定叶区域后，最优叶值就是该区域的残差均值。因此，平方误差下，用回归树拟合残差与逐叶优化可以自然衔接。

考虑特征 $1,2,3,4$ 和目标 $1,1,5,5$。初始常数为 3，残差为 $-2,-2,2,2$。一个在 $2.5$ 处划分的树得到两个叶值 $-2$ 和 2。

若 $\eta=0.5$，第一轮预测变为 $2,2,4,4$，均方误差从 4 降到 1。第二轮残差变为 $-1,-1,1,1$，同样的结构继续更新后，预测为 $1.5,1.5,4.5,4.5$，均方误差降到 $0.25$。

### 2.4 二分类交叉熵

二分类时，树的累加结果表示对数几率得分：

$$
p_i=\sigma(F(\mathbf x_i))
$$

损失采用与[逻辑回归](/dl/logistic_regression/)一致的交叉熵：

$$
\ell(y,F)=\log(1+\exp(F))-yF
$$

若训练正类比例为 $\pi\in(0,1)$，最优初始常数为：

$$
F_0=\log\frac{\pi}{1-\pi}
$$

一阶导数和二阶导数分别为：

$$
g_i=p_i-y_i,\qquad h_i=p_i(1-p_i)
$$

因此负梯度为 $y_i-p_i$。它不是“标签减去预测类别”，也不是“标签减去当前对数几率得分”。

树仍然是回归树，因为训练目标是连续的伪残差。推理时先累加所有树的得分，再执行逻辑函数得到概率，而不是把各树概率直接相加。

### 2.5 叶值的牛顿近似

对固定叶区域，交叉熵的最优常数增量通常需要数值求解。使用一次牛顿近似：

$$
\gamma_{jm}\approx
\frac{\sum_{i\in R_{jm}}(y_i-p_i)}
{\sum_{i\in R_{jm}}p_i(1-p_i)}
$$

分母很小时，近似更新可能过大，需要结合正则、最小叶样本数和更新幅度约束。

如果对当前增量增加 $\lambda\gamma^2/2$ 惩罚，二阶近似中的分母变为 $\sum h_i+\lambda$。这是针对该叶增量的局部近似，不应直接解释为对最终累加模型施加了同一个平方范数惩罚。

一阶梯度拟合、原损失逐叶优化和二阶近似是不同的实现选择。比较工具时，需要同时核对树结构的划分准则与叶值计算规则。

### 2.6 其他损失与多类别任务

| 任务或损失 | 初始常数 | 更新方向或特点 |
| --- | --- | --- |
| 平方误差回归 | 标签均值 | 负梯度为残差 |
| 绝对误差回归 | 标签中位数 | 使用残差符号的负次梯度，叶值需重新优化 |
| 分位数回归 | 指定分位数 | 不对称惩罚高估和低估 |
| 二分类交叉熵 | 正类比例的对数几率 | 负梯度为标签减概率 |
| 多类别交叉熵 | 按约定初始化各类得分 | 每轮通常需要多个输出或多棵树 |

多类别任务通过归一化指数函数将各类得分转换为概率。每轮树数取决于具体表示和实现，不能把二分类“一轮一棵树”无条件推广到多类别。

### 2.7 学习率、容量与随机化

学习率缩小每棵树的贡献。较小学习率通常需要更多轮次，不能仅减小学习率而保持轮数不变，就期望效果一定提升。

树深度和叶节点数控制单轮更新能表达的交互复杂度。一层树桩一次只使用一个特征，多棵树桩的加和可以拟合加性非线性关系，但不能直接表达任意跨特征交互。

限制最小叶样本数可以避免基于少量样本进行极端修正。按轮随机抽取部分样本构成随机梯度提升，特征采样也可降低成本和树间相关性，但都需要通过验证数据检查偏差与方差的变化。

### 2.8 早停与泛化

训练损失下降不保证验证损失持续下降。早停通过监控独立验证集，在连续若干轮没有改善后停止，并保留最佳轮次对应的模型。

实现早停时必须保存或回退到最佳树数量。只停止继续添加树，却保留最佳轮次之后的全部树，不等于恢复最佳模型。测试集不能承担轮数选择或超参数选择。

梯度提升通常使用浅树，但最终模型是多棵树的组合，不具有单条浅树路径同等直观的全局解释。特征重要性和局部贡献也需要结合相关性、数据分布和校准状态解释。

## 3. 模型实现

### 3.1 回归树桩

标准库实现以深度为一的回归树桩为基学习器，每轮拟合负梯度，叶值取伪残差均值。回归使用平方误差，二分类使用一阶交叉熵更新；不额外执行牛顿叶值修正。

```python
import math


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_loss(label, score):
    return max(score, 0.0) - label * score + math.log1p(math.exp(-abs(score)))


class RegressionStump:
    def __init__(self, min_samples_leaf=1):
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, targets):
        self.feature = None
        self.left_value = self.right_value = sum(targets) / len(targets)
        best_cost = sum((value - self.left_value) ** 2 for value in targets)
        for feature in range(len(X[0])):
            values = sorted(set(row[feature] for row in X))
            for lower, upper in zip(values, values[1:]):
                threshold = lower / 2.0 + upper / 2.0
                if not lower <= threshold < upper:
                    threshold = lower
                left = [targets[i] for i, row in enumerate(X) if row[feature] <= threshold]
                right = [targets[i] for i, row in enumerate(X) if row[feature] > threshold]
                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue
                left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
                cost = (sum((v - left_mean) ** 2 for v in left) +
                        sum((v - right_mean) ** 2 for v in right))
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    self.feature, self.threshold = feature, threshold
                    self.left_value, self.right_value = left_mean, right_mean
        return self

    def predict(self, X):
        if self.feature is None:
            return [self.left_value] * len(X)
        return [self.left_value if row[self.feature] <= self.threshold
                else self.right_value for row in X]
```

未找到有效划分时，树桩退化为常数预测器。各阈值重新扫描样本的写法便于检查计算过程，但没有使用排序前缀统计或直方图优化。

### 3.2 梯度提升训练器

```python
class GradientBoosting:
    def __init__(self, task="regression", n_estimators=50,
                 learning_rate=0.1, min_samples_leaf=1):
        if task not in {"regression", "classification"}:
            raise ValueError("invalid task")
        if (n_estimators < 1 or min_samples_leaf < 1 or
                not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("invalid parameters")
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        X, y = [list(row) for row in X], list(y)
        if not X or len(X) != len(y) or not X[0]:
            raise ValueError("invalid training shape")
        self.n_features = len(X[0])
        if any(len(row) != self.n_features for row in X):
            raise ValueError("inconsistent feature dimensions")
        if any(not math.isfinite(v) for row in X for v in row):
            raise ValueError("features must be finite")
        if any(not math.isfinite(value) for value in y):
            raise ValueError("targets must be finite")
        average = sum(y) / len(y)
        if self.task == "classification":
            if any(value not in (0, 1) for value in y) or not 0 < average < 1:
                raise ValueError("classification requires both binary classes")
            self.initial_score = math.log(average / (1.0 - average))
        else:
            self.initial_score = average

        scores = [self.initial_score] * len(y)
        self.trees = []
        self.train_loss = [self._loss(y, scores)]
        for _ in range(self.n_estimators):
            if self.task == "regression":
                residual = [label - score for label, score in zip(y, scores)]
            else:
                residual = [label - sigmoid(score) for label, score in zip(y, scores)]
            tree = RegressionStump(self.min_samples_leaf).fit(X, residual)
            update = tree.predict(X)
            scores = [score + self.learning_rate * delta
                      for score, delta in zip(scores, update)]
            self.trees.append(tree)
            self.train_loss.append(self._loss(y, scores))
        self.fitted = True
        return self

    def _loss(self, y, scores):
        if self.task == "regression":
            return sum(0.5 * (label - score) ** 2
                       for label, score in zip(y, scores)) / len(y)
        return sum(binary_loss(label, score)
                   for label, score in zip(y, scores)) / len(y)

    def decision_function(self, X):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        X = [list(row) for row in X]
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row)
               for row in X):
            raise ValueError("invalid prediction features")
        scores = [self.initial_score] * len(X)
        for tree in self.trees:
            scores = [score + self.learning_rate * delta
                      for score, delta in zip(scores, tree.predict(X))]
        return scores

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("probabilities require classification")
        return [sigmoid(score) for score in self.decision_function(X)]

    def predict(self, X):
        scores = self.decision_function(X)
        if self.task == "regression":
            return scores
        return [int(score >= 0.0) for score in scores]
```

该实现保存每轮训练损失，便于观察更新过程。它使用固定轮数，不包含早停、采样、样本权重和深树；替换基学习器时，需要保持输出为连续更新量。

### 3.3 训练示例

```python
X = [[1.0], [2.0], [3.0], [4.0]]

regressor = GradientBoosting(n_estimators=2, learning_rate=0.5)
regressor.fit(X, [1.0, 1.0, 5.0, 5.0])
print("regression:", regressor.predict(X))
print("half_mse:", regressor.train_loss)

classifier = GradientBoosting(
    task="classification", n_estimators=50, learning_rate=0.5
)
classifier.fit(X, [0, 0, 1, 1])
print("classification:", classifier.predict(X))
print("log_loss_decreased:", classifier.train_loss[-1] < classifier.train_loss[0])
```

输出为：

```text
regression: [1.5, 1.5, 4.5, 4.5]
half_mse: [2.0, 0.5, 0.125]
classification: [0, 0, 1, 1]
log_loss_decreased: True
```

回归损失记录的是均方误差的一半，与公式中的缩放一致。分类样例是可分数据，用于检查更新方向和预测，不作为泛化评估；真实任务需要独立验证和测试。

### 3.4 计算成本与数值边界

对 $n$ 个样本和 $d$ 个特征，直接树桩搜索的单轮最坏成本约为 $O(dn^2)$，$M$ 轮约为 $O(Mdn^2)$。排序与前缀统计可以减少阈值遍历中的重复计算，工程实现还可以使用直方图分桶。

每个树桩只保存一个特征、阈值和两个叶值，因此模型存储量为 $O(M)$，单样本预测为 $O(M)$。深树则需要按节点数和路径深度重新计算成本。

稳定逻辑函数和交叉熵避免直接计算大正数指数。有限数据不代表任意学习率都安全：过大更新可能使目标变差，甚至产生数值异常。样例采用固定且温和的学习率，生产训练仍需检查验证损失和参数配置。

### 3.5 与其他树集成方法的关系

| 方法 | 树之间的关系 | 主要思路 |
| --- | --- | --- |
| 随机森林 | 各树通常可独立训练 | 通过采样和平均降低方差 |
| 梯度提升树 | 后续树依赖当前整体预测 | 逐轮拟合下降方向并累加 |
| 二阶提升树 | 同时使用一阶与二阶统计 | 结合局部损失近似选择结构和叶值 |

XGBoost 等系统将正则化、二阶优化和高效划分结合到树提升框架中；经典梯度提升本身不要求必须使用二阶导数。不同实现的区别需要落实到目标函数、结构选择、叶值更新与工程算法。

## 4. 总结

梯度提升树用加法模型逐步降低损失。平方误差下，新树拟合当前残差；交叉熵下，新树拟合标签与概率之差，最终在累加得分上进行概率变换。叶值可以直接使用一阶拟合结果，也可以按原始损失重新优化或进行牛顿近似。

学习率、树容量和轮数共同决定模型复杂度。验证集早停、合理的特征处理、数值稳定计算和独立评估，是将训练损失下降转化为泛化收益的关键。

## 5. 参考文献

[1] Friedman J H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001, 29(5): 1189–1232.

[2] Friedman J H. Stochastic Gradient Boosting. Computational Statistics & Data Analysis, 2002, 38(4): 367–378.

[3] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd Edition. Springer, 2009.
