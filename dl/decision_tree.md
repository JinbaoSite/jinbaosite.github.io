# 决策树：从特征划分到分类与回归

决策树通过一系列特征条件递归划分样本空间，在叶节点输出类别或连续值。每个内部节点对应一次条件判断，从根到叶的路径构成一组可追溯的规则。

与线性模型依赖特征加权求和不同，决策树可以直接学习非线性边界和特征交互。模型训练的核心是选择划分条件，并控制树的复杂度，避免把训练噪声也编码成规则。

## 1. 背景

预测一个用户是否购买商品，可以先判断浏览次数，再根据价格或停留时长继续划分。满足不同条件的样本进入不同叶节点，每个叶节点依据训练样本给出预测。

对数值特征，常用二元规则为：

$$
x_j\le t\quad\text{或}\quad x_j>t
$$

其中 $j$ 是特征索引，$t$ 是阈值。分类树预测类别，回归树预测连续值。两者共享递归划分框架，主要区别在于衡量划分质量的准则和叶节点输出。

一棵树可以表达多个条件共同作用。例如，某特征只在另一特征超过阈值后才影响预测，这种关系通过不同层级的分支自然形成。

## 2. 模型原理

### 2.1 递归划分与贪心搜索

设节点包含样本集合 $S$，候选规则 $(j,t)$ 将其分为：

$$
S_L=\{i\in S:x_{ij}\le t\},\qquad
S_R=\{i\in S:x_{ij}>t\}
$$

用 $H(S)$ 表示节点不纯度，划分收益为：

$$
\Delta(j,t)=H(S)-\frac{|S_L|}{|S|}H(S_L)-\frac{|S_R|}{|S|}H(S_R)
$$

训练时枚举候选条件，选择收益最大的合法划分，再对左右子节点重复操作，直到满足停止条件。

这种搜索是逐节点贪心的，不保证找到全局最优树。当前收益很小的划分可能为后续有价值的规则创造条件；过早停止也可能错过这种结构。

### 2.2 分类树的划分准则

设节点中类别 $k$ 的比例为 $p_k$。基尼不纯度为：

$$
H_{\mathrm{Gini}}(S)=1-\sum_{k=1}^{C}p_k^2
$$

节点只有一个类别时不纯度为零，类别混合时不纯度增加。二分类中两类各占一半时，基尼不纯度为 $0.5$。

信息熵为：

$$
H_{\mathrm{entropy}}(S)=-\sum_{k=1}^{C}p_k\log_2p_k
$$

约定 $0\log_20=0$。将信息熵代入划分收益，得到信息增益。基尼与熵都偏好类别更集中的子节点，但具体排序和最终树结构可能不同。

叶节点类别通常取出现次数最多的类别。类别概率可以用该叶节点的类别频率估计；样本很少的叶节点容易产生接近 0 或 1 的极端概率，需要结合最小叶样本数、平滑或概率校准控制。

### 2.3 分类划分的手算示例

有六个样本，单个特征依次为 $1,2,3,4,5,6$，标签依次为 $0,0,0,1,1,1$。根节点两类各占一半：

$$
H(S)=1-\left(\frac12\right)^2-\left(\frac12\right)^2=\frac12
$$

阈值 $t=3.5$ 把两类完全分开，左右子节点不纯度都为零，因此收益为 $0.5$。

若阈值取 $t=2.5$，左侧两个样本均为类别 0，右侧包含一个类别 0 和三个类别 1：

$$
H(S_R)=1-\left(\frac14\right)^2-\left(\frac34\right)^2=\frac38
$$

$$
\Delta=\frac12-\frac26\cdot0-\frac46\cdot\frac38=\frac14
$$

因此，在这些候选中，$3.5$ 比 $2.5$ 提供更大的不纯度下降。

### 2.4 回归树与平方误差

对于连续标签，节点预测为样本均值：

$$
\overline y_S=\frac1{|S|}\sum_{i\in S}y_i
$$

它最小化节点内的残差平方和。定义节点不纯度为均方误差：

$$
H(S)=\frac1{|S|}\sum_{i\in S}(y_i-\overline y_S)^2
$$

最小化左右节点的加权不纯度，等价于最小化两侧的残差平方和之和。

例如，特征为 $1,2,3,4$，目标为 $1,1,5,5$。根节点均值为 3，均方误差为 4；在 $t=2.5$ 处划分后，两个叶节点分别预测 1 和 5，均方误差降为零。

回归树的输出是分段常数。预测输入超出训练特征范围时，仍会落入某个已有叶节点，通常无法像线性回归那样沿某个斜率外推。

### 2.5 连续与类别特征

数值特征排序后，相邻不同取值之间都可以产生候选阈值。在有限训练集上，这些阈值足以表示所有可能的二元排序划分；相同取值的样本不能被同一个阈值拆开。

类别特征可以使用多路划分或类别子集的二元划分，也可以先做独热编码。把无序类别直接编码为整数会引入人为顺序，限制单次阈值能够表达的类别组合。

单变量精确阈值搜索主要依赖取值顺序，因此通常无需标准化。严格单调变换保持训练样本的顺序和可选划分集合，但有限精度、阈值表示和近似搜索可能带来差异。

### 2.6 常见算法路线

| 算法 | 核心思路 | 适用范围与特点 |
| --- | --- | --- |
| ID3 | 使用信息增益选择特征 | 经典形式主要处理离散分类，容易偏向取值多的特征 |
| C4.5 | 引入增益率，并扩展连续特征与剪枝 | 使用额外选择规则，避免单纯依赖增益率的偏差 |
| CART | 递归二元划分 | 同时覆盖分类与回归，分类常用基尼，回归常用平方误差 |

对于产生多个分支的特征，增益率以划分本身的熵归一化信息增益：

$$
\operatorname{GainRatio}=\frac{\operatorname{Gain}}
{-\sum_v\frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}}
$$

分母为零时，该划分无效。增益率能缓解信息增益对多取值特征的偏好，但分母过小时也可能放大比值，需要结合有效增益筛选。

### 2.7 预剪枝与后剪枝

**（1）预剪枝**

在生长过程中设置约束，例如最大深度、最小划分样本数、最小叶样本数，以及最小不纯度下降。

最小划分样本数限制一个节点是否可以尝试分裂，最小叶样本数限制分裂后每个子节点的规模。前者不能替代后者：一个包含很多样本的节点也可能分出只有一个样本的叶子。

**（2）后剪枝**

先生成较大的树，再删除收益不足的子树。代价复杂度目标可以写为：

$$
R_{\alpha}(T)=R(T)+\alpha|\mathcal L(T)|
$$

其中 $R(T)$ 是明确约定的训练误差或加权叶不纯度，$|\mathcal L(T)|$ 是叶节点数。$\alpha$ 越大，对复杂度的惩罚越强；其数值含义取决于 $R(T)$ 的定义和缩放。

剪枝路径可以由训练数据产生，但最终深度和剪枝强度应通过验证集或交叉验证选择。测试集保留作最终评估。

### 2.8 缺失值、权重与特征重要性

缺失值可以通过训练集统计量填补并增加缺失指示特征，也可以由支持该机制的实现学习缺失方向或替代划分。是否支持原生缺失值取决于具体算法与实现。

存在样本权重时，类别比例、均值和左右分支权重都应统一改用加权统计。类别加权会影响规则选择及叶节点概率，输出需要结合部署分布检查校准。

按不纯度下降累计得到的特征重要性容易偏向候选切分较多的特征。验证集上的置换重要性可以补充观察，但相关特征之间也会分担贡献。无论哪种重要性，都不能直接解释为因果效应。

## 3. 模型实现

### 3.1 节点与不纯度

实现使用 Python 标准库，支持有限数值特征、整数类别标签和回归目标。分类采用基尼不纯度，回归采用均方误差；树深度从根节点的 0 开始计数。

```python
import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class Node:
    value: object
    probability: dict | None = None
    feature: int | None = None
    threshold: float | None = None
    left: object = None
    right: object = None


class DecisionTree:
    def __init__(self, task="classification", max_depth=3,
                 min_samples_split=2, min_samples_leaf=1,
                 min_gain=0.0):
        if task not in {"classification", "regression"}:
            raise ValueError("invalid task")
        if (max_depth < 0 or min_samples_split < 2 or
                min_samples_leaf < 1 or not math.isfinite(min_gain) or min_gain < 0):
            raise ValueError("invalid tree parameters")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root = None

    def impurity(self, values):
        if self.task == "classification":
            counts = Counter(values)
            return 1.0 - sum((count / len(values)) ** 2 for count in counts.values())
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def fit(self, X, y):
        self.root = None
        X = [list(row) for row in X]
        y = list(y)
        if not X or len(X) != len(y) or not X[0]:
            raise ValueError("invalid training shape")
        self.n_features = len(X[0])
        if any(len(row) != self.n_features for row in X):
            raise ValueError("inconsistent feature dimensions")
        if any(not math.isfinite(value) for row in X for value in row):
            raise ValueError("features must be finite")
        if self.task == "regression" and any(not math.isfinite(value) for value in y):
            raise ValueError("targets must be finite")
        self.root = self._grow(X, y, depth=0)
        return self
```

### 3.2 候选划分与递归生长

递归函数直接加入同一个类定义，运行时可将以下方法接在 `fit` 后面。

```python
    def _grow(self, X, y, depth):
        if self.task == "classification":
            counts = Counter(y)
            probability = {label: count / len(y) for label, count in sorted(counts.items())}
            value = min(counts, key=lambda label: (-counts[label], label))
        else:
            probability = None
            value = sum(y) / len(y)
        node = Node(value=value, probability=probability)
        parent_impurity = self.impurity(y)
        if (depth >= self.max_depth or len(y) < self.min_samples_split or
                parent_impurity <= 1e-15):
            return node

        best = None
        best_gain = self.min_gain
        for feature in range(self.n_features):
            values = sorted(set(row[feature] for row in X))
            for lower, upper in zip(values, values[1:]):
                threshold = lower / 2.0 + upper / 2.0
                # 极端浮点数舍入时，使用下界保持左右集合非空。
                if not lower <= threshold < upper:
                    threshold = lower
                left = [i for i, row in enumerate(X) if row[feature] <= threshold]
                right = [i for i, row in enumerate(X) if row[feature] > threshold]
                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue
                cost = (len(left) * self.impurity([y[i] for i in left]) +
                        len(right) * self.impurity([y[i] for i in right])) / len(y)
                gain = parent_impurity - cost
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feature, threshold, left, right)

        if best is None:
            return node
        node.feature, node.threshold, left, right = best
        node.left = self._grow([X[i] for i in left], [y[i] for i in left], depth + 1)
        node.right = self._grow([X[i] for i in right], [y[i] for i in right], depth + 1)
        return node

    def _leaf(self, row):
        if self.root is None:
            raise RuntimeError("fit must be called before prediction")
        if len(row) != self.n_features or any(not math.isfinite(x) for x in row):
            raise ValueError("invalid prediction features")
        node = self.root
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        return [self._leaf(row).value for row in X]

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("probabilities require classification")
        return [self._leaf(row).probability.copy() for row in X]
```

`min_gain` 使用当前节点内归一化的不纯度下降，没有再乘以节点样本数占全体样本数的比例。它的数值不能直接与使用全局加权下降的工具参数比较。收益相同时保留最先遇到的特征和阈值。

概率输出使用类别到频率的映射，未在叶节点出现的类别概率视为零。实现只接受有限数值输入，缺失值需在调用前处理；后剪枝、样本权重和类别子集划分属于独立扩展。

### 3.3 分类与回归示例

```python
classifier = DecisionTree(max_depth=2)
classifier.fit([[1], [2], [3], [4], [5], [6]], [0, 0, 0, 1, 1, 1])
print("threshold:", classifier.root.threshold)
print("classification:", classifier.predict([[2.5], [4.5]]))
print("probability:", classifier.predict_proba([[2.5], [4.5]]))

regressor = DecisionTree(task="regression", max_depth=1)
regressor.fit([[1], [2], [3], [4]], [1.0, 1.0, 5.0, 5.0])
print("regression:", regressor.predict([[1.5], [3.5], [10.0]]))
```

输出为：

```text
threshold: 3.5
classification: [0, 1]
probability: [{0: 1.0}, {1: 1.0}]
regression: [1.0, 5.0, 5.0]
```

样例用于核对划分规则和叶节点输出。概率为 1 只是对应叶子的训练频率，不代表真实任务中预测绝对可靠；超出训练范围的输入 10 也只是进入最右侧叶子。

### 3.4 计算成本与实现边界

在包含 $m$ 个样本、$d$ 个特征的节点上，直接实现会枚举约 $O(dm)$ 个阈值，每个阈值重新扫描样本并计算不纯度，因此最坏约为 $O(dm^2)$。不能用优化版树算法的复杂度描述这种逐阈值重算实现。

工程实现可对特征排序，再通过前缀类别计数或前缀和更新左右统计，减少重复扫描。直方图方法进一步把连续取值压缩到有限桶内，以一定精度损失换取速度和内存收益。

预测沿根到叶访问节点，单样本成本为 $O(h)$，其中 $h$ 是实际路径深度。深树增加预测开销，也容易产生小样本叶子；树结构与节点规则需要随预处理配置一起保存。

### 3.5 评估与常见问题

1. **使用独立验证数据选择容量**：深度、最小叶样本数和剪枝强度都属于超参数；测试集用于最终评价。
2. **按任务选择指标**：分类观察精确率、召回率、宏平均分数和概率损失，回归观察均方根误差、平均绝对误差与残差结构。
3. **注意高方差**：训练样本的微小变化可能改变根节点选择，进而改变整棵树。随机森林通过集成降低这种不稳定性。
4. **识别局部贪心限制**：标准异或数据的任意单特征首层划分收益可以为零。要求严格正收益的实现会直接停止，即使深度增加后本可表达正确规则。
5. **检查类别与缺失编码**：类别整数编码和固定填充值可能引入人为划分关系，特征处理必须与业务语义一致。

## 4. 总结

决策树通过递归选择特征与阈值，把样本空间划分为多个区域。分类使用类别不纯度评价规则，回归通过降低平方误差选择规则，最终由叶节点输出类别频率或目标均值。

贪心划分使训练过程直观，但也带来局部最优和高方差问题。深度约束、最小叶样本数和后剪枝用于控制容量，独立验证和合理指标用于判断泛化能力。理解单棵树，也为随机森林和梯度提升树提供基础。

## 5. 参考文献

[1] Breiman L, Friedman J H, Olshen R A, Stone C J. Classification and Regression Trees. Wadsworth, 1984.

[2] Quinlan J R. Induction of Decision Trees. Machine Learning, 1986, 1: 81–106.

[3] Quinlan J R. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.
