# 随机森林：从自助采样到多树集成

随机森林通过自助采样生成不同训练子集，并在每个节点随机选择部分特征搜索划分，再聚合多棵树的预测。样本随机性和特征随机性降低树之间的相关性，使单棵决策树的不稳定性通过集成得到缓解。

分类任务可以使用多数投票或类别概率平均，回归任务通常对预测值求平均。各树的训练互不依赖，适合并行执行；袋外样本则为训练过程提供额外的评估路径。

## 1. 背景

[决策树](/dl/decision_tree/)对训练数据变化比较敏感。少量样本变化可能改变根节点的选择，继而改变整棵树。直接增加单棵树的深度能够降低训练误差，却也可能放大这种高方差。

如果训练多棵不同的树，再对结果聚合，个别树的波动可以相互抵消。但如果每棵树都使用同样的数据和特征，学到的结构高度相似，增加树数的收益就会受到限制。

随机森林在自助采样集成的基础上增加节点级特征随机化，使树保持一定预测能力的同时，减少树之间重复犯错的程度。

## 2. 模型原理

### 2.1 自助采样

设训练集包含 $n$ 个样本。每棵树从训练集有放回抽取 $n$ 次，组成自己的自助样本。相同样本可能重复出现，也可能完全没有被选中。

某个样本在一次抽取中未被选中的概率为 $1-1/n$，在整次自助采样中未被选中的概率为：

$$
q_n=\left(1-\frac1n\right)^n\longrightarrow e^{-1}\approx0.368
$$

因此，当样本数较大时，一棵树平均使用约 $63.2\%$ 的不同训练样本，其余约 $36.8\%$ 为该树的袋外样本。抽样记录仍有 $n$ 条，只是其中存在重复，不能把自助采样实现为无放回抽取 $63.2\%$ 的样本。

### 2.2 节点级随机特征选择

在每个内部节点，从 $d$ 个特征中随机选择 $m$ 个候选，再只在这些特征上搜索最佳划分。不同节点重新采样，而不是为整棵树固定同一组特征。

$m$ 较大时，单次划分容易找到强特征，但各树可能更加相似；$m$ 较小时，随机性增加，但也可能错过有效特征，使单树变弱。这个权衡需要通过验证数据或袋外评估选择。

随机选择特征不等于随机选择阈值。标准随机森林通常仍在候选特征上寻找优质阈值；极端随机树还会进一步随机化阈值。

### 2.3 分类与回归聚合

设森林有 $B$ 棵树，第 $b$ 棵树的分类输出为 $h_b(\mathbf x)$。硬投票为：

$$
\widehat y=\arg\max_k\sum_{b=1}^{B}\mathbb I(h_b(\mathbf x)=k)
$$

另一种常见方式是平均叶节点类别频率。令 $p_{bk}(\mathbf x)$ 为单树对类别 $k$ 的概率：

$$
\overline p_k(\mathbf x)=\frac1B\sum_{b=1}^{B}p_{bk}(\mathbf x),\qquad
\widehat y=\arg\max_k\overline p_k(\mathbf x)
$$

两种规则并不总是等价。三棵树的正类概率若为 $0.51,0.51,0.01$，硬投票选择正类，概率平均却低于 $0.5$。实现需要明确聚合的是类别还是概率。

回归任务直接对连续预测取均值：

$$
\widehat y=\frac1B\sum_{b=1}^{B}h_b(\mathbf x)
$$

类别概率平均能缓和单树的极端输出，但不保证模型已经校准。小样本叶子、类别加权与分布漂移仍可能影响概率解释。

### 2.4 平均如何降低方差

在固定输入处，假设各树预测方差都为 $\sigma^2$，任意两棵树的相关系数均为 $\rho$，则平均预测的方差为：

$$
\operatorname{Var}\left(\frac1B\sum_bh_b\right)
=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2
$$

这是用于解释的等方差、等相关近似。增加树数降低第二项，却不能消除相关性带来的第一项，因此减少树间相关性很重要。该公式直接描述数值预测的平均，不能原样当作多数投票分类错误率公式。

树数增加通常使随机波动逐渐稳定，但会增加存储和推理成本，也不能修复数据泄漏、错误标签或不合理的数据划分。

### 2.5 袋外评估

对训练样本 $i$，只使用未抽到该样本的树进行预测。设这些树的索引为 $\mathcal B_i$，回归袋外预测为：

$$
\widehat y_i^{\mathrm{OOB}}=\frac1{|\mathcal B_i|}
\sum_{b\in\mathcal B_i}h_b(\mathbf x_i)
$$

分类时也只平均这些树的类别概率。袋外预测不能混入见过该样本的树，否则退化为偏乐观的训练集评价。

树数较少时，某些样本可能没有袋外树，必须报告覆盖率并排除这些无有效预测的样本，而不是填入默认类别。独立自助采样下，样本在 $B$ 棵树中均无袋外机会的概率为 $(1-q_n)^B$。

袋外评估有助于观察泛化，但反复依据袋外分数选择参数也会引入选择偏差。时间依赖、用户分组等任务仍需采用符合业务场景的数据划分，不能用随机袋外机制替代时间或分组隔离。

### 2.6 容量与数据处理

| 参数 | 控制内容 | 常见影响 |
| --- | --- | --- |
| 树数量 | 集成规模 | 影响随机波动、模型体积和推理成本 |
| 最大深度 | 单树复杂度 | 深树能学习复杂交互，也可能产生小样本叶子 |
| 最小叶样本数 | 叶节点统计量稳定性 | 增大后预测通常更平滑 |
| 候选特征数 | 单树强度与树间相关性 | 过小可能丢失有效划分 |
| 自助样本大小 | 单树使用的数据规模 | 改变样本多样性与袋外覆盖 |

数值特征通常无需标准化。无序类别需要合理编码，缺失值是否能原生处理取决于具体实现；训练依赖的填补和编码规则只能从训练数据拟合。

### 2.7 特征重要性与适用边界

按不纯度下降汇总的重要性容易偏向可切分取值较多的特征。独立验证集上的置换重要性可以作为补充，但强相关特征会分担贡献，单独打乱其中一个特征未必反映整个特征组的价值。

随机森林适合包含非线性与交互的表格数据，能够作为分类和回归基线。其回归预测由叶均值再平均得到，通常不能沿训练范围之外的趋势外推；高维稀疏文本也不一定比线性模型或朴素贝叶斯更合适。

## 3. 模型实现

### 3.1 随机化决策树

标准库实现使用分类基尼不纯度和回归均方误差。在每个节点重新采样候选特征，并设置固定随机种子保证可复现。输入为有限数值特征，分类标签使用整数。

```python
import math
import random
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


class RandomizedTree:
    def __init__(self, task="classification", max_depth=3,
                 min_samples_split=2, min_samples_leaf=1,
                 min_gain=0.0, max_features=1, random_state=0):
        if task not in {"classification", "regression"}:
            raise ValueError("invalid task")
        if (max_depth < 0 or min_samples_split < 2 or
                min_samples_leaf < 1 or not math.isfinite(min_gain) or min_gain < 0):
            raise ValueError("invalid tree parameters")
        if not isinstance(max_features, int) or max_features < 1:
            raise ValueError("max_features must be a positive integer")
        self.max_features = max_features
        self.random_state = random_state
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
        self.rng = random.Random(self.random_state)
        self.root = self._grow(X, y, depth=0)
        return self

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
        candidates = self.rng.sample(
            range(self.n_features), min(self.max_features, self.n_features)
        )
        for feature in candidates:
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

### 3.2 自助采样与森林聚合

森林保存每棵树未见过的原始样本索引。分类统一按全局类别集合对齐概率，避免某棵树的自助样本缺少类别时发生错位。

```python
class RandomForest:
    def __init__(self, task="classification", n_estimators=30,
                 max_depth=5, max_features=1, min_samples_leaf=1,
                 random_state=42):
        if task not in {"classification", "regression"}:
            raise ValueError("invalid task")
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        X, y = [list(row) for row in X], list(y)
        # 统一检查原始数据，避免无效样本恰好未被抽到。
        validator = RandomizedTree(
            task=self.task, max_depth=0,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
        ).fit(X, y)
        self.n_features = validator.n_features
        self.classes = sorted(set(y)) if self.task == "classification" else None
        self.trees, self.oob_indices = [], []
        rng = random.Random(self.random_state)
        for _ in range(self.n_estimators):
            indices = [rng.randrange(len(X)) for _ in range(len(X))]
            selected = set(indices)
            tree = RandomizedTree(
                task=self.task, max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randrange(2 ** 32),
            ).fit([X[i] for i in indices], [y[i] for i in indices])
            self.trees.append(tree)
            self.oob_indices.append(set(range(len(X))) - selected)
        self.fitted = True
        self.oob_prediction = []
        self.oob_counts = []
        for i, row in enumerate(X):
            eligible = [tree for tree, excluded in zip(self.trees, self.oob_indices)
                        if i in excluded]
            self.oob_counts.append(len(eligible))
            self.oob_prediction.append(self._aggregate(row, eligible) if eligible else None)
        self.oob_coverage = sum(count > 0 for count in self.oob_counts) / len(X)
        return self

    def _aggregate(self, row, trees):
        if self.task == "regression":
            return sum(tree.predict([row])[0] for tree in trees) / len(trees)
        total = {label: 0.0 for label in self.classes}
        for tree in trees:
            for label, value in tree.predict_proba([row])[0].items():
                total[label] += value
        return {label: value / len(trees) for label, value in total.items()}

    def _outputs(self, X):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        return [self._aggregate(row, self.trees) for row in X]

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("probabilities require classification")
        return self._outputs(X)

    def predict(self, X):
        outputs = self._outputs(X)
        if self.task == "regression":
            return outputs
        return [max(self.classes, key=lambda label: probability[label])
                for probability in outputs]
```

相同类别概率时，按排序后的类别顺序决策。随机特征子集内没有合法正收益划分时，该节点停止，不额外搜索未选特征。实现不支持样本权重和原生缺失值，预测规则与预处理需要保持一致。

### 3.3 分类、回归与袋外评价

```python
data_rng = random.Random(7)
X = [[data_rng.uniform(-1, 1), data_rng.uniform(-1, 1)] for _ in range(240)]
labels = [int(row[0] + 0.6 * row[1] > 0) for row in X]
train_X, test_X = X[:180], X[180:]
train_y, test_y = labels[:180], labels[180:]

classifier = RandomForest(n_estimators=30, max_features=1, max_depth=5)
classifier.fit(train_X, train_y)
predictions = classifier.predict(test_X)
accuracy = sum(a == b for a, b in zip(predictions, test_y)) / len(test_y)
valid = [i for i, count in enumerate(classifier.oob_counts) if count > 0]
oob_correct = 0
for i in valid:
    probability = classifier.oob_prediction[i]
    label = max(classifier.classes, key=lambda value: probability[value])
    oob_correct += label == train_y[i]
print("test_accuracy:", round(accuracy, 3))
print("oob_coverage:", round(classifier.oob_coverage, 3))
print("oob_accuracy:", round(oob_correct / len(valid), 3) if valid else None)

values = [row[0] ** 2 + row[1] for row in X]
regressor = RandomForest(task="regression", n_estimators=30,
                         max_features=2, max_depth=6)
regressor.fit(train_X, values[:180])
regression_prediction = regressor.predict(test_X)
mse = sum((pred - target) ** 2 for pred, target in
          zip(regression_prediction, values[180:])) / len(test_X)
print("test_mse:", round(mse, 4))
```

数据按固定随机种子生成，测试集不参与训练与袋外统计。袋外覆盖不足时，只对有有效袋外预测的样本计算分数，并同时报告覆盖率。真实数据还需检查各类别指标、概率损失及目标分布下的误差。

### 3.4 计算成本

直接实现对每个候选阈值重新扫描节点数据。在包含 $m$ 个样本、候选特征数为 $k$ 的节点上，最坏成本约为 $O(km^2)$，还需累计所有树的各节点成本。工程实现通常使用排序前缀统计、紧凑数组和并行训练减少开销。

若每棵树平均有 $L$ 个节点，森林存储量约为 $O(BL)$。单样本预测需遍历所有树的路径，平均深度为 $h$ 时约为 $O(Bh)$。袋外索引和预测缓存也会占用额外内存。

各树可并行训练，但并发数受内存和计算资源约束。预测时也应权衡吞吐量与单请求延迟，避免仅依据树数量选择模型。

### 3.5 与梯度提升树的关系

| 方法 | 每棵树的训练目标 | 树之间关系 | 聚合方式 |
| --- | --- | --- | --- |
| 随机森林 | 自助样本上的原始标签 | 树可独立训练 | 投票或平均 |
| 梯度提升树 | 当前模型的负梯度或相关局部目标 | 后续轮次依赖已有预测 | 按学习率累加 |

随机森林侧重通过随机化和聚合减少方差，[梯度提升树](/dl/gradient_boosting/)通过顺序修正优化整体损失。两者都需要控制树容量、避免数据泄漏，并通过验证而非训练得分选择配置。

## 4. 总结

随机森林结合自助采样和节点级特征随机选择，训练多棵具有差异的决策树，再通过平均或投票形成预测。降低树间相关性能够提升集成的方差缩减收益，袋外样本则提供训练集内部的额外评估途径。

可靠实现需要保留重复采样的次数、对齐各树类别概率，并严格隔离袋外预测。树数、候选特征数和叶节点容量共同影响精度、稳定性与推理成本，最终配置应结合部署分布和独立评估确定。

## 5. 参考文献

[1] Breiman L. Random Forests. Machine Learning, 2001, 45: 5–32.

[2] Breiman L. Bagging Predictors. Machine Learning, 1996, 24: 123–140.

[3] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd Edition. Springer, 2009.
