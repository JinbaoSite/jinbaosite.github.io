# CatBoost：有序统计与类别特征树提升

CatBoost 是支持类别特征的梯度提升系统，通过有序目标统计构造数值特征，并通过有序提升缓解训练预测与新样本预测之间的分布偏移。对称树则为树结构施加统一的层级规则，便于紧凑存储与快速预测。

有序目标统计和有序提升解决不同环节的问题：前者控制类别编码使用哪些标签，后者控制生成梯度时依赖哪些历史样本。两者可以配合，但不能把普通目标编码加上树模型就直接视为完整 CatBoost。

## 1. 背景

商品类别、城市和用户分组等字段不具有自然数值顺序。独热编码在高基数下会产生大量稀疏列，直接整数编码又可能让树按照任意编号顺序搜索阈值。

另一种方法是用同类别的目标统计量编码特征。例如，用某城市的平均点击率作为城市特征。但如果一条样本自己的标签参与了它的编码，模型就可以从输入中间接读到答案。

对于只出现一次的类别，未经约束的类别均值恰好等于该样本标签。训练表现可能很好，新样本却无法获得同等信息，这构成目标泄漏及训练服务差异的重要来源。

## 2. 模型原理

### 2.1 普通目标统计的泄漏

设样本 $i$ 的类别值为 $c_i$，标签为 $y_i$。平滑类别均值为：

$$
\widehat t(c)=\frac{\sum_{j:c_j=c}y_j+ap}{\sum_{j:c_j=c}1+a}
$$

$p$ 是先验均值，$a>0$ 是先验强度。平滑降低小样本类别的方差，但若训练编码仍包含当前标签，泄漏并不会自动消失。

将全量数据编码后再划分训练测试集，还会直接泄漏测试标签。编码器必须在数据划分之后拟合，验证与测试编码不得使用它们自己的目标值。

### 2.2 有序目标统计

给训练样本生成排列 $\sigma$。位于排列第 $k$ 个位置的样本，只使用它之前同类别样本的标签：

$$
\widehat t_{\sigma_k}=
\frac{\displaystyle\sum_{j<k}\mathbb I(c_{\sigma_j}=c_{\sigma_k})y_{\sigma_j}+ap}
{\displaystyle\sum_{j<k}\mathbb I(c_{\sigma_j}=c_{\sigma_k})+a}
$$

计算顺序必须是先生成当前编码，再把当前标签加入统计。某类别首次出现时没有历史证据，编码退回先验 $p$。

例如，排列后的类别为 A、B、A、A，标签为 1、0、0、1，取 $p=0.5$、$a=1$，有序编码依次为：

$$
0.5,\quad0.5,\quad0.75,\quad0.5
$$

第三条 A 使用第一条 A 的标签，第四条 A 使用前两条 A 的标签；两者都不包含当前行标签。

多组排列可以减少单个排列带来的随机波动。真实系统还会使用不同统计形式、标签量化及特征组合，不仅限于一个类别均值公式。

### 2.3 先验、稀有类别与推理

对于验证、测试和服务样本，可使用训练数据积累的类别统计，而不使用待预测样本的标签。以同样的均值形式为例：

$$
\widehat t_{\mathrm{new}}(c)=\frac{\sum_{j\in\mathcal D_{train}:c_j=c}y_j+ap}
{\sum_{j\in\mathcal D_{train}:c_j=c}1+a}
$$

未知类别回退先验。训练前缀统计和推理完整训练统计的数据量不同，有序处理主要减少自身标签泄漏，并不使两者在有限数据下完全相同。

先验最好固定或由符合评估协议的数据估计。若使用全训练标签均值，当前标签也可能通过先验产生弱依赖；它与完整同类均值的泄漏强度不同，但不能再声称编码完全不依赖自身目标。

### 2.4 有序提升与预测偏移

即使类别编码已排除自身标签，普通提升在计算训练样本残差时，已有模型仍可能受该样本标签影响。模型对参与拟合的样本和全新样本产生不同的预测误差分布，这种现象称为预测偏移。

有序提升的概念形式是：对于排列位置 $k$ 的样本，使用仅依赖前缀样本的支持模型 $F_{m-1}^{<k}$ 计算梯度：

$$
g_{\sigma_k,m}=\left.
\frac{\partial\ell(y_{\sigma_k},F)}{\partial F}
\right|_{F=F_{m-1}^{<k}(\mathbf x_{\sigma_k})}
$$

当前样本标签参与它自身的损失梯度是正常监督学习，关键是用于求导的已有预测不提前使用该标签。

概念上为每个前缀维护完整模型代价很高，实际系统通过共享结构和前缀近似等机制控制开销。因此，仅实现排列编码并使用普通残差提升，没有实现完整有序提升。

### 2.5 对称树

对称树在同一深度的所有节点使用相同的特征条件。设深度为 $D$，第 $r$ 层判断为 $b_r(\mathbf x)\in\{0,1\}$，叶索引可以直接编码为：

$$
q(\mathbf x)=\sum_{r=0}^{D-1}2^rb_r(\mathbf x)
$$

树输出为对应叶值 $w_{q(\mathbf x)}$。深度 $D$ 对应最多 $2^D$ 个叶位置，其中一些可能没有训练样本。

该结构限制了每个局部节点独立选择规则的自由度，属于结构正则化，也有利于批量推理。代价是面对复杂局部关系时可能需要更多层或更多树，叶数组空间也随深度指数增长。

CatBoost 常使用对称树，但也支持其他生长策略，不能把所有配置下的模型都描述为对称结构。

### 2.6 类别组合与编码策略

多个类别字段的组合可能包含单字段无法表达的关系，例如城市与商品类目组合。组合统计可以增强表达能力，但会增加基数、稀疏性和存储成本，需要先验与容量约束。

低基数类别可能使用独热处理，高基数类别则可通过目标相关统计等方式处理。具体策略与任务、运行设备和配置有关；有序提升也不是所有设备与数据规模下的无条件默认值。

原生类别处理仍要求稳定的类别语义。训练和服务的字符串表示、缺失标记、字段顺序与组合规则必须保持一致。

### 2.7 训练配置与评估

| 配置 | 主要作用 | 注意事项 |
| --- | --- | --- |
| 迭代轮数与学习率 | 控制加法模型规模 | 结合验证集早停 |
| 树深度与生长策略 | 控制交互结构和资源成本 | 对称树的叶位置数随深度增长 |
| 叶值正则化 | 抑制叶输出过大 | 与损失缩放共同影响更新 |
| 类别统计与组合约束 | 控制类别表达 | 关注稀有类别和未知类别 |
| 提升模式 | 选择普通或有序机制 | 核对任务与设备支持 |
| 数据顺序配置 | 控制排列相关处理 | 时间任务仍需显式时间划分 |

随机排列不会让时间泄漏自动消失。预测未来行为时，特征、标签统计和数据划分都必须符合当时可见信息；用户分组任务也需要避免相关实体跨集合泄漏。

## 3. 模型实现

### 3.1 有序目标统计编码器

标准库实现面向二分类标签，使用固定先验，不从全体标签估计先验。训练转换使用排列前缀，预测转换使用训练类别统计。

```python
import math
import random
from collections import defaultdict


class OrderedTargetEncoder:
    def __init__(self, prior=0.5, strength=1.0, random_state=42):
        if not math.isfinite(prior) or not 0 <= prior <= 1:
            raise ValueError("prior must be in [0, 1]")
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be finite and positive")
        self.prior, self.strength = prior, strength
        self.random_state = random_state
        self.fitted = False

    def fit_transform(self, categories, labels):
        self.fitted = False
        categories, labels = list(categories), list(labels)
        if not categories or len(categories) != len(labels):
            raise ValueError("invalid input lengths")
        if any(value not in (0, 1) for value in labels):
            raise ValueError("binary labels are required")
        self.order = list(range(len(labels)))
        random.Random(self.random_state).shuffle(self.order)
        sums, counts = defaultdict(float), defaultdict(int)
        encoded = [0.0] * len(labels)
        for i in self.order:
            category = categories[i]
            encoded[i] = ((sums[category] + self.strength * self.prior) /
                          (counts[category] + self.strength))
            sums[category] += labels[i]
            counts[category] += 1
        self.sums, self.counts = dict(sums), dict(counts)
        self.fitted = True
        return encoded

    def transform(self, categories):
        if not self.fitted:
            raise RuntimeError("fit_transform must be called first")
        return [(self.sums.get(category, 0.0) + self.strength * self.prior) /
                (self.counts.get(category, 0) + self.strength)
                for category in categories]
```

当前标签只在编码生成之后写入统计；固定先验与固定排列下，改变一行标签不会改变该行自身的编码，但会改变同类别后续行的编码。

### 3.2 对称回归树

对称树通过每层全局选择一个规则，最小化所有子区域的残差平方和。代码只接受有限数值特征，用连续回归目标演示共享层规则和位索引；没有实现有序提升或完整类别组合搜索。

```python
class SymmetricRegressionTree:
    def __init__(self, depth=2, reg_lambda=1.0):
        if depth < 0 or not math.isfinite(reg_lambda) or reg_lambda < 0:
            raise ValueError("invalid tree parameters")
        self.depth, self.reg_lambda = depth, reg_lambda
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        X, y = [list(row) for row in X], list(y)
        if not X or not X[0] or len(X) != len(y):
            raise ValueError("invalid training shape")
        self.n_features = len(X[0])
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row)
               for row in X) or any(not math.isfinite(v) for v in y):
            raise ValueError("invalid values")
        indices, self.splits = [0] * len(X), []

        def score(groups, count):
            sums, sizes = [0.0] * count, [0] * count
            for group, value in zip(groups, y):
                sums[group] += value
                sizes[group] += 1
            return sum(s ** 2 / (n + self.reg_lambda) for s, n in zip(sums, sizes) if n)

        for level in range(self.depth):
            best_score = score(indices, 2 ** level)
            best = None
            for feature in range(self.n_features):
                values = sorted(set(row[feature] for row in X))
                for lower, upper in zip(values, values[1:]):
                    threshold = lower / 2.0 + upper / 2.0
                    if not lower <= threshold < upper:
                        threshold = lower
                    groups = [group | (int(row[feature] > threshold) << level)
                              for group, row in zip(indices, X)]
                    candidate = score(groups, 2 ** (level + 1))
                    if candidate > best_score + 1e-12:
                        best_score = candidate
                        best = feature, threshold, groups
            if best is None:
                break
            feature, threshold, indices = best
            self.splits.append((feature, threshold))

        count = 2 ** len(self.splits)
        sums, sizes = [0.0] * count, [0] * count
        for group, value in zip(indices, y):
            sums[group] += value
            sizes[group] += 1
        self.leaf_values = [s / (n + self.reg_lambda) if n else 0.0
                            for s, n in zip(sums, sizes)]
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        result = []
        for row in X:
            if len(row) != self.n_features or any(not math.isfinite(v) for v in row):
                raise ValueError("invalid prediction features")
            index = sum(int(row[feature] > threshold) << level
                        for level, (feature, threshold) in enumerate(self.splits))
            result.append(self.leaf_values[index])
        return result
```

评分来自平方损失加叶值平方惩罚后的最优目标，叶值为目标和除以样本数加正则系数。没有训练样本的叶位置返回零，这是示例的明确回退约定。

### 3.3 组件验证示例

```python
encoder = OrderedTargetEncoder(random_state=7)
training_values = encoder.fit_transform(["A", "B", "A", "A"], [1, 0, 0, 1])
print("ordered_encoding:", [round(value, 6) for value in training_values])
print("inference_encoding:", [round(value, 6) for value in encoder.transform(["A", "B", "new"])])

X = [[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0]]
tree = SymmetricRegressionTree(depth=2, reg_lambda=0.0)
tree.fit(X, [1.0, 1.0, 5.0, 5.0])
print("prediction:", tree.predict(X))
print("shared_splits:", tree.splits)
```

有序训练编码取决于排列，推理时 A、B 和未知类别的编码分别为 $0.625$、$0.25$ 和 $0.5$。树样例验证对称规则与叶输出，编码器与树是独立组件；将它们直接连接为普通监督模型，仍不等于实现完整有序提升。

### 3.4 计算成本与实现边界

单次有序编码的累计统计约为 $O(n)$，存储量与类别数成正比，多排列会增加成本。对称树深度为 $D$ 时，预测执行 $D$ 个共享条件并查找叶数组，叶数组大小最多为 $2^D$。

示例逐阈值扫描全部数据，并重复分配叶统计数组，不具备量化、候选缓存和并行实现的效率。完整训练还要处理多个统计特征、前缀支持模型、损失相关叶估计和模型序列化。

真实任务可以将原始类别字段交给 CatBoost 的类别接口，使用独立验证集选择轮数和容量。保存模型之外，还需保存字段顺序、类别表示约定和数据版本，避免服务输入与训练不一致。

### 3.5 与其他树提升系统的关系

| 系统 | 重点机制 | 需要区分的配置 |
| --- | --- | --- |
| XGBoost | 二阶目标、正则化与高效分裂搜索 | 树方法、生长策略和数据表示 |
| LightGBM | 直方图、叶子优先和稀疏优化 | 采样策略、叶预算与类别约束 |
| CatBoost | 有序类别统计、有序提升与对称结构 | 提升模式、生长策略与类别统计配置 |

三个系统均具有多种训练策略，不能用“一个支持类别、其他都不支持”概括区别。比较时应固定数据划分和资源预算，并报告训练时间、推理成本、泛化指标及校准状态。

## 4. 总结

CatBoost 通过有序目标统计控制类别编码的信息来源，通过有序提升缓解梯度计算中的预测偏移，对称树则提供共享层规则与紧凑叶索引。三者分别作用于特征、训练与模型结构。

类别基数、先验、排列顺序和部署未知类别都会影响实际结果。有序机制降低特定泄漏风险，但不能替代正确的数据划分、时间约束和独立评估。

## 5. 参考文献

[1] Prokhorenkova L, Gusev G, Vorobev A, Dorogush A V, Gulin A. CatBoost: Unbiased Boosting with Categorical Features. NeurIPS, 2018, 31.

[2] Dorogush A V, Ershov V, Gulin A. CatBoost: Gradient Boosting with Categorical Features Support. arXiv:1810.11363, 2018.

[3] Friedman J H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001, 29(5): 1189–1232.
