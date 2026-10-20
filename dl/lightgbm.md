# LightGBM：直方图与叶子优先的高效树提升

LightGBM 是基于梯度提升决策树的学习系统，通过直方图分桶、叶子优先生长和稀疏特征优化降低训练成本。梯度单边采样与互斥特征捆绑进一步减少参与计算的样本或特征，适合大规模表格数据上的分类、回归和排序任务。

它与 [XGBoost](/dl/xgboost/)共享加法树模型和梯度统计优化的基础，差异更多体现在树的生长调度、统计构建和数据表示上。理解这些机制，需要区分目标函数、近似搜索和工程实现三个层面。

## 1. 背景

[梯度提升树](/dl/gradient_boosting/)逐轮添加树来改善整体预测。数据量和特征维度增加后，训练成本往往集中在特征扫描、候选阈值搜索和统计量聚合上。

直方图方法把连续特征压缩为有限桶，叶子优先策略把有限叶预算用于当前收益最大的区域。两者分别优化候选搜索和树容量分配，但也会引入离散近似与局部过拟合风险。

模型形式仍为：

$$
F_M(\mathbf x)=F_0(\mathbf x)+\eta\sum_{m=1}^{M}f_m(\mathbf x)
$$

每轮计算当前预测对应的一阶、二阶统计量，用它们构建新增树，再以学习率 $\eta$ 缩放更新。

## 2. 模型原理

### 2.1 二阶叶目标

设损失的一阶导数为 $g_i$，二阶导数为 $h_i$。叶区域 $I_j$ 的统计量为：

$$
G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i
$$

使用叶权重平方惩罚时，局部二阶目标为：

$$
\widetilde{\mathcal L}_j=G_jw_j+\frac12(H_j+\lambda)w_j^2
$$

因此：

$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

二元划分的近似收益为：

$$
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{G_P^2}{H_P+\lambda}
\right]
$$

这里固定使用带 $1/2$ 的数学约定。不同实现可能对分裂评分作统一倍数缩放，与最小分裂增益等参数比较时需要核对定义。

### 2.2 直方图分桶

设特征 $x_j$ 被映射为桶编号 $b_j(x_j)\in\{0,\ldots,K_j-1\}$。节点中每个桶保存：

$$
G_{jk}=\sum_{i:b_j(x_{ij})=k}g_i,\qquad
H_{jk}=\sum_{i:b_j(x_{ij})=k}h_i
$$

扫描桶的前缀和即可得到候选划分的左右统计。对于单个节点，构建直方图需要读取样本，随后搜索成本约为每特征 $O(K_j)$，而不是逐个枚举原始取值并重新扫描样本。

分桶边界由训练数据确定，验证与预测复用相同边界。桶数越少，内存和搜索成本越低，但可能把有效阈值合并到同一桶。真实系统还需要处理缺失桶、稀疏存储和分桶构建的采样策略。

### 2.3 直方图相减

父节点直方图等于左右子节点直方图之和：

$$
\operatorname{Hist}(P)=\operatorname{Hist}(L)+\operatorname{Hist}(R)
$$

因此只需直接构建其中一个子节点的直方图，再用父节点减去它得到另一个子节点。通常优先构建样本较少的一侧，减少数据扫描量。

相减成立的前提是使用相同特征桶定义，并保持统计量与样本权重一致。数值舍入可能产生微小误差，也需要在实现中处理。

### 2.4 叶子优先生长

层级优先策略按深度扩展节点，叶子优先策略则维护所有可分叶子的最佳候选，每次选择收益最大的叶子继续分裂。

例如，根节点划分后，左叶最佳收益为 8，右叶为 2；先划分左叶后，新叶中又有收益为 6 的候选，叶子优先会继续选择收益 6 的叶子，而不因右叶更浅就先扩展右叶。

相同叶预算下，这种贪心分配通常能更充分降低当前训练目标，但不能保证全局最优或更好的泛化。树可能明显不平衡，需要同时约束叶数、深度、最小叶样本数和最小海森和。

设置最大深度后，模型仍可以按叶子优先调度，只是超过深度上限的叶子不再扩展。`num_leaves` 与 `max_depth` 控制不同约束，不能简单互相替代。

### 2.5 梯度单边采样

梯度单边采样根据梯度绝对值保留高梯度样本，并从其余样本中随机抽取一部分。高梯度表示当前损失对预测更敏感，并不等同于所有任务中的“大残差样本”或“正样本”。

定义高梯度集合占全部样本的比例为 $a$，从低梯度集合中抽取占全部样本比例为 $b$ 的样本。低梯度样本的入选概率为：

$$
q=\frac{b}{1-a}
$$

对入选低梯度样本乘以逆概率权重：

$$
\omega=\frac1q=\frac{1-a}{b}
$$

这个表达式中的 $b$ 是相对全体样本的比例；如果改定义为低梯度集合内部的抽取比例，补偿权重应改为 $1/b$。

在固定梯度与固定候选区域下，逆概率加权可保持线性统计量的期望。增益包含平方与除法，搜索还会选择最大值，因此不能据此声称分裂增益或最终树无偏。二阶目标下还需要一致处理海森统计，不能只修正梯度和而忽略分母。

梯度采样是一种可选策略，不是所有 LightGBM 训练都默认执行。不同版本与配置的预热和采样调度需要按实际设置核对。

### 2.6 互斥特征捆绑

高维稀疏数据中的某些特征不会同时非零，例如同一类别字段的不同独热指示。互斥特征捆绑把这些特征的非零桶映射到不同编号区间，用一个捆绑表示减少直方图构建工作。

假设特征 A 的非零桶为 1、2，特征 B 的非零桶为 1、2、3。可以把 B 的桶偏移到 3、4、5，并保留共享零状态。严格互斥时，一行只激活其中一段，因而可以区分非零值来自哪个特征。

这不是对多个特征直接求和，也不意味着模型丢弃了原特征的划分语义。允许少量冲突的近似捆绑需要处理信息碰撞，收益取决于稀疏程度和冲突控制；密集且频繁共同出现的特征不适合这种压缩。

### 2.7 类别特征与缺失值

类别特征可以在节点内聚合梯度统计，并依据目标相关统计排序后搜索类别子集划分，减少枚举所有子集的成本。实际实现还会施加类别平滑、正则化和小类别处理，不能把简单统计排序当作所有分类目标下的无条件全局最优保证。

类别编码需要稳定，训练和服务使用一致映射。负值或未知编码如何处理，取决于输入接口约定。原生类别支持不意味着可以任意改变类别 ID 而不保存映射。

缺失值通过专门的统计和默认分支处理。普通零、稀疏未存储项和缺失标记的含义受配置影响，尤其需要核对是否将零视为缺失。

### 2.8 参数与评估

| 参数类别 | 代表参数 | 作用 |
| --- | --- | --- |
| 树容量 | `num_leaves`、`max_depth` | 限制叶数与路径深度 |
| 叶稳定性 | `min_data_in_leaf`、`min_sum_hessian_in_leaf` | 控制样本证据和曲率统计 |
| 分桶 | `max_bin` | 权衡阈值精度与资源开销 |
| 正则化 | `lambda_l1`、`lambda_l2`、`min_gain_to_split` | 控制叶输出和分裂收益 |
| 更新步长 | `learning_rate`、训练轮数 | 控制累加模型规模 |
| 随机化 | 特征与数据采样相关参数 | 减少成本并改变偏差方差权衡 |

早停通过独立验证集选择最佳轮数，测试集留作最终评价。排序任务还需要正确组织查询分组，避免将同一查询的文档拆到不恰当的集合。

## 3. 模型实现

### 3.1 分桶与二阶直方图树

标准库实现展示固定分桶和叶子优先的平方误差回归。桶边界按训练取值的近似等频位置选择；实现不包含梯度采样、互斥捆绑、直方图相减、原生类别或缺失处理，和完整软件系统的能力边界明确区分。

```python
import bisect
import heapq
import math


class HistogramLeafTree:
    def __init__(self, num_leaves=4, max_depth=4,
                 min_samples_leaf=2, reg_lambda=1.0):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda

    def fit(self, bins, gradients, hessians, bin_counts):
        self.bins = bins
        self.gradients, self.hessians = gradients, hessians
        self.bin_counts = bin_counts
        self.nodes = []
        pending = []

        def add_node(indices, depth):
            total_g = sum(gradients[i] for i in indices)
            total_h = sum(hessians[i] for i in indices)
            node = {"value": -total_g / (total_h + self.reg_lambda)}
            node_id = len(self.nodes)
            self.nodes.append(node)
            if depth < self.max_depth:
                candidate = self._best_split(indices)
                if candidate is not None:
                    gain, feature, threshold = candidate
                    heapq.heappush(pending, (-gain, node_id, feature, threshold, indices, depth))
            return node_id

        add_node(list(range(len(bins))), 0)
        leaves = 1
        while pending and leaves < self.num_leaves:
            _, node_id, feature, threshold, indices, depth = heapq.heappop(pending)
            left = [i for i in indices if bins[i][feature] <= threshold]
            right = [i for i in indices if bins[i][feature] > threshold]
            left_id = add_node(left, depth + 1)
            right_id = add_node(right, depth + 1)
            self.nodes[node_id].update(feature=feature, threshold=threshold,
                                       left=left_id, right=right_id)
            leaves += 1
        self.leaf_count = leaves
        # 预测只需要节点规则，不保留训练数据引用。
        del self.bins, self.gradients, self.hessians
        return self

    def _best_split(self, indices):
        total_g = sum(self.gradients[i] for i in indices)
        total_h = sum(self.hessians[i] for i in indices)
        parent = total_g ** 2 / (total_h + self.reg_lambda)
        best = None
        for feature, count in enumerate(self.bin_counts):
            gs, hs, sizes = [0.0] * count, [0.0] * count, [0] * count
            for i in indices:
                bucket = self.bins[i][feature]
                gs[bucket] += self.gradients[i]
                hs[bucket] += self.hessians[i]
                sizes[bucket] += 1
            gl, hl, nl = 0.0, 0.0, 0
            for threshold in range(count - 1):
                gl, hl, nl = gl + gs[threshold], hl + hs[threshold], nl + sizes[threshold]
                if min(nl, len(indices) - nl) < self.min_samples_leaf:
                    continue
                gain = 0.5 * (gl ** 2 / (hl + self.reg_lambda) +
                              (total_g - gl) ** 2 / (total_h - hl + self.reg_lambda) - parent)
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, feature, threshold)
        return best

    def predict(self, bins):
        result = []
        for row in bins:
            node = self.nodes[0]
            while "feature" in node:
                branch = "left" if row[node["feature"]] <= node["threshold"] else "right"
                node = self.nodes[node[branch]]
            result.append(node["value"])
        return result
```

堆中的候选覆盖所有当前可分叶子，负增益让最小堆每次弹出收益最大的候选。达到叶数限制或没有正收益候选时停止。

### 3.2 直方图提升回归

```python
class HistogramBoostingRegressor:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_bins=16,
                 num_leaves=4, max_depth=4, min_samples_leaf=2, reg_lambda=1.0):
        if min(n_estimators, num_leaves, min_samples_leaf) < 1 or max_bins < 2 or max_depth < 0:
            raise ValueError("invalid tree parameters")
        if not 0 < learning_rate <= 1 or not math.isfinite(reg_lambda) or reg_lambda <= 0:
            raise ValueError("invalid optimization parameters")
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_bins, self.num_leaves, self.max_depth = max_bins, num_leaves, max_depth
        self.min_samples_leaf, self.reg_lambda = min_samples_leaf, reg_lambda
        self.fitted = False

    def _transform(self, X):
        X = [list(row) for row in X]
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row) for row in X):
            raise ValueError("invalid features")
        return [[bisect.bisect_left(cuts, value) for cuts, value in zip(self.cuts, row)] for row in X]

    def fit(self, X, y):
        self.fitted = False
        X, y = [list(row) for row in X], list(y)
        if not X or not X[0] or len(X) != len(y) or any(not math.isfinite(v) for v in y):
            raise ValueError("invalid training data")
        self.n_features = len(X[0])
        if any(len(row) != self.n_features or any(not math.isfinite(v) for v in row) for row in X):
            raise ValueError("invalid features")
        self.cuts = []
        for feature in range(self.n_features):
            values = sorted(row[feature] for row in X)
            cuts = {values[min(len(values) - 1, k * len(values) // self.max_bins)]
                    for k in range(1, self.max_bins)}
            self.cuts.append(sorted(v for v in cuts if v < values[-1]))
        bins = self._transform(X)
        counts = [len(cuts) + 1 for cuts in self.cuts]
        self.initial_score = sum(y) / len(y)
        prediction = [self.initial_score] * len(y)
        self.trees = []
        for _ in range(self.n_estimators):
            gradients = [p - target for p, target in zip(prediction, y)]
            tree = HistogramLeafTree(self.num_leaves, self.max_depth,
                                     self.min_samples_leaf, self.reg_lambda)
            tree.fit(bins, gradients, [1.0] * len(y), counts)
            prediction = [p + self.learning_rate * update
                          for p, update in zip(prediction, tree.predict(bins))]
            self.trees.append(tree)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        bins = self._transform(X)
        result = [self.initial_score] * len(bins)
        for tree in self.trees:
            result = [p + self.learning_rate * update
                      for p, update in zip(result, tree.predict(bins))]
        return result
```

训练时构建一次分桶边界，所有树及预测输入复用这些边界。平方损失的海森为 1，示例因此只需最小叶样本数；扩展到分类时还需考虑小海森和的稳定性。

### 3.3 训练示例与叶预算核对

```python
X = [[float(i)] for i in range(80)]
y = [1.0 if i < 25 else 5.0 if i < 55 else 2.0 for i in range(80)]
model = HistogramBoostingRegressor(n_estimators=50, num_leaves=3,
                                   max_depth=3, max_bins=16)
model.fit(X, y)
prediction = model.predict(X)
mse = sum((p - target) ** 2 for p, target in zip(prediction, y)) / len(y)
print("training_mse:", round(mse, 4))
print("max_leaves:", max(tree.leaf_count for tree in model.trees))
print("prediction_count:", len(prediction))
```

例子用于验证分桶、叶预算和逐轮拟合。分桶可能把真实跳变位置与相邻样本合并，因此不要求误差为零；独立测试与调参仍需要另行划分数据。

### 3.4 复杂度与工程边界

设节点有 $m$ 个样本、$d$ 个特征，每个特征最多 $K$ 桶。构建直方图约需 $O(dm)$，搜索候选约需 $O(dK)$。样例对子节点重新构建直方图，未使用相减优化；总训练成本还需累加各树所有节点的工作。

分桶数据约占 $O(nd)$ 空间，压缩整数表示能够减少常数开销。预测先分桶，再沿树路径查找叶值；实际系统也可以保存等价的原始阈值规则。

样例使用普通列表、固定轮数与回归目标，不能据此推断完整 LightGBM 的并行效率或分类效果。完整训练还需验证早停、采样策略、类别处理和模型序列化配置。

## 4. 总结

LightGBM 通过有限桶统计降低阈值搜索成本，通过叶子优先调度把叶预算分配给当前更有收益的区域。梯度单边采样减少样本计算，互斥特征捆绑利用稀疏结构减少特征统计工作。

这些优化各有适用条件。分桶精度、树容量、采样补偿、缺失语义与独立验证共同决定最终效果，不能只根据模型名称推断速度或精度优劣。

## 5. 参考文献

[1] Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017, 30.

[2] Friedman J H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001, 29(5): 1189–1232.

[3] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016: 785–794.
