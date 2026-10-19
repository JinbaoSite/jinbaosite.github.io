# 基于邻域的协同过滤

基于邻域的协同过滤从用户与物品的交互记录中寻找相似关系，再利用相似用户或相似物品生成推荐。它不需要物品文本、图片等内容特征，核心信息来自群体行为。

基于用户的协同过滤通过相似用户发现候选，基于物品的协同过滤通过历史物品扩展候选。两种方法都可以用于隐式反馈推荐；用于显式评分预测时，还需要考虑评分尺度和均值偏移。

## 1. 背景

假设用户甲看过电影 A、B，用户乙看过 A、B、C。两人共同看过的电影可以作为兴趣相似的证据，电影 C 因而成为向甲推荐的候选。这就是基于用户的协同过滤。

换一个视角，如果许多用户同时看过 A 和 C，就可以建立 A 与 C 的关联。甲看过 A，即使不实时查找相似用户，也可以通过 A 的相似物品表找到 C。这就是基于物品的协同过滤。

协同过滤中的“相似”来自行为共现，不一定代表内容相似。经常一起购买的咖啡机和咖啡豆也可能高度相关；反过来，两件内容相似但没有共同交互记录的新品，可能无法建立可靠的协同关系。

### 1.1 显式反馈与隐式反馈

| 类型 | 典型数据 | 建模时的注意事项 |
| --- | --- | --- |
| 显式反馈 | 星级、评分 | 不同用户的评分尺度可能不同，需要考虑均值偏移 |
| 隐式反馈 | 点击、收藏、购买、播放 | 未交互不代表不喜欢，交互也可能受到曝光和流行度影响 |

二元反馈建模将同一用户对同一物品的多次正向行为合并为一次二元交互。没有记录的位置只表示未知，不作为明确负反馈。若需要区分点击和购买，或加入时间衰减，应额外定义行为权重。

### 1.2 符号约定

设用户集合为 $\mathcal U$，物品集合为 $\mathcal I$，二元交互矩阵为：

$$
x_{ui}=\begin{cases}
1, & \text{用户 }u\text{ 与物品 }i\text{ 有正向交互}\\
0, & \text{没有观测到对应交互}
\end{cases}
$$

用 $I_u=\{i:x_{ui}=1\}$ 表示用户 $u$ 交互过的物品集合，用 $U_i=\{u:x_{ui}=1\}$ 表示物品 $i$ 的交互用户集合。$K$ 表示参与计算的邻居数，$N$ 表示最终返回的推荐数量，两者不是同一个参数。

## 2. 算法原理

### 2.1 基于用户的协同过滤

**（1）用户共现与相似度**

用户 $u$ 和 $v$ 的共现次数为：

$$
c_{uv}=|I_u\cap I_v|
$$

直接使用共现次数会偏向交互数量多的用户。二元交互向量的余弦相似度对用户活跃度进行归一化：

$$
w_{uv}=\frac{|I_u\cap I_v|}{\sqrt{|I_u||I_v|}}
$$

分母必须包含平方根。例如，甲的历史为 $\{A,B\}$，乙的历史为 $\{A,B,C\}$，相似度为 $2/\sqrt{6}\approx0.816$。空历史用户没有可计算的邻域，不能通过这个公式直接生成个性化推荐。

为了避免枚举所有用户对，可以先建立“物品到用户”的倒排表。对每件物品，只累加交互过它的用户对，再计算这些用户对的相似度。

**（2）候选生成与打分**

设 $S_K(u)$ 是与 $u$ 最相似的 $K$ 个其他用户。对用户尚未交互的候选物品 $i$，计算：

$$
s(u,i)=\sum_{v\in S_K(u)\cap U_i}w_{uv}x_{vi}
$$

二元反馈下，参与求和的 $x_{vi}$ 都等于 1。一个物品如果得到多个高相似度邻居支持，就会获得更高分数。最后过滤 $I_u$ 中的物品，按分数返回前 $N$ 个结果。

邻域得分用于相对排序，不是经过校准的点击概率。邻域无法覆盖的物品不会进入候选集合，需要由其他召回策略补充。

### 2.2 基于物品的协同过滤

**（1）物品共现与相似度**

物品 $i$ 和 $j$ 的共现次数以及二元余弦相似度为：

$$
c_{ij}=|U_i\cap U_j|
$$

$$
w_{ij}=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}
$$

计算时遍历每个用户的去重历史，将其中不同物品组成的物品对累加到共现表。相似物品表建立后，可为每个物品保留前 $K$ 个邻居。

**（2）从历史物品扩展候选**

设 $S_K(j)$ 是历史物品 $j$ 的前 $K$ 个相似物品，$a_{uj}$ 是该历史行为的权重：

$$
s(u,i)=\sum_{\substack{j\in I_u\\i\in S_K(j)}}a_{uj}w_{ji}
$$

基础实现取 $a_{uj}=1$。这表示遍历每个历史物品的邻居表，将到达同一候选的分数相加，再过滤已交互物品。

需要注意，逐行保留前 $K$ 个邻居以后，邻接关系可能不再对称：$i$ 是 $j$ 的近邻，不意味着 $j$ 一定进入 $i$ 的前 $K$ 个近邻。实现和公式必须采用相同的扩展方向。

### 2.3 一个完整的计算示例

考虑以下交互记录：

| 用户 | 历史物品 |
| --- | --- |
| 甲 | A、B |
| 乙 | A、B、C |
| 丙 | B、C |
| 丁 | D |

对于用户甲，用户乙与甲的相似度为 $2/\sqrt{6}$，用户丙与甲的相似度为 $1/2$。若保留两个邻居，只有 C 是未交互候选：

$$
s_{\mathrm{UserCF}}(\text{甲},C)=\frac{2}{\sqrt{6}}+\frac12\approx1.316
$$

从物品视角看，A、B、C 分别有 2、3、2 个交互用户。A 与 C 有 1 个共同用户，B 与 C 有 2 个共同用户，因此：

$$
w_{AC}=\frac12,\qquad w_{BC}=\frac{2}{\sqrt6}
$$

当两个历史物品的邻居表都包含 C 时，甲对 C 的物品邻域得分同样约为 1.316。这个数值相同只是当前数据的结果，两种算法一般不会产生相同的分数或排序。D 没有共现邻居，不会被这两条路径召回。

### 2.4 热门与活跃因素的修正

**（1）降低热门物品对用户相似度的贡献**

两个用户都点击过一个人人都看过的热门物品，不足以证明他们具有独特的共同兴趣。可以对共同物品按流行度降权：

$$
w_{uv}=\frac{\displaystyle\sum_{i\in I_u\cap I_v}\frac{1}{\log(1+|U_i|)}}{\sqrt{|I_u||I_v|}}
$$

它改变的是共同物品对用户相似度的贡献，不是直接给最终推荐结果减去一个热门惩罚。

**（2）降低活跃用户对物品相似度的贡献**

一个用户如果交互过大量物品，会产生很多物品对。可以降低这类用户对物品共现的贡献，通常称为逆用户频率修正：

$$
w_{ij}=\frac{\displaystyle\sum_{u\in U_i\cap U_j}\frac{1}{\log(1+|I_u|)}}{\sqrt{|U_i||U_j|}}
$$

两种修正分别处理热门物品和活跃用户，不能混为同一个公式。它们属于加权共现分数；分母仍使用未加权计数，因此不能再直接解释为标准二元余弦相似度，也不应假设分数一定落在 $[0,1]$。

**（3）抑制小样本下的不可靠相似度**

只有一个共同交互的物品对，可能偶然获得很高的归一化分数。可设置最小共现次数，或使用收缩系数：

$$
\widetilde w_{ij}=\frac{c_{ij}}{c_{ij}+\lambda}w_{ij},\qquad\lambda\ge0
$$

其中 $c_{ij}$ 是原始共同用户数。最小共现门槛过滤证据不足的邻接关系，收缩参数降低小样本相似度的权重。

### 2.5 行为权重与时间因素

推荐打分时可以让购买、收藏和点击拥有不同权重，并对较旧行为进行衰减：

$$
a_{uj}=b_{uj}\exp\bigl(-\gamma(t-t_{uj})\bigr)
$$

其中 $b_{uj}$ 是业务行为权重，$t_{uj}$ 是行为发生时间，$\gamma$ 控制衰减速度。该公式属于在邻域方法上的工程扩展，不意味着所有推荐场景都应快速遗忘长期兴趣。

还可以按行将相似度除以该行最大值，使不同历史物品的最大贡献接近。但这种处理会改变跨行分数尺度，并可能放大小样本物品的影响，不能保证一定提高覆盖率或推荐质量。

### 2.6 显式评分预测

如果目标是预测评分，而不是对隐式行为生成候选，需要考虑用户的评分习惯。常见的用户邻域预测形式为：

$$
\widehat r_{ui}=\overline r_u+
\frac{\displaystyle\sum_{v\in S_K(u)\cap U_i}w_{uv}(r_{vi}-\overline r_v)}
{\displaystyle\sum_{v\in S_K(u)\cap U_i}|w_{uv}|}
$$

$\overline r_u$ 和 $\overline r_v$ 是训练集中各用户的平均评分。相似度可以使用基于共同评分的相关系数；物品邻域也可以使用减去用户均值后的调整余弦，以缓解用户评分尺度差异 [1]。分母为零时，需要回退到用户均值或全局基线。

评分缺失不能作为真实的零分参与均值或相关系数计算。隐式反馈推荐的实现采用二元交互和邻域得分，不使用评分预测公式。

## 3. 算法实现

### 3.1 相似度构建

实现只依赖 Python 标准库。输入为“用户 ID 到物品 ID 集合”的映射，示例约定 ID 为字符串，以便在分数相同时使用 ID 保证排序稳定。共现构建使用稀疏字典，不分配完整的用户或物品方阵。

```python
import math
from collections import defaultdict
from itertools import combinations


def build_similarity(train, kind="item", discount=False,
                     min_common=1, shrinkage=0.0):
    if kind not in {"user", "item"}:
        raise ValueError("kind must be 'user' or 'item'")
    if min_common < 1 or shrinkage < 0:
        raise ValueError("invalid similarity parameters")

    history = {u: set(items) for u, items in train.items()}
    item_users = defaultdict(set)
    for user, items in history.items():
        for item in items:
            item_users[item].add(user)

    if kind == "user":
        groups = item_users.values()
        degree = {u: len(items) for u, items in history.items()}
    else:
        groups = history.values()
        degree = {i: len(users) for i, users in item_users.items()}

    common = defaultdict(int)
    weighted = defaultdict(float)
    for group in groups:
        if len(group) < 2:
            continue
        contribution = 1.0 / math.log1p(len(group)) if discount else 1.0
        for left, right in combinations(sorted(group), 2):
            pair = (left, right)
            common[pair] += 1
            weighted[pair] += contribution

    similarity = {entity: {} for entity in degree}
    for (left, right), count in common.items():
        if count < min_common:
            continue
        score = weighted[left, right] / math.sqrt(
            degree[left] * degree[right]
        )
        score *= count / (count + shrinkage)
        similarity[left][right] = score
        similarity[right][left] = score
    return similarity


def build_neighbors(similarity, k):
    if k < 1:
        raise ValueError("k must be positive")
    return {
        entity: sorted(row.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        for entity, row in similarity.items()
    }
```

`kind="user"` 时，共现组是一件物品的用户集合；`kind="item"` 时，共现组是一个用户的物品集合。`discount=True` 对当前共现组按规模降权，分别对应热门物品修正和活跃用户修正。

### 3.2 推荐与示例

```python
def recommend(train, user, neighbors, kind="item", n=10):
    if kind not in {"user", "item"}:
        raise ValueError("kind must be 'user' or 'item'")
    if n < 1:
        raise ValueError("n must be positive")
    seen = set(train.get(user, ()))
    if not seen:
        return []

    scores = defaultdict(float)
    if kind == "user":
        for other, weight in neighbors.get(user, ()):
            for item in sorted(set(train.get(other, ())) - seen):
                scores[item] += weight
    else:
        for item in sorted(seen):
            for candidate, weight in neighbors.get(item, ()):
                if candidate not in seen:
                    scores[candidate] += weight

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:n]


train = {
    "u1": {"A", "B"},
    "u2": {"A", "B", "C"},
    "u3": {"B", "C"},
    "u4": {"D"},
}

for kind in ("user", "item"):
    similarity = build_similarity(train, kind=kind)
    neighbors = build_neighbors(similarity, k=2)
    result = recommend(train, "u1", neighbors, kind=kind, n=3)
    print(kind, [(item, round(score, 3)) for item, score in result])
```

输出为：

```text
user [('C', 1.316)]
item [('C', 1.316)]
```

实现会过滤已交互物品，合并多个邻居对同一候选的贡献，并对空历史、无共现邻居和未知用户返回空列表。线上系统可接入热门、新品或内容召回作为兜底，而不是在协同分数中混入未说明的默认值。

### 3.3 计算成本与服务更新

用户邻域的共现构建成本取决于每件物品的交互用户数，物品邻域则取决于每个用户的历史长度：

$$
\operatorname{Cost}_{\mathrm{UserCF}}=O\left(\sum_{i\in\mathcal I}|U_i|^2\right)
$$

$$
\operatorname{Cost}_{\mathrm{ItemCF}}=O\left(\sum_{u\in\mathcal U}|I_u|^2\right)
$$

因此，选择方法不能只比较用户数和物品数。极热门物品会显著增加用户对枚举量，超活跃用户也会显著增加物品对枚举量。生产环境通常限制时间窗口、截断历史、采样大共现组，并使用分布式聚合构建邻居表。

示例先保存所有非零共现对，再对每行完整排序取前 $K$。邻居表的存储量可降到实体数乘以 $K$，但构建过程仍可能占用较大内存；大规模实现还需采用堆、分块或流式聚合控制中间结果。

物品邻域在线扩展至多访问 $|I_u|K$ 个邻居条目，聚合后还要筛选前 $N$ 个结果。新增行为可以立即成为新的扩展种子，前提是对应物品已有邻居表，且服务读取了最新历史。它不会自动更新共现统计，也不保证最终列表一定变化。

### 3.4 离线评估与常见问题

按时间划分训练、验证和测试数据。相似度、共现计数、热门统计及行为权重都只能由当时可见的历史计算，不能用未来交互构建邻居表后再评价过去的推荐。

推荐任务可关注 Recall@N、NDCG@N、覆盖率以及不同活跃度用户的表现；显式评分预测则可关注 MAE、RMSE。使用全量候选还是采样负例会显著影响指标，应固定候选协议，避免直接比较不同评估设置。

首次交互推荐需要过滤已交互物品；对于音乐复听、商品复购等场景，可能需要保留部分历史物品，并单独定义重复推荐规则。新用户没有邻域证据，新物品没有共现记录，两类冷启动都需要其他信息或策略补充。

## 4. 总结

基于用户的协同过滤利用相似用户扩展候选，基于物品的协同过滤利用历史物品的邻居表扩展候选。两者都依赖交互共现，基本流程是构建相似度、截取邻居、累加候选得分并过滤不合适的物品。

可靠实现需要明确反馈类型、相似度定义和邻居方向，同时处理热门偏置、小样本共现、时间泄漏与冷启动。邻域方法结构直观、推荐理由容易追溯，既可作为推荐系统的基础基线，也可作为多路召回中的一条候选来源。

## 5. 参考文献

[1] Sarwar B, Karypis G, Konstan J, Riedl J. Item-based Collaborative Filtering Recommendation Algorithms. WWW, 2001: 285–295.
