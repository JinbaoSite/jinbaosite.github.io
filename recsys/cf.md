# 协同过滤算法

## 00 协同过滤算法基本原理

**协同过滤：通过群体的行为来找到某种相似性（用户之间的相似性或者物品之间的相似性），通过相似性来为用户做决策和推荐**。

基于用户的协同过滤算法（UserCF）的基本原理：给用户推荐和他兴趣相似的其他用户喜欢的物品。

基于物品的协同过滤算法（ItemCF）的基本原理：给用户推荐和他之前喜欢的物品相似的物品。

---

## 01 基于用户的协同过滤算法（UserCF）

### 1.1 UserCF的算法流程

1. 根据用户的历史行为，获取用户之间的共现矩阵$C[u][v]= \vert N(u) \bigcap N(v) \vert$
2. 根据共现矩阵得到用户与用户之间的相似矩阵$W[u][v]= \frac{C[u][v]}{\vert N(u) \vert * \vert N(v) \vert}$
3. 根据用户之间的相似矩阵找到topK个相似的用户作为和目标用户兴趣相似的用户集合，并找到这个集合中属于这个集合中用户喜欢但目标用户没有出现的物品列表，对物品列表中的物品计算兴趣程度$p(u, i) = \sum {W[u][v]}$，按照兴趣程度排序推荐给目标用户

首先建立与物品发生行为的用户列表，令$C[u][v]$表示用户u和用户v相同物品的个数，$N[u]$表示用户u的物品数，遍历每个物品的用户列表，将用户列表中的两两用户对应的$C[u][v]$加1，同时计算$N[u]$，最后计算用户之间的相似矩阵$W[u][v]$；推荐物品时，先从相似矩阵中得到目标用户topK个相似用户，然后计算属于这topK个用户但不属于目标用户的所有物品的兴趣程度值即可。

### 1.2 UserCF的代码实现

```python
import math
def UserSimilarity(train):
    item_users = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(user)
    C = dict()
    N = dict()
    for item, users in item_users.items():
        for u in users:
            N[user] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W
def Recommand(user, train, W, K):
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items, 
                         key=lambda x: x[1], reverse=True)[:K]:
        for i, rvi in train[v].items:
            if i in interacted_items:
                continue
            rank[i] += wuv
    return rank
```

---

## 02 基于物品的协同过滤算法（ItemCF）

### 2.1 ItemCF的算法流程

1. 根据用户的历史行为，获取物品之间的共现矩阵$C[i][j]= \vert N(i) \bigcap N(j) \vert$
2. 根据共现矩阵得到物品之间的相似矩阵$W[i][j]= \frac{C[i][j]}{\vert N(i) \vert * \vert N(j) \vert}$
3. 根据相似矩阵找到目标用户发生行为的物品列表中每个物品对应的topK个物品，并计算目标用户对每个物品对应的topK个物品的兴趣程度$p(u, i)= \sum W[i][j]$，按照兴趣程度排序推荐给目标用户

### 2.2 ItemCF的代码实现

```python
import math
def ItemSimilarity(train):
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wij in sorted(W[i].items(), 
                             key=lambda x: x[1], reverse=True)[:K]:
            if j in ru:
                continue
            rank[j] += wij
    return rank
```

---

## 03 协同过滤算法的改进

### 3.1 相似度的改进IUF

用户（物品）相似度的计算公式为
$$
w_{uv} = \frac{\vert N(u) \bigcap N(v) \vert}{\sqrt{\vert N(u) \vert \vert N(v) \vert}}
$$
相似度的计算实际上更大程度会受活跃用户（物品）的影响比较大，所以计算相似度应该活跃用户（物品）对物品的相似度的贡献应该小于不活跃的用户（物品），现引入IUF(Inverse User Frequence)，即用户（物品）活跃度对数的倒数的参数，来修正相似度，公式如下：
$$
w_{uv} = \frac{\sum_{i \in N(u) \bigcap N(v)} \frac{1}{log(1+ \vert N(i) \vert)}}{\sqrt{ \vert N(u) \vert \vert N(v) \vert}}
$$

### 3.2 相似度的归一化Norm

一般来说，热门物品相似度比较大，所以推荐的物品大多数是推荐比较热门里的物品，因此，推荐的覆盖率就比较低，如果进行相似度的归一化，就可以提供推荐的覆盖率

相似度归一化公式为
$$
w_{ij} = \frac{w_{ij}}{max_{j} w_{ij}}
$$

## 04 ItemCF和UserCF的对比

| 指标 | UserCF | ItemCF |
| :-- | :-- | :-- |
| 性能 | 适用于用户较少的场合，如果用户很多，计算用户相似度矩阵代价很大 | 适用于物品数明显小于用户数的场合，如果物品很多，计算物品相似度矩阵代价很大 |
| 领域 | 时效性较强，用户个性化兴趣不太明显的领域 | 长尾物品丰富，用户个性化需求强烈的领域 |
| 实时性 | 用户有新行为，不一定造成推荐结果的立即变化 | 用户有新行为，一定会造成推荐结果的实时变化 |
| 冷启动 | 在新用户对很少的物品产生行为后，不能立即对他进行个性化推荐，因为用户相似度矩阵是每隔一段时间离线计算的 | 新用户只要对一个物品产生行为，就可以给他推荐和该物品相关的其他物品 |

## 05 参考资料

[1] [协同过滤推荐算法总结](https://zhuanlan.zhihu.com/p/25069367)

 [2] [最小推荐系统：协同过滤（Collaborative Filtering)](https://zhuanlan.zhihu.com/p/149152447)

 
