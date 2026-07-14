# FM因子分解机

## 1 背景

在现代工业级推荐系统中，预测用户对特定物品的点击率（Click-Through Rate, CTR）和转化率（Conversion Rate, CVR）是核心的任务之一。推荐场景下的输入数据不仅包含用户标识（User ID）和物品标识（Item ID），还涵盖了极其丰富的上下文特征（Context）、用户人口统计学属性、物品类别以及短期行为序列等多维度的稀疏特征。

为了能够直接处理这些多维特征，工业界通常采用 **One-Hot 编码** 将类别特征（Categorical Features）转换为高维稀疏向量。然而，这一预处理手段在提升特征表达空间的同时，引入了两个严峻的学术与工程瓶颈：

- **极其高维且稀疏的数据分布（Extreme Sparsity）**：One-Hot 编码后，特征向量的维度往往达到 $10^7$ 甚至 $10^9$ 级别，且每个样本中非零元素占比极低。
- **高阶特征交叉的计算瓶颈（Feature Interaction Bottleneck）**：传统的线性模型（如逻辑回归 Logistic Regression, LR）由于缺乏特征交叉能力，无法捕捉“时间+类别”或“性别+商品属性”等非线性关联。若采用多项式回归（Poly2）进行二阶显式交叉，其交叉项参数量将达 $O(d^2)$（$d$ 为特征维度）。在数据极度稀疏的场景下，**绝大多数特征组合的共现样本数（Co-occurrence）为零**，导致二阶交叉项的参数根本无法得到有效训练。

为了在“捕捉二阶特征交叉”与“缓解稀疏数据下的参数不收敛”之间取得理论平衡，Rendle 于 2010 年提出了因子分解机（Factorization Machines, FM）模型。FM 巧妙地将低秩矩阵分解的思想泛化至任意特征交叉领域，不仅打破了稀疏数据下高阶交叉的拟合瓶颈，也奠定了推荐系统从“浅层线性模型”向“自动特征工程”演进的理论基石。

## 2 FM算法原理

FM 模型的核心思想在于：**放弃对二阶交叉特征参数的独立估计，转而通过两个低维稠密隐向量的内积来表征特征交叉的权重。**

### 2.1 数学形式化表达

一个标准的二阶 FM 模型其预测输出公式定义如下：

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

其中：

* $w_0 \in \mathbb{R}$ 为全局偏置（Global Bias）。
* $\mathbf{w} \in \mathbb{R}^d$ 为一阶特征权重向量，$w_i$ 刻画了单一特征 $x_i$ 对预测目标的影响。
* $\mathbf{v}_i \in \mathbb{R}^k$ 为第 $i$ 个特征的**隐向量（Latent Vector）**，用来表征特征 $x_i$ 的语义分布。
* $k$（且 $k \ll d$）为隐向量的维度，是控制模型表达能力与防过拟合的关键超参数。
* $\mathbf{v}_i, \mathbf{v}_j$ 表示两个隐向量的内积（Inner Product），用于代替传统 Poly2 模型中独立的交叉权重 $w_{ij}$：

$$\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^k v_{i,f} v_{j,f}$$

### 2.2 稀疏数据下的泛化机制

在传统多项式模型中，交叉项参数 $w_{ij}$ 的训练必须依赖特征 $x_i$ 和 $x_j$ 在训练集中同时出现（即 $x_i x_j \neq 0$）。
而在 FM 模型中，交叉项的参数由 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 决定。即使在训练集中特征 $x_i$ 和 $x_j$ **从未共现过**，只要 $x_i$ 与其他特征共现过（从而训练了 $\mathbf{v}_i$），且 $x_j$ 也与其他特征共现过（从而训练了 $\mathbf{v}_j$），模型依然能够通过内积得出一个合理的、具备泛化性的交叉权重。这从根本上克服了稀疏数据对特征交叉参数训练的限制。

## 3 FM实现

在工程落地中，如何高效计算 FM 模型的二阶交叉项是能否实现工业级在线推断的关键。

### 3.1 核心数学推导：将复杂度从 $O(k d^2)$ 降至 $O(k d)$

直观上看，计算公式中二阶交叉项的显式双重循环需要对所有特征对进行两两组合，其计算复杂度为 $O(k d^2)$。Rendle 提出了一种极具工程美感的代数变形，将计算复杂度在线性时间（Linear Time）内完成重构。

通过展开平方项，二阶交叉项可以等价改写为：

$$\sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i,f} x_i \right)^2 - \sum_{i=1}^d v_{i,f}^2 x_i^2 \right]$$

**证明过程如下**：
根据代数恒等式 $\sum_{i<j} a_i a_j = \frac{1}{2} \left[ \left(\sum_i a_i\right)^2 - \sum_i a_i^2 \right]$，对于隐向量的第 $f$ 个维度，有：

$$
\begin{aligned}
\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} w_{ij}x_i x_j &=  \frac{1}{2} (\sum_{i=1}^n \sum_{j=1}^n w_{ij} x_i x_j - \sum_{i=1}^n w_{ii} x_i x_i)  \\
&= \frac{1}{2} (\sum_{i=1}^n \sum_{j=1}^n (V_i^T V_j) x_i x_j - \sum_{i=1}^n (V_i^T V_i) x_i x_i) \\
&= \frac{1}{2} (\sum_{i=1}^n \sum_{j=1}^n (\sum_{t=1}^k v_{it} v_{jt}) x_i x_j - \sum_{i=1}^n (\sum_{t=1}^k v_{it} v_{it}) x_i x_i) \\
&= \frac{1}{2} (\sum_{i=1}^n \sum_{j=1}^n \sum_{t=1}^k v_{it} v_{jt} x_i x_j - \sum_{i=1}^n \sum_{t=1}^k v_{it} v_{it} x_i x_i) \\
&= \frac{1}{2} \sum_{t=1}^k(\sum_{i=1}^n \sum_{j=1}^n v_{it} v_{jt} x_i x_j - \sum_{i=1}^n v_{it} v_{it} x_i x_i) \\
&= \frac{1}{2} \sum_{t=1}^k(\sum_{i=1}^n v_{it} x_i \sum_{j=1}^n v_{jt} x_j - \sum_{i=1}^n v_{it} v_{it} x_i x_i) \\
&= \frac{1}{2} \sum_{t=1}^k((\sum_{i=1}^n v_{it} x_i)^2 - \sum_{i=1}^n v_{it}^2 x_i^2)
\end{aligned}
$$

在将所有 $k$ 个维度求和后，即得证。由于化简后的公式内部仅包含单重循环，其整体计算复杂度被成功压缩至 **$O(k d)$**。由于在实际场景中特征向量 $\mathbf{x}$ 极度稀疏，非零元素个数 $N_z \ll d$，计算复杂度可进一步收敛至 **$O(k N_z)$**，这使得 FM 模型具备了极高的工业实时服务（Serving）效率。

### 3.2 损失函数与参数更新

根据推荐任务的不同（如回归任务或二分类 CTR 预测），FM 可采用不同的损失函数：

- **分类任务目标函数（交叉熵损失）**

$$\min_{\Theta} \mathcal{L} = \sum_{(\mathbf{x}, y) \in \mathcal{D}} \ln \left(1 + \exp(-y \cdot \hat{y}(\mathbf{x}))\right) + \frac{\lambda_{\theta}}{2} \Vert{}\Theta\Vert{}_2^2$$

其中 $y \in \{-1, +1\}$ 为真实标签，$\Theta = \{w_0, \mathbf{w}, \mathbf{V}\}$ 为待求解参数集，$\lambda_{\theta}$ 为 L2 正则化系数。

 - **参数优化更新（以 SGD 为例）**
通过链式法则，模型参数的一阶导数计算如下：

$$\frac{\partial \hat{y}(\mathbf{x})}{\partial \theta} = \begin{cases}  1, & \text{if } \theta = w_0 \\ x_i, & \text{if } \theta = w_i \\ x_i \sum_{j=1}^d v_{j,f} x_j - v_{i,f} x_i^2, & \text{if } \theta = v_{i,f} \end{cases}$$

利用该梯度公式，可通过随机梯度下降法（SGD）、AdaGrad 或 Adam 算法对模型进行高效的参数迭代更新。

## 4 总结

FM 因子分解机在推荐系统算法演进史上具有不可替代的承上启下作用，它完美地连接了浅层统计模型与深层表征学习。

### 4.1 核心价值与理论优势

- **突破性的稀疏泛化能力**：引入低维连续隐向量表征特征，打破了传统 Poly2 模型在稀疏数据下无法拟合未见交叉特征的瓶颈。
- **极佳的计算效率**：通过精妙的代数化简，将二阶特征交叉的计算复杂度降低至线性级 $O(k d)$，完全能够适配工业级大规模在线实时预估。
- **统一的特征框架**：相较于矩阵分解（MF）只能处理特定的“User-Item”二元关系，FM 提供了通用的多维特征输入框架，能够无缝融合上下文（时间、地点）、多域属性等任意特征。

### 4.2 局限性与改进演进

- **等同的交叉权重限制**：FM 中每个特征仅由一个隐向量 $\mathbf{v}_i$ 表征。在进行交叉时，无论该特征与何种域（Field）的特征相乘，其隐向量都是恒定不变的。这限制了其在精细化交叉场景下的表达力。为了解决这一问题，学者们推出了**场感知因子分解机（Field-aware Factorization Machines, FFM）**，为每个特征在不同的关联域中学习不同的隐向量。
- **高阶非线性捕获能力不足**：FM 仅完成了二阶显式特征交叉。在面对更高阶（三阶及以上）的复杂非线性模式时，FM 的人工推导复杂度将呈指数级上升。
