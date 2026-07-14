# 深度模型DeepFM模型

## 1 背景

在 CTR 预估任务中，特征交叉至关重要。例如，“在下午 5 点”与“下载外卖 App”这两个特征之间存在强关联（高阶交叉）；而“男”和“游戏”也存在天然的弱关联（低阶交叉）。

为了捕获这些交叉特征，业界经历了几代架构演进：

* **FM (Factorization Machines)**：擅长自动捕获二阶特征交叉，但在表达高阶（三阶及以上）非线性交叉时力不从心。
* **DNN (Deep Neural Networks)**：理论上可以学习任意高阶特征交叉，但容易忽略低阶的简单关联，且容易对稀疏特征过拟合。
* **Wide & Deep**：将线性模型（Wide 部分）与深度模型（Deep 部分）并联。然而，其 Wide 部分依然需要**人工构建交叉特征**（Cross-Product Transformations），这不仅耗时耗力，还依赖专家经验。

为了克服上述限制，华为诺亚方舟实验室于 2017 年提出了 **DeepFM**。其核心设计思想可以概括为两点：

1. **并行双核驱动**：结合 FM（提取低阶特征）与 DNN（提取高阶特征）。
2. **共享输入表征（Shared Embedding）**：FM 部分与 Deep 部分共享同一套稠密向量（Embedding）输入，无需任何人工特征工程，实现真正的端到端（End-to-End）训练。

## 2 DeepFM模型架构 (Model Architecture)

DeepFM 的整体架构非常优雅。它由 **FM Component** 和 **Deep Component** 两部分并联组成。

对于一个给定的输入特征向量 $x$，DeepFM 的预测输出 $\hat{y} \in (0, 1)$ 计算公式如下：

$$\hat{y} = \sigma(y_{FM} + y_{Deep})$$

其中，$\sigma$ 为 Sigmoid 激活函数，$y_{FM}$ 为 FM 部分的输出，$y_{Deep}$ 为 Deep 部分的输出。

![DeepFM模型结构](https://raw.githubusercontent.com/JinbaoSite/jinbaosite.github.io/master/img/deepfm.svg)

### 2.1 共享 Embedding 机制

DeepFM 的一大精妙之处在于其 **Shared Embedding**。

- 输入数据通常是高维稀疏的一阶特征（One-hot 编码）。
- 模型通过一个 Embedding 层将这些稀疏特征映射为低维稠密向量。
- **关键点**：FM 的隐向量 $v_i$ 与 Deep 网络第一层的输入 Embedding 向量是**完全共享且协同训练的**。这保证了模型在梯度反向传播时，低阶和高阶特征的学习能相互促进。
  
### 2.2 FM结构

![FM结构](https://raw.githubusercontent.com/JinbaoSite/jinbaosite.github.io/master/img/d2_fm.svg)

FM 部分负责学习一阶（Linear）特征以及二阶（Pairwise）特征交叉。其表达式为：

$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^{d} \sum_{j=i+1}^{d} \langle v_i, v_j \rangle x_i x_j$$

- **一阶部分** $\langle w, x \rangle$：反映单一特征对目标的影响。
- **二阶部分** $\sum \sum \langle v_i, v_j \rangle x_i x_j$：通过隐向量内积 $\langle v_i, v_j \rangle$ 自动计算特征 $i$ 和 $j$ 的二阶交叉强度。

### 2.3 Deep结构

![Deep结构](https://raw.githubusercontent.com/JinbaoSite/jinbaosite.github.io/master/img/d3_deep.svg)

1. **输入层**：将所有 Sparse 特征对应的共享 Embedding 向量拼接（Concatenate）在一起，得到：

$$a^{(0)} = [e_1, e_2, \dots, e_M]$$



其中 $M$ 是特征域（Field）的数量，$e_i$ 是第 $i$ 个域的 Embedding 向量。
2. **前向传播**：

$$a^{(l+1)} = \text{Activation}(W^{(l)} a^{(l)} + b^{(l)})$$


3. **输出层**：

$$y_{Deep} = W^{\vert{}H\vert{}+1} a^{(\vert{}H\vert{})} + b^{(\vert{}H\vert{}+1)}$$

## 3 DeepFM实现

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Input
from tensorflow.keras.models import Model

class FMInteraction(Layer):
    """FM 二阶特征交叉部分"""
    def __init__(self, **kwargs):
        super(FMInteraction, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs 维度: [batch_size, num_fields, embedding_size]
        # 和的平方
        sum_embeddings = tf.reduce_sum(inputs, axis=1) # [batch_size, embedding_size]
        sum_square = tf.square(sum_embeddings)
        
        # 平方的和
        square_embeddings = tf.square(inputs)
        square_sum = tf.reduce_sum(square_embeddings, axis=1) # [batch_size, embedding_size]
        
        # 二阶交叉公式: 0.5 * sum( (sum(v_i*x_i))^2 - sum((v_i*x_i)^2) )
        cross_term = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
        return cross_term

def DeepFM(linear_feature_num, dnn_feature_num, embedding_size, dnn_hidden_units):
    """
    linear_feature_num: 稀疏特征进行一阶线性映射的特征总数
    dnn_feature_num: 稀疏特征的 Field 数量 (即有多少个离散特征)
    embedding_size: 隐向量的维度
    dnn_hidden_units: DNN 隐藏层神经元列表，例如 [128, 64]
    """
    # 1. 输入定义 (这里假设输入已经过预处理，全为类别特征的单热点/长整型索引)
    # linear_inputs 用于一阶线性部分，dnn_inputs 用于二阶和深度部分
    linear_inputs = Input(shape=(dnn_feature_num,), dtype=tf.int32, name='linear_inputs')
    dnn_inputs = Input(shape=(dnn_feature_num,), dtype=tf.int32, name='dnn_inputs')
    
    # 2. FM 一阶线性部分 (Linear Part)
    # 用 Embedding(..., 1) 来模拟 w_i * x_i
    linear_embed = Embedding(input_dim=linear_feature_num, output_dim=1, name='linear_emb')(linear_inputs)
    linear_out = tf.reduce_sum(linear_embed, axis=1) # [batch_size, 1]
    
    # 3. FM 二阶交叉部分 (FM Interaction Part) 与 DNN 共享 Embedding
    feature_embed = Embedding(input_dim=linear_feature_num, output_dim=embedding_size, name='shared_emb')(dnn_inputs)
    # feature_embed 维度: [batch_size, dnn_feature_num, embedding_size]
    
    fm_out = FMInteraction()(feature_embed) # [batch_size, 1]
    
    # 4. Deep 深度部分 (DNN Part)
    dnn_in = tf.keras.layers.Flatten()(feature_embed) # 展开成一维向量
    for units in dnn_hidden_units:
        dnn_in = Dense(units, activation='relu')(dnn_in)
        # 可根据需要在此处添加 Dropout 或 BatchNormalization
    dnn_out = Dense(1, activation=None)(dnn_in) # [batch_size, 1]
    
    # 5. 结果融合 (Linear + FM + DNN)
    logits = linear_out + fm_out + dnn_out
    output = tf.keras.layers.Activation('sigmoid', name='output')(logits)
    
    model = Model(inputs=[linear_inputs, dnn_inputs], outputs=output)
    return model
```

## 4 优势与创新点

| 特性 \ 模型 | LR | FM | Wide & Deep | DeepFM |
| --- | --- | --- | --- | --- |
| **低阶特征提取** | 手工构建 | 自动二阶 | 手工构建 | **自动一阶 & 二阶** |
| **高阶特征提取** | 无法提取 | 无法提取 | 自动高阶 | **自动高阶** |
| **共享 Embedding** | 无 | 无 | 否（两套输入） | **是（Shared Embedding）** |
| **端到端训练** | 否 | 是 | 否（需要人工介入） | **是** |

DeepFM 凭借其**无需人工特征工程**、**低阶与高阶特征兼顾**以及**共享Embedding**的优雅设计，在工业界（尤其是在算力与工程落地受限的中小团队）得到了极广泛的应用。时至今日，它依然是点击率预估及推荐算法工程师面试与实战中必谈的经典模型。
