# 深度模型Wide&Deep模型

## 1 序言

在推荐系统（如 Google Play 应用商店）中，点击率（CTR）预估模型的性能直接决定了用户体验与商业收益。在 Wide & Deep 模型提出之前，学术界与工业界主要依赖以下两类范式：

- **Wide 组件（记忆能力）：** 基于大规模离散特征和交叉特征的线性模型（如 Logistic Regression）。通过频繁出现的特征组合（例如，`User_Installed_App="Netflix"` $\times$ `Impression_App="Hulu"`），模型能够直接记住历史高频共现的规则。这种方式简单高效，但**无法推荐在训练集中未曾出现或罕见的特征组合**。
- **Deep 组件（泛化能力）：** 基于深度神经网络（DNN）的模型。通过将高维稀疏的 ID 类特征映射为低维稠密向量（Embedding），再输入多层感知机。DNN 能够学习到特征之间的潜在语义关系（例如，通过向量相似度发现安装过 "Netflix" 的用户可能也会对 "Disney+" 感兴趣），从而具备强大的泛化能力。然而，当用户有极度特殊的个性化需求或面对**高度稀疏的长尾数据**时，过度泛化可能导致推荐偏离目标。

**Wide & Deep 的核心动机：** 打破“非此即彼”的局限，在一个框架内同时实现**记忆**与**泛化**。将两者的优势结合，使模型既能精准捕捉高频强关联规则，又能探索潜在的用户兴趣。


## 2 Wide&Deep模型结构

Wide & Deep 的核心架构可以用“双路并行”来概括，其最终输出通过联合训练（Joint Training）进行融合。

![Wide&Deep模型结构](https://s3.bmp.ovh/imgs/2022/04/27/cfd884b771cd37f7.png)

### 2.1 Wide部分

Wide 部分本质上是一个广义线性模型（Generalized Linear Model），其数学形式如下：

$$y = w^T x + b$$

其中，输入特征 $x$ 既包括原始的单热点（One-hot）特征，也包括极为关键的交叉积变换（Cross-product Transformation）特征。其定义为：

$$\phi_k(x) = \prod_{i=1}^d x_i^{c_{ki}}, \quad c_{ki} \in \{0, 1\}$$

> **示例：** 如果条件 $k$ 是“用户安装了 Netflix 且当前展示的是 Hulu”，那么当且仅当这两个原始特征都为 1 时，交叉特征 $\phi_k(x)$ 才为 1，否则为 0。这使得 Wide 部分能够精准捕捉特定组合的强信号。

### 2.2 Deep部分

Deep 部分是一个典型的前馈神经网络。对于类别特征（如 User ID、Device Type），首先通过一个嵌入层（Embedding Layer）将其转化为低维稠密向量：

$$a^{(0)} = [\text{embed}_1(x_1), \text{embed}_2(x_2), \dots, \text{embed}_m(x_m)]$$

随后，这些嵌入向量与连续型特征（经过常态化处理）拼接在一起，送入多层感知机（MLP）中进行非线性变换：

$$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)})$$

### 2.3 联合训练 (Joint Training)

Wide & Deep 采用的是 **联合训练 (Joint Training)**，而非集成（Ensemble/Stacking）。集成模型中各个子模型是独立训练的，而联合训练是在训练时同时优化 Wide 和 Deep 的参数，共同最小化最终的损失函数。

对于二分类逻辑回归任务，其最终的预测目标为：

$$P(Y=1\vert{}x) = \sigma(w_{wide}^T [x, \phi(x)] + w_{deep}^T a^{(L)} + b)$$

在优化器的选择上，论文采用了非对称的协同优化策略：

* **Wide 侧：** 采用带 $L_1$ 正则化的 **FTRL (Follow-the-Regularized-Leader)** 算法。FTRL 能够带来极高的稀疏性，从而使 Wide 侧保留最有效的高频特征规则，减小模型体积。
* **Deep 侧：** 采用 **AdaGrad** 或 **Adam** 优化器，加速稠密向量和深层网络的收敛。

## 4 Wide&Deep模型实现

```python
def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(shape=(1,), name=feature_name)
    return inputs

def encode_inputs(inputs, use_embedding=False):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary_size = \
            len(CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]) \+ 1
            if use_embedding:
                embedding_dims = int(math.sqrt(vocabulary_size))
            else:
                embedding_dims = 1
            x = layers.Embedding(vocabulary_size,
                                 embedding_dims)(inputs[feature_name])
            encoded_features.append(x)
        if feature_name in NUMERIC_FEATURE_NAMES:
            encoded_features.append(inputs[feature_name])
    return layers.Concatenate()(encoded_features)
    
def create_wide_and_deep_model():

    inputs = create_model_inputs()
    wide = encode_inputs(inputs)
    wide = layers.BatchNormalization()(wide)

    deep = encode_inputs(inputs, use_embedding=True)
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([wide, deep])
    outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

## 5 总结

Wide & Deep 模型的提出，是推荐系统架构演进史上的一个重要里程碑。它用一种极其优雅且符合直觉的“双路结构”，完美回答了工业界如何在“守正（记忆历史）”与“出奇（探索未知）”之间寻找平衡。

### 优点总结 (Strengths)

* **两全其美：** 同时兼顾了线性模型的记忆能力和深度模型的泛化能力。
* **扩展性极佳：** 结构简单清晰，允许工程师根据业务独立调优 Wide 侧的特征工程或 Deep 侧的网络深度。

### 后续演进 (Evolution)

尽管 Wide & Deep 表现优异，但它仍留下了一个痛点：**Wide 侧的交叉特征依然需要大量的人工特征工程（Manual Feature Engineering）**。
