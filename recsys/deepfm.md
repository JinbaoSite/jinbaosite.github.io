# 深度模型DeepFM模型

## 1 序言

Wide&Deep模型中Wide侧是一个广义线性模型，它具备捕获低阶特征交互的能力，通常是在特征向量中手动对低阶特征进行显性特征交互，但是它不能进行一些显性高阶特征交互以及一些很少出现在训练数据中的特征交互；而且Wide侧的输入依赖特征工程，更多需要人工进行设计特征输入，与Deep侧的输入无法进行输入共享。

因子分解机（FM）将成对特征交互建模为特征之间潜在向量的内积，并显示出非常有前景的结果；虽然原则上FM可以模拟高阶特征交互，但在实践中，由于高度复杂，通常只考虑二阶特征交互。

## 2 DeepFM模型

DeepFM模型是在原有Wide&Deep模型结构上改造而来，将Wide侧的LR模型替换成FM模型结构，从而组成了DeepFM模型，而且FM结构和Deep结构的输入是共享输入，可以在没有任何特征工程的情况下进行端到端的训练。DeepFM模型结构图如下

![DeepFM模型结构](https://raw.githubusercontent.com/JinbaoSite/jinbaosite.github.io/master/img/deepfm.svg)

DeepFM模型的输出有FM结构和Deep结构相加得到
$$
\begin{aligned}
y = sigmoid(y_{FM} + y_{Deep})
\end{aligned}
$$

### 2.1 FM结构

![FM结构](https://s1.ax1x.com/2022/04/28/LXue0O.png)
FM结构是由一阶的线性部分和二阶的交叉部分组成，一阶线性部分是给与每个特征一个权重，然后进行加权和；二阶交叉部分是对特征进行两两相乘，然后赋予权重加权求和。然后将两部分结果累加在一起即为FM的输出
$$
\begin{aligned}
y_{FM} = w_0 + \sum_{i=0}^n w_i x + \sum_{i=0}^{n} \sum_{j=i+1}^{n} <V_i, V_j> x_i x_j
\end{aligned}
$$

### 2.2 Deep结构

![Deep结构](https://s1.ax1x.com/2022/04/28/LXMgfJ.png)
Deep结构是一个DNN模型，主要用来学习高阶隐性特征交互，不同Sparse特征经过Embedding层映射成相同维度的Dense特征，经过多个隐藏层进行特征交互学习，最后经过sigmoid激活函数得到输出。


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

## 4 DeepFM优缺点

优点：

 - 可以同时学习低阶和高阶特征交互信息
 - 不需要像Wide&Deep模型那样需要特征工程

缺点：

 - 可以学习


## 5 参考资料

- [DeepFM: A Factorization-Machine based Nnetwork for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf)
- [推荐算法(四)——经典模型 DeepFM 原理详解及代码实践](https://zhuanlan.zhihu.com/p/361451464)
- [深度推荐模型之DeepFM](https://zhuanlan.zhihu.com/p/57873613)
- [基于DeepFM模型的Embedding](https://zhuanlan.zhihu.com/p/384156476)
