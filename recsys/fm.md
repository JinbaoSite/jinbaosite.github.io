# FM因子分解机

## 0 序言

传统的LR模型存在仅仅只能使用单一特征进行模型预估，不具备进行特征交叉生成高纬度组合特征的能力，忽略了特征与特征之间的关联信息，导致表达能力不强的缺点。

以往的方式是需要根据丰富的业务经验手动组合一些特征，然后再通过各种分析手段筛选组合特征，最后将有效的组合特征加入到LR模型中，整个过程异常复杂而且耗费人力巨大；再或者是通过暴力组合特征，即对任意两个特征进行两两特征组合，用数学公式来表达就是
$$
y = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} w_{ij}x_i x_j
$$
这种方式虽然在一定程度上解决了特征组合的问题，但存在两个问题：

 1.  LR模型通常采用one-hot处理的类别型特征会使得特征向量极度稀疏，当进行两两特征交叉时，会使得类别型特征更加稀疏，而且导致大部分交叉特征的权重因为有效训练数据的不足，无法完全收敛
 2. LR模型需要学习的参数由$n$上升到$n^2$，极大增加了训练复杂难度。

---

## 1 FM算法原理

因子分解机FM算法的基本原理是**权重系数$w_{ij}$可以分解成一个$k \times 1$向量的转置乘另一个$k \times 1$向量，或者两个$k \times 1$向量的内积**，即$w_{ij}=V_i^T V_j=<v_i , v_j>=\sum_{t=1}^k v_{it} v_{jt}$，分解出来的$k \times 1$向量$v_i$称之为特征$x_i$的**隐向量**，那么
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

FM算法为每个特征$x_i$学习了一个$k \times 1$的的隐向量，把原本要学习的参数$n^2$减少到了$nk$，极大的降低了训练开销；

同时FM能更好地解决数据稀疏性的问题，假定样本中有两个特征$(X,Y)$，对于训练样本$(x_1, y_1)$，FM算法可以学习到$x_1$的隐向量和$y_1$的隐向量，计算得到权重系数$w_{11}$，对于训练样本$(x_2, y_2)$，FM算法可以学习到$x_2$的隐向量和$y_2$的隐向量，计算得到权重系数$w_{22}$，那么对于没有出现的训练样本$(x_1, y_2), (x_2, y_1)$，可以通过对$x_1$的隐向量和$y_2$的隐向量的内积得到样本$(x_1, y_2)$的权重系数$w_{12}$，以及对$x_2$的隐向量和$y_1$的隐向量的内积得到样本$(x_2, y_1)$的权重系数$w_{21}$，所以FM算法可以学习到训练样本中没有出现组合特征的权重系数，更好地解决了数据稀疏性问题。

FM算法的缺点也非常明显，只扩展了二阶特征交叉，没办法进行更高阶的特征交叉。

---

## 2 FM实现

基于tensorflow/keras的实现
```
class FM_Layer(Layer):
    def __init__(self, layer_name="fm"):
        self.layer_name = layer_name
        super(FM_Layer, self).__init__()

    def call(self, inputs, training=None, mask=None):
        summed_features_emb = tf.reduce_sum(inputs, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(inputs)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        fm_output = 0.5 * (tf.subtract(summed_features_emb_square,
                                       squared_sum_features_emb))
        y_fm = tf.reduce_sum(fm_output, axis=-1, keepdims=True)
        return y_fm
```

---

## 3 参考资料

[1] 《深度学习推荐系统》
[2] [FM因子分解机的原理、公式推导、Python实现和应用](https://zhuanlan.zhihu.com/p/145436595)
[3] [因子分解机（Factorization Machine）详解（一）](https://blog.csdn.net/lijingru1/article/details/88623136)
[4] [【推荐系统】Factorization Machine](https://zhuanlan.zhihu.com/p/80726100)
