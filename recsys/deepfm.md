# 深度模型DeepFM模型

## 00 序言

Wide&Deep模型中Wide侧是一个广义线性模型，它具备捕获低阶特征交互的能力，通常是在特征向量中手动对低阶特征进行显性特征交互，但是它不能进行一些显性高阶特征交互以及一些很少出现在训练数据中的特征交互；而且Wide侧的输入依赖特征工程，更多需要人工进行设计特征输入，与Deep侧的输入无法进行输入共享。

因子分解机（FM）将成对特征交互建模为特征之间潜在向量的内积，并显示出非常有前景的结果；虽然原则上FM可以模拟高阶特征交互，但在实践中，由于高度复杂，通常只考虑二阶特征交互。

---

## 01 DeepFM模型

DeepFM模型是在原有Wide&Deep模型结构上改造而来，将Wide侧的LR模型替换成FM模型结构，从而组成了DeepFM模型，而且FM结构和Deep结构的输入是共享输入，可以在没有任何特征工程的情况下进行端到端的训练。DeepFM模型结构图如下
![DeepFM模型结构](https://s1.ax1x.com/2022/04/28/LXVpuD.png)
DeepFM模型的输出有FM结构和Deep结构相加得到
$$
\begin{aligned}
y = sigmoid(y_{FM} + y_{Deep})
\end{aligned}
$$

##### 1、FM结构

![FM结构](https://s1.ax1x.com/2022/04/28/LXue0O.png)
FM结构是由一阶的线性部分和二阶的交叉部分组成，一阶线性部分是给与每个特征一个权重，然后进行加权和；二阶交叉部分是对特征进行两两相乘，然后赋予权重加权求和。然后将两部分结果累加在一起即为FM的输出
$$
\begin{aligned}
y_{FM} = w_0 + \sum_{i=0}^n w_i x + \sum_{i=0}^{n} \sum_{j=i+1}^{n} <V_i, V_j> x_i x_j
\end{aligned}
$$

##### 2、Deep结构

![Deep结构](https://s1.ax1x.com/2022/04/28/LXMgfJ.png)
Deep结构是一个DNN模型，主要用来学习高阶隐性特征交互，不同Sparse特征经过Embedding层映射成相同维度的Dense特征，经过多个隐藏层进行特征交互学习，最后经过sigmoid激活函数得到输出。

---

## 03 DeepFM实现

```python

```

---

## 04 DeepFM优缺点

优点：

 - 可以同时学习低阶和高阶特征交互信息
 - 不需要像Wide&Deep模型那样需要特征工程

缺点：

 - 可以学习

---
 
## 05 参考资料

 - [DeepFM: A Factorization-Machine based Nnetwork for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf)

 - [推荐算法(四)——经典模型 DeepFM 原理详解及代码实践](https://zhuanlan.zhihu.com/p/361451464)

 - [深度推荐模型之DeepFM](https://zhuanlan.zhihu.com/p/57873613)

 - [基于DeepFM模型的Embedding](https://zhuanlan.zhihu.com/p/384156476)
 



