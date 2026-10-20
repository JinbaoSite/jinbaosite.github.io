# 朴素贝叶斯：从条件概率到文本分类

朴素贝叶斯通过类别先验和类条件分布计算类别后验概率，用于文本分类、垃圾邮件识别等任务。模型对给定类别下的特征作条件独立假设，将高维联合概率的估计转化为多个一维条件分布的估计。

不同变体使用不同的特征分布：多项式模型适合词频等计数数据，伯努利模型适合二元特征，高斯模型适合用正态分布描述的连续特征。训练通常只需要累计计数或估计均值、方差，无需梯度下降。

## 1. 背景

判断一封邮件是否为垃圾邮件时，“中奖”“优惠”等词语可能提供证据。单个词不能决定类别，但多个词的组合可以改变类别的相对可能性。

设类别为 $y\in\{1,\ldots,C\}$，特征向量为 $\mathbf x=(x_1,\ldots,x_d)$。贝叶斯公式为：

$$
P(y=c\mid\mathbf x)=\frac{P(y=c)P(\mathbf x\mid y=c)}{P(\mathbf x)}
$$

其中 $P(y=c)$ 是类别先验，$P(\mathbf x\mid y=c)$ 是类条件分布，$P(y=c\mid\mathbf x)$ 是类别后验。连续特征下，对应的特征分布应理解为概率密度。

难点在于估计高维联合分布。即使每个特征只有两个可能值，$d$ 个特征也有 $2^d$ 种组合，有限样本难以覆盖。朴素贝叶斯用条件独立假设降低参数量，以更强的结构假设换取更简单的估计。

## 2. 模型原理

### 2.1 条件独立假设

给定类别 $c$ 后，各特征条件独立：

$$
P(\mathbf x\mid y=c)=\prod_{j=1}^{d}P(x_j\mid y=c)
$$

因此：

$$
P(y=c\mid\mathbf x)\propto P(y=c)\prod_{j=1}^{d}P(x_j\mid y=c)
$$

条件独立不等于无条件独立。两个特征可以因为共享类别因素而相关；假设只约束固定类别后的分布。现实文本中的词语即使在同一类别内也经常相关，因此该假设通常是近似。

模型属于生成式分类方法：先估计类别及特征生成机制，再由贝叶斯公式得到分类概率。[逻辑回归](/dl/logistic_regression/)则直接对类别条件概率建模。

### 2.2 分类与对数概率

按最大后验概率分类：

$$
\widehat y=\arg\max_c\left[P(y=c)\prod_jP(x_j\mid y=c)\right]
$$

连乘许多小概率容易发生浮点数下溢。取对数后，乘积变为求和：

$$
s_c=\log P(y=c)+\sum_j\log P(x_j\mid y=c)
$$

$$
\widehat y=\arg\max_cs_c
$$

仅比较类别时可以省略共同的归一化项；需要后验概率时，则必须进行归一化。令 $m=\max_cs_c$，稳定计算为：

$$
P(y=c\mid\mathbf x)=\frac{\exp(s_c-m)}{\sum_k\exp(s_k-m)}
$$

归一化后的数值符合模型后验定义，但不保证在真实数据上已经校准。强相关特征可能重复计入相似证据，使模型产生过高置信度。

### 2.3 类别先验

训练集共有 $n$ 个样本，类别 $c$ 有 $n_c$ 个样本，先验的频率估计为：

$$
\widehat\pi_c=\frac{n_c}{n}
$$

先验由文档数或样本数估计，不能用该类别的总词数替代。长文档会影响词频统计，但每篇文档在类别样本计数中只贡献一次。

若训练数据经过类别重采样，频率先验反映的是采样后的类别比例。部署类别比例不同时，应根据采样机制和验证数据调整先验，并检查概率校准。

### 2.4 多项式朴素贝叶斯

**（1）词袋与词频**

设词表大小为 $V$，$x_j$ 表示词语 $j$ 在文档中的出现次数，文档长度为 $L=\sum_jx_j$。给定类别和长度，多项分布为：

$$
P(\mathbf x\mid y=c,L)=\frac{L!}{\prod_jx_j!}\prod_{j=1}^{V}\theta_{cj}^{x_j}
$$

其中 $\theta_{cj}$ 是类别 $c$ 下生成词语 $j$ 的概率，并满足 $\sum_j\theta_{cj}=1$。文档可看作从类条件词分布反复抽取词语，再将结果汇总为词频；词频分量受总长度约束，不应进一步解释为相互独立的计数变量。

标准多项式文本分类模型通常不单独建模类别相关的文档长度分布。多项式系数对同一文档的各类别相同，分类时可省略：

$$
s_c=\log\pi_c+\sum_{j=1}^{V}x_j\log\theta_{cj}
$$

**（2）计数估计与加性平滑**

令 $N_{cj}$ 为类别 $c$ 的全部训练文档中词语 $j$ 的总次数，$N_c=\sum_jN_{cj}$ 为该类别总词数。加性平滑为：

$$
\widehat\theta_{cj}=\frac{N_{cj}+\alpha}{N_c+\alpha V},\qquad\alpha>0
$$

$\alpha=1$ 称为拉普拉斯平滑。分子和分母必须同时加入伪计数，才能保持词概率之和为 1。没有平滑时，训练中未在某类出现的一个词就可能把整篇文档在该类下的似然变为零。

平滑只作用于已定义的词表。预测时遇到词表外词，可以忽略，也可以使用训练时定义好的未知词标记；不能临时扩充词表而不更新分母和模型参数。

### 2.5 多项式模型的手算示例

考虑已经分词的四篇训练文档：

| 类别 | 文档 |
| --- | --- |
| 垃圾 | 优惠 优惠 中奖 |
| 垃圾 | 优惠 中奖 |
| 正常 | 会议 项目 |
| 正常 | 会议 会议 项目 |

词表为“优惠、中奖、会议、项目”，$V=4$，两个类别先验均为 $1/2$。垃圾类总词数为 5，正常类总词数也为 5。

取 $\alpha=1$，得到：

| 词语 | 垃圾类概率 | 正常类概率 |
| --- | --- | --- |
| 优惠 | $4/9$ | $1/9$ |
| 中奖 | $3/9$ | $1/9$ |
| 会议 | $1/9$ | $4/9$ |
| 项目 | $1/9$ | $3/9$ |

对“优惠 中奖”，两个类别的未归一化分数分别为：

$$
q_{\mathrm{垃圾}}=\frac12\cdot\frac49\cdot\frac39=\frac6{81}
$$

$$
q_{\mathrm{正常}}=\frac12\cdot\frac19\cdot\frac19=\frac{1}{162}
$$

共同的多项式系数已省略。垃圾类相对正常类的分数比为 12，因此后验概率为：

$$
P(y=\mathrm{垃圾}\mid\mathbf x)=\frac{12}{13}\approx0.9231
$$

### 2.6 伯努利朴素贝叶斯

伯努利模型令 $x_j\in\{0,1\}$ 表示某个词是否出现：

$$
P(\mathbf x\mid y=c)=\prod_{j=1}^{V}
\theta_{cj}^{x_j}(1-\theta_{cj})^{1-x_j}
$$

若 $D_{cj}$ 是类别 $c$ 中包含词语 $j$ 的文档数，则：

$$
\widehat\theta_{cj}=\frac{D_{cj}+\alpha}{n_c+2\alpha}
$$

每个特征是二元分布，因此分母加入 $2\alpha$，而非 $\alpha V$。一篇文档中同一个词重复十次，也只对文档出现计数贡献一次。

类别得分同时包括出现和未出现的特征：

$$
s_c=\log\pi_c+\sum_j\left[x_j\log\theta_{cj}
+(1-x_j)\log(1-\theta_{cj})\right]
$$

因此，将词频截断为 0 和 1 后继续使用多项式得分，并不等价于伯努利模型。前者没有纳入词语缺失的证据。

### 2.7 高斯朴素贝叶斯

连续特征可以在各类别内使用高斯分布：

$$
p(x_j\mid y=c)=\frac{1}{\sqrt{2\pi\sigma_{cj}^2}}
\exp\left(-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\right)
$$

训练时按类别和特征估计均值与最大似然方差：

$$
\widehat\mu_{cj}=\frac1{n_c}\sum_{i:y_i=c}x_{ij},\qquad
\widehat\sigma_{cj}^2=\frac1{n_c}\sum_{i:y_i=c}(x_{ij}-\widehat\mu_{cj})^2
$$

使用方差下界或小幅方差平滑可避免常数特征导致除零。极小方差也可能使微小偏移主导类别得分，平滑尺度需要结合数据量纲选择。

| 模型 | 特征类型 | 类条件参数 | 关键区别 |
| --- | --- | --- | --- |
| 多项式 | 非负计数 | 各类别的词概率 | 使用重复次数，标准得分省略共同多项式系数 |
| 伯努利 | 二元指示 | 各类别的出现概率 | 同时使用出现和缺失信息 |
| 高斯 | 连续数值 | 各类别各特征的均值与方差 | 使用概率密度，需要处理退化方差 |

词频标准化后得到的负值不适合多项式模型。非负的词频逆文档频率特征有时也用于多项式分类，但不再具有严格的整数词频生成解释。

## 3. 模型实现

### 3.1 词频统计与训练

多项式实现只依赖 Python 标准库。输入是已经分词的词语列表，标签为字符串。词表只从训练文档构建，预测时忽略词表外词；空文档或全部为未知词的文档根据类别先验分类。

```python
import math
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = alpha
        self.fitted = False

    def fit(self, documents, labels):
        self.fitted = False
        documents = [list(document) for document in documents]
        labels = list(labels)
        if not documents or len(documents) != len(labels):
            raise ValueError("documents and labels must have equal nonzero length")

        self.classes = sorted(set(labels))
        self.vocabulary = set(word for document in documents for word in document)
        if not self.vocabulary:
            raise ValueError("training vocabulary must not be empty")

        class_count = Counter(labels)
        word_count = {label: Counter() for label in self.classes}
        for document, label in zip(documents, labels):
            word_count[label].update(document)

        self.log_prior = {
            label: math.log(class_count[label] / len(labels))
            for label in self.classes
        }
        self.log_word_probability = {}
        for label in self.classes:
            total = sum(word_count[label].values())
            denominator = total + self.alpha * len(self.vocabulary)
            self.log_word_probability[label] = {
                word: math.log((word_count[label][word] + self.alpha) / denominator)
                for word in self.vocabulary
            }
        self.fitted = True
        return self

    def log_scores(self, document):
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        counts = Counter(word for word in document if word in self.vocabulary)
        return {
            label: self.log_prior[label] + sum(
                count * self.log_word_probability[label][word]
                for word, count in sorted(counts.items())
            )
            for label in self.classes
        }

    def predict_proba(self, document):
        scores = self.log_scores(document)
        maximum = max(scores.values())
        weights = {label: math.exp(score - maximum) for label, score in scores.items()}
        total = sum(weights.values())
        return {label: weight / total for label, weight in weights.items()}

    def predict(self, document):
        scores = self.log_scores(document)
        return max(self.classes, key=lambda label: scores[label])
```

相同得分下，`predict` 按排序后的类别顺序返回第一个类别。所有类别在训练集中均有样本，因此频率先验为正；某个类别没有任何词时，加性平滑使该类别的词分布成为均匀分布。

### 3.2 文本分类示例

```python
train_documents = [
    ["优惠", "优惠", "中奖"],
    ["优惠", "中奖"],
    ["会议", "项目"],
    ["会议", "会议", "项目"],
]
train_labels = ["垃圾", "垃圾", "正常", "正常"]

model = MultinomialNaiveBayes(alpha=1.0)
model.fit(train_documents, train_labels)

query = ["优惠", "中奖"]
print("prediction:", model.predict(query))
print("probability:", {
    label: round(value, 6)
    for label, value in model.predict_proba(query).items()
})

print("unknown:", model.predict_proba(["未知词"]))
```

输出为：

```text
prediction: 垃圾
probability: {'垃圾': 0.923077, '正常': 0.076923}
unknown: {'垃圾': 0.5, '正常': 0.5}
```

该数据用于核对计数和平滑公式，不构成泛化效果评估。真实任务需要独立的验证集与测试集，并在划分之后构建词表、过滤规则及其他数据依赖的预处理参数。

### 3.3 计算成本与增量更新

设训练语料总词数为 $T$，类别数为 $C$，词表大小为 $V$。计数成本约为 $O(T)$，构建全部类别词概率的成本为 $O(CV)$，存储量同阶。对于包含 $m$ 个不同有效词的文档，各类别打分的主要计算量为 $O(Cm)$；示例为稳定累加顺序额外排序，需付出 $O(m\log m)$ 的成本。

多项式模型可以通过累加类别文档数和类别词计数实现增量更新。固定词表下，只需重新计算变化类别的分母与词概率；若扩展词表，所有类别的平滑分母都会变化，必须同步更新。模型参数和分词、词表版本需要一起发布。

### 3.4 评价与适用边界

文本分类可观察各类别精确率、召回率、宏平均调和分数和混淆矩阵，概率输出还可用对数损失与可靠性曲线检查。类别不平衡时，整体准确率可能掩盖少数类表现。

朴素贝叶斯适合参数估计成本低、稀疏高维、需要快速建立基线的场景。其限制主要来自以下方面：

1. **相关特征重复提供证据**：近义词、重复字段或强关联词组可能导致过度置信，需要结合特征处理和概率校准评估。
2. **词袋丢失顺序**：“喜欢”和“不喜欢”可能共享词语，词组特征可以补充局部语义，但会增加词表和稀疏性。
3. **短文本与未知词**：有效证据较少时，先验影响更大；全部未知词的结果仅反映先验，不代表模型理解了文本。
4. **训练与部署分布变化**：词汇和类别比例漂移都会改变预测质量，需要监控未知词比例和分类别指标。

## 4. 总结

朴素贝叶斯通过条件独立结构简化类条件分布估计，再结合类别先验完成分类。多项式模型累计词频，伯努利模型统计出现与缺失，高斯模型估计连续特征的类内均值和方差。

可靠实现需要匹配特征分布、正确设置平滑分母，并在对数空间中计算得分。词表边界、类别先验、训练测试划分和概率校准同样决定模型的实际可用性。

## 5. 参考文献

[1] Bishop C M. Pattern Recognition and Machine Learning. Springer, 2006.

[2] Manning C D, Raghavan P, Schütze H. Introduction to Information Retrieval. Cambridge University Press, 2008.

[3] McCallum A, Nigam K. A Comparison of Event Models for Naive Bayes Text Classification. AAAI Workshop on Learning for Text Categorization, 1998.
