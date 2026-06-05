# AutoInt：用自注意力机制自动学习特征交互

> 论文：*AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*（Song et al., CIKM 2019）

在点击率（CTR）预估、推荐排序这类任务里,**特征交互（feature interaction）** 几乎是决定模型上限的关键。"年轻 + 男性 + 篮球类目"这样的组合往往比任何单一特征都更能说明用户意图。问题在于:有用的交叉特征藏在海量组合里,靠人工去枚举既费力又容易漏。AutoInt 给出的答案很直接——**让模型自己用自注意力机制把特征交互学出来**,而且学得既高阶、又可解释。

下面从问题背景讲起,一步步拆开它的结构。


## 一、为什么特征交互这么难

CTR 数据有两个让人头疼的特点:

1. **特征又多又稀疏**。用户 ID、商品 ID、城市、设备型号……做完 one-hot 之后动辄几百万维,而每条样本里真正非零的只有寥寥几个。
2. **类别特征和数值特征混在一起**。"年龄=25"是数值,"性别=男"是类别,二者要在同一个空间里参与交互才有意义。

在 AutoInt 之前,业界主要有几条路线,但都各有遗憾:

| 方法 | 思路 | 局限 |
|------|------|------|
| LR / 人工交叉 | 手动设计交叉特征喂给线性模型 | 极度依赖专家经验,组合爆炸 |
| FM | 用隐向量内积建模二阶交互 | 只能到二阶,且对所有交互一视同仁 |
| Wide & Deep | 线性部分 + DNN 部分 | wide 侧仍需人工交叉 |
| DeepFM / NFM | FM + DNN 联合建模 | 高阶交互由 DNN **隐式**完成,不可解释 |
| AFM | 给二阶交互加注意力权重 | 仍止步于二阶 |

可以看到,核心矛盾是:**要么交互阶数受限,要么高阶交互是 DNN"黑箱"里隐式发生的,既说不清哪些组合起了作用,也无法保证学到的就是有意义的交叉**。AutoInt 想同时解决这两件事。


## 二、AutoInt 的核心思想

一句话概括:**把每个特征看成一个"token",用 Transformer 式的多头自注意力,让特征之间互相"打分"并加权融合,从而显式地建模任意阶的特征交互。**

它带来三个好处:

- **显式建模**:交互发生在注意力层,而不是被埋进一堆全连接里。
- **可解释**:注意力权重直接告诉你"哪两个特征的组合更重要"。
- **高阶可控**:每叠一层自注意力,交互阶数就升高一级,层数决定最高阶数。


## 三、模型架构逐层拆解

整体结构是四层:**输入层 → 嵌入层 → 交互层（可堆叠）→ 输出层**。

### 1. 输入层

把一条样本表示成各个 field 的拼接。类别特征用 one-hot(或 multi-hot),数值特征保留原值:

$$
\mathbf{x} = [\mathbf{x}_1; \mathbf{x}_2; \dots; \mathbf{x}_M]
$$

其中 $M$ 是 field 的数量,$\mathbf{x}_i$ 是第 $i$ 个 field 的表示。

### 2. 嵌入层:把类别和数值都映射到同一空间

这是 AutoInt 处理"异构特征"的关键设计。

- **类别特征**:每个 field 有一张嵌入表 $\mathbf{V}_i$,取出对应的低维向量
  $$\mathbf{e}_i = \mathbf{V}_i \mathbf{x}_i$$
- **数值特征**:为每个数值 field 准备一个嵌入向量 $\mathbf{v}_m$,再用标量值缩放
  $$\mathbf{e}_m = v_m \cdot \mathbf{v}_m$$

经过这一层,**所有特征——无论原本是类别还是数值——都变成了维度相同的稠密向量**,于是它们可以平等地参与后续的注意力计算。

### 3. 交互层:多头自注意力(核心)

这是 AutoInt 的心脏。它借用了 Transformer 的 multi-head self-attention,但目的不是建模序列,而是建模**特征两两之间的关联**。

对某个特征 $m$,在第 $h$ 个注意力头下,它与特征 $k$ 的相关性用 query–key 内积衡量:

$$
\alpha^{(h)}_{m,k} = \frac{\exp\big(\psi^{(h)}(\mathbf{e}_m, \mathbf{e}_k)\big)}{\sum_{l=1}^{M}\exp\big(\psi^{(h)}(\mathbf{e}_m, \mathbf{e}_l)\big)},
\qquad
\psi^{(h)}(\mathbf{e}_m, \mathbf{e}_k) = \langle \mathbf{W}^{(h)}_{Q}\mathbf{e}_m,\; \mathbf{W}^{(h)}_{K}\mathbf{e}_k \rangle
$$

得到注意力分布后,把所有特征的 value 加权求和,得到特征 $m$ 在该头下的新表示:

$$
\tilde{\mathbf{e}}^{(h)}_m = \sum_{k=1}^{M}\alpha^{(h)}_{m,k}\,\big(\mathbf{W}^{(h)}_{V}\mathbf{e}_k\big)
$$

**多头**意味着在多个不同的子空间里同时学习交互,再把各头结果拼起来:

$$
\tilde{\mathbf{e}}_m = \tilde{\mathbf{e}}^{(1)}_m \oplus \tilde{\mathbf{e}}^{(2)}_m \oplus \cdots \oplus \tilde{\mathbf{e}}^{(H)}_m
$$

### 4. 残差连接:别忘了原始特征

只做注意力会丢掉特征本身的信息(以及低阶交互)。AutoInt 加了一个标准残差连接:

$$
\mathbf{e}^{Res}_m = \mathrm{ReLU}\big(\tilde{\mathbf{e}}_m + \mathbf{W}_{Res}\,\mathbf{e}_m\big)
$$

$\mathbf{W}_{Res}$ 用来对齐维度。这样每个特征的最终表示里,既有"和别人交互后的信息",也保留了"自己原本的信息"。

### 5. 堆叠:阶数随层数增长

把交互层的输出再喂给下一个交互层,就能建模更高阶的交互:
- 一层之后,特征 $m$ 的表示融合了所有二阶交互;
- 两层之后,这些二阶组合再彼此交互,得到三阶、四阶……

**层数 = 可建模的最高交互阶数**,这正是"高阶可控"的来源。

### 6. 输出层

把最后一层所有特征的表示拼接,过一个线性层加 sigmoid 得到点击概率:

$$
\hat{y} = \sigma\Big(\mathbf{w}^{\top}\big(\mathbf{e}^{Res}_1 \oplus \cdots \oplus \mathbf{e}^{Res}_M\big) + b\Big)
$$

损失函数就是常规的 Logloss(二元交叉熵)。


## 四、一段代码看懂交互层

下面用 PyTorch 写一个最小化的交互层,帮助把上面的公式落地(省略了 batch 维度的细节注释):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractingLayer(nn.Module):
    def __init__(self, embed_dim, att_dim, num_heads=2, use_residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.att_dim = att_dim
        self.use_residual = use_residual

        # 每个头共享一组 Q/K/V 投影,这里用一个大矩阵一次算出所有头
        self.W_Q = nn.Linear(embed_dim, att_dim * num_heads, bias=False)
        self.W_K = nn.Linear(embed_dim, att_dim * num_heads, bias=False)
        self.W_V = nn.Linear(embed_dim, att_dim * num_heads, bias=False)
        if use_residual:
            self.W_Res = nn.Linear(embed_dim, att_dim * num_heads, bias=False)

    def forward(self, x):
        # x: [B, M, embed_dim]，M 为 field 数
        B, M, _ = x.shape
        H, D = self.num_heads, self.att_dim

        # 投影并拆成多头: [B, H, M, D]
        Q = self.W_Q(x).view(B, M, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, M, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, M, H, D).transpose(1, 2)

        # 特征两两打分并归一化: [B, H, M, M]
        scores = torch.matmul(Q, K.transpose(-1, -2))
        attn = F.softmax(scores, dim=-1)

        # 加权求和: [B, H, M, D] -> 拼回 [B, M, H*D]
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, M, H * D)

        # 残差 + 非线性
        if self.use_residual:
            out = out + self.W_Res(x)
        return F.relu(out)
```

把若干个 `InteractingLayer` 串起来,前面接嵌入层、后面接输出层,就是一个完整的 AutoInt。


## 五、实验结论:它到底好在哪

作者在四个公开数据集上做了验证:**Criteo、Avazu、KDD12、MovieLens-1M**,覆盖广告点击和电影评分两类场景。主要结论可以归纳为几点:

- **效果有竞争力**:在 AUC / Logloss 上,AutoInt 优于 LR、FM、AFM、DeepCrossing、NFM、CrossNet 等一批基线,与当时最强的几个深度模型相当或更好。
- **和 DNN 互补**:把 AutoInt 与一个普通 DNN 拼成双塔(类似 Wide & Deep 的思路),效果还能进一步提升——说明显式交互和隐式交互捕捉的信息不完全重叠。
- **效率不差**:自注意力的计算复杂度相对可控,在大规模稀疏数据上仍然实用。
- **可解释性是亮点**:把注意力权重可视化成热力图,可以直观看到模型认为"哪些特征组合更重要"。论文里就展示过类似"性别 × 类目""年龄 × 时段"这类被高亮的有意义组合。


## 六、和其他模型放在一起看

| 模型 | 交互方式 | 最高阶数 | 可解释性 |
|------|----------|----------|----------|
| FM | 隐向量内积 | 2 阶 | 中 |
| AFM | 二阶 + 注意力权重 | 2 阶 | 较好 |
| DeepFM | FM + DNN | DNN 隐式高阶 | 弱 |
| Deep & Cross | 显式 cross network | 受层数控制 | 中 |
| **AutoInt** | **多头自注意力** | **层数控制,可高阶** | **强(注意力权重)** |

AutoInt 的独特定位在于:**把"显式""高阶""可解释"三者同时拿到手**。Deep & Cross 也做显式高阶,但 AutoInt 借自注意力额外获得了"每个交互有多重要"的权重信号。


## 七、实践中的几点提醒

- **嵌入维度**通常取 16 左右就有不错效果,不必太大。
- **交互层数**一般 2~3 层即可;层数过多容易过拟合,收益递减。
- **多头数量**和注意力维度需要一起调,常见组合是 2 头 / 每头维度 16~32。
- **数值特征的处理**别忽视:除了论文里的"标量 × 嵌入向量",实践中也常先做分桶离散化,效果有时更稳。
- **可以叠一个 DNN 分支**:如果线上指标卡瓶颈,加一个并行 DNN 往往能再榨出一点提升。


## 小结

AutoInt 的贡献,本质上是把 Transformer 的自注意力优雅地搬到了**特征交互建模**这件事上:

1. 用嵌入层把类别和数值特征统一到同一空间;
2. 用多头自注意力**显式**学习特征之间的交互,并给出可解释的权重;
3. 用残差连接保住原始信息,用堆叠层数控制交互阶数。

它既不像 FM 那样被二阶束缚,也不像纯 DNN 那样把交互藏进黑箱里。如果你正在做 CTR / 排序,又希望模型"既准又能讲清楚为什么",AutoInt 是一个非常值得放进候选名单的结构。
