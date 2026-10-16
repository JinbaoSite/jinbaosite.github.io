# CCFormer：腾讯工业推荐中的高效长序列建模

## 1. 背景

近年来，工业推荐系统逐渐从 DLRM、DeepFM 等传统特征交叉模型转向基于 Transformer 的统一序列建模架构。增加用户行为序列长度和模型规模通常能够持续提升推荐效果，但标准 Self-Attention 的计算复杂度为 $O(L_s^2)$，当用户历史扩展到数千个行为时，训练成本、显存占用和在线推理延迟都会快速增长。另一方面，将用户画像、历史行为和候选物品全部视为同质 Token 进行 Self-Attention，也忽略了不同特征域之间天然存在的语义差异。

腾讯提出 **CCFormer（Efficient Cross-Field Interaction and Hierarchical Sequence Compression）**，核心目标是在工业级延迟和资源约束下，同时解决**跨特征域交互**与**超长用户行为序列建模**问题。模型已经应用于腾讯视频推荐和广告排序场景。

## 2. 模型架构

CCFormer 将推荐特征明确拆分为三个语义域：

- 用户画像：$\mathbf{U}^{(0)}\in\mathbb{R}^{B\times L_u\times d}$
- 用户行为序列：$\mathbf{S}^{(0)}\in\mathbb{R}^{B\times L_s\times d}$
- 目标候选物品：$\mathbf{T}^{(0)}\in\mathbb{R}^{B\times L_t\times d}$

模型堆叠多个 CCFormer Block：

$$
(\mathbf U^{(\ell+1)},\mathbf S^{(\ell+1)},\mathbf T^{(\ell+1)})
=
\operatorname{CCFormerBlock}_{\ell}
(\mathbf U^{(\ell)},\mathbf S^{(\ell)},\mathbf T^{(\ell)})
$$

最终将三个域的表示聚合并送入 Click、Like、Share 等多任务预测头。

整体架构如下：

<svg width="1000" height="650" viewBox="0 0 1000 650"
     xmlns="http://www.w3.org/2000/svg">

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10"
            refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#333"/>
    </marker>
  </defs>

  <style>
    .box { fill:#f7f7f7; stroke:#333; stroke-width:2; rx:10; }
    .module { fill:#e8f1ff; stroke:#3578c8; stroke-width:2; rx:10; }
    .seq { fill:#eaf7ea; stroke:#3b8c45; stroke-width:2; rx:10; }
    .output { fill:#fff0dc; stroke:#d88727; stroke-width:2; rx:10; }
    .txt { font-family:Arial,sans-serif; font-size:16px; text-anchor:middle; }
    .small { font-family:Arial,sans-serif; font-size:13px; text-anchor:middle; }
    .arrow { stroke:#333; stroke-width:2; fill:none; marker-end:url(#arrow); }
  </style>

  <!-- Inputs -->
  <rect class="box" x="50" y="40" width="200" height="60"/>
  <text class="txt" x="150" y="67">User Profile</text>
  <text class="small" x="150" y="88">User Tokens U</text>

  <rect class="box" x="400" y="40" width="200" height="60"/>
  <text class="txt" x="500" y="67">Behavior Sequence</text>
  <text class="small" x="500" y="88">Sequence Tokens S</text>

  <rect class="box" x="750" y="40" width="200" height="60"/>
  <text class="txt" x="850" y="67">Target Item</text>
  <text class="small" x="850" y="88">Target Tokens T</text>

  <!-- Cross-field -->
  <rect class="module" x="90" y="160" width="820" height="90"/>
  <text class="txt" x="500" y="190">Feature-Field Separated Cross Attention</text>
  <text class="small" x="500" y="218">
    User → Sequence | Target → Sequence | Target → User
  </text>

  <path class="arrow" d="M150 100 L250 160"/>
  <path class="arrow" d="M500 100 L500 160"/>
  <path class="arrow" d="M850 100 L750 160"/>

  <!-- Sequence -->
  <rect class="seq" x="315" y="300" width="370" height="65"/>
  <text class="txt" x="500" y="327">Relative Time &amp; Position Encoding</text>
  <text class="small" x="500" y="350">Local Subspace O(Ls · m)</text>

  <path class="arrow" d="M500 250 L500 300"/>

  <rect class="seq" x="315" y="400" width="370" height="65"/>
  <text class="txt" x="500" y="427">Subspace Token Mixing</text>
  <text class="small" x="500" y="450">Per-channel FFN</text>

  <path class="arrow" d="M500 365 L500 400"/>

  <rect class="seq" x="315" y="500" width="370" height="65"/>
  <text class="txt" x="500" y="527">Hierarchical Token Compression</text>
  <text class="small" x="500" y="550">Conv1D / Progressive Downsampling</text>

  <path class="arrow" d="M500 465 L500 500"/>

  <!-- Output -->
  <rect class="output" x="740" y="500" width="210" height="65"/>
  <text class="txt" x="845" y="527">Multi-task Heads</text>
  <text class="small" x="845" y="550">Click / Like / Share / ...</text>

  <path class="arrow" d="M685 532 L740 532"/>

  <!-- loop -->
  <path class="arrow" d="M315 532 C240 532,240 205,90 205"/>
  <text class="small" x="205" y="385">Stack × L</text>

</svg>

### 2.1 Feature-Field Separated Cross Attention

CCFormer 没有把所有 Token 放入一次全局 Self-Attention，而是根据推荐业务中的语义关系构造三条定向信息流：

1. **User → Sequence**：根据用户画像从历史行为中检索兴趣；
2. **Target → Sequence**：寻找与当前候选物品相关的历史行为；
3. **Target → User**：建模候选物品与用户画像的直接匹配关系。

其核心仍然是标准 Attention：

$$
\operatorname{Attn}(\mathbf Q,\mathbf K,\mathbf V;\mathbf M)
=
\operatorname{softmax}
\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right)\mathbf V
$$

三条信息流分别为：

$$
\begin{aligned}
\mathbf O_{u\rightarrow s}
&=\operatorname{Attn}(\mathbf Q_u,\mathbf K_s,\mathbf V_s;\mathbf M),\\
\mathbf O_{t\rightarrow s}
&=\operatorname{Attn}(\mathbf Q_{t\rightarrow s},\mathbf K_s,\mathbf V_s;\mathbf M),\\
\mathbf O_{t\rightarrow u}
&=\operatorname{Attn}(\mathbf Q_{t\rightarrow u},\mathbf K_u,\mathbf V_u).
\end{aligned}
$$

关键区别在于：行为序列主要作为被查询的 Key/Value，而不是让全部历史行为之间执行全局 Self-Attention。

因此跨域交互复杂度由传统的

$$
O(L_s^2)
$$

变为近似

$$
O(L_uL_s+L_tL_s).
$$

工业场景通常满足 $L_u,L_t\ll L_s$，因此该部分相对于行为序列长度基本呈线性增长。

### 2.2 Relative Time & Position Encoding

推荐系统中的行为具有很强的时间属性：用户几分钟前点击的商品通常比几个月前的行为更能反映当前兴趣。

CCFormer 不计算完整的 $L_s\times L_s$ 时间关系，而是把行为序列划分为长度为 $m$ 的局部子空间。

对于第 $p$ 个子空间中的行为 $i,j$，时间权重定义为：

$$
W^{time}_{p,i,j}
=
\alpha\cdot
\beta^{|t_{p,i}-t_{p,j}|^\gamma},
\qquad 0<\beta<1.
$$

其中 $\alpha,\gamma$ 为可学习参数。随着两个行为时间距离增加，权重逐渐衰减。

同时加入可学习的相对位置编码：

$$
W^{pos}_{p,i,j}
=
W^{pos\_origin}[i-j+m-1].
$$

最终：

$$
S^{tp}_p=
(W^{time}_p+W^{pos}_p)S_p.
$$

由于计算仅发生在长度为 $m$ 的局部窗口中，复杂度从全局关系建模的 $O(L_s^2)$ 降低到：

$$
O(L_s m).
$$


### 2.3 Subspace Token Mixing

仅解决跨域 Attention 还不够，行为序列内部仍然需要进行信息交互。CCFormer 的关键设计是不用 Self-Attention，而使用 **Subspace Token Mixing**。

首先将：

$$
S\in\mathbb R^{B\times L_s\times d}
$$

重新组织为：

$$
X=
\operatorname{Reshape}(S)
\in
\mathbb R^{
B\times\frac{L_s}{m}\times\frac d n\times mn
}.
$$

这样，每个长度为 $mn$ 的子空间同时包含 $m$ 个相邻行为和 $n$ 个隐藏维度，使 Token 信息与 Channel 信息能够在一个紧凑空间中直接交互。

然后对每个 Channel Group 使用独立的门控 PFFN：

$$
\operatorname{PFFN}_c(x)
=
W_c^o
\left[
\phi(xW_c^g)
\odot
(xW_c^v)
\right].
$$

得到：

$$
Z_{b,p,c}
=
\operatorname{PFFN}_c(X_{b,p,c}),
$$

最后恢复原来的序列结构：

$$
\hat S=\operatorname{Restore}(Z)
\in\mathbb R^{B\times L_s\times d}.
$$

这相当于把昂贵的“所有行为两两 Attention”，替换为局部子空间中的 Token/Channel Mixing，使序列建模复杂度相对于 $L_s$ 保持线性。

### 2.4 Hierarchical Token Compression

即使 Token Mixing 已经是线性复杂度，当历史行为达到数千条时，每一层都处理完整序列仍然昂贵。

CCFormer 因此进一步使用 Conv1D 对行为序列逐层压缩：

$$
S^{(\ell+1)}
=
\operatorname{Conv1D}_{k,s}
\left(
\hat S^{(\ell)}
\right),
$$

其中 $k$ 为卷积核大小，$s$ 为步长。

其思想类似层次化的时间信息聚合：

```text
原始行为序列
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
        ↓ Conv1D
    ●   ●   ●   ●   ●   ●   ●   ●
        ↓ Conv1D
        ●       ●       ●       ●
        ↓ Conv1D
            ●               ●
        ↓ Conv1D
                    ●
```

浅层 Token 保留局部、细粒度和短期兴趣；随着网络加深，相邻行为不断聚合，每个 Token 的感受野逐渐扩大。论文示例中感受野可以沿层级从 **3 → 7 → 15 → 31** 扩展，因此深层 Token 能表示越来越长时间尺度上的用户兴趣，而不是像直接截断序列那样丢弃早期行为。

从整体上看，CCFormer 实际进行了三层复杂度优化：

$$
\boxed{
\text{Global Self-Attention}
\rightarrow
\text{Directed Cross Attention}
\rightarrow
\text{Subspace Mixing}
\rightarrow
\text{Hierarchical Compression}
}
$$

也就是说，它并非简单设计一种“更快的 Attention”，而是重新拆解工业推荐中的信息交互路径：**跨域关系使用 Cross Attention，序列内部关系使用 Token Mixing，超长历史则通过层次压缩逐层降低计算量。**

## 3. 总结

CCFormer 的核心价值在于重新思考了工业推荐 Transformer 中“哪些 Token 真正需要互相 Attention”。它将用户画像、行为序列和目标物品拆分为独立语义域，用定向 Cross Attention 完成跨域交互，用 Subspace Token Mixing 替代行为序列中的二次复杂度 Self-Attention，再利用 Conv1D 分层压缩长序列，使模型能够兼顾细粒度兴趣建模与工业部署效率。

论文在腾讯生产环境中的结果尤其值得关注：相较 HSTU，CCFormer 训练速度提升 **2.21×**；线上视频推荐场景取得 **3.57% CTR 提升**，广告排序场景取得 **1.71% 广告收入提升**，并已部署到对应生产推荐系统的主流量。对于超长行为序列推荐而言，CCFormer 展示了一条比单纯扩大 Transformer 更工程化的 Scaling 路径。
