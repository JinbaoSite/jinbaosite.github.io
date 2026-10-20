# 激活函数：从逻辑函数到门控线性单元

激活函数在仿射变换之间引入非线性，使神经网络能够表达复杂的输入输出关系。不同函数对正负输入、饱和区间和梯度的处理不同，会影响表示能力、优化条件与计算成本。

从逻辑函数和双曲正切，到修正线性单元、平滑自门控函数，再到双分支门控结构，激活机制逐渐从逐元素变换扩展为可学习的特征调制模块。

## 1. 背景

[多层感知机](/dl/mlp/)的多个仿射层若没有非线性，组合后仍是一个仿射变换。激活函数改变中间表示，允许网络构造非线性边界和特征交互。

设 $\mathbf z=W\mathbf x+\mathbf b$，隐藏表示为 $\mathbf h=\phi(\mathbf z)$。按照[反向传播](/dl/backpropagation/)的链式法则：

$$
\frac{\partial L}{\partial\mathbf z}=
\frac{\partial L}{\partial\mathbf h}\odot\phi'(\mathbf z)
$$

局部导数影响梯度如何穿过隐藏层，但梯度稳定性还取决于权重、网络深度、归一化和残差路径，不能仅凭激活函数判断整个网络会否梯度消失或爆炸。

## 2. 函数原理

### 2.1 逻辑函数

逻辑函数为：

$$
\sigma(z)=\frac1{1+e^{-z}},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

输出位于 $(0,1)$，在 $z=0$ 处为 $0.5$，导数最大为 $1/4$。输入绝对值很大时，导数接近零，进入饱和区域。

逻辑函数适合表达二分类概率或有界门值。隐藏层连续使用它可能产生较弱梯度和非零中心输出，但这不是禁止使用的充分理由，具体效果仍依赖结构与优化配置。

训练二分类交叉熵时，通常直接把得分传入 `BCEWithLogitsLoss`，避免先计算概率再取对数导致数值不稳定。推理需要概率时，再执行逻辑函数。

### 2.2 双曲正切

双曲正切为：

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},\qquad
\tanh'(z)=1-\tanh^2(z)
$$

输出位于 $(-1,1)$，关于原点对称，原点导数为 1。它保留正负信号，但两端仍会饱和。

两种函数满足 $\tanh(z)=2\sigma(2z)-1$。实现应使用框架算子，而不是直接按指数分式手写，以降低溢出和消减误差风险。

### 2.3 修正线性单元

修正线性单元为：

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

$$
\operatorname{ReLU}'(z)=
\begin{cases}0,&z<0\\1,&z>0\end{cases}
$$

零点不可微，具体实现需要约定反向值，PyTorch 在零点采用零。正半轴不饱和，计算简单；负半轴梯度为零，可能使某些神经元长期处于不激活状态。

这个现象与输入、偏置和更新幅度有关。观察到输出为零，不代表神经元一定永久失活；需要检查一段训练过程中的输入分布和梯度。

保留负半轴斜率可得到泄漏修正线性单元：

$$
\operatorname{LeakyReLU}(z)=\begin{cases}z,&z\ge0\\az,&z<0\end{cases}
$$

$a$ 通常取较小正数；若让该斜率可学习，则得到参数化变体。零点在两侧斜率不同时仍不可微。

### 2.4 高斯误差线性单元

高斯误差线性单元为：

$$
\operatorname{GELU}(z)=z\Phi(z)
$$

$\Phi$ 是标准正态分布的累积分布函数：

$$
\Phi(z)=\frac12\left[1+\operatorname{erf}\left(\frac z{\sqrt2}\right)\right]
$$

导数为：

$$
\operatorname{GELU}'(z)=\Phi(z)+z\varphi(z)
$$

其中 $\varphi$ 是标准正态密度。大正数区域接近恒等映射，大负数区域输出趋近零，中间区域保留部分负值。它是平滑且非单调的函数，不应描述为整个实数范围都保持正导数。

常用双曲正切近似为：

$$
\operatorname{GELU}(z)\approx\frac z2\left[1+\tanh\left(\sqrt{\frac2\pi}(z+0.044715z^3)\right)\right]
$$

精确形式与近似形式不完全相同，复现模型时应保存对应配置。标准形式是确定性变换，不是在每次前向中随机丢弃神经元。

### 2.5 逻辑加权线性单元

逻辑加权线性单元为：

$$
\operatorname{SiLU}(z)=z\sigma(z)
$$

导数为：

$$
\operatorname{SiLU}'(z)=\sigma(z)+z\sigma(z)(1-\sigma(z))
$$

它用输入自身生成软门值，因此也可称为自门控。输出平滑、可为负，且同样不是全局单调。函数值在大负数处趋近零，在大正数处接近输入。

更一般的形式为 $z\sigma(\beta z)$。$\beta=1$ 时与 SiLU 相同；讨论 Swish 时需要明确是否使用这个固定参数约定。

### 2.6 门控线性单元

门控线性单元使用两个分支：

$$
\mathbf a=W_a\mathbf x+\mathbf b_a,\qquad
\mathbf g=W_g\mathbf x+\mathbf b_g
$$

$$
\operatorname{GLU}(\mathbf x)=\mathbf a\odot\sigma(\mathbf g)
$$

它不只是对输入应用一个逐元素函数，而是先通过独立投影产生内容与门值。对输出上游梯度 $\mathbf d$：

$$
\frac{\partial L}{\partial\mathbf a}=\mathbf d\odot\sigma(\mathbf g)
$$

$$
\frac{\partial L}{\partial\mathbf g}=\mathbf d\odot\mathbf a\odot
\sigma(\mathbf g)(1-\sigma(\mathbf g))
$$

门接近零会抑制内容路径，门饱和也可能削弱门分支梯度，因此门控不是天然消除梯度消失的保证。

PyTorch 的 `nn.GLU` 只负责把输入沿指定维度分成两半，计算第一半乘以第二半的逻辑函数，不会自动创建两组线性投影。输入对应维度必须为偶数，输出宽度减半。

### 2.7 平滑门控变体

将门分支激活替换为其他函数：

$$
\operatorname{GEGLU}(\mathbf x)=\mathbf a\odot\operatorname{GELU}(\mathbf g)
$$

$$
\operatorname{SwiGLU}(\mathbf x)=\mathbf a\odot\operatorname{SiLU}(\mathbf g)
$$

门分支的值不再局限于 $(0,1)$，可以为负或大于 1，因此“门”表示可学习调制，不必解释为概率或简单开关。

前馈模块通常再接输出投影：

$$
\operatorname{FFN}_{\mathrm{SwiGLU}}(\mathbf x)=
W_o\left[(W_a\mathbf x)\odot\operatorname{SiLU}(W_g\mathbf x)\right]
$$

SiLU 由同一个标量自门控，SwiGLU 则使用两个独立可学习分支；两者不能只因名称接近就视为相同模块。

### 2.8 参数量与选择依据

忽略偏置，普通前馈层的输入输出宽度为 $d$，隐藏宽度为 $m$，参数量为 $2dm$。双分支门控前馈层的隐藏宽度为 $k$，参数量为 $3dk$。

相近参数预算下，可取：

$$
k\approx\frac23m
$$

如果保持相同隐藏宽度，门控模块会增加参数和乘法成本；公平比较时需要说明是否匹配参数量、训练量或推理预算。

| 位置或需求 | 常见选择 | 检查重点 |
| --- | --- | --- |
| 隐藏层基础网络 | ReLU、Tanh | 输入尺度、饱和或失活 |
| 平滑前馈模块 | GELU、SiLU | 精确或近似实现、计算开销 |
| 双分支前馈模块 | GEGLU、SwiGLU | 投影数、隐藏宽度和预算 |
| 二分类输出 | 得分配合交叉熵，推理时 Sigmoid | 避免重复概率变换 |
| 回归输出 | 按目标约束选择 | 激活是否限制预测范围 |

没有单个函数在所有任务上都占优。替换预训练模型中的激活会改变计算函数，不能仅凭局部平滑性就假设权重无需重新训练。

## 3. 函数实现

### 3.1 公式与自动求导核对

代码使用双精度，对平滑函数的手推导数与自动求导逐点比较。修正线性单元单独检查零点约定。

```python
import math
import torch
from torch import nn
from torch.nn import functional as F


x = torch.tensor([-5., -1., 0., 1., 5.], dtype=torch.float64,
                 requires_grad=True)
probability = torch.sigmoid(x)
normal_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
normal_pdf = torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)

cases = {
    "sigmoid": (probability, probability * (1.0 - probability)),
    "tanh": (torch.tanh(x), 1.0 - torch.tanh(x).square()),
    "gelu": (x * normal_cdf, normal_cdf + x * normal_pdf),
    "silu": (x * probability, probability + x * probability * (1.0 - probability)),
}
for name, (values, expected_gradient) in cases.items():
    gradient, = torch.autograd.grad(values.sum(), x, retain_graph=True)
    assert torch.allclose(gradient, expected_gradient, atol=1e-12)
    print(name, "derivative_check:", True)

relu_input = torch.tensor([-1., 0., 1.], requires_grad=True)
F.relu(relu_input).sum().backward()
print("relu_gradient:", relu_input.grad.tolist())
```

对逐元素函数求输出之和再反传，能同时获得各位置导数。该技巧依赖输出元素之间没有交叉依赖，不适用于任意向量函数的完整雅可比求解。

### 3.2 门控前馈模块

```python
class GatedFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, gate="swiglu"):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("dimensions must be positive")
        if gate not in {"glu", "geglu", "swiglu"}:
            raise ValueError("invalid gate")
        self.gate = gate
        self.content = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, inputs):
        content = self.content(inputs)
        gate_value = self.gate_projection(inputs)
        if self.gate == "glu":
            gate_value = torch.sigmoid(gate_value)
        elif self.gate == "geglu":
            gate_value = F.gelu(gate_value, approximate="none")
        else:
            gate_value = F.silu(gate_value)
        return self.output(content * gate_value)


torch.manual_seed(42)
inputs = torch.randn(2, 4, 12, requires_grad=True)
module = GatedFeedForward(input_dim=12, hidden_dim=32, gate="swiglu")
outputs = module(inputs)
outputs.square().mean().backward()
print("output_shape:", tuple(outputs.shape))
print("parameter_count:", sum(parameter.numel() for parameter in module.parameters()))
print("finite_input_gradient:", torch.isfinite(inputs.grad).all().item())
```

输出形状为 `(2,4,12)`，参数量为 $3\times12\times32=1152$。前两维可表示批次和序列位置，模块只在最后一维执行投影，不混合不同位置。

### 3.3 框架门控算子核对

```python
packed = torch.randn(3, 10, dtype=torch.float64)
content_half, gate_half = packed.chunk(2, dim=-1)
manual_glu = content_half * torch.sigmoid(gate_half)
framework_glu = F.glu(packed, dim=-1)
assert torch.allclose(manual_glu, framework_glu)
print("glu_shape:", tuple(framework_glu.shape))

check_input = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(F.silu, (check_input,))
assert torch.autograd.gradcheck(lambda value: F.gelu(value, approximate="none"),
                               (check_input,))
print("smooth_gradient_checks:", True)
```

该门控算子的输入宽度是 10，输出宽度是 5。它与双投影模块中的逐元素门控部分对应，但不包含输入投影和输出投影。

### 3.4 数值与训练诊断

极端输入下优先使用框架稳定算子，避免直接执行可能溢出的指数表达式。低精度训练还需监控激活范围、非有限值和梯度范数，平滑函数不保证任意输入尺度下都稳定。

若更换激活，建议同时检查初始化、学习率、归一化和参数预算。统计隐藏层输出分布与梯度分布，比单独观察损失更容易定位饱和、长期失活或门控幅度异常。

原地激活可能覆盖反向传播仍需使用的中间值，尤其在分支和共享张量中需要谨慎。默认使用非原地形式通常更便于调试，再依据内存测量决定是否优化。

## 4. 总结

逻辑函数和双曲正切提供有界非线性，修正线性单元以简单分段形式保留正向梯度，GELU 与 SiLU 通过平滑调制保留部分负信号。门控线性单元进一步引入两个可学习分支，GEGLU 和 SwiGLU 则扩展门分支的变换形式。

选择激活函数时，需要同时考虑输出语义、梯度、初始化与计算预算。公式核对、自动求导检查及真实训练评估共同决定模块是否适合目标网络。

## 5. 参考文献

[1] Nair V, Hinton G E. Rectified Linear Units Improve Restricted Boltzmann Machines. ICML, 2010: 807–814.

[2] Hendrycks D, Gimpel K. Gaussian Error Linear Units. arXiv:1606.08415, 2016.

[3] Dauphin Y N, Fan A, Auli M, Grangier D. Language Modeling with Gated Convolutional Networks. ICML, 2017: 933–941.

[4] Shazeer N. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.
