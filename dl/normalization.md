# 归一化：从批次统计到均方根缩放

归一化层通过指定范围内的均值、方差或均方根调整激活尺度，再使用可学习参数恢复所需表示。不同方法的核心区别是统计量沿哪些维度计算，以及训练与推理是否使用同一套统计。

批归一化、层归一化、组归一化、实例归一化与均方根归一化分别适应不同的数据形态。理解具体张量轴，比仅按模型类别选择归一化更可靠。

## 1. 背景

网络激活尺度影响梯度与优化条件。归一化通过重新参数化中间表示，往往有助于稳定训练，但不会把任意输入强制变成高斯分布，也不保证所有网络都因此收敛更快。

内部协变量偏移是批归一化提出时的重要动机，但不是解释其效果的唯一机制。训练收益还涉及尺度不变性、优化几何及批统计带来的随机性，应通过具体实验判断。

输入标准化与网络归一化层也不同。前者通常使用训练集固定统计处理原始特征，后者位于计算图内部，参与[反向传播](/dl/backpropagation/)，并可能具有可学习参数或运行状态。

## 2. 归一化原理

### 2.1 统一表达与统计范围

设当前归一化集合为 $S$，包含 $M$ 个元素：

$$
\mu_S=\frac1M\sum_{i\in S}x_i,\qquad
v_S=\frac1M\sum_{i\in S}(x_i-\mu_S)^2
$$

$$
\widehat x_i=\frac{x_i-\mu_S}{\sqrt{v_S+\epsilon}},\qquad
y_i=\gamma_i\widehat x_i+\beta_i
$$

$\epsilon>0$ 防止零方差除零。仿射参数的共享方式由具体层决定，并非每个输入位置都有独立参数。

即使不使用仿射变换，归一化后的方差也为 $v_S/(v_S+\epsilon)$，通常只是接近 1。仿射变换后，输出更不必保持零均值或单位方差。

对于图像输入 $[N,C,H,W]$ 和序列输入 $[N,L,D]$：

| 方法 | 每组统计沿哪些轴计算 | 是否混合不同样本 |
| --- | --- | --- |
| 批归一化 | 图像中每个通道沿 $N,H,W$ | 是 |
| 层归一化 | 序列常见配置中每个位置沿 $D$ | 否 |
| 组归一化 | 每样本每组沿 $C/G,H,W$ | 否 |
| 实例归一化 | 每样本每通道沿 $H,W$ | 否 |
| 均方根归一化 | 常见序列配置沿 $D$ 计算平方均值 | 否 |

### 2.2 批归一化

对图像输入，固定通道 $c$，使用 $M=NHW$ 个元素计算均值与方差。常见仿射参数为每通道一组 $\gamma_c,\beta_c$。

默认训练模式使用当前批统计，并更新运行均值和运行方差。PyTorch 的运行均值约定为：

$$
\mu_{run}\leftarrow(1-m)\mu_{run}+m\mu_{batch}
$$

这里的 $m$ 是统计更新比例，与[优化器](/dl/optimizers/)动量系数的常见写法不同。

PyTorch 训练前向使用分母为 $M$ 的有偏方差，写入运行方差时则使用无偏估计：

$$
s_{batch}^2=\frac{M}{M-1}v_{batch},\qquad M>1
$$

默认评价模式使用运行统计，而不重新估计当前批均值与方差。若关闭运行统计跟踪，训练与评价都会使用当前输入统计，推理结果仍可能依赖批次组成。

### 2.3 有效样本数与分布式统计

批大小为 1 不代表统计集合只有一个元素。图像批归一化在空间尺寸较大时仍有 $HW$ 个元素；但空间相关性可能降低统计的有效独立信息量。

对于形状 $[1,C]$ 或 $[1,C,1,1]$，每通道统计集合只有一个元素，训练批归一化无法按常规方式估计方差，框架通常会拒绝这种输入。评价模式若使用已有运行统计，则是另一种情况。

分布式数据并行不会自动让普通批归一化使用全局批统计。同步批归一化可以在指定进程组内聚合统计，代价是额外通信；它也不能消除训练与部署分布差异。

梯度累积只扩大一次参数更新所覆盖的样本量，并不会自动扩大每次前向的归一化统计集合。

### 2.4 层归一化

层归一化沿配置指定的尾部维度计算统计。对序列 $[N,L,D]$，`nn.LayerNorm(D)` 为每个样本的每个序列位置，单独沿最后的 $D$ 维归一化。

若使用 `nn.LayerNorm((L,D))`，才会同时覆盖序列与隐藏维度。这两种配置语义不同，后者还把参数形状绑定到对应序列长度。

PyTorch 的逐元素仿射参数形状与 `normalized_shape` 一致。对于 `LayerNorm(D)`，参数在不同样本和序列位置之间共享，训练与评价均使用当前输入统计，没有运行均值和方差。

逐位置层归一化不需要对不同位置共同计算统计，但填充位置依然可能经偏置变成非零向量。注意力与损失中的填充掩码仍需单独处理。

### 2.5 组归一化

将 $C$ 个通道划分为 $G$ 组，要求 $C$ 能被 $G$ 整除。每个样本、每组独立归一化，统计元素数为：

$$
M=\frac CGHW
$$

它不跨批次混合样本，训练与评价均使用当前输入。仿射参数通常仍按通道设置，而不是每组只有一个缩放值。

$G=1$ 时，统计范围与对整张 $[C,H,W]$ 做层归一化一致，但默认仿射参数不同：组归一化按通道，整图层归一化可按位置设置。因此只有关闭仿射或适当约束参数后才能讨论完整输出等价。

$G=C$ 时，统计范围与实例归一化一致；还需匹配仿射配置、稳定项及运行统计策略，不能仅根据组数声称两个模块在所有模式下完全相同。

### 2.6 实例归一化

实例归一化对每个样本、每个通道独立处理空间元素，强调样本内部的相对空间变化。它在某些图像生成和风格变换任务中有用，但不能泛化为“完整去除图像风格”。

PyTorch 的 `InstanceNorm2d` 默认不启用仿射参数，也不跟踪运行统计，训练与评价都使用当前实例统计。启用相应选项后，模块行为会改变，比较方法时必须固定这些配置。

空间只有一个元素时，当前实例统计同样退化。不跨批次并不意味着任何形状下都能获得有意义的方差。

### 2.7 均方根归一化

均方根归一化不减均值，沿指定特征范围计算：

$$
r^2=\frac1D\sum_{j=1}^{D}x_j^2,\qquad
y_j=\gamma_j\frac{x_j}{\sqrt{r^2+\epsilon}}
$$

它通常只保留可学习缩放，不使用平移。与层归一化不同，输入整体加一个常数会改变输出；“取消均值”也不是假设输入均值始终为零。

忽略稳定项时，它对正比例缩放具有不变性；有限 $\epsilon$ 会打破精确不变性。计算减少可能带来性能收益，但实际速度取决于融合算子、设备和形状，不能使用统一百分比描述所有场景。

### 2.8 残差结构中的归一化位置

对残差子层 $F$，前置归一化为：

$$
\mathbf x_{l+1}=\mathbf x_l+F(\operatorname{Norm}(\mathbf x_l))
$$

后置归一化为：

$$
\mathbf x_{l+1}=\operatorname{Norm}(\mathbf x_l+F(\mathbf x_l))
$$

两种结构的梯度路径不同。前置形式保留了不经过当前归一化算子的残差路径，常有利于深层优化，但不能据此直接断言后置结构表达力更强或某种布局始终更优。

归一化位置属于模型结构的一部分。使用预训练权重时，不应随意交换顺序或替换归一化方法而不重新验证。

## 3. 公式实现

### 3.1 批统计与运行状态核对

双精度示例同时核对训练输出和运行方差。将统计动量设为 1，使本次运行统计直接等于本批估计，便于验证。

```python
import torch
from torch import nn


torch.manual_seed(42)
eps = 1e-5
image = torch.randn(2, 4, 3, 3, dtype=torch.float64)
bn = nn.BatchNorm2d(4, eps=eps, momentum=1.0, affine=False).double()
bn.train()
bn_output = bn(image)
axes = (0, 2, 3)
mean = image.mean(dim=axes, keepdim=True)
variance = image.var(dim=axes, correction=0, keepdim=True)
manual_bn = (image - mean) * torch.rsqrt(variance + eps)
assert torch.allclose(bn_output, manual_bn, atol=1e-12)
assert torch.allclose(bn.running_var, image.var(dim=axes, correction=1))

bn.eval()
evaluation = bn(image)
manual_eval = ((image - bn.running_mean[None, :, None, None]) *
               torch.rsqrt(bn.running_var[None, :, None, None] + eps))
assert torch.allclose(evaluation, manual_eval, atol=1e-12)
print("batch_statistics_and_running_state:", True)
```

即使评价输入与训练输入相同，两次输出也可能不同，因为这里训练前向与运行方差采用不同估计方式。普通训练中还叠加了运行平均历史。

### 3.2 层、组与实例统计核对

```python
sequence = torch.randn(2, 5, 4, dtype=torch.float64)
ln = nn.LayerNorm(4, eps=eps, elementwise_affine=False).double()
sequence_mean = sequence.mean(dim=-1, keepdim=True)
sequence_variance = sequence.var(dim=-1, correction=0, keepdim=True)
manual_ln = (sequence - sequence_mean) * torch.rsqrt(sequence_variance + eps)
assert torch.allclose(ln(sequence), manual_ln, atol=1e-12)

# 分组后形状为 [样本数, 组数, 组内通道数, 高, 宽]。
grouped = image.reshape(2, 2, 2, 3, 3)
group_mean = grouped.mean(dim=(2, 3, 4), keepdim=True)
group_variance = grouped.var(dim=(2, 3, 4), correction=0, keepdim=True)
manual_gn = ((grouped - group_mean) * torch.rsqrt(group_variance + eps)).reshape_as(image)
gn = nn.GroupNorm(2, 4, eps=eps, affine=False).double()
assert torch.allclose(gn(image), manual_gn, atol=1e-12)

in_layer = nn.InstanceNorm2d(4, eps=eps, affine=False,
                            track_running_stats=False).double()
gn_channels = nn.GroupNorm(4, 4, eps=eps, affine=False).double()
assert torch.allclose(in_layer(image), gn_channels(image), atol=1e-12)
gn_single = nn.GroupNorm(1, 4, eps=eps, affine=False).double()
ln_image = nn.LayerNorm((4, 3, 3), eps=eps, elementwise_affine=False).double()
assert torch.allclose(gn_single(image), ln_image(image), atol=1e-12)
print("normalization_axes_and_special_cases:", True)
```

这些核对关闭仿射变换，并固定稳定项与运行状态，验证的是明确条件下的等价关系，而不是默认模块之间的无条件互换。

### 3.3 均方根缩放与低精度处理

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        if dim < 1 or eps <= 0:
            raise ValueError("invalid dimensions or epsilon")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, inputs):
        if inputs.shape[-1] != self.weight.numel():
            raise ValueError("last dimension does not match weight")
        compute = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
        inverse_rms = torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + self.eps)
        result = compute * inverse_rms * self.weight.to(compute.dtype)
        return result.to(inputs.dtype)


rms = RMSNorm(4).double()
rms_input = sequence.clone().requires_grad_(True)
rms_output = rms(rms_input)
expected = rms_input / torch.sqrt(rms_input.square().mean(dim=-1, keepdim=True) + rms.eps)
assert torch.allclose(rms_output, expected)
rms_output.square().mean().backward()
assert torch.isfinite(rms_input.grad).all()
print("rms_formula_and_gradient:", True)
```

半精度输入先以单精度累计平方均值，最后转换回输入精度。该做法减少统计溢出风险，但极端输入、最终类型转换和参数精度仍需单独检查；它不等于对任意数值都安全。

### 3.4 训练、推理与部署检查

1. **先核对轴再核对数值**：最后一维是否确实为隐藏维度，组数是否整除通道数，统计集合是否退化。
2. **区分模式与梯度开关**：`eval()` 控制运行统计等层行为，`no_grad()` 控制求导记录，二者不能互相替代。
3. **保存运行状态**：批归一化的运行均值、方差属于模型状态，部署时需与权重一同加载。
4. **保持掩码语义**：跨序列位置计算统计时，填充值可能污染均值与方差；默认归一化层通常不自动理解填充掩码。
5. **依据验证选择方法**：小批次、域偏移和空间相关性都会影响统计，不能只凭任务名称确定某种方法必然最好。

固定运行统计后，批归一化可与紧邻的仿射层按条件折叠，减少推理算子。使用输入自身统计的层归一化等方法则不能以相同方式变成一组固定权重。

## 4. 总结

归一化方法的差异首先在于统计轴：批归一化跨样本，层归一化沿指定尾部维度，组归一化在样本内分组，实例归一化独立处理各通道空间，均方根归一化则省略均值中心化。

正确实现还需匹配方差估计、仿射参数、稳定项与训练推理状态。明确这些条件，再通过逐轴公式核对和独立验证判断效果，能够避免名称相近或统计范围相似导致的错误替换。

## 5. 参考文献

[1] Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015: 448–456.

[2] Ba J L, Kiros J R, Hinton G E. Layer Normalization. arXiv:1607.06450, 2016.

[3] Wu Y, He K. Group Normalization. ECCV, 2018: 3–19.

[4] Ulyanov D, Vedaldi A, Lempitsky V. Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv:1607.08022, 2016.

[5] Zhang B, Sennrich R. Root Mean Square Layer Normalization. NeurIPS, 2019, 32.
