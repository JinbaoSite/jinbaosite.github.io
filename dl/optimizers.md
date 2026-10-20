# 优化器：从梯度下降到自适应更新

优化器根据梯度与历史状态调整参数，决定每次更新的方向和幅度。梯度下降提供基本规则，动量平滑方向，自适应方法根据梯度统计缩放不同坐标，权重衰减则额外约束参数规模。

[反向传播](/dl/backpropagation/)负责计算导数，优化器使用这些导数执行更新。选择优化器时，需要同时考虑损失缩放、学习率、状态内存和验证表现，而不能仅比较算法名称。

## 1. 背景

设模型参数为 $\boldsymbol\theta$，训练目标为经验损失。第 $t$ 步在当前参数 $\boldsymbol\theta_{t-1}$ 上计算梯度：

$$
\mathbf g_t=\nabla_{\boldsymbol\theta}\mathcal L_{\mathcal B_t}(\boldsymbol\theta_{t-1})
$$

$\mathcal B_t$ 可以是完整训练集，也可以是小批量样本。基础更新为：

$$
\boldsymbol\theta_t=\boldsymbol\theta_{t-1}-\eta_t\mathbf g_t
$$

不同特征和参数方向的曲率可能差异很大，同一学习率可能在某些方向过大、另一些方向过小。小批量梯度还带有采样噪声，优化器因此需要在步长、历史信息和资源开销之间权衡。

## 2. 更新原理

### 2.1 梯度下降与随机梯度下降

全批量梯度下降每步使用全部数据。随机梯度下降通常指单样本或小批量梯度更新；PyTorch 的 `SGD` 可以接收任意来源的梯度，实际是否随机取决于数据如何组织。

学习率过大可能导致震荡或发散，过小则更新缓慢。梯度下降并非一定收敛慢，也不能仅凭自适应算法就保证更少训练时间，效果取决于曲率、噪声和配置。

### 2.2 动量与前瞻更新

采用累加梯度的动量约定：

$$
\mathbf v_t=\mu\mathbf v_{t-1}+\mathbf g_t,\qquad
\boldsymbol\theta_t=\boldsymbol\theta_{t-1}-\eta_t\mathbf v_t
$$

其中 $0\le\mu<1$，初始状态为零。持续一致的方向会累积，符号反复变化的分量则部分抵消。

另一种常见写法在新梯度前乘 $1-\mu$。两种状态尺度不同，学习率和初始化也需要相应调整，不能直接混用公式和代码配置。

前瞻动量的一种参数化形式在预估位置求梯度。框架实现也可以采用等价或相关重参数化；例如 PyTorch 在无阻尼设定下，可写为先更新缓冲区，再使用 $\mathbf g_t+\mu\mathbf v_t$ 作为更新方向。比较实现时需要核对缓冲区初始化、阻尼和参数表示。

### 2.3 累积平方梯度

AdaGrad 累积逐坐标平方梯度：

$$
\mathbf s_t=\mathbf s_{t-1}+\mathbf g_t^2,\qquad
\boldsymbol\theta_t=\boldsymbol\theta_{t-1}
-\eta_t\frac{\mathbf g_t}{\sqrt{\mathbf s_t}+\epsilon}
$$

平方、除法与开方均为逐元素操作。经常获得大梯度的坐标被更强地缩放，适合某些稀疏特征场景。由于状态只增不减，长期训练时有效步长可能持续下降，但不能据此推断算法最终必然停止学习。

稳定项放在平方根内外会改变计算结果。公式与实际工具的 $\epsilon$ 位置必须保持一致。

### 2.4 指数平均平方梯度

RMSprop 用指数移动平均替代长期累积：

$$
\mathbf s_t=\rho\mathbf s_{t-1}+(1-\rho)\mathbf g_t^2
$$

$$
\boldsymbol\theta_t=\boldsymbol\theta_{t-1}
-\eta_t\frac{\mathbf g_t}{\sqrt{\mathbf s_t}+\epsilon}
$$

这使缩放更侧重近期梯度。动量、中心化平方统计和稳定项位置等选项会产生不同变体，复现实验时需要固定对应约定。

### 2.5 一阶矩与二阶矩自适应更新

Adam 维护两组指数平均：

$$
\mathbf m_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t
$$

$$
\mathbf v_t=\beta_2\mathbf v_{t-1}+(1-\beta_2)\mathbf g_t^2
$$

$\mathbf v_t$ 是未中心化二阶矩的估计，不是梯度方差。零初始化会使前期统计偏向零，标准偏差修正为：

$$
\widehat{\mathbf m}_t=\frac{\mathbf m_t}{1-\beta_1^t},\qquad
\widehat{\mathbf v}_t=\frac{\mathbf v_t}{1-\beta_2^t}
$$

参数更新为：

$$
\boldsymbol\theta_t=\boldsymbol\theta_{t-1}
-\eta_t\frac{\widehat{\mathbf m}_t}{\sqrt{\widehat{\mathbf v}_t}+\epsilon}
$$

自适应缩放不意味着无需调整学习率。Adam 的训练速度和泛化表现取决于任务与超参数，不能将差异单一归因于“找到尖锐极小值”。

### 2.6 平方惩罚与解耦权重衰减

在损失中加入 $\lambda\|\boldsymbol\theta\|_2^2/2$，梯度变为：

$$
\widetilde{\mathbf g}_t=\mathbf g_t+\lambda\boldsymbol\theta_{t-1}
$$

对无动量的普通梯度下降：

$$
\boldsymbol\theta_t=(1-\eta_t\lambda)\boldsymbol\theta_{t-1}-\eta_t\mathbf g_t
$$

它与直接参数衰减等价。但在自适应优化器中，把惩罚项加入梯度还会改变一阶、二阶统计，并被逐坐标缩放，因此不再等价于直接收缩参数。

AdamW 使用不含衰减项的梯度更新矩估计，再执行：

$$
\boldsymbol\theta_t=(1-\eta_t\lambda)\boldsymbol\theta_{t-1}
-\eta_t\frac{\widehat{\mathbf m}_t}{\sqrt{\widehat{\mathbf v}_t}+\epsilon}
$$

“解耦”表示衰减不进入矩估计，不表示它与学习率无关。最终累计收缩量还取决于学习率调度和训练步数。

带动量的梯度惩罚也会进入动量状态，因此普通无动量梯度下降的等价结论不能无条件推广。不同优化器的衰减系数不存在统一的倍数换算规则。

### 2.7 分块信任比

LAMB 在自适应方向上增加按参数块计算的信任比。一个常用约定为：

$$
\mathbf u_t^{(l)}=
\frac{\widehat{\mathbf m}_t^{(l)}}{\sqrt{\widehat{\mathbf v}_t^{(l)}}+\epsilon}
+\lambda\boldsymbol\theta_{t-1}^{(l)}
$$

$$
r_t^{(l)}=\frac{\phi(\|\boldsymbol\theta_{t-1}^{(l)}\|_2)}{\|\mathbf u_t^{(l)}\|_2},\qquad
\boldsymbol\theta_t^{(l)}=\boldsymbol\theta_{t-1}^{(l)}-\eta_tr_t^{(l)}\mathbf u_t^{(l)}
$$

$\phi$ 可以约束参数范数。常见变体直接使用范数，并在参数范数或更新范数为零时令信任比为 1，避免零参数永远无法更新。具体实现还可能对部分参数关闭信任比。

分母必须与实际缩放的更新方向一致。若方向包含衰减项，却只用不含衰减项的方向计算范数，就变成另一种更新规则。

参数块可能是单个张量，不一定等于完整网络层。LAMB 面向大批量训练的稳定性问题，但批量达到某个阈值并不意味着必须使用它。

### 2.8 符号动量更新

Lion 保存一组动量状态，并用混合方向的符号更新：

$$
\mathbf c_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t
$$

$$
\boldsymbol\theta_t=(1-\eta_t\lambda)\boldsymbol\theta_{t-1}
-\eta_t\operatorname{sign}(\mathbf c_t)
$$

$$
\mathbf m_t=\beta_2\mathbf m_{t-1}+(1-\beta_2)\mathbf g_t
$$

符号函数取值为 $-1,0,1$，零分量不会产生符号步进。更新先使用旧动量，再更新状态，交换顺序会改变算法。

只保留一组状态可以减少相对 Adam 的矩状态内存，但不等于总训练显存减半。符号方向的幅度也不同，不能直接沿用 AdamW 的学习率和衰减配置。

### 2.9 状态内存与比较方式

设有 $P$ 个参数，且状态与参数使用相同精度，额外持久状态的简化估计为：

| 优化器 | 主要额外状态 | 约需状态元素数 |
| --- | --- | --- |
| 无动量 SGD | 无 | $0$ |
| 动量 SGD | 动量 | $P$ |
| AdaGrad | 平方梯度累积 | $P$ |
| 基础 RMSprop | 平方梯度平均 | $P$ |
| Adam、AdamW | 两组矩估计 | $2P$ |
| 常见 LAMB | 两组矩估计 | $2P$ |
| Lion | 一组动量 | $P$ |

表格不包括参数、梯度、激活、临时张量、主权重和分布式通信缓存。实际内存还受混合精度、状态分片与融合实现影响。

公平比较需要固定数据划分，并为不同优化器分别调整学习率、衰减和调度。训练步数、样本量、计算预算与最终验证指标应同时报告。

## 3. 更新实现

### 3.1 手动实现与框架核对

双精度示例在固定梯度序列上比较手写 AdamW 与框架更新，覆盖偏差修正、矩状态和解耦衰减。

```python
import torch
from torch import nn


def manual_adamw_step(parameter, gradient, state, lr, betas, eps, weight_decay):
    beta1, beta2 = betas
    state["step"] += 1
    state["m"] = beta1 * state["m"] + (1 - beta1) * gradient
    state["v"] = beta2 * state["v"] + (1 - beta2) * gradient.square()
    step = state["step"]
    m_hat = state["m"] / (1 - beta1 ** step)
    v_hat = state["v"] / (1 - beta2 ** step)
    return parameter * (1 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps)


manual = torch.tensor([1., -2.], dtype=torch.float64)
parameter = nn.Parameter(manual.clone())
settings = dict(lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)
optimizer = torch.optim.AdamW([parameter], **settings, foreach=False)
state = {"step": 0, "m": torch.zeros_like(manual), "v": torch.zeros_like(manual)}
gradients = [[0.3, -0.4], [-0.2, 0.1], [0.0, 0.0]]
for values in gradients:
    gradient = torch.tensor(values, dtype=torch.float64)
    manual = manual_adamw_step(manual, gradient, state, **settings)
    parameter.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.allclose(parameter.detach(), manual, atol=1e-12, rtol=1e-12)
print("adamw_update_check:", True)
```

零梯度张量仍会触发状态更新和权重衰减，已有动量也可能继续影响参数。`.grad` 为 `None` 则通常使该参数在当步被跳过，两者不能视为同一种情况。

### 3.2 完整训练示例

合成回归任务使用固定种子，训练、验证与测试样本独立，最佳权重由验证损失决定。偏置单独放入不衰减参数组，展示参数组配置；这是一种实验选择，不是所有模型必须遵循的规则。

```python
import copy


torch.manual_seed(42)
X = torch.randn(600, 4)
true_weight = torch.tensor([1.5, -2., 0.5, 3.])
y = X @ true_weight + 0.7 + 0.1 * torch.randn(600)
X_train, X_valid, X_test = X[:360], X[360:480], X[480:]
y_train, y_valid, y_test = y[:360], y[360:480], y[480:]
model = nn.Linear(4, 1)
optimizer = torch.optim.AdamW([
    {"params": [model.weight], "weight_decay": 0.01},
    {"params": [model.bias], "weight_decay": 0.0},
], lr=0.03)
criterion = nn.MSELoss()
best_loss, best_state, waiting = float("inf"), None, 0

for epoch in range(800):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(X_train).squeeze(-1), y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = criterion(model(X_valid).squeeze(-1), y_valid).item()
    if valid_loss < best_loss - 1e-6:
        best_loss = valid_loss
        best_state = copy.deepcopy(model.state_dict())
        waiting = 0
    else:
        waiting += 1
        if waiting >= 50:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_mse = criterion(model(X_test).squeeze(-1), y_test).item()
print("test_mse:", round(test_mse, 6))
```

示例验证更新流程，不构成优化器优劣排名。恢复模型后若要继续训练，还需恢复匹配时刻的优化器状态、学习率调度状态及必要的随机状态，而不只是模型权重。

### 3.3 训练诊断与工程边界

1. **检查学习率与损失归约**：从求和改成均值会改变梯度尺度，迁移配置时需要重新核对更新幅度。
2. **区分梯度累加与每步更新**：微批损失按实际样本数缩放，在预定更新边界执行清空和参数更新。
3. **按需裁剪梯度**：裁剪是控制异常梯度的工具，并非某类模型无条件必需。混合精度下，范数裁剪应作用于反缩放后的梯度。
4. **监控状态与非有限值**：梯度、参数和矩状态都需要检查，单次非有限值可能污染后续更新。
5. **保存完整训练配置**：稳定项、偏差修正、衰减方式、参数组和状态精度都会影响复现。

学习率调度改变时间维度上的全局步长，自适应缩放改变坐标尺度，两者可以同时使用。具体调度机制可结合[学习率调度](/dl/lr_schedule/)与训练预算选择。

## 4. 总结

优化器在梯度方向、历史平滑、坐标缩放和参数约束之间进行组合。动量方法积累方向，AdaGrad 与 RMSprop 使用平方统计调节尺度，Adam 同时维护两组矩估计，AdamW 将衰减从矩估计中分离。LAMB 增加分块信任比，Lion 使用符号更新减少状态需求。

公式约定、状态处理和训练配置共同决定真实行为。通过固定梯度序列核对实现，再通过独立验证比较训练效果，比依据算法标签预设泛化与速度结论更可靠。

## 5. 参考文献

[1] Duchi J, Hazan E, Singer Y. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 2011, 12: 2121–2159.

[2] Kingma D P, Ba J. Adam: A Method for Stochastic Optimization. ICLR, 2015.

[3] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. ICLR, 2019.

[4] You Y, Li J, Reddi S, et al. Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes. ICLR, 2020.

[5] Chen X, Liang C, Huang D, et al. Symbolic Discovery of Optimization Algorithms. NeurIPS, 2023, 36.
