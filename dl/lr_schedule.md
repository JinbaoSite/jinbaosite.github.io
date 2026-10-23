# 学习率调度：从预热到周期退火

学习率调度按照训练进度或验证指标调整优化器的全局步长。预热控制训练初期的更新幅度，衰减在后期逐步减小步长，周期策略则在预算内安排一次或多次升降。

调度器不替代[优化器](/dl/optimizers/)。动量与自适应统计决定更新方向和坐标缩放，学习率决定整体更新尺度；两者的配置、调用顺序和状态共同影响实际训练轨迹。

## 1. 背景

对基础梯度下降：

$$
\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t\mathbf g_t
$$

$\eta_t$ 是本次参数更新使用的学习率。训练初期的统计与激活尺度尚未稳定，较温和的更新可能有益；后期减小步长则可能降低随机梯度导致的波动。但这些不是所有任务必须遵循的固定阶段。

较小步长不意味着模型一定到达“更尖锐且泛化更好”的极小值，训练损失下降也不保证验证指标提升。峰值学习率、衰减终点和总更新预算需要通过独立验证共同选择。

## 2. 调度原理

### 2.1 明确时间单位

调度时间可以是参数更新次数、完整数据遍历次数或已处理词元数量，三者不能混用。按步调度时，以真正执行的优化器更新为基准，梯度累积期间的每个微批不单独推进调度。

设数据加载器每轮有 $K$ 个微批，每 $A$ 个微批更新一次，若每轮处理最后不足 $A$ 个的分组，则 $E$ 轮更新预算为：

$$
T=E\left\lceil\frac KA\right\rceil
$$

若丢弃末尾分组或跨轮累积，公式需要改变。混合精度出现溢出并跳过更新时，也要明确调度是否按成功更新推进，不能只依据循环次数假设参数已经更新。

### 2.2 阶梯与指数衰减

以零开始的调度索引 $t$，阶梯衰减为：

$$
\eta_t=\eta_0\gamma^{\lfloor t/S\rfloor}
$$

$S$ 是衰减间隔，$0<\gamma<1$。指定多个衰减里程碑时，每跨过一个里程碑乘一次 $\gamma$。指数衰减则为：

$$
\eta_t=\eta_0\gamma^t
$$

同样的 $\gamma$ 若按更新步调用与按整轮调用，会产生完全不同的下降速度。PyTorch 的调度器并不会自行知道一个 `step()` 应对应哪种训练单位。

### 2.3 线性预热

预热从较小学习率逐渐升至峰值。它可以缓解某些配置下的早期不稳定，但不能修复错误标签、非有限输入或任意过大的峰值学习率。

为了对齐离散训练代码，规定计划执行 $T$ 次更新，第 $t$ 次更新使用 $\eta(t)$，索引为 $t=0,\ldots,T-1$。预热包含 $W\ge2$ 次更新，起始比例为 $r_s$：

$$
\eta(t)=\eta_{peak}\left[r_s+(1-r_s)\frac{t}{W-1}\right],\qquad0\le t<W
$$

第一步使用 $r_s\eta_{peak}$，第 $W$ 次更新达到峰值。这是明确的离散约定，并非唯一正确的预热公式；其他实现可能把峰值放在不同索引上。

### 2.4 预热与余弦衰减

预热结束后，以：

$$
p(t)=\frac{t-(W-1)}{T-W}
$$

表示衰减进度，使用：

$$
\eta(t)=\eta_{min}+\frac{\eta_{peak}-\eta_{min}}2
\left[1+\cos(\pi p(t))\right],\qquad W\le t<T
$$

这样在 $t=W-1$ 达到峰值，在最后一次计划更新 $t=T-1$ 使用最低学习率。无预热时改用 $p(t)=t/(T-1)$，从峰值直接衰减。

实现还应把超出预算的索引限制在终点，避免余弦继续变化。PyTorch 的 `CosineAnnealingLR` 没有保证超过 `T_max` 后永久保持 `eta_min`；持续推进可能重新升高。自定义曲线与内置调度器的边界约定需要分别核对。

### 2.5 周期退火与热重启

周期余弦在每个周期内部从较高值降到较低值，再在新周期恢复较高学习率。若当前周期起点为 $s_i$，长度为 $T_i$，则：

$$
\eta(t)=\eta_{min}+\frac{\eta_{max}-\eta_{min}}2
\left[1+\cos\left(\pi\frac{t-s_i}{T_i}\right)\right]
$$

周期长度可固定或逐次扩展，不能在变长周期中始终对全局索引取同一个周期长度的模。

学习率热重启通常不意味着重新初始化参数或清空优化器状态。它提供不同更新节奏，是否改善泛化需要实验，不能保证一定跳出不良区域。

### 2.6 单周期调度

OneCycle 在有限预算内先升高、再降低学习率，还可以反向调整动量。PyTorch 的默认配置采用两阶段安排；启用 `three_phase=True` 才采用三阶段形式。

参数关系为：

$$
\eta_{initial}=\frac{\eta_{max}}{\mathrm{div\_factor}},\qquad
\eta_{final}=\frac{\eta_{initial}}{\mathrm{final\_div\_factor}}
$$

最终值的分母包含两个因子，不能只用最大值除以 `final_div_factor`。`pct_start` 控制上升阶段占比，离散端点还受总步数与实现插值规则影响。

`cycle_momentum=True` 时，调度器可能改变 SGD 动量或 Adam 类优化器的一阶矩衰减系数。若只想观察学习率效应，应显式关闭该选项。更高峰值与更快收敛只在某些配置下成立，不存在统一的训练加速倍数。

### 2.7 指标驱动衰减

`ReduceLROnPlateau` 观察验证指标，在持续没有改善时降低学习率。它需要输入评价值，而非仅接收步数。

验证损失通常使用最小化模式，准确率等指标通常使用最大化模式。指标方向、改善阈值、耐心轮数和冷却期必须与验证频率一起设置。

指标驱动调度在验证完成后推进，和每次参数更新后推进的固定时间曲线具有不同调用节奏。测试集不能用于决定何时衰减。

### 2.8 学习率范围测试

范围测试短暂地增大学习率，观察损失和梯度的变化，以缩小搜索区间。指数递增的一种形式为：

$$
\eta_t=\eta_{start}\left(\frac{\eta_{end}}{\eta_{start}}\right)^{t/(T-1)}
$$

记录的学习率必须是产生当前损失和更新时的值，不能更新调度后再把下一步学习率与上一批损失配对。

测试会改变模型和优化器状态。正式训练前应恢复匹配的初始状态、随机状态及必要的数据迭代状态，或直接使用独立的试验副本。发散前某个固定比例并不是普适的最优学习率选择规则。

## 3. 调度实现

### 3.1 明确边界的预热余弦函数

函数返回相对峰值的倍率，供 `LambdaLR` 使用。采用第二部分的离散约定，禁止只有一个预热点等含糊边界，并在预算结束后保持最低比例。

```python
import math
import copy
import torch
from torch import nn


def warmup_cosine_factor(index, total_steps, warmup_steps,
                         start_ratio=0.1, min_ratio=0.05):
    if total_steps < 2 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("invalid step budget")
    if warmup_steps == 1:
        raise ValueError("use zero or at least two warmup steps")
    if not (0 < start_ratio <= 1 and 0 <= min_ratio <= 1):
        raise ValueError("invalid learning-rate ratios")
    index = min(max(index, 0), total_steps - 1)
    if warmup_steps and index < warmup_steps:
        return start_ratio + (1 - start_ratio) * index / (warmup_steps - 1)
    if warmup_steps:
        progress = (index - (warmup_steps - 1)) / (total_steps - warmup_steps)
    else:
        progress = index / (total_steps - 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


schedule_config = dict(total_steps=200, warmup_steps=20,
                       start_ratio=0.1, min_ratio=0.05)


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda index: warmup_cosine_factor(index, **config)
    )
```

`LambdaLR` 接收倍率而非绝对学习率，各参数组以各自基础学习率乘以相同倍率。若希望各组采用不同曲线，可以显式提供不同倍率函数。

### 3.2 按更新步推进的训练循环

固定随机种子生成回归数据，用全批量更新展示调用顺序。调度器创建后已设置初始学习率，每次先使用当前值更新，再推进到下一步。

```python
torch.manual_seed(42)
X = torch.randn(160, 3, dtype=torch.float64)
y = X @ torch.tensor([1.5, -2.0, 0.7], dtype=torch.float64) + 0.3


def make_training_state(config):
    model = nn.Linear(3, 1).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    scheduler = make_scheduler(optimizer, config)
    return model, optimizer, scheduler


model, optimizer, scheduler = make_training_state(schedule_config)
criterion = nn.MSELoss()
used_lrs = []
checkpoint = None

for step in range(schedule_config["total_steps"]):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(X).squeeze(-1), y)
    loss.backward()
    used_lrs.append(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()

    if step == 79:
        checkpoint = copy.deepcopy({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_steps": step + 1,
            "schedule_config": schedule_config,
        })

print("first_peak_last_lr:", used_lrs[0], used_lrs[19], used_lrs[-1])
print("completed_updates:", len(used_lrs))
```

首步、峰值与最后一步学习率分别为 $0.005$、$0.05$ 和 $0.0025$。记录发生在 `optimizer.step()` 之前，因此反映真实使用值，而非预先设置的下一步值。

### 3.3 断点续训核对

恢复时先创建优化器和调度器，再加载匹配状态。匿名倍率函数并不会作为完整训练配置保存在状态字典中，因此必须保存并重建曲线配置。

```python
resumed_model, resumed_optimizer, resumed_scheduler = make_training_state(
    checkpoint["schedule_config"]
)
resumed_model.load_state_dict(checkpoint["model"])
resumed_scheduler.load_state_dict(checkpoint["scheduler"])
resumed_optimizer.load_state_dict(checkpoint["optimizer"])
resumed_lrs = []

for step in range(checkpoint["completed_steps"], schedule_config["total_steps"]):
    resumed_optimizer.zero_grad(set_to_none=True)
    loss = criterion(resumed_model(X).squeeze(-1), y)
    loss.backward()
    resumed_lrs.append(resumed_optimizer.param_groups[0]["lr"])
    resumed_optimizer.step()
    resumed_scheduler.step()

assert resumed_lrs == used_lrs[checkpoint["completed_steps"]:]
assert all(torch.allclose(left, right, atol=1e-12, rtol=1e-12)
           for left, right in zip(model.parameters(), resumed_model.parameters()))
print("resume_matches_continuous_training:", True)
```

示例没有随机失活或随机小批抽取，恢复训练不需要额外重放随机运算。实际训练还需保存随机数、采样器、混合精度缩放器等状态，明确检查点位于更新前还是更新后。

### 3.4 单周期调用示例

```python
cycle_parameter = nn.Parameter(torch.tensor(1.0))
cycle_optimizer = torch.optim.SGD([cycle_parameter], lr=0.01, momentum=0.9)
cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    cycle_optimizer, max_lr=0.1, total_steps=20,
    pct_start=0.3, div_factor=10.0, final_div_factor=100.0,
    cycle_momentum=False, three_phase=False,
)
cycle_used_lrs = []
for _ in range(20):
    cycle_optimizer.zero_grad(set_to_none=True)
    cycle_parameter.square().backward()
    cycle_used_lrs.append(cycle_optimizer.param_groups[0]["lr"])
    cycle_optimizer.step()
    cycle_scheduler.step()
print("onecycle_first_last:", cycle_used_lrs[0], cycle_used_lrs[-1])
```

起始与最终使用值约为 $0.01$ 和 $0.0001$。总更新次数不能超过配置预算，也不应根据最后一次 `scheduler.step()` 之后的学习率，反推最后一次更新使用了哪个值。

### 3.5 配置与诊断

| 情况 | 优先检查 |
| --- | --- |
| 首步学习率与预期不同 | 调度器初始化行为、预热索引和参数组基础值 |
| 衰减过快 | 每批与每轮调用单位是否混淆 |
| 后期学习率重新升高 | 是否超过曲线预算或启用了周期重启 |
| 恢复后轨迹改变 | 优化器状态、调度状态、曲线配置与检查点边界 |
| 更新跳过但曲线继续 | 混合精度溢出处理与调度推进策略 |
| 验证指标驱动调度无效 | 指标方向、验证频率和传入值 |

改变总训练预算会改变曲线含义，不能只加载旧步数就认为新计划等价。持续训练需要明确是继续原计划、重新预热，还是设计新的衰减段，并独立评价其效果。

## 4. 总结

学习率调度通过训练时间或验证反馈改变整体步长。预热、阶梯衰减、余弦与单周期各有不同预算安排，实际行为取决于离散索引、调用频率及参数组配置。

可靠实现需要记录真实使用值，在成功更新边界推进调度，并保存可重建的曲线和训练状态。边界测试与连续训练、断点恢复的对照验证，可以发现许多仅观察损失难以定位的调度错误。

## 5. 参考文献

[1] Loshchilov I, Hutter F. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR, 2017.

[2] Smith L N. Cyclical Learning Rates for Training Neural Networks. WACV, 2017: 464–472.

[3] Smith L N, Topin N. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, 2019, 11006.

[4] Goyal P, Dollár P, Girshick R, et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677, 2017.
