# 反向传播：链式法则与自动求导

反向传播沿计算图的反方向应用链式法则，高效计算标量目标对大量参数的梯度。自动求导系统在执行前向运算时记录依赖关系，再按算子的局部求导规则传播梯度，从而支持神经网络训练。

[多层感知机](/dl/mlp/)展示了非线性网络的组成，反向传播则回答每个权重应如何获得梯度。它负责求导，优化器负责根据梯度更新参数；梯度计算与参数更新需要分别理解。

## 1. 背景

设模型有数百万个参数，训练目标为标量损失 $L$。若分别扰动每个参数再重新执行前向计算，求出全部梯度的成本会随参数数量显著增加。

神经网络由矩阵乘法、加法、激活和损失等基本运算组合而成。各运算的局部导数已知，链式法则允许复用中间结果，一次反向遍历即可把损失梯度传播到所有相关参数。

自动求导不是通过微小扰动近似导数，也通常不需要把完整函数展开成符号表达式。它沿实际执行的计算路径，组合基本算子的导数规则；结果仍受到浮点精度与算子求导约定的影响。

## 2. 基本原理

### 2.1 标量链式法则

对于复合函数 $u=g(x)$、$L=f(u)$：

$$
\frac{dL}{dx}=\frac{dL}{du}\frac{du}{dx}
$$

如果一个变量通过多条路径影响目标，各路径的贡献需要相加。设 $L=f(u,v)$，$u=g(x)$，$v=h(x)$：

$$
\frac{dL}{dx}=\frac{\partial L}{\partial u}\frac{du}{dx}
+\frac{\partial L}{\partial v}\frac{dv}{dx}
$$

因此，反向传播既包含沿路径相乘，也包含分支汇合处求和。忽略累加会让共享参数、残差连接和重复使用变量的梯度出错。

### 2.2 计算图与手算示例

考虑：

$$
u=ab,\qquad v=u+c,\qquad L=v^2+u
$$

取 $a=2$、$b=3$、$c=1$，前向得到 $u=6$、$v=7$、$L=55$。变量 $u$ 同时通过平方分支和直接加法分支影响损失。

从 $\partial L/\partial L=1$ 开始反传：

$$
\frac{\partial L}{\partial v}=2v=14
$$

$$
\frac{\partial L}{\partial u}=
\frac{\partial L}{\partial v}\frac{\partial v}{\partial u}+1=15
$$

$$
\frac{\partial L}{\partial a}=15b=45,\qquad
\frac{\partial L}{\partial b}=15a=30,\qquad
\frac{\partial L}{\partial c}=14
$$

反向顺序必须满足依赖关系：先累计一个节点从所有下游路径收到的梯度，再向它的输入传播。

### 2.3 向量与雅可比矩阵

设 $\mathbf y=f(\mathbf x)$，$\mathbf x\in\mathbb R^n$，$\mathbf y\in\mathbb R^m$，雅可比矩阵为：

$$
J_{ij}=\frac{\partial y_i}{\partial x_j}
$$

使用列向量梯度约定，上游梯度 $\mathbf v=\nabla_{\mathbf y}L$ 反传为：

$$
\nabla_{\mathbf x}L=J^T\mathbf v
$$

这通常称为向量与雅可比乘积；在行向量约定下写为 $\mathbf v^TJ$。自动求导通常直接计算这个乘积，无需显式保存完整雅可比矩阵。

当输出不是标量时，必须说明希望对输出做怎样的加权。提供向量 $\mathbf v$ 等价于求 $\mathbf v^T\mathbf y$ 对输入的梯度，不会自动返回每个输出对应的全部导数。

### 2.4 全连接层与广播

批量输入按行排列，设：

$$
Z=XW^T+\mathbf b,\qquad
X\in\mathbb R^{B\times d},\quad W\in\mathbb R^{h\times d}
$$

上游梯度为 $G=\partial L/\partial Z\in\mathbb R^{B\times h}$，则：

$$
\frac{\partial L}{\partial X}=GW,\qquad
\frac{\partial L}{\partial W}=G^TX,\qquad
\frac{\partial L}{\partial\mathbf b}=\sum_{i=1}^{B}G_{i,:}
$$

偏置在前向时广播到整个批次，反向就需要沿被广播的维度求和。梯度最终形状必须与对应参数形状一致。

若损失使用样本均值，上游梯度已包含 $1/B$；不能在每层反传时重复除以批大小。多输出损失还需要核对归约是按样本数还是全部元素数求均值。

### 2.5 激活函数与损失层

逐元素激活 $H=\phi(Z)$ 的梯度为：

$$
\frac{\partial L}{\partial Z}=\frac{\partial L}{\partial H}\odot\phi'(Z)
$$

例如 $\tanh'(z)=1-\tanh^2(z)$。修正线性激活在正值处导数为 1、负值处为 0，在零点不可微，需要使用实现约定的导数值。

二分类交叉熵以得分 $z$ 为输入时：

$$
\frac{\partial\ell}{\partial z}=\sigma(z)-y
$$

互斥多分类中，归一化指数函数与交叉熵组合后，对类别得分的梯度为预测概率减标签指示向量。融合损失实现可以同时简化计算并改善数值稳定性。

### 2.6 反向模式与前向模式

前向模式传播输入扰动方向，计算 $J\mathbf u$；反向模式传播输出加权，计算 $J^T\mathbf v$。输入维度较大、目标为标量的网络训练适合反向模式，因为一次反向就能覆盖大量输入参数。

反向传播需要保存或重算部分前向中间量。通常一次前向和一次反向的计算成本同阶，但激活存储可能成为内存瓶颈。激活检查点通过反向时重算部分前向结果，用额外计算换取内存节省。

### 2.7 自动求导的图与梯度存储

PyTorch 的计算图随前向执行动态构建。启用梯度的叶张量通常是用户创建的输入或模型参数，反向时梯度默认累加到这些张量的 `.grad`。

中间张量也参与梯度传播，但其 `.grad` 默认不保留；需要观察时可调用 `retain_grad()`。`torch.autograd.grad` 可以直接返回指定输入的导数，通常不把结果写入输入的 `.grad`。

调用 `backward()` 默认会释放反传所需的部分保存量。对同一图再次反传可能需要 `retain_graph=True`，但常规训练每轮重新执行前向，不应为避免报错而无条件保留图。

`create_graph=True` 用于建立导数本身的计算图，支持高阶求导；它与保留原图的用途不同，也会增加资源成本。

### 2.8 梯度累加与阻断

多个损失或多个批次可以贡献到同一组参数，累加行为因此有明确用途。普通每步更新前通常先清空梯度；若有意累积微批次，则需按最终目标正确缩放损失。

不同微批次大小不一致时，将每个微批均值简单除以微批数量，并不等于对全部样本求均值。按样本数加权，才能得到一致目标。含批统计、随机失活等操作时，微批累积也不保证与完整大批前向等价。

`detach()` 切断对应路径的梯度依赖，`torch.no_grad()` 禁止上下文内常规运算记录梯度。`model.eval()` 切换部分层的行为，并不自动关闭求导；`model.train()` 同样不自动执行反向传播。

## 3. 原理实现

### 3.1 标量反向求导引擎

标准库实现支持加法、乘法和平方。节点保存局部反传函数，通过拓扑排序保证下游贡献完整汇合。仅用于无环标量图的机制演示，不具备张量广播、设备管理与高阶求导能力。

```python
class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def square(self):
        return self * self

    def backward(self):
        order, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        # 演示引擎每次重新求导，显式清空整张图的梯度。
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


a, b, c = Value(2), Value(3), Value(1)
u = a * b
loss = (u + c).square() + u
loss.backward()
print("loss:", loss.data)
print("gradients:", a.grad, b.grad, c.grad)
```

输出为损失 55 和梯度 $45,30,14$。平方通过同一变量重复参与乘法实现，反传必须对两个输入位置分别累加；不能因它们指向同一节点就丢掉其中一条贡献。

该引擎每次清空整张图，与 PyTorch 默认叶梯度跨调用累加的行为不同。前向后不得修改节点数据，否则局部反传读取的数值可能与前向不一致。

### 3.2 手算、有限差分与自动求导

中心差分近似为：

$$
\frac{\partial L}{\partial\theta_j}\approx
\frac{L(\boldsymbol\theta+\epsilon\mathbf e_j)-L(\boldsymbol\theta-\epsilon\mathbf e_j)}{2\epsilon}
$$

平滑函数下，其截断误差通常为 $O(\epsilon^2)$，但 $\epsilon$ 过小会放大浮点消减误差。梯度检查通常使用双精度并避开不可微点。

```python
import torch


def objective(a, b, c):
    u = a * b
    return (u + c) ** 2 + u


point = [2.0, 3.0, 1.0]
epsilon = 1e-5
finite_difference = []
for index in range(3):
    plus, minus = point.copy(), point.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    derivative = (objective(*plus) - objective(*minus)) / (2 * epsilon)
    finite_difference.append(derivative)

parameters = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
              for value in point]
torch_loss = objective(*parameters)
torch_loss.backward()
automatic = [parameter.grad.item() for parameter in parameters]

print("finite_difference:", [round(v, 6) for v in finite_difference])
print("autograd:", automatic)
```

三种求导结果应在数值误差内一致。有限差分适合小规模核对，不适合逐参数检查大型网络；随机方向差分可以降低检查成本。

### 3.3 非标量输出与梯度累加

```python
x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
outputs = x.square()
vector = torch.tensor([3., 4.], dtype=torch.float64)
weighted_gradient, = torch.autograd.grad(outputs, x, grad_outputs=vector)
print("weighted_gradient:", weighted_gradient.tolist())

weight = torch.tensor(2.0, requires_grad=True)
(weight * 3).backward()
(weight * 4).backward()  # 新建前向图，累加到同一个叶参数。
print("accumulated_gradient:", weight.grad.item())
weight.grad = None
(weight * 4).backward()
print("fresh_gradient:", weight.grad.item())
```

输出依次为 `[6.0,16.0]`、7 和 4。第一个结果是加权输出对输入的梯度，不是完整雅可比矩阵；后两次反传使用不同前向图，因此无需保留旧图。

### 3.4 高阶求导与求导检查

```python
x_high = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
first, = torch.autograd.grad(x_high ** 3, x_high, create_graph=True)
second, = torch.autograd.grad(first, x_high)
print("first_second:", first.item(), second.item())

inputs = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
               for value in point)
passed = torch.autograd.gradcheck(objective, inputs, eps=1e-6, atol=1e-5)
print("gradcheck:", passed)
```

在 $x=2$ 处，$x^3$ 的一阶和二阶导数均为 12。`gradcheck` 比较自动导数与数值近似，适合检查自定义可微运算；它不能替代模型的数据验证或泛化评估。

### 3.5 常见问题与训练诊断

| 现象 | 可能原因 | 检查方式 |
| --- | --- | --- |
| 参数没有梯度 | 路径被切断、参数未参与损失或未启用求导 | 检查依赖关系及梯度配置 |
| 梯度持续放大 | 忘记清空、损失缩放错误或训练不稳定 | 核对累加边界、归约和梯度范数 |
| 重复反传报错 | 重用已释放保存量的计算图 | 常规训练重新前向，按需保留图 |
| 原地修改报错 | 反向需要的中间量被覆盖 | 避免破坏保存张量 |
| 数值梯度不一致 | 不可微点、精度不足或实现错误 | 双精度检查平滑小例子 |

通过 `.item()` 将损失转换成普通数值后，不能再从该数值回到原图求导。将中间结果重新包装成新张量，也不会自动恢复此前的依赖。

梯度为零与梯度不存在不同：零梯度可以是合法导数，`.grad` 为 `None` 则可能表示未计算、已清空或未保留。排查时应结合图结构，避免仅凭一个属性判断训练是否正常。

## 4. 总结

反向传播按计算图逆序组合局部导数，在分支汇合处累加贡献，并通过向量与雅可比乘积避免显式构建巨大导数矩阵。自动求导将这些规则与实际前向运算结合，使大量参数的梯度计算可复用、可检查。

正确训练还需要明确损失归约、梯度清空边界、图生命周期和参数更新时机。手算、中心差分与自动求导交叉验证，可以在小例子上定位求导错误，再将可信的运算扩展到复杂网络。

## 5. 参考文献

[1] Rumelhart D E, Hinton G E, Williams R J. Learning Representations by Back-propagating Errors. Nature, 1986, 323: 533–536.

[2] Baydin A G, Pearlmutter B A, Radul A A, Siskind J M. Automatic Differentiation in Machine Learning: A Survey. Journal of Machine Learning Research, 2018, 18(153): 1–43.

[3] Goodfellow I, Bengio Y, Courville A. Deep Learning. MIT Press, 2016.
