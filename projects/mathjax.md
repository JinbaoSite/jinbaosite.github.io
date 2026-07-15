# MathJax语法

## 1 基础语法

### 1.1 显示公式

- 在行中显示的 (inline mode)，就用 `$ ... $`
- 单独一行显示 (display mode)，则用 `$$ ... $$`

### 1.2 希腊字母

- 要显示小写希腊字母，可以用 `\alpha`, `\beta`, …, `\omega`，输出：$\alpha, \beta, \dots, \omega$
- 想要显示大写的话，就用 `\Gamma`, `\Delta`, …, `\Omega`，输出：$\Gamma, \Delta, \dots, \Omega$

### 1.3 上下标

上下标可用 `^` 和 `_`，上下标符号只能用于接下来一个「Group」（即单个字符，或一组花括号内的东西）。

- 例如：`\log_2 x` 显示为 $\log_2 x$
- 如果要写 $10^{10}$，必须写成 `10^{10}`

### 1.4 括号

- 小括号、方括号直接输入，花括号要用 `\{` 和 `\}`。
- 由于括号不会自动伸缩，直接写 `(\frac{\sqrt x}{y^3})` 会得到 $(\frac{\sqrt x}{y^3})$。
- 如果需要自动伸缩，需要使用 `\left` 和 `\right` 进行包裹。例如：`\left(\frac{\sqrt x}{y^3}\right)` 会得到 $\left(\frac{\sqrt x}{y^3}\right)$。
- `\left` 和 `\right` 还可以配合以下符号使用：
  - 绝对值：`\left\vert x \right\vert` ($\left\vert x \right\vert$)
  - 范数：`\left\Vert x \right\Vert` ($\left\Vert x \right\Vert$)
  - 尖角：`\left\langle x \right\rangle` ($\left\langle x \right\rangle$)
  - 向上取整：`\left\lceil x \right\rceil` ($\left\lceil x \right\rceil$)
  - 向下取整：`\left\lfloor x \right\rfloor` ($\left\lfloor x \right\rfloor$)
- 如果只需显示单侧符号，可以用 `.` 表示另一边为空。例如：`\left. \frac 1 2 \right \rbrace` 显示为 $\left. \frac 1 2 \right \rbrace$。
- 也可以手动调整括号大小：`\Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)` 显示为 $\Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)$。

### 1.5 求和与积分

- 求和：`\sum_1^n` 显示为 $\sum_1^n$
- 积分：`\int_1^n` 显示为 $\int_1^n$
- 连乘：`\prod` ($\prod$)
- 并集：`\bigcup` ($\bigcup$)
- 交集：`\bigcap` ($\bigcap$)
- 多重积分：`\iint` ($\iint$)

*(注：上下限不止一位时需要用花括号包裹)*

### 1.6 分数（fraction）

有两种方法来显示分数：
1. 使用 `\frac a b` 显示为 $\frac a b$
2. 使用 `\over`，如 `{a+1 \over b+1}` 显示为 ${a+1 \over b+1}$

### 1.7 字体

- 使用 `\mathbb` 或 `\Bbb` 选择 blackboard bold 字体：如 `\mathbb {R}` 显示为 $\mathbb {R}$
- 使用 `\mathbf` 选择 boldface（粗体）：如 `\mathbf {x}` 显示为 $\mathbf {x}$
- 使用 `\mathtt` 选择 typewriter 字体：如 `\mathtt {A}` 显示为 $\mathtt {A}$
- 使用 `\mathrm` 选择 roman（正体）：如 `\mathrm {A}` 显示为 $\mathrm {A}$
- 使用 `\mathsf` 选择 sans-serif 字体：如 `\mathsf {A}` 显示为 $\mathsf {A}$
- 使用 `\mathcal` 选择 calligraphic 字体：如 `\mathcal {A}` 显示为 $\mathcal {A}$
- 使用 `\mathscr` 选择 script 字体：如 `\mathscr {A}` 显示为 $\mathscr {A}$
- 使用 `\mathfrak` 选择 Fraktur 字体：如 `\mathfrak {A}` 显示为 $\mathfrak {A}$

### 1.8 根号

- 平方根：`\sqrt {x^3}` 显示为 $\sqrt {x^3}$
- 高次根：`\sqrt[3] {\frac x y}` 显示为 $\sqrt[3] {\frac x y}$

### 1.9 三角函数、极限和对数

像 “lim”, “sin”, “max”, “ln” 等符号已包括在 roman 字体中，直接用斜杠调用即可：

- 例如：`\lim_{x\to 0}` 显示为 $\lim_{x\to 0}$

### 1.10 特殊符号和记号

- 不等号：`\lt` ($<$)、`\gt` ($>$)、`\le` ($\le$)、`\ge` ($\ge$)、`\neq` ($\neq$)。还可以在符号前加 `\not`，如 `\not\lt` ($\not\lt$)。
- 算术运算符：`\times` ($\times$)、`\div` ($\div$)、`\pm` ($\pm$)、`\mp` ($\mp$)。点乘用 `\cdot` 表示，如 `x \cdot y` 显示为 $x \cdot y$。
- 集合类符号：`\cup` ($\cup$)、`\cap` ($\cap$)、`\setminus` ($\setminus$)、`\subset` ($\subset$)、`\subseteq` ($\subseteq$)、`\subsetneq` ($\subsetneq$)、`\supset` ($\supset$)、`\in` ($\in$)、`\notin` ($\notin$)、`\emptyset` ($\emptyset$)、`\varnothing` ($\varnothing$)。
- 组合数：`{n+1 \choose 2k}` 或 `\binom{n+1}{2k}` 显示为 $\binom{n+1}{2k}$。
- 箭头：`\to` ($\to$)、`\rightarrow` ($\rightarrow$)、`\left` ($\leftarrow$)、`\Rightarrow` ($\Rightarrow$)、`\Leftarrow` ($\Leftarrow$)、`\mapsto` ($\mapsto$)。
- 逻辑运算符：`\land` ($\land$)、`\lor` ($\lor$)、`\lnot` ($\lnot$)、`\forall` ($\forall$)、`\exists` ($\exists$)、`\top` ($\top$)、`\bot` ($\bot$)、`\vdash` ($\vdash$)、`\vDash` ($\vDash$)。
- 其他运算符：`\star` ($\star$)、`\ast` ($\ast$)、`\oplus` ($\oplus$)、`\circ` ($\circ$)、`\bullet` ($\bullet$)。
- 关系符：`\approx` ($\approx$)、`\sim` ($\sim$)、`\simeq` ($\simeq$)、`\cong` ($\cong$)、`\equiv` ($\equiv$)、`\prec` ($\prec$)、`\lhd` ($\lhd$)。
- 无穷与微积分：`\infty` ($\infty$)、`\aleph_0` ($\aleph_0$)、`\nabla` ($\nabla$)、`\partial` ($\partial$)、`\Im` ($\Im$)、`\Re` ($\Re$)。
- 取模：用 `\pmod`，如 `a \equiv b\pmod n` 表示 $a \equiv b\pmod n$。
- 省略号：
  - 底点省略号（如逗号间）：`\ldots`，如 `a_1, a_2, \ldots ,a_n` 表示 $a_1, a_2, \ldots ,a_n$。
  - 中间位置省略号（如加号间）：`\cdots`，如 `a_1 + a_2 + \ldots + a_n` 表示 $a_1 + a_2 + \ldots + a_n$。

### 1.11 空格

MathJax 中直接输入空格不会改变表达式。如果想在表达式中加入空格，可以使用：

- 极小空格：`\,`
- 小空格：`\;`
- 大空格：`\quad`
- 特大空格：`\qquad`
- 插入文本：如果想在公式中加入一段正常文本，可用 `\text{…}`。例如：`\{x \in s \mid x \text{ is extra large}\}` 显示为 $\{x \in s \mid x \text{ is extra large}\}$。在 `\text{…}` 里面还可以嵌套数学符号 `$…$`。

### 1.12 Accents (重音符) and diacritical (变音符) marks

- 单字符重音：`\hat x` ($\hat x$)，变音：`\widehat {xy}` ($\widehat {xy}$)
- 平均值/长横线：`\bar x` ($\bar x$)，`\overline {xyz}` ($\overline {xyz}$)
- 向量/箭头：`\vec x` ($\vec x$)，`\overrightarrow {xy}` ($\overrightarrow {xy}$)，`\overleftrightarrow {xy}` ($\overleftrightarrow {xy}$)
- 导数点号：`\dot x` ($\dot x$)，`\ddot x` ($\ddot x$)

### 1.13 转义符

一般情况下可用 `\` 来作转义。但如果想要表示 `\` 本身，必须使用 `\backslash`，因为双斜杠 `\\` 在 MathJax 中表示换行。

## 2 矩阵

### 2.1 矩阵表示

可以用 `\begin{matrix} … \end{matrix}` 来表示矩阵。将矩阵元素放在 `\begin` 和 `\end` 之间，用 `\\` 来分割行，用 `&` 来分割同一行的矩阵元素。

$$
\begin{matrix}
	1 & x & x^2 \\
	1 & y & y^2 \\
	1 & z & z^2 \\
\end{matrix}
$$

### 2.2 矩阵两端的括号

除了用 `\left…\right` 之外，还可以通过替换 `matrix` 关键字来快速实现不同样式的矩阵：

- 圆括号 `pmatrix`：$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$
- 方括号 `bmatrix`：$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
- 花括号 `Bmatrix`：$\begin{Bmatrix} 1 & 2 \\ 3 & 4 \end{Bmatrix}$
- 单竖线 `vmatrix`：$\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix}$
- 双竖线 `Vmatrix`：$\begin{Vmatrix} 1 & 2 \\ 3 & 4 \end{Vmatrix}$

### 2.3 在中间省略一些项

可以用 `\cdots` ($\cdots$)、`\ddots` ($\ddots$)、`\vdots` ($\vdots$) 来表示省略项。

$$
\begin{pmatrix}
     1 & a_1 & a_1^2 & \cdots & a_1^n \\
     1 & a_2 & a_2^2 & \cdots & a_2^n \\
     \vdots  & \vdots& \vdots & \ddots & \vdots \\
     1 & a_m & a_m^2 & \cdots & a_m^n    
\end{pmatrix}
$$

### 2.4 增广矩阵 augmented matrix

写增广矩阵需要用到 `array` 语法。

```latex
\left [
    \begin {array} {cc|c}
      1&2&3\\
      4&5&6
    \end {array}
\right ]
```
显示为：

$$\left [
    \begin {array} {cc|c}
      1&2&3\\
      4&5&6
    \end {array}
\right ]$$

*(注：`{cc|c}` 的作用是在第二列和第三列之间画一条垂直分割线，其中的 `c` 表示列内容居中对齐)*

### 2.5 在行内画小矩阵

如果需要在行内插入小矩阵，可以使用 `\bigl(\begin{smallmatrix} ... \end{smallmatrix}\bigr)`。

* 例如：$\bigl( \begin{smallmatrix} a & b \\ c & d \end{smallmatrix} \bigr)$


## 3 对齐等式

如果有一系列的等式需要写，并且等号需要对齐，可以使用 `\begin{align} … \end{align}`。每次换行用 `\\`，并在需要对齐的符号（如等号）前加上 `&`：

```latex
\begin{align}
\sqrt{37} & = \sqrt{ \frac{73^2-1}{12^2}} \\
& = \sqrt{ \frac{73^2}{12^2} \cdot \frac{73^2-1}{73^2}} \\ 
& = \sqrt{ \frac{73^2}{12^2}}\sqrt{ \frac{73^2-1}{73^2}} \\
& = \frac{73}{12} \sqrt{1 - \frac{1}{73^2}} \\ 
& \approx \frac{73}{12} \left(1 - \frac{1}{2 \cdot73^2} \right)
\end{align}
```

其显示效果会自动为每一行附带编号（在支持 align 编号的 MathJax 环境中）。在使用 align 时，外层的 `$$` 符号可以省略。

## 4 分段函数 piecewise functions

分段函数使用 `cases` 语法：`\begin{cases} … \end{cases}`，用 `\\` 换行，用 `&` 对齐条件。

$$f(n) =
\begin{cases}
n/2,  & \text{if $n$ is even} \\
3n+1, & \text{if $n$ is odd}
\end{cases}$$

如果想把大括号放在右边，可以这样写：

```latex
\left.
\begin{array}{l}
\text{if $n$ is even:}&n/2\\
\text{if $n$ is odd:}&3n+1
\end{array}
\right\}
=f(n)

```

如果想让两行之间的行间距更大一些，可以用 `\\[2ex]` 代替 `\\`（`ex` 是指字母 x 的高度，`2ex` 表示两倍的字母 x 高度）。


## 5 Array (数组与表格)

在 `\begin{array}` 之后，必须用一对花括号 `{}` 定义每一列的对齐方式和分隔线：

* `c` 表示居中对齐
* `l` 表示左对齐
* `r` 表示右对齐
* `|` 表示垂直分隔线
* 用 `&` 分割单元格，用 `\\` 换行
* 如需水平分隔线，在当前行前加上 `\hline` 即可

$$\begin{array} {c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i
\end{array}$$

`array` 还支持嵌套，用以形成复杂的矩阵和表格。


## 6 方程组

要建立方程组，有以下几种常用方式：

1. 使用 `array` 结合 `\left\{` 和 `\right.` 隐藏右括号：
$$\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right. $$


2. 直接使用 `cases` 语法：
$$\begin{cases}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{cases}$$


3. 如果想把等号对齐，可嵌套 `aligned` 语法：
$$
\left\{
\begin{aligned} 
a_1x+b_1y+c_1z &= d_1+e_1 \\ 
a_2x+b_2y &= d_2 \\ 
a_3x+b_3y+c_3z &= d_3 
\end{aligned} 
\right.
$$


## 7 连续分式

书写连续分式（连分数）时，请使用 `\cfrac` 代替普通的 `\frac`，以保证层级比例正常。

* **推荐做法 (`\cfrac`)**：
$$x = a_0 + \cfrac{1^2}{a_1
          + \cfrac{2^2}{a_2
          + \cfrac{3^2}{a_3 + \cfrac{4^4}{a_4 + \cdots}}}}$$


* **不推荐做法 (`\frac` 会导致字体逐层缩小且挤压)**：
$$x = a_0 + \frac{1^2}{a_1
          + \frac{2^2}{a_2
          + \frac{3^2}{a_3 + \frac{4^4}{a_4 + \cdots}}}}$$

## 8 打 Tag 和引用公式

在 MathJax 中可以使用标记与引用系统。

* 使用 `\tag{yourtag}` 给公式打标签。
* 如果后续需要引用，在 `\tag` 后面加上 `\label{somelabel}`。

例如，先对公式打标：

$$a := x^2-y^3 \tag{*}\label{*}$$

在后文中，可以使用 `\eqref{*}` 带有括号地引用它：

$$a+y^3 \stackrel{\eqref{*}}= x^2$$

或者使用 `\ref{*}` 进行无括号引用：

$$a+y^3 \stackrel{\ref{*}}= x^2$$

## 9 Commutative diagrams (交换图表)

交换图表（AMScd）必须以 `\require{AMScd}` 开头引入：

```latex
\require{AMScd}
\begin{CD}
    A @>a>> B\\
    @V b V V= @VV c V\\
    C @>>d> D
\end{CD}

```

显示为：

$$\require{AMScd}
\begin{CD}
    A @>a>> B\\
    @V b V V= @VV c V\\
    C @>>d> D
\end{CD}$$

**符号说明：**

* `@>>>` 向右箭头
* `@<<<` 向左箭头
* `@VVV` 向下箭头
* `@AAA` 向上箭头
* `@=` 水平双线
* `@|` 垂直双线
* `@.` 空白（无箭头）
