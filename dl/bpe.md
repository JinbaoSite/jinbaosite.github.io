# 字节对编码(BPE):一种简单而高效的开放词表分词方法

## 摘要

词级分词面临词表过大与未登录词(OOV)问题,字符级分词则序列过长、语义稀薄。字节对编码(Byte Pair Encoding, BPE)以数据驱动的贪心合并在二者间取得折中,现已成为大语言模型的标准分词方案。

## 1 引言

神经网络模型处理的是离散符号 ID 序列,因此文本处理的首要步骤是将字符串映射为整数序列,即分词(tokenization)。分词单元的选择直接影响词表规模、序列长度与模型的泛化能力。词级分词以完整单词为单位,语义明确,但词表规模随形态变化迅速膨胀,且无法表示训练中未出现的词(OOV),只能退化为统一的未知符号;字符级分词词表极小、无 OOV 问题,却导致序列过长且单元语义稀薄。

理想方案应介于二者之间:高频完整词保留为单一单元,低频或未见词则切分为有意义的子词片段。BPE 以频率驱动的合并机制自动学习这样一套切分,从而在词表规模、序列长度与泛化能力间取得平衡。


## 2 背景与相关工作

### 2.1 算法起源

BPE 最初并非为自然语言处理设计。Gage(1994)将其作为一种数据压缩算法提出:其核心思想是反复识别数据中出现频率最高的一对相邻字节,并以一个未使用的新字节替换之,通过迭代实现压缩。

Sennrich 等(2016)首次将该算法引入神经机器翻译,用以构建子词词表。他们将操作单元由"字节"替换为"字符",将优化目标由"压缩数据"替换为"学习开放词表的切分",从而使模型能够以有限的词表处理罕见词与未登录词。此后,BPE 及其变体成为神经网络文本处理的标准组件,被 GPT 系列、RoBERTa、LLaMA 等模型广泛采用。

### 2.2 核心思想

BPE 的核心可概括为一条迭代规则:

> 从最小的符号单元出发,反复将训练语料中共现频率最高的相邻符号对合并为一个新符号,直至词表达到预设规模。

该过程不依赖任何语言学规则或人工标注,完全由语料中的统计频率驱动。


## 3 算法描述

### 3.1 形式化定义

设训练语料 $C$ 为词的多重集合,其中每个词 $w$ 以频率 $\mathrm{freq}(w)$ 出现。初始时,每个词被表示为一个基础符号序列(字符或字节),并在词尾附加特殊边界标记(记为 `_`),用以区分子词出现在词中还是词尾。令符号词表 $V$ 初始化为全部基础符号的集合。

在第 $t$ 轮迭代中,算法执行如下步骤:

1. **统计共现频率。** 对所有相邻符号对 $(a, b)$,计算其加权共现频率:

   $$F(a, b) = \sum_{w \in C} \mathrm{freq}(w) \cdot \mathrm{count}\big((a, b),\, w\big)$$

   其中 $\mathrm{count}\big((a, b),\, w\big)$ 为对 $(a, b)$ 在词 $w$ 中作为相邻符号出现的次数。

2. **选择最优对。** 取频率最高者

   $$(a^\ast, b^\ast) = \operatorname*{arg\,max}_{(a, b)} F(a, b)$$

   (频率相同时按实现约定打破平局)。

3. **合并并更新。** 构造新符号 $a^\ast b^\ast = \mathrm{concat}(a^\ast, b^\ast)$,加入词表 $V$;将语料中所有相邻的 $(a^\ast, b^\ast)$ 替换为 $a^\ast b^\ast$;并将该合并记录入有序规则表 $M$。

迭代执行 $k$ 轮后终止。最终词表规模约为 $|V_0| + k$(其中 $V_0$ 为基础符号集),合并轮数 $k$ 是控制词表大小的唯一超参数。

算法的两项产物为:词表 $V$,以及**有序的**合并规则表 $M$。规则的顺序至关重要,因为后产生的符号依赖于先前的合并结果。

### 3.2 训练过程

训练过程的伪代码如算法 1 所示。

```
算法 1:BPE 训练
输入:语料 C(词及其频率),合并轮数 k
输出:合并规则表 M
1.  将 C 中每个词拆分为基础符号序列(附加边界标记)
2.  M ← 空列表
3.  for t = 1 to k do
4.      统计所有相邻符号对的加权频率 F
5.      若不存在任何相邻对,则终止
6.      (a*, b*) ← argmax F
7.      在 C 中将所有 (a*, b*) 合并为 a*b*
8.      将 (a*, b*) 追加至 M
9.  end for
10. return M
```

### 3.3 编码过程

对新文本进行切分时,首先将其按相同方式拆分为基础符号序列,随后**严格按照规则表 $M$ 中记录的顺序**依次应用每一条合并规则。由于子词片段在训练阶段已被学习,即使整词未曾出现,也能被切分为已知的子词组合,从而避免未登录词问题(详见第 4 节示例)。


## 4 示例分析

为说明合并规则的产生,考虑一个微型语料。设语料经统计后得到下列词及其频率,各词已拆分为字符并附加边界标记 `_`:

```
5   l o w _
2   l o w e s t _
6   n e w e r _
3   w i d e r _
2   n e w _
```

**第一轮。** 统计加权共现频率,部分关键结果为:*F(e, r)* = 6 + 3 = 9(出现于 `newer` 与 `wider`),*F(r, _)* = 9,*F(n, e)* = 6 + 2 = 8,*F(l, o)* = 5 + 2 = 7。频率最高者为 *(e, r)*,故执行首次合并 *e + r → er*。语料更新为:

```
5   l o w _
2   l o w e s t _
6   n e w er _
3   w i d er _
2   n e w _
```

**第二轮。** 此时 *(er, _)* 的频率为 9(出现于 `newer` 与 `wider`),成为最高频对,执行合并 *er + _ → er\_*。

**后续迭代。** 继续迭代可依次得到表 1 所示的合并规则。值得注意的是,合并产生的符号将作为后续迭代的原子单元参与更大的合并,因而 `n → ne → new → newer_` 是逐层累积的结果。

**表 1. 微型语料上前 8 轮的合并规则**

| 轮次 | 合并对 | 新符号 |
|:----:|:------:|:------:|
| 1 | (e, r) | er |
| 2 | (er, _) | er_ |
| 3 | (n, e) | ne |
| 4 | (ne, w) | new |
| 5 | (l, o) | lo |
| 6 | (lo, w) | low |
| 7 | (new, er_) | newer_ |
| 8 | (low, _) | low_ |

**对未登录词的处理。** 考虑训练集中未完整出现的词 `newest`。其初始序列为 `n e w e s t _`,按表 1 的顺序应用规则后,*(n, e)* 与 *(ne, w)* 被依次合并,得到 `new e s t _`。尽管 `newest` 整体从未出现,其主要语义仍通过已学习的子词 `new` 得以保留,而非退化为单一未知符号。这正是 BPE 解决 OOV 问题的机制。

## 5 参考实现

算法 1 可由数十行 Python 代码实现。下列实现改编自 Sennrich 等(2016)给出的经典版本。

```python
import re
import collections


def get_stats(vocab):
    """统计所有相邻符号对的加权共现频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab_in):
    """将指定符号对在整个语料中合并为新符号"""
    vocab_out = {}
    bigram = re.escape(' '.join(pair))
    # 负向断言确保仅匹配被空格分隔的完整符号
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab_in:
        merged = pattern.sub(''.join(pair), word)
        vocab_out[merged] = vocab_in[word]
    return vocab_out


# 初始语料:每词拆为字符,附加边界标记 </w>
vocab = {
    'l o w </w>': 5,
    'l o w e s t </w>': 2,
    'n e w e r </w>': 6,
    'w i d e r </w>': 3,
    'n e w </w>': 2,
}

num_merges = 8
merge_rules = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merge_rules.append(best)
    print(f"第 {i + 1} 轮合并: {best} -> {''.join(best)}")
```

该实现由三个要素构成:`get_stats` 负责统计各相邻对的加权频率,作为每轮决策依据;`merge_vocab` 通过正则替换在全语料范围内合并选定的对,其中负向断言 `(?<!\S)` 与 `(?!\S)` 保证仅匹配完整符号;主循环重复"统计—选择—合并"过程共 `num_merges` 轮。其输出与第 4 节的手算结果一致。

## 6 字节级字节对编码

第 3—5 节描述的是字符级 BPE。Radford 等(2019)在 GPT-2 中引入了一种重要变体——字节级 BPE(Byte-Level BPE),其差异在于基础符号的选取。

字节级 BPE 不以 Unicode 字符为起点,而以原始字节为起点。任意文本均可编码为 UTF-8 字节序列,而字节仅有 256 种取值。由此,基础词表仅需 256 个字节即可表示任意语言、符号及表情字符,从根本上消除了未登录词问题。无论输入为中文、阿拉伯文或表情符号,在字节层面均表现为若干 0–255 的整数,BPE 均可在其上学习合并规则。其代价是部分非拉丁文字(如中文,单字符通常占 3 个 UTF-8 字节)会被切分为更多单元。

## 7 讨论

### 7.1 优势与局限

BPE 的主要优势在于:词表规模可由合并轮数精确控制;通过子词共享显著提升了对罕见词与词形变化的处理能力;字节级变体从根本上规避了 OOV 问题;算法本身简单、训练高效,且完全由数据驱动,无需语言学规则或人工标注。

其局限同样源于其简单性:合并完全由频率决定,所得片段不保证具备语言学意义,切分边界有时不符合语素结构;训练与编码均采用贪心策略,不保证全局最优;对形态丰富的语言,其效果有时不及基于语言学的方法。

### 7.2 相关方法

针对上述局限,后续研究提出了若干相关方法。WordPiece(Schuster 与 Nakajima, 2012)在合并时不以频率为准则,而以合并对语言模型似然的提升为准则,被 BERT 采用。Unigram 语言模型(Kudo, 2018)则采取相反思路,从一个较大的候选词表出发,通过迭代裁剪保留最优子集。SentencePiece(Kudo 与 Richardson, 2018)将上述方法封装为语言无关、可直接处理原始文本的工业级工具。尽管准则各异,这些方法在"以数据驱动学习子词切分"这一核心思想上,与 BPE 一脉相承。

## 8 结论

BPE 以一条朴素的迭代规则——反复合并共现频率最高的相邻符号对——有效解决了分词中词表规模、序列长度与泛化能力之间的取舍。该算法不依赖语法或语义知识,仅凭频率统计,却恰好落在上述权衡的有利区间内。其字节级变体进一步消除了未登录词问题,使其成为当代大规模语言模型的标准分词方案。理解 BPE,有助于理解现代语言模型如何将任意文本转化为可处理的离散表示。

## 参考文献

[1] Gage, P. (1994). A New Algorithm for Data Compression. *The C Users Journal*, 12(2), 23–38.

[2] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 1715–1725.

[3] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.

[4] Schuster, M., & Nakajima, K. (2012). Japanese and Korean Voice Search. In *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5149–5152.

[5] Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 66–75.

[6] Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations*, 66–71.
