---
layout: listpage
title: 深度学习
subtitle: 神经网络基础、CNN、Transformer、训练优化与机器学习算法
article-list:
  - article-title: LeNet-5 与 AlexNet：CNN 的奠基之作
    article-url: /dl/lenet_alexnet
    article-date: 2026-07-08
    article-desc: 拆解 LeNet-5 的「卷积-池化-全连接」范式与 AlexNet 的 ReLU+Dropout+GPU 三大创新,梳理从 MNIST 到 ImageNet 的范式跃迁。
    article-tags: [CNN, LeNet, AlexNet, 历史]
  - article-title: VGG 与 GoogLeNet：更深更宽的探索
    article-url: /dl/vgg_googlenet
    article-date: 2026-07-08
    article-desc: VGG 用 3×3 卷积堆叠把网络做到 16-19 层,GoogLeNet 用 1×1 降维 + Inception 多分支做到 22 层仅 500 万参数。
    article-tags: [VGG, GoogLeNet, Inception, 历史]
  - article-title: ResNet 与 DenseNet：残差连接的胜利
    article-url: /dl/resnet_densenet
    article-date: 2026-07-08
    article-desc: ResNet 用加法捷径让网络深到 152 层,DenseNet 用拼接捷径让每层复用所有历史特征,共同确立了「跨层连接 = 现代 CNN 标配」。
    article-tags: [ResNet, DenseNet, 残差连接]
  - article-title: "卷积运算详解：从 二维卷积到深度可分离卷积"
    article-url: /dl/convolution
    article-date: 2026-07-08
    article-desc: 系统讲解 2D 卷积、1×1 卷积、空洞卷积、转置卷积与深度可分离卷积的原理、PyTorch 实现与计算量对比。
    article-tags: [CNN, 卷积, MobileNet]
  - article-title: 混合精度训练与分布式训练（DDP）
    article-url: /dl/mixed_precision_ddp
    article-date: 2026-07-08
    article-desc: 详解 FP16/BF16 混合精度与 DistributedDataParallel 的原理、PyTorch 实战与踩坑指南,LLM 训练六件套。
    article-tags: [AMP, DDP, 分布式]
  - article-title: "Transformer 注意力机制"
    article-url: /dl/attention
    article-date: 2026-07-07
    article-desc: 详解 Scaled Dot-Product Attention, Multi-Head Attention (MHA), Multi-Query Attention (MQA)及Grouped-Query Attention (GQA)四种经典机制。
    article-tags: [Atttention, MHA, MQA, GQA]
  - article-title: "位置编码"
    article-url: /dl/positional_encoding
    article-date: 2026-07-06
    article-desc: 详解 Learnable、Sinusoidal、RoPE 及 YaRN 四种主流位置编码方案的数学原理与代码实现，梳理从绝对位置编码到相对位置编码的演进脉络。
    article-tags: [Sinusoidal, RoPE, YaRN]
  - article-title: 字节对编码(BPE):一种简单而高效的开放词表分词方法
    article-url: /dl/bpe
    article-date: 2026-06-17
    article-desc: 从最小的符号单元出发,反复将训练语料中共现频率最高的相邻符号对合并为一个新符号,直至词表达到预设规模。
    article-tags: [BPE, tokenizer]
  - article-title: "语言模型评估：困惑度详解"
    article-url: /dl/perplexity
    article-date: 2025-05-08
    article-desc: Perplexity衡量模型在给定文本上的困惑程度，即模型对文本预测的不确定程度。
    article-tags: [指标]
  - article-title: 学习率调度：从预热到周期退火
    article-url: /dl/lr_schedule
    article-date: 2026-07-08
    article-desc: 学习率调度按照更新步数或验证反馈调整步长，涵盖预热、阶梯与余弦衰减、周期重启及 OneCycle，提供明确边界的 PyTorch 实现与断点续训核对。
    article-tags: [学习率, Cosine, OneCycle]
  - article-title: 优化器：从梯度下降到自适应更新
    article-url: /dl/optimizers
    article-date: 2026-07-08
    article-desc: 优化器通过动量、梯度统计与权重衰减调整参数，涵盖 SGD、AdaGrad、RMSprop、Adam、AdamW、LAMB 和 Lion，提供更新公式核对、状态内存分析及完整训练示例。
    article-tags: [优化器, AdamW, LAMB]
  - article-title: 归一化：从批次统计到均方根缩放
    article-url: /dl/normalization
    article-date: 2026-07-08
    article-desc: 归一化通过指定轴上的统计调整激活尺度，涵盖批、层、组、实例及均方根归一化，厘清运行方差、训练推理差异和等价条件，并提供公式与梯度核对实现。
    article-tags: [归一化, LayerNorm, RMSNorm]
  - article-title: 激活函数：从逻辑函数到门控线性单元
    article-url: /dl/activation_functions/
    article-date: 2026-10-20
    article-desc: 激活函数为网络引入非线性，涵盖 Sigmoid、Tanh、ReLU、GELU、SiLU 及 GLU、GEGLU、SwiGLU，说明梯度、饱和与参数预算，并提供公式核对及门控前馈模块实现。
    article-tags: [激活函数, 门控, 神经网络]
  - article-title: 反向传播：链式法则与自动求导
    article-url: /dl/backpropagation/
    article-date: 2026-10-20
    article-desc: 反向传播通过计算图与链式法则高效计算参数梯度，涵盖分支累加、矩阵与广播求导、梯度存储及高阶导数，并提供标量求导引擎、有限差分和 PyTorch 核对代码。
    article-tags: [反向传播, 自动求导, 链式法则]
  - article-title: 多层感知机：从线性模型到非线性神经网络
    article-url: /dl/mlp/
    article-date: 2026-10-20
    article-desc: 多层感知机通过仿射层与非线性激活学习特征表示，涵盖异或构造、反向传播、输出任务、初始化与容量控制，并提供包含验证集早停的完整 PyTorch 实现。
    article-tags: [多层感知机, 神经网络, 反向传播]
  - article-title: DNN：深度神经网络的模型结构与实现
    article-url: /dl/dnn/
    article-date: 2026-10-26
    article-desc: DNN 由输入编码、隐藏主干与任务输出组成，结合三层全连接架构图详解模型结构、张量形状与参数量，逐步推导前向计算和反向梯度，并提供数值算例、自动求导核对及完整训练代码。
    article-tags: [DNN, 模型结构, 前向传播, 反向传播]
  - article-title: CatBoost：有序统计与类别特征树提升
    article-url: /dl/catboost/
    article-date: 2026-10-20
    article-desc: CatBoost 通过有序目标统计和有序提升控制类别标签泄漏与预测偏移，结合对称树提升预测效率，涵盖先验、类别组合与数据划分，并提供可运行的核心组件实现。
    article-tags: [CatBoost, 类别特征, 有序提升]
  - article-title: LightGBM：直方图与叶子优先的高效树提升
    article-url: /dl/lightgbm/
    article-date: 2026-10-20
    article-desc: LightGBM 通过直方图分桶与叶子优先生长提升训练效率，涵盖直方图相减、梯度单边采样、互斥特征捆绑及类别处理，并提供可运行的直方图提升回归实现。
    article-tags: [LightGBM, 集成学习, 直方图]
  - article-title: XGBoost：二阶优化与正则化树提升
    article-url: /dl/xgboost/
    article-date: 2026-10-20
    article-desc: XGBoost 结合二阶损失近似与树正则化，推导叶权重、分裂增益及海森约束，说明直方图搜索与缺失值默认方向，并提供可运行的二阶提升树桩实现。
    article-tags: [XGBoost, 集成学习, 二阶优化]
  - article-title: 随机森林：从自助采样到多树集成
    article-url: /dl/random_forest/
    article-date: 2026-10-20
    article-desc: 随机森林通过自助采样和节点级随机特征选择降低树间相关性，涵盖分类概率聚合、回归平均、袋外评估与方差分析，并提供完整 Python 实现。
    article-tags: [随机森林, 集成学习, 袋外评估]
  - article-title: 梯度提升树：从负梯度拟合到加法模型
    article-url: /dl/gradient_boosting/
    article-date: 2026-10-20
    article-desc: 梯度提升树通过逐轮拟合负梯度构建加法模型，涵盖平方误差回归、二分类交叉熵、叶值优化、学习率与早停，并提供基于回归树桩的完整 Python 实现。
    article-tags: [梯度提升树, 集成学习, GBDT]
  - article-title: 决策树：从特征划分到分类与回归
    article-url: /dl/decision_tree/
    article-date: 2026-10-20
    article-desc: 决策树通过递归划分特征空间完成分类与回归，涵盖基尼不纯度、信息增益、平方误差、预剪枝与后剪枝，并提供支持分类和回归的完整 Python 实现。
    article-tags: [决策树, 分类, 回归]
  - article-title: 朴素贝叶斯：从条件概率到文本分类
    article-url: /dl/naive_bayes/
    article-date: 2026-10-20
    article-desc: 朴素贝叶斯利用类别先验和类条件分布完成分类，涵盖条件独立假设、多项式、伯努利与高斯模型、加性平滑及对数概率，并提供可运行的文本分类实现。
    article-tags: [朴素贝叶斯, 概率模型, 文本分类]
  - article-title: 逻辑回归：从概率建模到交叉熵优化
    article-url: /dl/logistic_regression/
    article-date: 2026-10-20
    article-desc: 逻辑回归通过线性对数几率估计分类概率，涵盖伯努利似然、交叉熵梯度、数值稳定性、决策阈值、正则化与概率校准，并提供手动梯度和自动求导的完整 PyTorch 实现。
    article-tags: [逻辑回归, 分类, 交叉熵]
  - article-title: 线性回归：从最小二乘到梯度下降
    article-url: /dl/linear_regression/
    article-date: 2026-10-20
    article-desc: 线性回归通过特征加权求和预测连续目标，涵盖平方损失、正规方程、梯度下降、标准化与正则化，并提供最小二乘、手动梯度和自动求导的完整 PyTorch 实现。
    article-tags: [线性回归, 最小二乘, 梯度下降]

---
