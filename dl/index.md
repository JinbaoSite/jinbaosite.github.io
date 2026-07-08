---
layout: listpage
title: 深度学习
subtitle: CNN、Transformer、训练优化
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
  - article-title: 卷积运算详解：从 2D 卷积到深度可分离卷积
    article-url: /dl/convolution
    article-date: 2026-07-08
    article-desc: 系统讲解 2D 卷积、1×1 卷积、空洞卷积、转置卷积与深度可分离卷积的原理、PyTorch 实现与计算量对比。
    article-tags: [CNN, 卷积, MobileNet]
  - article-title: 归一化层：BatchNorm / LayerNorm / GroupNorm
    article-url: /dl/normalization
    article-date: 2026-07-08
    article-desc: 对比 BatchNorm、LayerNorm、GroupNorm、InstanceNorm、RMSNorm 五种归一化方法,给出按任务类型选择的工程指南。
    article-tags: [BatchNorm, LayerNorm, GroupNorm]
  - article-title: 优化器演进：SGD → Adam → AdamW → LAMB
    article-url: /dl/optimizers
    article-date: 2026-07-08
    article-desc: 系统梳理 SGD、Momentum、Adam、AdamW、LAMB、Lion 的算法、PyTorch 实现与适用场景,Transformer 默认 AdamW。
    article-tags: [优化器, AdamW, LAMB]
  - article-title: 学习率调度：Warmup + Cosine Annealing + OneCycle
    article-url: /dl/lr_schedule
    article-date: 2026-07-08
    article-desc: 讲解 Warmup、Step Decay、Exponential、Cosine Annealing、OneCycle 五种调度策略,附 LR Range Test 找最优学习率。
    article-tags: [学习率, Cosine, OneCycle]
  - article-title: 混合精度训练与分布式训练（DDP）
    article-url: /dl/mixed_precision_ddp
    article-date: 2026-07-08
    article-desc: 详解 FP16/BF16 混合精度与 DistributedDataParallel 的原理、PyTorch 实战与踩坑指南,LLM 训练六件套。
    article-tags: [AMP, DDP, 分布式]
---