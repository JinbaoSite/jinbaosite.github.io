---
layout: listpage
title: 推荐算法
subtitle: 召回、排序、重排、序列推荐、多目标优化、生成式推荐
article-list:
  - article-title: 深度交叉网络（Deep & Cross Network，DCN）
    article-url: /recsys/dcn
    article-date: 2026-10-08
    article-desc: DCN 通过 Cross Network 显式学习有界阶数的特征交叉，并与 Deep Network 的非线性表达并行融合，在保持参数量和计算复杂度随特征维度线性增长的同时减少人工特征工程，本文系统讲解 Cross Layer 公式、高阶交叉原理、模型结构、PyTorch 实现与工程实践。
    article-tags: [DCN, 特征交叉, CTR]
  - article-title: CCFormer：腾讯工业推荐中的高效长序列建模
    article-url: /recsys/ccformer
    article-date: 2026-09-30
    article-desc: CCFormer 面向工业推荐中的超长用户行为序列，通过 Feature-Field Separated Cross Attention 建模用户、序列与候选物品的定向交互，并结合相对时间位置编码、Subspace Token Mixing 和分层序列压缩，将传统 Self-Attention 的平方复杂度降低为近似线性复杂度。
    article-tags: [长序列, CCFormer, Attention]
  - article-title: 神经协同过滤算法解析：从矩阵分解到深度神经网络
    article-url: /recsys/ncf
    article-date: 2026-09-22
    article-desc: NCF 用神经网络替代传统矩阵分解中的线性内积，通过用户与物品 Embedding、GMF 和多层感知机学习复杂的非线性交互，并以 NeuMF 融合两条建模路径，本文涵盖核心结构、负采样损失、应用场景与 PyTorch 实现。
    article-tags: [NCF, 协同过滤, NeuMF]
  - article-title: RankMixer：工业级推荐排序模型的规模化之路
    article-url: /recsys/rankmixer
    article-date: 2026-07-20
    article-desc: RankMixer 用无参数的 Multi-Head Token Mixing（切分 head 后跨 Token 重组）替代平方复杂度的 Self-Attention 完成全局特征交互，再用 Per-token FFN 让每个特征子空间拥有独立参数、避免高频特征淹没长尾特征，并可扩展为 ReLU Routing + DTSI 的 Sparse-MoE 变体，用可堆叠矩阵乘架构大幅提升 GPU 利用率，实现推荐排序模型的低成本参数扩容。
    article-tags: [精排, RankMixer]
  - article-title: AutoInt：用自注意力机制自动学习特征交互
    article-url: /recsys/autoint
    article-date: 2026-06-05
    article-desc: AutoInt 通过多头自注意力机制显式建模特征间的交互,用残差连接保留原始信息、用堆叠层数控制交互阶数,在自动学习高阶特征组合的同时,借注意力权重提供可解释性。
    article-tags: [精排, 特征交叉]
  - article-title: 多任务模型MMoE
    article-url: /recsys/mmoe
    article-date: 2020/12/13
    article-desc: 通过门控网络来学习多个专家模型的权重，提高模型的多任务学习能力
    article-tags: [排序]
  - article-title: CVR预估模型ESMM
    article-url: /recsys/esmm
    article-date: 2020/08/24
    article-desc: 通过多任务学习，同时学习ctr和cvr，在完整样本空间上进行训练，避免了传统CVR模型经常遭遇的样本选择偏差和训练数据稀疏的问题
    article-tags: [排序]
  - article-title: 深度兴趣网络Deep Interest Network (DIN)
    article-url: /recsys/din
    article-date: 2023/2/14
    article-desc: DIN（深度兴趣网络）的核心原理在于引入局部激活机制（Local Activation）。 它改变了传统模型用固定向量粗暴压缩用户历史的行为，而是针对当前具体的候选广告，自适应地计算历史行为的注意力权重。候选广告与某个历史行为越相关，该行为被赋予的权重就越高，从而帮助模型精准提取出与当前广告匹配的局部用户兴趣。
    article-tags: [DIN,排序]
  - article-title: 深度模型DeepFM模型
    article-url: /recsys/deepfm
    article-date: 2023/2/7
    article-desc: DeepFM融合FM与DNN,采用并行双流架构，能自动捕获低阶与高阶特征交叉,两部分共享底层Embedding，无需人工特征工程，实现了高效、端到端的联合训练。
    article-tags: [DeepFM,排序]
  - article-title: 深度模型Wide&Deep模型
    article-url: /recsys/wdl
    article-date: 2023/1/28
    article-desc: 将协同过滤和深度学习结合，捕捉用户和物品的隐式联系和高阶特征
    article-tags: [Wide&Deep,排序]
  - article-title: YouTube 深度神经网络推荐系统架构
    article-url: /recsys/youtube_dnn
    article-date: 2023/1/21
    article-desc: YouTube 深度神经网络推荐系统架构采用“召回+排序”的两阶段推荐：召回阶段将海量视频检索简化为“极端多分类”问题，引入 Example Age 特征消除时间偏置，通过高效近邻检索（ANN）粗滤出候选集；排序阶段则巧妙利用加权逻辑回归，将优化目标从点击率转化为“期望观看时长”，对候选视频进行精准打分。
    article-tags: [DNN,ANN]
  - article-title: FM因子分解
    article-url: /recsys/fm
    article-date: 2023/1/14
    article-desc: 将用户和物品的特征进行线性组合，并引入二次项来捕捉特征之间的交互关系
    article-tags: [FM,因子分解]
  - article-title: 协同过滤推荐系统中矩阵分解
    article-url: /recsys/mf
    article-date: 2023/1/7
    article-desc: 将用户行为矩阵分解为两个矩阵的乘积，通过用户向量和物品向量的内积来表示用户对物品的偏好
    article-tags: [MF,矩阵分解]
  - article-title: 基于邻域的协同过滤
    article-url: /recsys/cf
    article-date: 2023/1/1
    article-desc: 基于用户的协同过滤算法根据用户对物品的偏好，计算用户与其他用户的相似度，根据用户的相似度，推荐与用户兴趣相似的物品。基于物品的协同过滤算法根据物品之间的相似度，推荐与用户之前喜欢的物品相似的物品
    article-tags: [itemcf,usercf,协调过滤]
---
