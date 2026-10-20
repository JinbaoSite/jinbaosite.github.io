---
layout: listpage
title: 推荐算法
subtitle: 召回、排序、重排、序列推荐、多目标优化、生成式推荐
# article-date 按论文正式出版日期填写；只有预印本时采用首次公开日期。
# 正式出版记录缺少日时，以会议首日作为排序日期，并在条目中注明。
# 基础综述采用条目中注明的代表性文献，不代表该研究方向最早提出的时间。
article-list:
  - article-title: CCFormer：腾讯工业推荐中的高效长序列建模
    article-url: /recsys/ccformer
    article-date: 2026-07-30
    article-desc: CCFormer 面向工业推荐中的超长用户行为序列，通过 Feature-Field Separated Cross Attention 建模用户、序列与候选物品的定向交互，并结合相对时间位置编码、Subspace Token Mixing 和分层序列压缩，将传统 Self-Attention 的平方复杂度降低为近似线性复杂度。
    article-tags: [长序列, CCFormer, Attention]
  - article-title: RankMixer：工业级推荐排序模型的规模化之路
    article-url: /recsys/rankmixer
    article-date: 2025-07-21
    article-desc: RankMixer 用无参数的 Multi-Head Token Mixing（切分 head 后跨 Token 重组）替代平方复杂度的 Self-Attention 完成全局特征交互，再用 Per-token FFN 让每个特征子空间拥有独立参数、避免高频特征淹没长尾特征，并可扩展为 ReLU Routing + DTSI 的 Sparse-MoE 变体，用可堆叠矩阵乘架构大幅提升 GPU 利用率，实现推荐排序模型的低成本参数扩容。
    article-tags: [精排, RankMixer]
  - article-title: DCN V2：矩阵交叉与低秩混合专家
    article-url: /recsys/dcnv2/
    article-date: 2021-04-19
    article-desc: DCN V2 将向量式交叉扩展为矩阵式交叉，通过低秩分解与混合专家平衡表达能力和计算成本。本文讲解交叉层公式、串行与并行结构、交互次数边界，以及三种交叉模块的 PyTorch 实现。
    article-tags: [DCN V2, 特征交叉, 精排]
  - article-title: SIM：基于搜索的终身序列兴趣建模
    article-url: /recsys/sim/
    article-date: 2020-10-19
    article-desc: SIM 面向点击率预估中的终身行为序列，以通用搜索单元从上万条历史中检索候选相关行为，再由精确搜索单元融合时间间隔并进行多头注意力建模。本文讲解硬搜索、软搜索、联合训练、在线索引及可运行的 PyTorch 实现。
    article-tags: [SIM, 长序列, 精排]
  - article-title: AutoInt：用自注意力机制自动学习特征交互
    article-url: /recsys/autoint
    article-date: 2019-11-03
    article-desc: AutoInt 通过多头自注意力机制显式建模特征间的交互,用残差连接保留原始信息、用堆叠层数控制交互阶数,在自动学习高阶特征组合的同时,借注意力权重提供可解释性。
    article-tags: [精排, 特征交叉]
  - article-title: MIND：动态路由多兴趣网络
    article-url: /recsys/mind/
    article-date: 2019-11-03
    article-desc: MIND 面向召回阶段，通过行为到兴趣动态路由将用户历史聚合为多个兴趣向量，训练时以标签感知注意力对齐目标物品，服务时由多个兴趣分别查询向量索引。本文包含动态兴趣数量、采样训练和 PyTorch 实现。
    article-tags: [MIND, 多兴趣, 召回]
  - article-title: DSIN：深度会话兴趣网络
    article-url: /recsys/dsin/
    article-date: 2019-08-10
    article-desc: DSIN 将用户历史划分为多个会话，通过偏置编码和多头自注意力提取会话兴趣，再使用 Bi-LSTM 建模跨会话关系，并根据候选物品聚合原始与上下文化兴趣。本文包含两级掩码和空历史处理的 PyTorch 实现。
    article-tags: [DSIN, 会话兴趣, 精排]
  - article-title: DIEN：深度兴趣演化网络
    article-url: /recsys/dien/
    article-date: 2019-07-17
    article-desc: DIEN 使用 GRU 提取逐时刻兴趣状态，以辅助损失提供下一行为监督，再通过 AUGRU 根据候选物品控制兴趣演化。本文讲解辅助任务、AIGRU、AGRU、AUGRU 及完整的 PyTorch 实现。
    article-tags: [DIEN, 兴趣演化, 精排]
  - article-title: SASRec：基于自注意力的序列推荐
    article-url: /recsys/sasrec/
    article-date: 2018-11-17
    article-desc: SASRec 通过因果自注意力、可学习位置向量和逐位置前馈网络建模用户历史，并以共享物品 Embedding 完成下一物品预测。本文结合 SASRec架构图，讲解序列错位、因果与补零掩码、负采样 BCE，以及可运行的 PyTorch 实现。
    article-tags: [SASRec, 序列推荐, 自注意力]
  - article-title: xDeepFM：显式与隐式特征交互融合
    article-url: /recsys/xdeepfm/
    article-date: 2018-07-19
    article-desc: xDeepFM 通过压缩交互网络 CIN 显式建模向量级高阶交互，并融合线性分支与深度网络。本文讲解交互与压缩、阶数推导、多层池化、计算成本，以及完整的 PyTorch 实现。
    article-tags: [xDeepFM, 特征交叉, 精排]
  - article-title: "MMoE：多门控混合专家模型"
    article-url: /recsys/mmoe
    article-date: 2018-07-19
    article-desc: 通过门控网络来学习多个专家模型的权重，提高模型的多任务学习能力
    article-tags: [排序]
  - article-title: DIN：深度兴趣网络
    article-url: /recsys/din
    article-date: 2018-07-19
    article-desc: DIN 根据候选物品激活相关历史行为，通过未归一化加权求和构建目标相关兴趣。本文讲解兴趣激活、Dice、小批量感知正则化及模型边界，并提供支持补零掩码和空历史的 PyTorch 实现。
    article-tags: [DIN,排序]
  - article-title: "ESMM：全空间多任务模型"
    article-url: /recsys/esmm
    article-date: 2018-06-27
    article-desc: 通过多任务学习，同时学习ctr和cvr，在完整样本空间上进行训练，避免了传统CVR模型经常遭遇的样本选择偏差和训练数据稀疏的问题
    article-tags: [排序]
  - article-title: 深度模型DeepFM模型
    article-url: /recsys/deepfm
    article-date: 2017-08-19
    article-desc: DeepFM融合FM与DNN,采用并行双流架构，能自动捕获低阶与高阶特征交叉,两部分共享底层Embedding，无需人工特征工程，实现了高效、端到端的联合训练。
    article-tags: [DeepFM,排序]
  - article-title: "DCN：深度交叉网络"
    article-url: /recsys/dcn
    article-date: 2017-08-14
    article-desc: DCN 通过 Cross Network 显式学习有界阶数的特征交叉，并与 Deep Network 的非线性表达并行融合，在保持参数量和计算复杂度随特征维度线性增长的同时减少人工特征工程，本文系统讲解 Cross Layer 公式、高阶交叉原理、模型结构、PyTorch 实现与工程实践。
    article-tags: [DCN, 特征交叉, CTR]
  - article-title: 神经协同过滤算法解析：从矩阵分解到深度神经网络
    article-url: /recsys/ncf
    article-date: 2017-04-03
    article-desc: NCF 用神经网络替代传统矩阵分解中的线性内积，通过用户与物品 Embedding、GMF 和多层感知机学习复杂的非线性交互，并以 NeuMF 融合两条建模路径，本文涵盖核心结构、负采样损失、应用场景与 PyTorch 实现。
    article-tags: [NCF, 协同过滤, NeuMF]
  - article-title: 深度模型Wide&Deep模型
    article-url: /recsys/wdl
    article-date: 2016-09-15
    article-desc: 将协同过滤和深度学习结合，捕捉用户和物品的隐式联系和高阶特征
    article-tags: [Wide&Deep,排序]
  - article-title: YouTube 深度神经网络推荐系统架构
    article-url: /recsys/youtube_dnn
    article-date: 2016-09-07
    article-desc: YouTube 深度神经网络推荐系统架构采用“召回+排序”的两阶段推荐：召回阶段将海量视频检索简化为“极端多分类”问题，引入 Example Age 特征消除时间偏置，通过高效近邻检索（ANN）粗滤出候选集；排序阶段则巧妙利用加权逻辑回归，将优化目标从点击率转化为“期望观看时长”，对候选视频进行精准打分。
    article-tags: [DNN,ANN]
  - article-title: "FM因子分解机"
    article-url: /recsys/fm
    article-date: 2010-12-13
    article-desc: 将用户和物品的特征进行线性组合，并引入二次项来捕捉特征之间的交互关系
    article-tags: [FM,因子分解]
  - article-title: 协同过滤推荐系统中矩阵分解
    article-url: /recsys/mf
    article-date: 2009-08-07
    article-desc: 将用户行为矩阵分解为两个矩阵的乘积，通过用户向量和物品向量的内积来表示用户对物品的偏好
    article-tags: [MF,矩阵分解]
  - article-title: 基于邻域的协同过滤
    article-url: /recsys/cf
    article-date: 2001-04-01
    article-desc: 基于邻域的协同过滤通过用户或物品共现生成推荐，涵盖相似度计算、候选打分、热门与活跃因素修正、显式评分预测，以及可运行的 Python 实现与时间划分评估方法。
    article-tags: [UserCF, ItemCF, 协同过滤]
---
