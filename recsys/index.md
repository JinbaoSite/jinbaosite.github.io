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
    # 日期依据：arXiv:2607.28070 v1；首次公开日期
    article-date: 2026-07-30
    article-desc: CCFormer 面向工业推荐中的超长用户行为序列，通过 Feature-Field Separated Cross Attention 建模用户、序列与候选物品的定向交互，并结合相对时间位置编码、Subspace Token Mixing 和分层序列压缩，将传统 Self-Attention 的平方复杂度降低为近似线性复杂度。
    article-tags: [长序列, CCFormer, Attention]
  - article-title: RankMixer：工业级推荐排序模型的规模化之路
    article-url: /recsys/rankmixer
    # 日期依据：arXiv:2507.15551 v1；首次公开日期
    article-date: 2025-07-21
    article-desc: RankMixer 用无参数的 Multi-Head Token Mixing（切分 head 后跨 Token 重组）替代平方复杂度的 Self-Attention 完成全局特征交互，再用 Per-token FFN 让每个特征子空间拥有独立参数、避免高频特征淹没长尾特征，并可扩展为 ReLU Routing + DTSI 的 Sparse-MoE 变体，用可堆叠矩阵乘架构大幅提升 GPU 利用率，实现推荐排序模型的低成本参数扩容。
    article-tags: [精排, RankMixer]
  - article-title: DCN V2：矩阵交叉与低秩混合专家
    article-url: /recsys/dcnv2/
    # 日期依据：WWW 2021；DOI:10.1145/3442381.3450078；正式出版日期
    article-date: 2021-04-19
    article-desc: DCN V2 将向量式交叉扩展为矩阵式交叉，通过低秩分解与混合专家平衡表达能力和计算成本。本文讲解交叉层公式、串行与并行结构、交互次数边界，以及三种交叉模块的 PyTorch 实现。
    article-tags: [DCN V2, 特征交叉, 精排]
  - article-title: AutoInt：用自注意力机制自动学习特征交互
    article-url: /recsys/autoint
    # 日期依据：CIKM 2019；DOI:10.1145/3357384.3357925；正式出版日期
    article-date: 2019-11-03
    article-desc: AutoInt 通过多头自注意力机制显式建模特征间的交互,用残差连接保留原始信息、用堆叠层数控制交互阶数,在自动学习高阶特征组合的同时,借注意力权重提供可解释性。
    article-tags: [精排, 特征交叉]
  - article-title: SASRec：基于自注意力的序列推荐
    article-url: /recsys/sasrec/
    # 日期依据：ICDM 2018；DOI:10.1109/ICDM.2018.00035；出版记录精确到月，按会议首日排序
    article-date: 2018-11-17
    article-desc: SASRec 通过因果自注意力、可学习位置向量和逐位置前馈网络建模用户历史，并以共享物品 Embedding 完成下一物品预测。本文结合 SASRec架构图，讲解序列错位、因果与补零掩码、负采样 BCE，以及可运行的 PyTorch 实现。
    article-tags: [SASRec, 序列推荐, 自注意力]
  - article-title: xDeepFM：显式与隐式特征交互融合
    article-url: /recsys/xdeepfm/
    # 日期依据：KDD 2018；DOI:10.1145/3219819.3220023；正式出版日期
    article-date: 2018-07-19
    article-desc: xDeepFM 通过压缩交互网络 CIN 显式建模向量级高阶交互，并融合线性分支与深度网络。本文讲解交互与压缩、阶数推导、多层池化、计算成本，以及完整的 PyTorch 实现。
    article-tags: [xDeepFM, 特征交叉, 精排]
  - article-title: 多任务模型MMoE
    article-url: /recsys/mmoe
    # 日期依据：KDD 2018；DOI:10.1145/3219819.3220007；正式出版日期
    article-date: 2018-07-19
    article-desc: 通过门控网络来学习多个专家模型的权重，提高模型的多任务学习能力
    article-tags: [排序]
  - article-title: DIN：深度兴趣网络
    article-url: /recsys/din
    # 日期依据：KDD 2018；DOI:10.1145/3219819.3219823；正式出版日期
    article-date: 2018-07-19
    article-desc: DIN 根据候选物品激活相关历史行为，通过未归一化加权求和构建目标相关兴趣。本文讲解兴趣激活、Dice、小批量感知正则化及模型边界，并提供支持补零掩码和空历史的 PyTorch 实现。
    article-tags: [DIN,排序]
  - article-title: CVR预估模型ESMM
    article-url: /recsys/esmm
    # 日期依据：SIGIR 2018；DOI:10.1145/3209978.3210104；正式出版日期
    article-date: 2018-06-27
    article-desc: 通过多任务学习，同时学习ctr和cvr，在完整样本空间上进行训练，避免了传统CVR模型经常遭遇的样本选择偏差和训练数据稀疏的问题
    article-tags: [排序]
  - article-title: 深度模型DeepFM模型
    article-url: /recsys/deepfm
    # 日期依据：IJCAI 2017；DOI:10.24963/ijcai.2017/239；出版记录精确到月，按会议首日排序
    article-date: 2017-08-19
    article-desc: DeepFM融合FM与DNN,采用并行双流架构，能自动捕获低阶与高阶特征交叉,两部分共享底层Embedding，无需人工特征工程，实现了高效、端到端的联合训练。
    article-tags: [DeepFM,排序]
  - article-title: 深度交叉网络（Deep & Cross Network，DCN）
    article-url: /recsys/dcn
    # 日期依据：AdKDD 2017；DOI:10.1145/3124749.3124754；正式出版日期
    article-date: 2017-08-14
    article-desc: DCN 通过 Cross Network 显式学习有界阶数的特征交叉，并与 Deep Network 的非线性表达并行融合，在保持参数量和计算复杂度随特征维度线性增长的同时减少人工特征工程，本文系统讲解 Cross Layer 公式、高阶交叉原理、模型结构、PyTorch 实现与工程实践。
    article-tags: [DCN, 特征交叉, CTR]
  - article-title: 神经协同过滤算法解析：从矩阵分解到深度神经网络
    article-url: /recsys/ncf
    # 日期依据：WWW 2017；DOI:10.1145/3038912.3052569；正式出版日期
    article-date: 2017-04-03
    article-desc: NCF 用神经网络替代传统矩阵分解中的线性内积，通过用户与物品 Embedding、GMF 和多层感知机学习复杂的非线性交互，并以 NeuMF 融合两条建模路径，本文涵盖核心结构、负采样损失、应用场景与 PyTorch 实现。
    article-tags: [NCF, 协同过滤, NeuMF]
  - article-title: 深度模型Wide&Deep模型
    article-url: /recsys/wdl
    # 日期依据：DLRS 2016；DOI:10.1145/2988450.2988454；正式出版日期
    article-date: 2016-09-15
    article-desc: 将协同过滤和深度学习结合，捕捉用户和物品的隐式联系和高阶特征
    article-tags: [Wide&Deep,排序]
  - article-title: YouTube 深度神经网络推荐系统架构
    article-url: /recsys/youtube_dnn
    # 日期依据：RecSys 2016；DOI:10.1145/2959100.2959190；正式出版日期
    article-date: 2016-09-07
    article-desc: YouTube 深度神经网络推荐系统架构采用“召回+排序”的两阶段推荐：召回阶段将海量视频检索简化为“极端多分类”问题，引入 Example Age 特征消除时间偏置，通过高效近邻检索（ANN）粗滤出候选集；排序阶段则巧妙利用加权逻辑回归，将优化目标从点击率转化为“期望观看时长”，对候选视频进行精准打分。
    article-tags: [DNN,ANN]
  - article-title: FM因子分解
    article-url: /recsys/fm
    # 日期依据：ICDM 2010；DOI:10.1109/ICDM.2010.127；出版记录精确到月，按会议首日排序
    article-date: 2010-12-13
    article-desc: 将用户和物品的特征进行线性组合，并引入二次项来捕捉特征之间的交互关系
    article-tags: [FM,因子分解]
  - article-title: 协同过滤推荐系统中矩阵分解
    article-url: /recsys/mf
    # 日期依据：基础综述按代表文献 Matrix Factorization Techniques for Recommender Systems 排序；DOI:10.1109/MC.2009.263；IEEE 发表日期
    article-date: 2009-08-07
    article-desc: 将用户行为矩阵分解为两个矩阵的乘积，通过用户向量和物品向量的内积来表示用户对物品的偏好
    article-tags: [MF,矩阵分解]
  - article-title: 基于邻域的协同过滤
    article-url: /recsys/cf
    # 日期依据：基础综述按代表文献 Item-based collaborative filtering recommendation algorithms 排序；DOI:10.1145/371920.372071；明尼苏达大学出版记录日期
    article-date: 2001-04-01
    article-desc: 基于用户的协同过滤算法根据用户对物品的偏好，计算用户与其他用户的相似度，根据用户的相似度，推荐与用户兴趣相似的物品。基于物品的协同过滤算法根据物品之间的相似度，推荐与用户之前喜欢的物品相似的物品
    article-tags: [itemcf,usercf,协调过滤]
---
