---
layout: listpage
title: 推荐算法
subtitle: 召回、排序、重排、序列推荐、多目标优化、生成式推荐
article-list:
  - article-title: AutoInt：用自注意力机制自动学习特征交互
    article-url: /recsys/autoint
    article-date: 2026-06-05
    article-desc: AutoInt 通过多头自注意力机制显式建模特征间的交互,用残差连接保留原始信息、用堆叠层数控制交互阶数,在自动学习高阶特征组合的同时,借注意力权重提供可解释性。。
    article-tags: [精排, 特征交叉]
  - article-title: 深度模型Wide&Deep模型
    article-url: /recsys/wdl
    article-date: 2023/9/28
    article-desc: 将协同过滤和深度学习结合，捕捉用户和物品的隐式联系和高阶特征。
    article-tags: [召回]
  - article-title: 深度模型Wide&Deep模型
    article-url: /recsys/wdl
    article-date: 2023/9/28
    article-desc: 将协同过滤和深度学习结合，捕捉用户和物品的隐式联系和高阶特征
    article-tags: [排序]
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
