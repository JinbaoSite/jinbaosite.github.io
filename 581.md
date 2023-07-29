# 返乡发展人群预测

## 一、赛题介绍

比赛地址：[返乡发展人群预测](https://www.datafountain.cn/competitions/581)

### 1.1 任务介绍

基于中国联通的大数据能力，通过使用对联通的信令数据、通话数据、互联网行为等数据进行建模，对个人是否会返乡工作进行判断

### 1.2 数据简介

`train.csv`:包含全量数据集的70%（dataNoLabel是训练集的一部分，选手可以自己决定是否使用）
`test.csv`:包含全量数据集的30%
位置类特特征：基于联通基站产生的用户信令数据；
互联网类特征：基于联通用户上网产生的上网行为数据；
通话类特征：基于联通用户日常通话、短信产生的数据

### 1.3 评估指标

使用ROC曲线下面积AUC（Area Under Curve）作为评价指标，AUC越大，预测越准确

## 二、解题思路

这是一个二分类任务，主要是从数据清洗、特征构造和筛选以及模型融合的思路

- 数据清洗
- 特征构造和筛选
- 模型融合

最终提交结果：

| 榜单 | 分数 |
| :--- | :--- |
| A榜  | 0.91171720 |

### 1.1 库导入

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
```

### 1.2 数据读取

```python
train_data = pd.read_csv('data/dataTrain.csv')
test_data = pd.read_csv('data/dataA.csv')
submission = pd.read_csv('data/submit_example_A.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')
print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape  = {test_data.shape}')
```

### 1.3 特征构造

自己DIY的特征
```python
train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']
```
来自网上的特征（来源：https://github.com/librauee/CCF2022/blob/main/FX/FX_baseline.py）
```python
# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)
```
ID类特征数值化
```
cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])
```
最后构造出训练集和测试集
```
num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]
```

### 1.4 模型训练代码

常用的交叉验证模型框架
```python
def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold
```

### 1.5 数据清洗

在训练数据中存在很大一部分的干扰数据，导致模型训练存在很多噪声数据，所以需要对噪声数据进行清洗

这里我使用一个简易的模型，将数据切割成60份，对每一份数据单独作为验证集，如果验证集的AUC几乎接近于0.5，则验证集中的数据大多数为干扰数据。

```
gbc = GradientBoostingClassifier()
gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 60)
```

测试结果为

| KFlod | AUC |
| :---- | :-- |
|  0 |  0.9034 |
|  1 |  0.9112 |
|  2 |  0.9045 |
|  3 |  0.9006 |
|  4 |  0.9013 |
|  5 |  0.8986 |
|  6 |  0.9009 |
|  7 |  0.9116 |
|  8 |  0.9245 |
|  9 |  0.8902 |
|  10 |  0.8995 |
|  11 |  0.9047 |
|  12 |  0.9291 |
|  13 |  0.8980 |
|  14 |  0.9279 |
|  15 |  0.8995 |
|  16 |  0.9080 |
|  17 |  0.8942 |
|  18 |  0.9179 |
|  19 |  0.9044 |
|  20 |  0.8937 |
|  21 |  0.9138 |
|  22 |  0.9024 |
|  23 |  0.9091 |
|  24 |  0.8937 |
|  25 |  0.9173 |
|  26 |  0.9047 |
|  27 |  0.9010 |
|  28 |  0.9047 |
|  29 |  0.9144 |
|  30 |  0.8984 |
|  31 |  0.9079 |
|  32 |  0.9240 |
|  33 |  0.8899 |
|  34 |  0.8780 |
|  35 |  0.8942 |
|  36 |  0.9112 |
|  37 |  0.9221 |
|  38 |  0.9273 |
|  39 |  0.9137 |
|  40 |  0.9206 |
|  41 |  0.9053 |
|  42 |  0.9033 |
|  43 |  0.9193 |
|  44 |  0.9141 |
|  45 |  0.9087 |
|  46 |  0.8957 |
|  47 |  0.9142 |
|  48 |  0.9179 |
|  49 |  0.9128 |
|  50 |  0.5608 |
|  51 |  0.5328 |
|  52 |  0.5020 |
|  53 |  0.5067 |
|  54 |  0.4888 |
|  55 |  0.5065 |
|  56 |  0.5163 |
|  57 |  0.5019 |
|  58 |  0.5235 |
|  59 |  0.4842 |

很明显可以看出，50~59的数据为干扰数据
所以剔除干扰数据
```
train = train[:50000]
label = label[:50000]
```

### 1.6 模型融合

选取了
- `GradientBoostingClassifier` 
- `HistGradientBoostingClassifier`
- `XGBClassifier`
- `LGBMClassifier`
- `CatBoostClassifier`
这6个树模型作为基础模型
```python
gbc = GradientBoostingClassifier(
    n_estimators=50, 
    learning_rate=0.1,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6, 
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05, 
    n_estimators=100, 
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210, 
    depth=6, 
    learning_rate=0.03, 
    l2_leaf_reg=1, 
    loss_function='Logloss', 
    verbose=0
)
```
通过`StackingClassifier`将6个模型进行Stack，Stack模型用`LogisticRegression`
```python
estimators = [
    ('gbc', gbc),
    ('hgbc', hgbc),
    ('xgbc', xgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression()
)
```

### 1.7 特征筛选

特征筛选思路：先将模型训练好，然后对验证集进行测试得到基础AUC，之后循环遍历所有特征，在验证集上对单个特征进行mask后，得到mask后的AUC，评估两个AUC的差值，差值越大，则说明特征重要性越高。

先将训练数据划分成训练集和验证集
```python
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)
```
然后用组合模型进行训练和验证
```python
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)
```
循环遍历特征，对验证集中的特征进行mask
```
ff = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))
```
特征重要性TOP20

| 特征 | AUC | diff |
| :--- | :-- | :--- |
|  f46  | 0.91137151 | -0.00207482 |
| f2/f4  | 0.91193439 | -0.00151194 |
| f2/f6  | 0.9121758 | -0.00127053 |
| f4/f5  | 0.91224485 | -0.00120148 |
|  f47  | 0.91252212 | -0.00092421 |
| f5-f6  | 0.91257545 | -0.00087088 |
| f4-f5  | 0.91269045 | -0.00075589 |
| f44/f45  | 0.91274186 | -0.00070447 |
|   f3  | 0.9127596 | -0.00068673 |
|  f19  | 0.91277603 | -0.00067031 |
| f4-f6  | 0.91288475 | -0.00056158 |
| f2-f4  | 0.91292365 | -0.00052268 |
| f44/f46  | 0.91296255 | -0.00048378 |
| f45/f46  | 0.91296478 | -0.00048156 |
| f2/f5  | 0.91303568 | -0.00041066 |
| f5/f6  | 0.91307167 | -0.00037466 |
| f1/f6  | 0.91309919 | -0.00034714 |
|   f8  | 0.91310869 | -0.00033764 |
| f1/f5  | 0.91315365 | -0.00029269 |
| f1-f4  | 0.91318568 | -0.00026066 |

这里选取所有差值为负的特征，对比特征筛选后的特征提升
```python
clf.fit(X_train[ff], y_train)
y_pred = clf.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)
```

| 特征筛选前 | 特征筛选后 |
| :--------- | :--------- |
| 0.91344633 | 0.91385538 |


### 1.8 模型训练

```
train = train[ff]
test = test[ff]

clf_test_preds = model_train(clf, "StackingClassifier", 10)

submission['label'] = clf_test_preds
submission.to_csv('submission.csv', index=False)
```

最后线下训练结果为

| KFold | AUC |
| :---- | :--- |
|    0 |  0.9069 |
|    1 |  0.9091 |
|    2 |  0.9156 |
|    3 |  0.9059 |
|    4 |  0.9075 |
|    5 |  0.9083 |
|    6 |  0.9007 |
|    7 |  0.9173 |
|    8 |  0.9160 |
|    9 |  0.9146 |
| overall | 0.9100 |

## 三、总结

最后线上A榜的结果为**0.91171720**

后续优化思路：

- 半监督学习, 使用pseudo_labeling
- 特征工程，使用PolynomialFeatures进行多项式特征组合
