# ML-basics

#### Bias-variance tradeoff
  - 模型評估指標: Bias + Variance
    - Bias: 模型預估值和真實Label間的差異，過低代表模型欠擬合。
    - Variance: 模型預估值的分散程度，過高代表模型對於小變動很敏感，容易導致泛化能力差，通常代表過擬合。
  - 解決方案
    - 欠擬合 (high bias):
      - 增加訓練次數、特徵、模型複雜度
      - 如果有使用正則化技術，先移除
    - 過擬合 (high variance) 
      - early stopping
      - 減少特徵維度、模型複雜度
      - 嘗試加入 dropout、正則化技術

#### Logistic regression
  - 邏輯回歸是假設資料符合 Bernoulli 分布，並透過 Maximum likelihood estimation 求解，得到常見的 logloss (或稱 cross-entroy)
  - Entropy (熵): 用來描述事件的不確定性，熵越高則越混亂，熵越低則越秩序

#### Training skills
  - Used when model training encounter overfitting situation
    - L1 (Lasso)
      - reduce weights to zero and leads model sparse
    - L2 (Ridge)
      - weight decay
      - not leads model sparse
    - dropout
  - Used when model training encounter gradient vanish
    - batch normalization
      - resolve internal covariate shift
  - Used when model training encounter dataset is not enough or lack of variablity
    - data augmentation

#### Ensemble learning
  - 旨在訓練多個弱分類器 (weak learners)，得到一個強分類器 (strong learners)
  - Bagging
    - 各模型獨立訓練，可平行進行; 對訓練數據進行有放回的抽樣，生成多個不同的訓練子集
    - 預測結果為各分類器投票 (分類任務) 或者平均 (回歸任務)
    - 旨在減少方差 (Variance)
    - e.g. Random forest
  - Boosting
    - 在每一次訓練過程中，根據樣本權重訓練一個弱分類器，強調錯誤分類的樣本; 根據分類器的錯誤率更新樣本權重，使得下一次訓練，模型更加關注被前一次分類錯誤的樣本
    - 預測結果為各分類器的權重和
    - 旨在減少偏差 (Bias)
    - e.g. Adaboost
     
#### Decision tree/random forest

#### k-means/k-nn/gaussian mixture model
  - k-means
    - Unsupervised learning
    - Clustering algorithm that tries to partition a set of points into K sets (clusters) such that the points in each cluster tend to be near each other. It is unsupervised because the points have no external ground truth.
  - k-nn
    - Supervised learning
    - Combining the classification of the K nearest points. It is supervised because you are trying to classify a point based on the known classification of other points.  

#### evaluation metrics
  - AUC, ROC, PR
  - recall, precision, F-score
  - cross-entropy (logloss)

#### PCA

#### SVM

#### cross validation
  - k-fold

#### convolution
  - kernel size
  - receptive field

#### activation functions
  - relu, sigmoid, tanh, softmax

#### optimizer
  - Batch, stochastic, mini-batch gradient descent
    - Batch:
      - compute the gradient from the entrie training dataset.
    - Stochastic:
      - compute the gradient from each training sample.
      - suitable for online learning.
    - Mini-batch:
      - compute the gradient for each n training samples.
      - reduce high variance of stochastic method, and increase the updating perfromance of batch method.
  - SGD
    - compute gradient from each training sample, good choice for online learning.
  - Momentum
    - SGD will face oscillation conidition, add previous gradient with fraction in current updating step.
  - Nesterov (NAG)
    - Momentum may go over, NAG pre-compute next approximate gradient for solving such situation.
  - Adagrad
    - adaptive learning rate for each weight
    - lower learning rate for frequent features, larger learning rate for infrequent features
    - learning rate will shrink eventually, cause denominator is always positive
  - RMSProp
    - For solving drawbacks of Adagrad, take exponential moving average of gradients as denominator.
  - Adadelta
  - Adam
    - combine Adagrad & Momentum method.
  - [Overview](https://ruder.io/optimizing-gradient-descent/)

#### droput
