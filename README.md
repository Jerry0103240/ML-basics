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
     
#### Decision tree/Random forest
  - Entropy (熵): 用來描述事件的不確定性，熵越高則越混亂，熵越低則越秩序
  - Information gain (訊息增益): 描述有多少的熵被移除，增益越大越好
  - Decision tree
    - ID3 (Iterative Dichotomiser 3)
      - 利用 Information gain 作為特徵的選擇，越高就越優先做為決策條件
    - C4.5
      - ID3 有個缺點，如果有個 feature (例如流水編號) 把所有資料都切開，則 information gain 就會很大，但實際上沒有作用
      - 基於 ID3 的理念，額外考慮了分支數的資訊，是為 gain ratio
    - CART
      - 使用基尼不純度（Gini impurity）進行衡量，旨在描述某個類別被分錯的機率，因此每個 root 只會有兩個 child

#### K-means/KNN
  - K-means
    - 無監督訓練 (無 Label)，聚類算法，旨在將資料分為 K 個群體。
    - 隨機選擇 K 筆資料 (群心) 作為初始點，計算所有資料與該 K 筆群心的歐式距離，每筆資料會被判定給距離最近的群心，接著更新群心，不斷迭代直到資料不會有太大變動為止。
    - 對於超參數 K 的選擇與初始位置敏感。
  - k-NN
    - 監督訓練
    - 找尋與該筆資料最近的 K 個鄰居，並利用此 K 個鄰居進行預測。

#### Evaluation metrics
  - AUC-ROC
    - 縱軸為 TPR，橫軸為 FPR
    - 物理意義為，隨機抽取一對正負樣本，模型對正樣本估計值高於負樣本估計值的機率; 旨在衡量模型把正樣本排序在負樣本前面的能力，不考慮預測精度，並且不受分布、採樣、閥值影響。
  - AUC-PR
    - 縱軸為 Precision，橫軸為 Recall
    - 旨在關注正樣本的預測能力，baseline 為 positive / total samples
  - ref.: https://zhuanlan.zhihu.com/p/349516115

#### PCA

#### SVM

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
