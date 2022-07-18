# ML-basics

#### Bias-variance tradeoff
  - Model total error = bias + variance
  - Bias is the distance between the predictions of a model and the true values.
  - Variance of model predictions for given data points that tell us the spread of our data.
  - High bias leads to underfitting; high variance leads to overfitting.
  - Solution for underfitting (high bias):
    - increasing training iterations or features or model complexity
    - tuning model parameters
    - removing regularization if used
  - Solution for overfitting (high variance):
    - early stopping
    - tuning model parameters
    - increasing training data
    - decreasing training features dimensions or model complexity
    - adding dropout or regularization if not used

#### Linear vs logisitc regression
  - Linear regression: for regression problems, range can exceed 0 or 1
  - Logistic regression: for classification problems, range between 0 and 1
  - Logistic regression is based on the concept of maximum likelihood estimation
    - MLE is a technique for estimate parameters of model

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

#### bagging/boosting/Adaboost

#### decision tree/random forest

#### k-means/k-nn/gaussian mixture model

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
  - SGD
  - Momentum
  - Adagrad
  - RMSProp
  - Adadelta
  - Adam
  - [Overview](https://ruder.io/optimizing-gradient-descent/)
#### droput
