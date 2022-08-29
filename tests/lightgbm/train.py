import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score

pd.options.display.max_columns = 999

# loading the breast cancer dataset
X = load_breast_cancer()
Y = X.target

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.data, Y, test_size=0.2, random_state=0)

# converting the dataset into proper LGB format
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

# Specifying the parameter
params = {}
params['learning_rate'] = 0.1  # 学习率
params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
params['objective'] = 'binary'  # 这里选择二分类
params['metric'] = 'binary_logloss'  # 二分类损失函数
params['max_depth'] = 8  # 树的最大深度
params['num_leaves'] = 256
params['feature_fraction'] = 0.8
params['bagging_fraction'] = 0.8
params['num_threads'] = 4  # 线程数，建议等于CPU内核数。
params['verbosity'] = 20
params['early_stopping_round'] = 5

# train the model
clf = lgb.train(params, d_train, 1000, verbose_eval=4, valid_sets=[d_train, d_test])  # train the model on 100 epocs

# prediction on the test set
y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
clf.save_model('model.txt')
threshold = 0.5
pred_result = []
for mypred in y_pred:
    if mypred > threshold:
        pred_result.append(1)
    else:
        pred_result.append(0)
pred_result = np.array(pred_result)
print(np.sum(pred_result == y_test) / (y_test.shape))