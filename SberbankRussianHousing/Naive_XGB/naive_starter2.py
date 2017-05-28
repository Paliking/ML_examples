# https://www.kaggle.com/reynaldo/naive-xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import pickle
#now = datetime.datetime.now()

train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')
macro = pd.read_csv('../inputs/macro.csv')
id_test = test.id

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values) + list(x_test[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))



xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

print('train shape', x_train.shape)
# x_train.to_csv('../subs/naive2.csv', index=True)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=20, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


num_boost_rounds = len(cv_output)
print('num rounds:', num_boost_rounds)
model = xgb.train(xgb_params, dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, height=0.5, ax=ax)
plt.show()

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('../subs/xgbSub_seed255.csv', index=False)