import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import string

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


df_train = pd.read_csv('../data_prepared/train_ManBild.csv')
df_test = pd.read_csv('../data_prepared/test_ManBild.csv')


predictors = [ i for i in df_train.columns if not i in ['interest_level', 'manager_level_low', 'manager_level_medium', 'manager_level_high']]
targetname = 'interest_level'

train_X = df_train[predictors].values
train_y = df_train[targetname].values
test_X = df_test.values




# # option 1 
# cv_scores = []
# kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
# for dev_index, val_index in kf.split(range(train_X.shape[0])):
#         dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
#         dev_y, val_y = train_y[dev_index], train_y[val_index]
#         preds, model = runXGB(dev_X, dev_y, val_X, val_y)
#         cv_scores.append(log_loss(val_y, preds))
#         print(cv_scores)


# preds, model = runXGB(train_X, train_y, test_X, num_rounds=300)
# out_df = pd.DataFrame(preds)
# out_df.columns = ["low", "medium", "high"]
# out_df["listing_id"] = df_test.listing_id.values
# out_df.to_csv("../sub/xgb_LtIsLit_noMan.csv", index=False)




# option 2, automatic detection of the best num_rounds

SEED = 444
NFOLDS = 5

params = {
    'eta':.1,
    'colsample_bytree':.8,
    'subsample':.8,
    'seed':0,
    'max_depth':6,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3,
    'silent':1
}

dtrain = xgb.DMatrix(data=train_X, label=train_y)
dtest = xgb.DMatrix(data=test_X)

bst = xgb.cv(params, dtrain, 10000, NFOLDS, early_stopping_rounds=50, verbose_eval=25)
best_rounds = np.argmin(bst['test-mlogloss-mean'])
bst = xgb.train(params, dtrain, best_rounds)

preds = bst.predict(dtest)
preds = pd.DataFrame(preds)

cols = ['low', 'medium', 'high']
preds.columns = cols
preds['listing_id'] = df_test.listing_id.values
preds.to_csv('../sub/xgb_LtIsLit_bild.csv', index=None)