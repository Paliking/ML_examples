# Inspired by:
#SRK script
#https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python
#Faron script

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import pickle


SEED = 444




with open('data4stack/data_level2.pickle', 'rb') as handle:
    level2 = pickle.load(handle)
x_train, x_test = level2

# y train som zabudol dat do level2
with open('data4stack/data_perpared_wleak.pickle', 'rb') as handle:
    sets_prepared = pickle.load(handle)
_, y_train, X_test = sets_prepared






print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.02,
    'max_depth':3,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'silent':1
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=2000, nfold=5, seed=SEED,
             early_stopping_rounds=20, show_stdv=True, verbose_eval=10)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]


print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

# out_df = pd.DataFrame(gbdt.predict(dtest))
# out_df.columns = ["low", "medium", "high"]
# out_df["listing_id"] = X_test.listing_id.values
# out_df.to_csv('../sub/stacker2_starter7_2.csv', index=False)