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







with open('data4stack/data_perpared_wleak.pickle', 'rb') as handle:
    sets_prepared = pickle.load(handle)


X_train, y, X_test = sets_prepared


y_train = y
x_train = X_train.values
x_test = X_test.values

SEED = 42
NFOLDS = 5
n_classes=3

ntrain = x_train.shape[0]
ntest = x_test.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


## Creating Classes for stacking


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict_proba(self, x):
        proba = self.clf.predict_proba(x)
        return proba


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 30)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict_proba(self, x):
        proba = self.gbdt.predict(xgb.DMatrix(x))
        return proba


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,n_classes))
    oof_test = np.zeros((ntest,n_classes))
    oof_test_skf = np.empty((ntest, NFOLDS*n_classes))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[:,3*i: 3*i+3] = clf.predict_proba(x_test)
        
    for i in range(3):
        oof_test[:,i] = (oof_test_skf[:,i]+oof_test_skf[:,i+3]+oof_test_skf[:,i+6]+oof_test_skf[:,i+9]+oof_test_skf[:,i+12])/5

    
    return oof_train, oof_test


et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_features': 0.7,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': -1,
    'n_estimators': 400,
    'max_features': 0.2,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.01,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'nrounds': 3200
}

xgb_params2 = {
    'objective': 'multi:softprob',
    'eta':0.01,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'nrounds': 3500,
    'silent':1
}

xgb_params3 = {
    'objective': 'multi:softprob',
    'eta':0.02,
    'max_depth':5,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.9,
    'nrounds': 1600
}

xgb_params4 = {
    'objective': 'multi:softprob',
    'eta':0.05,
    'max_depth':7,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.9,
    'nrounds': 600,
    'silent':1
}


xgb_params5 = {
    'objective': 'multi:softprob',
    'eta':0.02,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'nrounds': 1500
}


xgb_params6 = {
    'objective': 'multi:softprob',
    'eta':0.02,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'nrounds': 1500,
    'silent':1
}

rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}


top_mans_build = []
manager_level = []
building_level = []
display_address_level = []
man_stats = []
street_adress = []
future_count = []
future_count_gr = []
leakage = []
for i in X_train.columns:
    print(i)
    if i.startswith('top'):
        top_mans_build.append(i)
    elif i.startswith('manager_level'):
        manager_level.append(i)
    elif i.startswith('building_level'):
        building_level.append(i)
    elif i.startswith('display_address_level'):
        display_address_level.append(i)
    elif i.startswith('man_'):
        man_stats.append(i)
    elif i.startswith('street_adress'):
        street_adress.append(i)
    elif i.startswith('future_count_gr'):
        future_count_gr.append(i)
    elif i.startswith('future_count_'):
        future_count.append(i)
    elif i.startswith('img_date') or i.startswith('time_stamp') or i.startswith('img_days'):
        leakage.append(i)
    else:
        pass


hcc = []
for i in X_train.columns:
    if 'medium' in i or 'high' in i:
        hcc.append(i)

xg = XgbWrapper(seed=SEED, params=xgb_params)
# et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
xg2 = XgbWrapper(seed=SEED, params=xgb_params2)
xg3 = XgbWrapper(seed=SEED, params=xgb_params3)
xg4 = XgbWrapper(seed=SEED, params=xgb_params4)
xg5 = XgbWrapper(seed=SEED, params=xgb_params5)
xg6 = XgbWrapper(seed=SEED, params=xgb_params6)
# rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

# xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test)
# et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
# # rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
# xg2_oof_train, xg2_oof_test = get_oof(xg2, x_train, y_train, x_test)
# xg3_oof_train, xg3_oof_test = get_oof(xg3, x_train, y_train, x_test)


features1 = [ i for i in X_train.columns if i not in display_address_level+building_level+top_mans_build+street_adress+future_count_gr+future_count]
features2 = [ i for i in X_train.columns if i not in manager_level+man_stats+top_mans_build+future_count_gr+future_count]
# features3 = [ i for i in X_train.columns if i not in manager_level+display_address_level+building_level+top_mans_build+street_adress]
features4 = [ i for i in X_train.columns if i not in manager_level+display_address_level+building_level+top_mans_build+street_adress]
features5 = [ i for i in X_train.columns if i not in leakage]
features6 = [ i for i in X_train.columns if i not in hcc]


xg_oof_train, xg_oof_test = get_oof(xg, X_train[features1].values, y_train, X_test[features1].values)
# et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
xg2_oof_train, xg2_oof_test = get_oof(xg2, X_train[features2].values, y_train, X_test[features2].values)
xg3_oof_train, xg3_oof_test = get_oof(xg3, x_train, y_train, x_test)
xg4_oof_train, xg4_oof_test = get_oof(xg4, X_train[features4].values, y_train, X_test[features4].values)
xg5_oof_train, xg5_oof_test = get_oof(xg5, X_train[features5].values, y_train, X_test[features5].values)
xg6_oof_train, xg6_oof_test = get_oof(xg6, X_train[features6].values, y_train, X_test[features6].values)



print("XG-CV: {}".format(log_loss(y_train, xg_oof_train)))
# print("ET-CV: {}".format(log_loss(y_train, et_oof_train)))
# print("RF-CV: {}".format(log_loss(y_train, rf_oof_train)))
print("XG2-CV: {}".format(log_loss(y_train, xg2_oof_train)))
print("XG3-CV: {}".format(log_loss(y_train, xg3_oof_train)))
print("XG4-CV: {}".format(log_loss(y_train, xg4_oof_train)))
print("XG5-CV: {}".format(log_loss(y_train, xg5_oof_train)))
print("XG6-CV: {}".format(log_loss(y_train, xg6_oof_train)))



x_train = np.concatenate((xg_oof_train, xg2_oof_train, xg3_oof_train, xg4_oof_train, xg5_oof_train, xg6_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, xg2_oof_test, xg3_oof_test, xg4_oof_test, xg5_oof_test, xg6_oof_test), axis=1)

with open('data4stack/data_level2.pickle', 'wb') as handle:
    pickle.dump([x_train, x_test], handle)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.1,
    'max_depth':2,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=300, nfold=5, seed=SEED,
             early_stopping_rounds=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]


print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

out_df = pd.DataFrame(gbdt.predict(dtest))
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = X_test.listing_id.values
out_df.to_csv('../sub/stacker2_starter7.csv', index=False)