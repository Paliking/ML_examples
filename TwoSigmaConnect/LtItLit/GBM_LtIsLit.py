import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection 
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score



def runGBM(clf, train_X, train_y, val_X):
    clf.fit(train_X, train_y)
    pred_test_y = clf.predict_proba(val_X)
    pred_classes = clf.predict(val_X)
    pred_train_y = clf.predict_proba(train_X)
    return pred_test_y, pred_classes, pred_train_y


df_train = pd.read_csv('../data_prepared/train_ManBild_exp.csv')
df_test = pd.read_csv('../data_prepared/test_ManBild_exp.csv')


predictors = [ i for i in df_train.columns if not i in ['interest_level']]
targetname = 'interest_level'


train_X = df_train[predictors].values
train_y = df_train[targetname].values
test_X = df_test[predictors].values

parameters = {'n_estimators':800, 'learning_rate':0.05, 'min_samples_leaf':2,'max_features':'sqrt', 'subsample':0.8, 'random_state':10,
                'max_depth':6}
clf = GradientBoostingClassifier(**parameters)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, pred_classes, pred_train_y = runGBM(clf, dev_X, dev_y, val_X)
    cv_scores.append(log_loss(val_y, preds))
    print(cv_scores)
    print('accuracy score: {}'.format(accuracy_score(val_y, pred_classes)))
    print('train logloss: {}'.format(log_loss(dev_y, pred_train_y)))
print(sum(cv_scores)/len(cv_scores))

preds, pred_classes, pred_train_y = runGBM(clf, train_X, train_y, test_X)
out_df = pd.DataFrame(preds)
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = df_test.listing_id.values
out_df.to_csv("../sub/gbm_ltislit_MBroom.csv", index=False)

