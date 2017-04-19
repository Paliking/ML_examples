# https://www.kaggle.com/justindeed/d/uciml/iris/iris-with-ensemble-stacking/


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

iris = pd.read_csv("../input/Iris.csv")

# split data

stacking_X_train, stacking_X_test, stacking_y_train, stacking_y_test = train_test_split(iris.ix[:, :'PetalWidthCm'], iris['Species'], random_state=0)

y_map = {'Iris-versicolor': 2, 'Iris-virginica': 1, 'Iris-setosa': 0}
stacking_y_train = stacking_y_train.map(y_map)
stacking_y_test = stacking_y_test.map(y_map)

# Stacking concept as from https://github.com/vecxoz/vecstack
# We want to predict train and test sets with some 1-st level model(s), and then use this predictions as features for 2-nd level model.
# Any model can be used as 1-st level model or 2-nd level model.
# To avoid overfitting (for train set) we use cross-validation technique and in each fold we predict out-of-fold part of train set.
# The common practice is to use from 3 to 10 folds.
# In each fold we predict full test set, so after completion of all folds we need to find mean (mode) of all test set predictions made in each fold.
# As an example we look at stacking implemented with single 1-st level model and 3-fold cross-validation.
# We can repeat this cycle using other 1-st level models to get more features for 2-nd level model.


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        self.classes_= None

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        self.classes_ = self.clf.classes_

    def predict_proba(self, x):
        proba = self.clf.predict_proba(x)
        return proba

    def get_name(self):
        return self.clf.__class__.__name__

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 30)
        self.training_data = None
        self.classes_ = None

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.training_data = dtrain
        self.classes_ = [int(x) for x in list(set(self.training_data.get_label())) if x.dtype == 'float32']
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, verbose_eval=False)

    def predict_proba(self, x):
        proba = self.gbdt.predict(xgb.DMatrix(x))
        return proba

    def get_name(self):
        return 'XGBClassifier'

class EnsembleStacking:
    def __init__(self, X_train, y_train, X_test, base_models, stacker, cv):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._base_models = base_models
        self._stacker = stacker
        self._cv = cv
        self._correlations = pd.DataFrame()

    def get_correlations(self):
        return self._correlations

    #get out-of-fold predictions
    def get_oof(self, clf, X_train_oof, y_train_oof, X_test_oof):

        clf.fit(X_train_oof, y_train_oof)
        preds_train = clf.predict_proba(X_test_oof)
        preds_test = clf.predict_proba(self._X_test)#use test set

        return preds_train, preds_test

    def fit_and_predict_proba(self):

        stacking_predictions_training = pd.DataFrame()
        stacking_predictions_test = pd.DataFrame()
        stacking_predictions_oof_avg = {}

        # in my words, per base model do the following.
        # split the data according to the cv (cross validation) parameter. fit the base model
        # and predict probabilities for the out-of-fold share of the cross validation.
        # take these probabilities for each out of fold prediction, they will sum up to what
        # amounts a full prediction from within the training set. as you predict the out-of-fold
        # samples, use the same fitted model to predict probabilities of the complete test set. once
        # you finished your cross validation (after n times), average the n test set predictions.
        # do these steps for each of the m base models and you receive m predictions of training
        # data (made up from the n-fold cross validation) and m averaged test data predictions.
        # because we calculate probabilities for l classes, each training prediction contains
        # m times l columns, as does the test prediction. this is all you need to train the
        # final model and receive a test prediction with l columns.

        last_feature = ''
        label_len_for_corr = []
        for i, clf in enumerate(self._base_models):
            stacking_predictions_oof_aggr = pd.DataFrame()
            skf = StratifiedKFold(n_splits=self._cv, random_state=0, shuffle=True)
            counter = 0
            for train_index, test_index in skf.split(self._X_train, self._y_train):
                # print("TRAIN:", train_index, "TEST:", test_index)
                if self._X_train.__class__.__name__ == 'DataFrame':
                    X_train_oof, X_test_oof = self._X_train.iloc[train_index], self._X_train.iloc[test_index]
                    y_train_oof, y_test_oof = self._y_train.iloc[train_index], self._y_train.iloc[test_index]
                else:
                    X_train_oof, X_test_oof = self._X_train[train_index], self._X_train[test_index]
                    y_train_oof, y_test_oof = self._y_train[train_index], self._y_train[test_index]

                # collect and append the predictions
                clf = self._base_models[i]
                # print('Fitting model',  '...', clf.__class__.__name__, repr(i))
                preds_train, preds_test = self.get_oof(clf=clf, X_train_oof=X_train_oof, y_train_oof=y_train_oof, X_test_oof=X_test_oof)

                print(log_loss(y_true=y_test_oof, y_pred=preds_train))

                preds_oof_df = pd.DataFrame()
                for j, label in enumerate(clf.classes_):
                    preds_oof_df[clf.get_name() + repr(i) + '_' + repr(label)] = preds_train[:, j]
                    last_feature = clf.get_name() + repr(i) + '_' + repr(label)
                preds_oof_df['index'] = test_index  # keep order of entries with an index
                stacking_predictions_oof_aggr = stacking_predictions_oof_aggr.append(preds_oof_df)

                # since our k-fold stacking predictions will have only three columns, also our test set needs to change
                # save each test set prediction fitted from all training data, later averaged
                stacking_predictions_oof_avg[counter] = pd.DataFrame()
                for j, label in enumerate(clf.classes_):
                    stacking_predictions_oof_avg[counter][clf.get_name() + repr(i) + '_' + repr(label)] = preds_test[:, j]
                counter += 1

            # add predict_proba columns to final prediction features
            stacking_predictions_oof_aggr = stacking_predictions_oof_aggr.sort_values(['index'], ascending=[1])
            stacking_predictions_oof_aggr.drop(['index'], axis=1, inplace=True)
            for j, column in enumerate(stacking_predictions_oof_aggr):
                stacking_predictions_training[column] = stacking_predictions_oof_aggr[column]
            # print('These should be equal in rows:', stacking_X_train.shape, stacking_predictions_kfold.shape)

            # add test set predictions average to final prediction features
            panel = pd.Panel(stacking_predictions_oof_avg)
            df_test_prediction_avg = panel.mean(axis=0)
            # print('These should be equal in rows:', stacking_X_test.shape, df_test_prediction_avg.shape)
            for j, column in enumerate(df_test_prediction_avg):
                stacking_predictions_test[column] = df_test_prediction_avg[column]

        # train a second-layer model on these predictions
        if self._y_train.__class__.__name__ == 'Series':
            stacking_predictions_training['target'] = self._y_train.as_matrix()
        else:
            stacking_predictions_training['target'] = self._y_train

        self._stacker.fit(stacking_predictions_training.ix[:, :last_feature], stacking_predictions_training['target'])
        preds = self._stacker.predict_proba(stacking_predictions_test)
        solution = pd.DataFrame()
        for index, label in enumerate(self._stacker.classes_):
            label_len_for_corr.append(repr(label))
            solution[label] = preds[:, index]

        # calculate correlation matrix
        temp_num_classes = int(len(stacking_predictions_training.columns) / len(self._base_models))
        temp_column_name = []
        temp_corr_mat = {}
        for l in range(0, temp_num_classes):
            temp_corr_df = pd.DataFrame()
            for k, clf in enumerate(self._base_models):
                # iterate over linked columns
                temp_corr_df[self._base_models[k].get_name()] = stacking_predictions_training.ix[:,l + (temp_num_classes * k)]
            # calculate correlation matrix for each label, then average the correlation matrices
            temp_corr_mat[l] = pd.DataFrame(np.corrcoef(temp_corr_df.T))
        for k, clf in enumerate(self._base_models):
            # add name to columns, index
            temp_column_name.append(self._base_models[k].get_name())
        temp_panel = pd.Panel(temp_corr_mat)
        self._correlations = temp_panel.mean(axis=0)
        self._correlations.columns = temp_column_name
        self._correlations.index = temp_column_name

        return solution



# specify the different parameter and models
et_params = {
    'n_jobs': 4,
    'n_estimators': 10,
    'max_features': 0.5,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 4,
    'n_estimators': 10,
    'max_features': 0.2,
    'max_depth': 12,
    #'min_samples_leaf': 2,
}

xgb_params = {
    'objective': 'multi:softprob',
    'eta':0.1,
    'max_depth':6,
    'num_class':3,
    'eval_metric':"mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}

rd_params={
    'alpha': 10
}

lr_params={
    'random_state': 1
}

xg = XgbWrapper(seed=0, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=0, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=0, params=rf_params)
lr = SklearnWrapper(clf=LogisticRegression, seed=0, params=lr_params)

base_models = [xg, et, rf, lr]

final_model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                            gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6,
                            min_child_weight=1, missing=None, n_estimators=310, nthread=-1,
                            objective='multi:softprob', reg_alpha=0, reg_lambda=1,
                            scale_pos_weight=1, seed=0, silent=True, subsample=1)

ensemble = EnsembleStacking(X_train=stacking_X_train, y_train=stacking_y_train, X_test=stacking_X_test,
                            base_models=base_models, stacker=final_model, cv=5)

solution = ensemble.fit_and_predict_proba()


#export results
# solution.to_csv("solution_stacking.csv", index = False)

print(ensemble.get_correlations().head())