import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold   #For K-fold cross validation
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# from sklearn.svm import SVC
# from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import cross_validation, metrics


def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(Xtrain, ytrain)
    print ("BEST PARAMS", gs.best_params_)
    best = gs.best_estimator_
    return best


def train_best(clf, parameters, indf, featurenames, targetname, standardize=False):
    subdf=indf[featurenames]
    if standardize:
        subdfstd=(subdf - subdf.mean())/subdf.std()
    else:
        subdfstd=subdf
    X=subdfstd.values
    y=indf[targetname].values
    clf = cv_optimize(clf, parameters, X, y)
    clf=clf.fit(X, y)
    training_accuracy = clf.score(X, y)
    # test_accuracy = clf.score(Xtest, ytest)
    print ("Accuracy on training data: %0.2f" % (training_accuracy))
    # print ("Accuracy on test data:     %0.2f" % (test_accuracy))
    return clf


def runGBM(clf, train_X, train_y, val_X):
    clf.fit(train_X, train_y)
    pred_test_y = clf.predict_proba(val_X)
    return pred_test_y


df_train = pd.read_csv('data_prepared/train_python.csv')
df_test = pd.read_csv('data_prepared/test_python.csv')
# --------------------GBM--------------------------------------
# clf = GradientBoostingClassifier()

predictors = [ i for i in df_train.columns if not i in ['interest_level']]
targetname = 'interest_level'

# parameters = {'n_estimators':list(range(5,30,10))}


# clf_best = train_best(clf, parameters, df_train, predictors, targetname, standardize=False)
##modelfit(clf, df_train, predictors, targetname, performCV=True, printFeatureImportance=True, cv_folds=5)


# ideme zistovant number of trees. Ostatne sme zvolili predbezne a intuitivne.
# n_jobs mi na tomto PC funguje len ked je 1

# param_test1 = {'n_estimators':list(range(400,500,50))}
# estimator = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=2016)
# gsearch1 = GridSearchCV(estimator = estimator, param_grid = param_test1,iid=False, cv=5, scoring='neg_log_loss')
# gsearch1.fit(df_train[predictors],df_train[targetname])
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


# najlepsie je min_samples_split=1 a max_depth=6 (alter 1 a 5)
# param_test2 = {'max_depth':list(range(2,7,1)), 'min_samples_split':list(range(1,5,1))}
# estimator = GradientBoostingClassifier(n_estimators= 30, learning_rate=0.1, min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=10)
# gsearch2 = GridSearchCV(estimator = estimator, param_grid = param_test2,n_jobs=1,iid=False, cv=5)
# gsearch2.fit(df_train[predictors],df_train[targetname])
# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

## najlepsie je min_samples_split=1 a min_samples_leaf=50
# param_test3 = {'min_samples_split':list(range(1,4,1)), 'min_samples_leaf':list(range(30,71,10))}
# estimator = GradientBoostingClassifier(n_estimators= 30, learning_rate=0.1,max_depth=3,max_features='sqrt',subsample=0.8,random_state=10)
# gsearch3 = GridSearchCV(estimator = estimator, param_grid = param_test3,n_jobs=1,iid=False, cv=5)
# gsearch3.fit(df_train[predictors],df_train[targetname])
# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)


# max_features=3 (alter. 5)
# param_test4 = {'max_features':list(range(2,8,1))}
# estimator = GradientBoostingClassifier(n_estimators= 30, learning_rate=0.1, min_samples_split=1, max_depth=3,min_samples_leaf=50,subsample=0.8,random_state=10)
# gsearch4 = GridSearchCV(estimator = estimator, param_grid = param_test4,n_jobs=1,iid=False, cv=5)
# gsearch4.fit(df_train[predictors],df_train[targetname])
# print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)


# subsample = 0.9
#param_test5 = {'subsample':[ 0.8, 0.85, 0.9, 0.95, 0.1]}
#estimator = GradientBoostingClassifier(n_estimators= 80, learning_rate=0.1, min_samples_split=1, max_depth=6,min_samples_leaf=50,random_state=10)
#gsearch5 = GridSearchCV(estimator = estimator, param_grid = param_test5,n_jobs=1,iid=False, cv=5)
#gsearch5.fit(df_train[predictors],df_train[targetname])
#print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

##gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=160,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
##modelfit(gbm_tuned_1, train, predictors)




# # estimator = GradientBoostingClassifier(n_estimators= 30, learning_rate=0.1, min_samples_split=1, max_depth=3,min_samples_leaf=50, max_features=4,subsample=0.8, random_state=10)
# estimator = GradientBoostingClassifier(n_estimators= 100, learning_rate=0.1, min_samples_split=1, max_depth=5,min_samples_leaf=50, max_features='sqrt',subsample=0.8, random_state=10)

# estimator.fit(df_train[predictors], df_train[targetname])
# df_test['target'] = estimator.predict(df_test[predictors])

# modelfit(estimator, df_train, predictors, targetname, performCV=True, printFeatureImportance=True, cv_folds=5)


# ------neyuziva sa--------------
##Predict on testing data:
##df_test['target'] = clf_best.predict(df_test[predictors])
# ------neyuziva sa--------------




# df_test['target'] = df_test['target'].apply(lambda x: 'Y' if x==1 else 'N')

# submission = pd.DataFrame()
# submission['Loan_ID'] = df_test['Loan_ID']
# submission['Loan_Status'] = df_test['target']
# submission.to_csv('submission_GBM.csv', index=False)



from sklearn import model_selection 
from sklearn.metrics import log_loss

train_X = df_train[predictors].values
train_y = df_train[targetname].values
test_X = df_test[predictors].values

parameters = {'n_estimators':800, 'learning_rate':0.05, 'min_samples_leaf':50,'max_features':'sqrt', 'subsample':0.8, 'random_state':10,
                'max_depth':6}
clf = GradientBoostingClassifier(**parameters)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds = runGBM(clf, dev_X, dev_y, val_X)
    cv_scores.append(log_loss(val_y, preds))
    print(cv_scores)
print(sum(cv_scores)/len(cv_scores))

# preds = runGBM(clf, train_X, train_y, test_X)
# out_df = pd.DataFrame(preds)
# out_df.columns = ["low", "medium", "high"]
# out_df["listing_id"] = df_test.listing_id.values
# out_df.to_csv("sub/gbm_ltislit.csv", index=False)