# https://www.kaggle.com/reynaldo/naive-xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()


def make_CV(x_train, y_train, x_test, xgb_params, mode='timesplit'):
    '''
    Make choosen CV type.
    Returns best number of rounds of XGB model and dtrain, dtest as xgb.DMatrix

    inputs
    x_train, y_train, x_test: data are dfs
    mode: str; CV type. 'kfold' - random k-fold splits
                        'timesplit' - last part of training set used as validation
    '''

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    # k-fold CV
    if mode == 'kfold':
        cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
            verbose_eval=10, show_stdv=True, nfold=5)
        # cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
        # plt.show()
        num_boost_rounds = np.argmin(cv_output['test-rmse-mean'])
        # num_boost_rounds = len(cv_output)
    # time split CV
    elif mode == 'timesplit':
        # Create a validation set, with last 20% of data
        num_val = int(len(x_train) * 0.2)

        X_train_CV = x_train[:-num_val]
        X_val_CV = x_train[-num_val:]
        y_train_CV = y_train[:-num_val]
        y_val_CV = y_train[-num_val:]

        dtrain_CV = xgb.DMatrix(X_train_CV, y_train_CV, feature_names=x_train.columns)
        dval_CV = xgb.DMatrix(X_val_CV, y_val_CV, feature_names=x_train.columns)

        partial_model = xgb.train(xgb_params, dtrain_CV, num_boost_round=1000, evals=[(dval_CV, 'val')],
                               early_stopping_rounds=10, verbose_eval=20)

        num_boost_rounds = partial_model.best_iteration

    return num_boost_rounds, dtrain, dtest




# metric is RMSLE. We can use RMSE in XGB model, but target value is "loged" first
use_log = True



train = pd.read_csv('../inputs/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../inputs/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('../inputs/macro.csv', parse_dates=['timestamp'])
id_test = test.id


# sposob ako vyrovnat score z CV a z LB
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717
# undersample trainset 10, 3 and 2 times based on price_doc and product_type
# and exclude times after 1.1.2015
trainsub = train[train.timestamp < '2015-01-01']
trainsub = trainsub[trainsub.product_type=="Investment"]

ind_1m = trainsub[trainsub.price_doc <= 1000000].index
ind_2m = trainsub[trainsub.price_doc == 2000000].index
ind_3m = trainsub[trainsub.price_doc == 3000000].index

train_index = set(train.index.copy())

for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
    ind_set = set(ind)
    ind_set_cut = ind.difference(set(ind[::gap]))

    train_index = train_index.difference(ind_set_cut)

train = train.loc[train_index]




# # other option is to drop all <=1M investment rows (look worse ?)
# mask_excl = (train['product_type'] == 'Investment') & (train['price_doc'] <= 1000000)
# train = train[~mask_excl]






if use_log:
    y_train = np.log1p(train['price_doc'].values)
else:
    y_train = train["price_doc"]

# plt.plot(train["timestamp"], train["price_doc"].rolling(40).mean())
# plt.show()

train = train.drop(["price_doc"], axis=1)

# -----------------------FE-------------------------
df_all = pd.concat([train, test])

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month-year count
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add tiem features
df_all['month'] = df_all.timestamp.dt.month
df_all["week_of_year"] = df_all["timestamp"].dt.weekofyear
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
# df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
# df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
# df_all['balcony'] = df_all['full_sq'] - df_all['life_sq']
df_all['room_sq'] = (df_all['life_sq'] - df_all['kitch_sq']) / df_all['num_room']

# encode ecology with logical order
ecology_map = {'poor': 1, 'satisfactory': 2, 'good': 3, 'excellent': 4, 'no data': np.NaN}
df_all['ecology'] = df_all['ecology'].apply(lambda x: ecology_map[x])

# split data back to train and test
train = df_all[:len(train)]
test = df_all[len(train):]


x_train = train.drop(["id", "timestamp"], axis=1)
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



num_boost_rounds, dtrain, dtest = make_CV(x_train, y_train, x_test, xgb_params, mode='kfold')
# num_boost_rounds, dtrain, dtest = make_CV(x_train, y_train, x_test, xgb_params, mode='timesplit')
model = xgb.train(xgb_params, dtrain, num_boost_round= num_boost_rounds)






# fig, ax = plt.subplots(1, 1, figsize=(8, 13))
# xgb.plot_importance(model, height=0.5, ax=ax)
# plt.show()

y_predict = model.predict(dtest)

if use_log:
    y_pred = np.exp(y_predict) - 1
else:
    y_pred = y_predict

output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

# output.to_csv('../subs/xgb_log_1315.csv', index=False)