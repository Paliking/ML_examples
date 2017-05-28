# https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]


df_train = pd.read_csv("../inputs/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../inputs/test.csv", parse_dates=['timestamp'])


y_train = df_train["price_doc"]
x_train = df_train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = df_test.drop(["id", "timestamp"], axis=1)


# for c in x_train.columns:
#     if x_train[c].dtype == 'object':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(x_train[c].values) + list(x_test[c].values)) 
#         x_train[c] = lbl.transform(list(x_train[c].values))
#         x_test[c] = lbl.transform(list(x_test[c].values))

df_all = pd.concat([x_train, x_test])

# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

x_train = df_values[:len(x_train)]
x_test = df_values[len(x_train):]

# df_macro = pd.read_csv("../inputs/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
# ylog_train_all = np.log1p(df_train['price_doc'].values)
# try directly price
ylog_train_all = df_train['price_doc'].values

id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

# # Add month-year
# month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
# month_year_cnt_map = month_year.value_counts().to_dict()
# df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# # Add week-year count
# week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
# week_year_cnt_map = week_year.value_counts().to_dict()
# df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# # Add month and day-of-week
# df_all['month'] = df_all.timestamp.dt.month
# df_all['dow'] = df_all.timestamp.dt.dayofweek

# # Other feature engineering
# df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
# df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

# Create a validation set, with last 20% of data
num_val = int(num_train * 0.2)

X_train_all = X_all[:num_train]
X_train = X_all[:num_train-num_val]
X_val = X_all[num_train-num_val:num_train]
ylog_train = ylog_train_all[:-num_val]
ylog_val = ylog_train_all[-num_val:]

X_test = X_all[num_train:]

df_columns = df_values.columns

print('X_train_all shape is', X_train_all.shape)
print('X_train shape is', X_train.shape)
print('y_train shape is', ylog_train.shape)
print('X_val shape is', X_val.shape)
print('y_val shape is', ylog_val.shape)
print('X_test shape is', X_test.shape)
# df_values[:num_train].to_csv('../subs/naive1.csv', index=True)


# test
ddtrain = xgb.DMatrix(x_train, y_train)
ddtest = xgb.DMatrix(x_test)




dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


# test
model = xgb.train(xgb_params, ddtrain, num_boost_round= 384)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, height=0.5, ax=ax)
plt.show()

y_predict = model.predict(ddtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('../subs/sub_changed.csv', index=False)








# # Uncomment to tune XGB `num_boost_rounds`
# partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
#                        early_stopping_rounds=20, verbose_eval=20)

# num_boost_round = partial_model.best_iteration





# fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(partial_model, height=0.5, ax=ax)
# plt.show()


# model = xgb.train(xgb_params, dtrain_all, num_boost_round=384)


# fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(model, height=0.5, ax=ax)
# plt.show()

# ylog_pred = model.predict(dtest)
# # y_pred = np.exp(ylog_pred) - 1
# # change based on task (log or not)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# y_pred = ylog_pred

# df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

# df_sub.to_csv('../subs/sub3.csv', index=False)