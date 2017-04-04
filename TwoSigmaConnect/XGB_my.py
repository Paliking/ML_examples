# https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python
# https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/improve-perfomances-using-manager-features/run/884450

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


data_path = "input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"

train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# concat for valuecounting for manager_id later
ntrain = train_df.shape[0]
train_test = pd.concat((train_df, test_df), axis=0)



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])



train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())

# add manager_skill feature (pozor na leak)
y_train = train_df["interest_level"]
# compute fractions and count for each manager
temp = pd.concat([train_df.manager_id,pd.get_dummies(y_train)], axis = 1).groupby('manager_id').mean()
print(temp.columns)
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = train_df.groupby('manager_id').count().iloc[:,1]

# remember the manager_ids look different because we encoded them in the previous step 
print(temp.tail(10))

# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

# get ixes for unranked managers...
unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
print(temp.tail(10))

# inner join to assign manager features to the managers in the training dataframe
train_df = train_df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
print(train_df.head())

# add the features computed on the training dataset to the test dataset
test_df = test_df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = test_df['high_frac'].isnull()
test_df.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

# add manager fractions and skills to the features to use
# features_to_use.extend(['high_frac','low_frac', 'medium_frac','manager_skill'])


# # features ideas
# # https://www.kaggle.com/visnaga/two-sigma-connect-rental-listing-inquiries/xgboost-for-the-millionth-time-0-54724-lb
# train_df['Zero_building_id'] = train_df['building_id'].apply(lambda x: 1 if x == '0' else 0)
# test_df['Zero_building_id'] = test_df['building_id'].apply(lambda x: 1 if x == '0' else 0)

# # description
# train_df['desc'] = train_df['description']
# train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
# train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))

# string.punctuation.__add__('!!')
# string.punctuation.__add__('(')
# string.punctuation.__add__(')')

# remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
# print('punc map: ', remove_punct_map)

# train_df['desc'] = train_df['desc'].apply(lambda x: x.translate(remove_punct_map))
# train_df['desc_letters_count'] = train_df['description'].apply(lambda x: len(x.strip()))
# train_df['desc_words_count'] = train_df['desc'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

# test_df['desc'] = test_df['description']
# test_df['desc'] = test_df['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
# test_df['desc'] = test_df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))
# test_df['desc'] = test_df['desc'].apply(lambda x: x.translate(remove_punct_map))
# test_df['desc_letters_count'] = test_df['description'].apply(lambda x: len(x.strip()))
# test_df['desc_words_count'] = test_df['desc'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

# features_to_use.extend(['Zero_building_id','desc_letters_count', 'desc_words_count'])


# # adress
# train_df['address1'] = train_df['display_address']
# train_df['address1'] = train_df['address1'].apply(lambda x: x.lower())
# test_df['address1'] = test_df['display_address']
# test_df['address1'] = test_df['address1'].apply(lambda x: x.lower())

# address_map = {
#     'w': 'west',
#     'st.': 'street',
#     'ave': 'avenue',
#     'st': 'street',
#     'e': 'east',
#     'n': 'north',
#     's': 'south'
# }


# def address_map_func(s):
#     s = s.split(' ')
#     out = []
#     for x in s:
#         if x in address_map:
#             out.append(address_map[x])
#         else:
#             out.append(x)
#     return ' '.join(out)


# train_df['address1'] = train_df['address1'].apply(lambda x: x.translate(remove_punct_map))
# train_df['address1'] = train_df['address1'].apply(lambda x: address_map_func(x))
# test_df['address1'] = test_df['address1'].apply(lambda x: x.translate(remove_punct_map))
# test_df['address1'] = test_df['address1'].apply(lambda x: address_map_func(x))

# new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

# for col in new_cols:
#     train_df[col] = train_df['address1'].apply(lambda x: 1 if col in x else 0)
#     test_df[col] = test_df['address1'].apply(lambda x: 1 if col in x else 0)

# train_df['other_address'] = train_df[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
# test_df['other_address'] = test_df[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

# new_features = ['Zero_building_id','desc_letters_count', 'desc_words_count', 'other_address'] + new_cols
# features_to_use.extend(new_features)

# label encode (no one hot, too many values...)
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)


# top percentils

managers_count = train_test['manager_id'].value_counts()

train_test['top_10_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
train_test['top_25_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
train_test['top_5_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
train_test['top_50_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
train_test['top_1_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
train_test['top_2_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
train_test['top_15_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
train_test['top_20_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
train_test['top_30_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 70)] else 0)

buildings_count = train_test['building_id'].value_counts()

train_test['top_10_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
train_test['top_25_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
train_test['top_5_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
train_test['top_50_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
train_test['top_1_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
train_test['top_2_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
train_test['top_15_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
train_test['top_20_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
train_test['top_30_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)


percentil_features = [ i for i in train_test.columns.values if i.startswith('top') ]
print(percentil_features)

print(train_test.shape)
x_train = train_test.iloc[:ntrain, :].reset_index(drop=True)
x_test = train_test.iloc[ntrain:, :].reset_index(drop=True)
print(x_train.shape, x_test.shape)
print(train_df.shape, test_df.shape)

train_df = train_df.merge(x_train[percentil_features], left_index=True, right_index=True)
test_df = test_df.merge(x_test[percentil_features], left_index=True, right_index=True)
print(train_df.shape, test_df.shape)


features_to_use = [ i for i in features_to_use if i not in ['high_frac','low_frac', 'medium_frac', 'manager_id' ] ]
# NLP
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)



cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)


preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("sub/xgb_my6.csv", index=False)