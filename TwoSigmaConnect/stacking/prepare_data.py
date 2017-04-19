import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import KFold
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random
from math import exp
import datetime as dt
import pickle
print(check_output(["ls", "../input"]).decode("utf8"))



X_train = pd.read_json("../input/train.json")
X_test = pd.read_json("../input/test.json")

interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])
X_test['interest_level'] = -1

X_train['price'].ix[X_train['price']>13000] = 13000

#add features
feature_transform = CountVectorizer(stop_words='english', max_features=150)
X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
feature_transform.fit(list(X_train['features']) + list(X_test['features']))

train_size = len(X_train)
low_count = len(X_train[X_train['interest_level'] == 0])
medium_count = len(X_train[X_train['interest_level'] == 1])
high_count = len(X_train[X_train['interest_level'] == 2])

def find_objects_with_only_one_record(feature_name):
    temp = pd.concat([X_train[feature_name].reset_index(), 
                      X_test[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]

managers_with_one_lot = find_objects_with_only_one_record('manager_id')
buildings_with_one_lot = find_objects_with_only_one_record('building_id')
addresses_with_one_lot = find_objects_with_only_one_record('display_address')

lambda_val = None
k=5.0
f=1.0
r_k=0.01 
g = 1.0

def categorical_average(variable, y, pred_0, feature_name):
    def calculate_average(sub1, sub2):
        s = pd.DataFrame(data = {
                                 variable: sub1.groupby(variable, as_index = False).count()[variable],                              
                                 'sumy': sub1.groupby(variable, as_index = False).sum()['y'],
                                 'avgY': sub1.groupby(variable, as_index = False).mean()['y'],
                                 'cnt': sub1.groupby(variable, as_index = False).count()['y']
                                 })
                                 
        tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable) 
        del tmp['index']                       
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + exp((cnt - k) / f))
            
        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis = 1)
            
        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],
                                   axis = 1)
                                   
        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
        tmp['random'] = np.random.uniform(size = len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] *(1 + (row['random'] - 0.5) * r_k),
                                   axis = 1)
    
        return tmp['adj_avg'].ravel()
     
    #cv for training set 
    k_fold = StratifiedKFold(5)
    X_train[feature_name] = -999 
    for (train_index, cv_index) in k_fold.split(np.zeros(len(X_train)),
                                                X_train['interest_level'].ravel()):
        sub = pd.DataFrame(data = {variable: X_train[variable],
                                   'y': X_train[y],
                                   'pred_0': X_train[pred_0]})
            
        sub1 = sub.iloc[train_index]        
        sub2 = sub.iloc[cv_index]
        
        X_train.loc[cv_index, feature_name] = calculate_average(sub1, sub2)
    
    #for test set
    sub1 = pd.DataFrame(data = {variable: X_train[variable],
                                'y': X_train[y],
                                'pred_0': X_train[pred_0]})
    sub2 = pd.DataFrame(data = {variable: X_test[variable],
                                'y': X_test[y],
                                'pred_0': X_test[pred_0]})
    X_test.loc[:, feature_name] = calculate_average(sub1, sub2)                               

def transform_data(X):
    #add features    
    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([ pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(), X1.reset_index()], axis = 1)
    del X['index']
    
    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed'] = X['price'] / X['bedrooms']    
    X['price_per_bath'] = X['price'] / X['bathrooms']
    X['price_per_room'] = X['price'] / (X['bathrooms'] + X['bedrooms'] )
    X["bedPerBath"] = X['bedrooms'] / X['bathrooms']
    X["bedBathDiff"] = X['bedrooms'] - X['bathrooms']
    X["bedBathSum"] = X["bedrooms"] + X['bathrooms']
    X["bedsPerc"] = X["bedrooms"] / (X['bedrooms'] + X['bathrooms'])

    # added from other script
    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    X['created_hour'] = X["created"].dt.hour
    X['created_weekday'] = X['created'].dt.weekday
    X['created_week'] = X['created'].dt.week
    X['created_quarter'] = X['created'].dt.quarter
    X['created_weekend'] = ((X['created_weekday'] == 5) & (X['created_weekday'] == 6))
    X['created_wd'] = ((X['created_weekday'] != 5) & (X['created_weekday'] != 6))
    X['created'] = X['created'].map(lambda x: float((x - dt.datetime(1899, 12, 30)).days) + (float((x - dt.datetime(1899, 12, 30)).seconds) / 86400))
    
    X['low'] = 0
    X.loc[X['interest_level'] == 0, 'low'] = 1
    X['medium'] = 0
    X.loc[X['interest_level'] == 1, 'medium'] = 1
    X['high'] = 0
    X.loc[X['interest_level'] == 2, 'high'] = 1
    
    X['display_address'] = X['display_address'].apply(lambda x: x.lower().strip())
    X['street_address'] = X['street_address'].apply(lambda x: x.lower().strip())
    
    X['pred0_low'] = low_count * 1.0 / train_size
    X['pred0_medium'] = medium_count * 1.0 / train_size
    X['pred0_high'] = high_count * 1.0 / train_size
    
    X.loc[X['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()), 
          'manager_id'] = "-1"
    X.loc[X['building_id'].isin(buildings_with_one_lot['building_id'].ravel()), 
          'building_id'] = "-1"
    X.loc[X['display_address'].isin(addresses_with_one_lot['display_address'].ravel()), 
          'display_address'] = "-1"
          
    return X

def normalize_high_cordiality_data():
    high_cardinality = ["building_id", "manager_id"]
    for c in high_cardinality:
        categorical_average(c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(c, "high", "pred0_high", c + "_mean_high")

def transform_categorical_data():
    categorical = ['building_id', 'manager_id', 
                   'display_address', 'street_address']
                   
    for f in categorical:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[f]) + list(X_test[f])) 
        X_train[f] = encoder.transform(X_train[f].ravel())
        X_test[f] = encoder.transform(X_test[f].ravel())
                  

def remove_columns(X):
    columns = ["photos", "pred0_high", "pred0_low", "pred0_medium",
               "description", "low", "medium", "high",
               "interest_level", "created"]
    for c in columns:
        del X[c]

def add_manager_level_weaker_leakage(train_df, test_df):
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)

    for i in range(5):
        building_level={}
        for j in train_df['manager_id'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level']==0:
                building_level[temp['manager_id']][0]+=1
            if temp['interest_level']==1:
                building_level[temp['manager_id']][1]+=1
            if temp['interest_level']==2:
                building_level[temp['manager_id']][2]+=1
        for j in test_index:
            temp=train_df.iloc[j]
            if sum(building_level[temp['manager_id']])!=0:
                a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
                b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
                c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
    train_df['manager_level_low']=a
    train_df['manager_level_medium']=b
    train_df['manager_level_high']=c


    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']==0:
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']==2:
            building_level[temp['manager_id']][2]+=1

    for i in test_df['manager_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['manager_level_low']=a
    test_df['manager_level_medium']=b
    test_df['manager_level_high']=c
    return train_df, test_df


def add_builing_level_weaker_leakage(train_df, test_df):
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)

    for i in range(5):
        building_level={}
        for j in train_df['building_id'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level']==0:
                building_level[temp['building_id']][0]+=1
            if temp['interest_level']==1:
                building_level[temp['building_id']][1]+=1
            if temp['interest_level']==2:
                building_level[temp['building_id']][2]+=1
        for j in test_index:
            temp=train_df.iloc[j]
            if sum(building_level[temp['building_id']])!=0:
                a[j]=building_level[temp['building_id']][0]*1.0/sum(building_level[temp['building_id']])
                b[j]=building_level[temp['building_id']][1]*1.0/sum(building_level[temp['building_id']])
                c[j]=building_level[temp['building_id']][2]*1.0/sum(building_level[temp['building_id']])
    train_df['building_level_low']=a
    train_df['building_level_medium']=b
    train_df['building_level_high']=c


    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df['building_id'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']==0:
            building_level[temp['building_id']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['building_id']][1]+=1
        if temp['interest_level']==2:
            building_level[temp['building_id']][2]+=1

    for i in test_df['building_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['building_level_low']=a
    test_df['building_level_medium']=b
    test_df['building_level_high']=c
    return train_df, test_df


def add_adress_level_weaker_leakage(train_df, test_df):
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)

    for i in range(5):
        building_level={}
        for j in train_df['display_address'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level']==0:
                building_level[temp['display_address']][0]+=1
            if temp['interest_level']==1:
                building_level[temp['display_address']][1]+=1
            if temp['interest_level']==2:
                building_level[temp['display_address']][2]+=1
        for j in test_index:
            temp=train_df.iloc[j]
            if sum(building_level[temp['display_address']])!=0:
                a[j]=building_level[temp['display_address']][0]*1.0/sum(building_level[temp['display_address']])
                b[j]=building_level[temp['display_address']][1]*1.0/sum(building_level[temp['display_address']])
                c[j]=building_level[temp['display_address']][2]*1.0/sum(building_level[temp['display_address']])
    train_df['display_address_level_low']=a
    train_df['display_address_level_medium']=b
    train_df['display_address_level_high']=c


    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df['display_address'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']==0:
            building_level[temp['display_address']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['display_address']][1]+=1
        if temp['interest_level']==2:
            building_level[temp['display_address']][2]+=1

    for i in test_df['display_address'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['display_address_level_low']=a
    test_df['display_address_level_medium']=b
    test_df['display_address_level_high']=c
    return train_df, test_df


def add_street_adress_level_weaker_leakage(train_df, test_df):
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)

    for i in range(5):
        building_level={}
        for j in train_df['street_address'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level']==0:
                building_level[temp['street_address']][0]+=1
            if temp['interest_level']==1:
                building_level[temp['street_address']][1]+=1
            if temp['interest_level']==2:
                building_level[temp['street_address']][2]+=1
        for j in test_index:
            temp=train_df.iloc[j]
            if sum(building_level[temp['street_address']])!=0:
                a[j]=building_level[temp['street_address']][0]*1.0/sum(building_level[temp['street_address']])
                b[j]=building_level[temp['street_address']][1]*1.0/sum(building_level[temp['street_address']])
                c[j]=building_level[temp['street_address']][2]*1.0/sum(building_level[temp['street_address']])
    train_df['street_address_level_low']=a
    train_df['street_address_level_medium']=b
    train_df['street_address_level_high']=c


    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df['street_address'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']==0:
            building_level[temp['street_address']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['street_address']][1]+=1
        if temp['interest_level']==2:
            building_level[temp['street_address']][2]+=1

    for i in test_df['street_address'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['street_address_level_low']=a
    test_df['street_address_level_medium']=b
    test_df['street_address_level_high']=c
    return train_df, test_df


def add_stats_for_manager(variable, train_df, test_df):
    '''
    Groupby manager_id and calculate 'sum', 'mean', 'count', 'median' of selected variable.
    '''
    train = train_df.copy()
    train['source'] = 'train'
    test = test_df.copy()
    test['source'] = 'test'
    df = pd.concat([train, test])
    grouped = df.groupby('manager_id')[variable]
    # ind2exclude = grouped.filter(lambda x: len(x) < excl_shorter).index
    # train_df.loc[ind2exclude, new_features] = np.nan
    functions = ['sum', 'mean', 'count', 'median']
    for function in functions:
        col_name = 'man_{}_{}'.format(variable, function)
        df[col_name] = grouped.transform(function)

    new_features = [ 'man_{}_{}'.format(variable, j) for j in functions]
    train_df[new_features] = df[df['source'] == 'train'][new_features]
    test_df[new_features] = df[df['source'] == 'test'][new_features]
    return train_df, test_df


def merge_same_info(df, dic, exclude_cols):
    ''' Marge the smae columns with different names based on dic'''
    for new_feature, old_features in dic.items():
        merged_column = df[old_features].sum(axis=1)
        merged_column[merged_column > 1] = 1
        df.drop(old_features, axis=1, inplace=True)
        df[new_feature] = merged_column
    df.drop(exclude_cols, axis=1, inplace=True)
    return df


encoder = {'dishwasher': ['dishwasher', '_dishwasher_'],
            'dryer': ['dryer_in_building', 'dryer_in_unit', 'dryer'],
            'fitness': ['fitness', 'fitness_center', 'gym_in_building', 'gym'],
            'doorman': ['ft_doorman', 'doorman'],
            'garage': ['garage', 'full_service_garage', 'site_garage'],
            'high_ceiling': ['high_ceiling', 'high_ceilings'],
            'highrise': ['highrise', 'hi_rise'],
            'laundry_room': ['laundry_in_unit', 'laundry_room', 'laundry_in_building', 'site_laundry'],
            'lounge': ['lounge', 'lounge_room', 'residents_lounge'],
            'outdoor': ['outdoor', 'outdoor_areas', 'outdoor_entertainment_space', 'outdoor_space'],
            'parking': ['parking', 'parking_space', 'site_parking', 'site_parking_lot'],
            'pets_ok': ['_pets_ok_', 'pet_friendly', 'dogs_allowed', 'pets_on_approval'],
            'post': ['post', 'post_war'], 
            'roof_deck': ['roofdeck', 'roof_deck'],
            'swimming_pool': ['swimming_pool', 'pool'],
            'washer': ['washer', 'washer_', 'washer_in_unit'],
            'wheelchair_access': ['wheelchair_access', 'wheelchair_ramp']}

exclude_cols = ['multi', 'pre']


def add_percentils(train_df, test_df):
    train = train_df.copy()
    train['source'] = 'train'
    test = test_df.copy()
    test['source'] = 'test'
    df = pd.concat([train, test])

    managers_count = df['manager_id'].value_counts()

    df['top_10_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
    df['top_25_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
    df['top_5_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
    df['top_50_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
    df['top_1_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
    df['top_2_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
    df['top_15_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
    df['top_20_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
    df['top_30_manager'] = df['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
        managers_count.values >= np.percentile(managers_count.values, 70)] else 0)

    buildings_count = df['building_id'].value_counts()

    df['top_10_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
    df['top_25_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
    df['top_5_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
    df['top_50_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
    df['top_1_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
    df['top_2_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
    df['top_15_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
    df['top_20_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
    df['top_30_building'] = df['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
        buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)

    percentil_features = [ i for i in df.columns.values if i.startswith('top') ]
    train_df[percentil_features] = df[df['source'] == 'train'][percentil_features]
    test_df[percentil_features] = df[df['source'] == 'test'][percentil_features]
    return train_df, test_df




print("Starting transformations")

X_train, X_test = add_percentils(X_train, X_test)


X_train = transform_data(X_train)    
X_test = transform_data(X_test) 
y = X_train['interest_level'].ravel()

print("Normalizing high cordiality data...")
normalize_high_cordiality_data()
transform_categorical_data()

X_train, X_test = add_manager_level_weaker_leakage(X_train, X_test)
X_train, X_test = add_builing_level_weaker_leakage(X_train, X_test)
X_train, X_test = add_adress_level_weaker_leakage(X_train, X_test)
X_train, X_test = add_street_adress_level_weaker_leakage(X_train, X_test)
X_train, X_test = add_stats_for_manager('price', X_train, X_test)
X_train, X_test = add_stats_for_manager('bedrooms', X_train, X_test)
X_train, X_test = add_stats_for_manager('bathrooms', X_train, X_test)
X_train, X_test = add_stats_for_manager('price_per_room', X_train, X_test)
X_train, X_test = add_stats_for_manager('bedBathSum', X_train, X_test)


# X_train = merge_same_info(X_train, encoder, exclude_cols)
# X_test = merge_same_info(X_test, encoder, exclude_cols)

remove_columns(X_train)
remove_columns(X_test)



# train_X = X_train.values
# train_y = y
# test_X = X_test.values

# for other models than xgb
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)




with open('data4stack/data_perpared_all2.pickle', 'wb') as handle:
    pickle.dump([X_train, y, X_test], handle)