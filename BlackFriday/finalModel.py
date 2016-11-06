import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import os


gender_dict = {'F':0, 'M':1}
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
city_dict = {'A':0, 'B':1, 'C':2}
stay_dict = {'0':0, '1':1, '2':2, '3':3, '4+':4}





# path with data
parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
data_path = os.path.join(parent_parent, os.path.join('data_examples', 'BlackFriday'))
train_file = os.path.join(data_path, "train_mod.csv")
test_file = os.path.join(data_path, "test_mod.csv")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print('train and test shapes', train_df.shape, test_df.shape)




train_y = np.array(train_df["Purchase"])
test_user_id = np.array(test_df["User_ID"])
test_product_id = np.array(test_df["Product_ID"])

train_df.drop(["Purchase"], axis=1, inplace=True)

cat_columns_list = ["User_ID", "Product_ID"]
for var in cat_columns_list:
        lb = LabelEncoder()
        full_var_data = pd.concat((train_df[var],test_df[var]),axis=0).astype('str')
        temp = lb.fit_transform(np.array(full_var_data))
        train_df[var] = lb.transform(np.array( train_df[var] ).astype('str'))
        test_df[var] = lb.transform(np.array( test_df[var] ).astype('str'))

train_X = np.array(train_df).astype('float')
test_X = np.array(test_df).astype('float')
print (train_X.shape, test_X.shape)

print('train', train_df)
print('test', test_df)

print ("Running model..")
params = {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2,
          'learning_rate': 0.1, 'subsample':0.8}
regressor = GradientBoostingRegressor(random_state=0, **params)
regressor.fit(train_X, train_y)
pred_test_y = regressor.predict(test_X)
pred_test_y[pred_test_y<0] = 1

out_df = pd.DataFrame({"User_ID":test_user_id})
out_df["Product_ID"] = test_product_id
out_df["Purchase"] = pred_test_y
out_df.to_csv("sub20.csv", index=False)