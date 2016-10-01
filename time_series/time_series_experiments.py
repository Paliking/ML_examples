# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 21:10:09 2016

@author: Pablo


zdroj: https://github.com/amueller/introduction_to_ml_with_python/blob/master/04-representing-data-feature-engineering.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor




# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    # split the given features into a training and test set
    X_train, X_test = features[:n_train], features[n_train:]
    # split also the 
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

#    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
#               ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
             label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()
    
    
    
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
ts = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse)

plt.plot(ts)
plt.show()

ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()

# use the first n data points for training, the rest for testing
n_train = 110
# extract the target values (number of rentals)
y = ts.values
X_hour = ts.index.hour.reshape(-1, 1)


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X_hour, y, regressor)


X_hour_week = np.hstack([ts.index.dayofweek.reshape(-1, 1),
                         ts.index.hour.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)

X_month = ts.index.month.reshape(-1, 1)
eval_on_features(X_month, y, regressor)

X_month_year = np.hstack([ts.index.month.reshape(-1, 1),
                         ts.index.year.reshape(-1, 1)])
eval_on_features(X_month_year, y, regressor)


y = ts_log.values
X_hour = ts_log.index.hour.reshape(-1, 1)
X_month_year = np.hstack([ts_log.index.month.reshape(-1, 1),
                         ts_log.index.year.reshape(-1, 1)])
eval_on_features(X_month_year, y, regressor)

# test na detrendnutych datach

model = np.polyfit(np.arange(len(ts)), ts.values, 1)
predicted = np.polyval(model, np.arange(len(ts)))
ts_detrend = ts['#Passengers'] - predicted
eval_on_features(X_month_year, ts_detrend.values, regressor)


model = np.polyfit(np.arange(len(ts_log)), ts_log.values, 1)
predicted = np.polyval(model, np.arange(len(ts_log)))
ts_detrend = ts_log['#Passengers'] - predicted
eval_on_features(X_month_year, ts_detrend.values, regressor)