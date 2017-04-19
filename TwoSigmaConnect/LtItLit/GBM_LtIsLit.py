import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection 
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score



def runGBM(clf, train_X, train_y, val_X, plot_fi=False, feature_names=None):
    clf.fit(train_X, train_y)
    pred_test_y = clf.predict_proba(val_X)
    pred_classes = clf.predict(val_X)
    pred_train_y = clf.predict_proba(train_X)
    if plot_fi:
        coefs = pd.Series(clf.feature_importances_, feature_names).sort_values(ascending=False)
        coefs.plot(kind='bar', title='Feature Importances')
        # plt.xticks(rotation=70)
        plt.show()
    return pred_test_y, pred_classes, pred_train_y


def merge_same_info(df, dic):
    ''' Marge the smae columns with different names based on dic'''
    for new_feature, old_features in dic:
        merged_column = df[old_features].sum(axis=1)
        merged_column[merged_column > 1] = 1
        del df.old_features
        df[new_feature] = merged_column
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


df_train = pd.read_csv('../data_prepared/train_ManStatsList.csv')
df_test = pd.read_csv('../data_prepared/test_ManStatsList.csv')


predictors = [ i for i in df_train.columns if not i in ['interest_level']]
targetname = 'interest_level'


train_X = df_train[predictors].values
train_y = df_train[targetname].values
test_X = df_test[predictors].values

parameters = {'n_estimators':800, 'learning_rate':0.1, 'min_samples_leaf':2,'max_features':'sqrt', 'subsample':0.7, 'random_state':10,
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

preds, pred_classes, pred_train_y = runGBM(clf, train_X, train_y, test_X, plot_fi=True, feature_names=predictors)
out_df = pd.DataFrame(preds)
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = df_test.listing_id.values
# out_df.to_csv("../sub/gbm_ltislit_MBroom.csv", index=False)

