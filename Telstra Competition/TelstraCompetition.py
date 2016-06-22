# zdroj: http://www.analyticsvidhya.com/blog/2016/03/complete-solution-top-11-telstra-network-disruptions-kaggle-competition/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

# je tu len priprava dat a feature engineering. prediktivny model nie (v priklade je XGB)

import pandas as pd
import numpy as np
from scipy.stats.mstats import mode
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv('data/train.csv') 
test = pd.read_csv('data/test.csv')
event_type = pd.read_csv('data/event_type.csv')
log_feature = pd.read_csv('data/log_feature.csv')
resource_type = pd.read_csv('data/resource_type.csv')
severity_type = pd.read_csv('data/severity_type.csv')



# spojenie train a test setu, nech robim na nich rovnake upravy
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test], ignore_index=True)

# nerovnomerne rozdelenie
print(data['fault_severity'].value_counts())




# priprava dat
#-----------------event type---------------

event_type = event_type.merge(data, on='id')
event_type_unq = pd.DataFrame(event_type['event_type'].value_counts())

#Determine % of training samples: (percento elementu v training sete = tr/(tr+te)) - pomer daneho elementu v training a testing sete
event_type_unq['PercTrain'] = event_type.pivot_table(values='source',index='event_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))


#Determine the mode of each:
event_type_unq['Mode_Severity'] = event_type.loc[event_type['source']=='train'].pivot_table(values='fault_severity',
                                                                                            index='event_type', aggfunc=lambda x: mode(x).mode[0])
#define action for each:
#initialize:
event_type_unq['preprocess'] = event_type_unq.index.values

#remove the ones not present in train:
##event_type_unq.loc[event_type_unq['PercTrain']==0, 'preprocess'] = 'Remove'

#replace the lower ones with mode:
top_unchange = 33
event_type_unq['preprocess'].iloc[top_unchange:] = event_type_unq['Mode_Severity'].iloc[top_unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'event_type others_%d'%int(x))

#Merge preprocess into original and then into train:
event_type = event_type.merge(event_type_unq[['preprocess']], left_on='event_type',right_index=True)
print(event_type['preprocess'].value_counts())

event_type_merge = event_type.pivot_table(values='event_type',index='id',columns='preprocess',aggfunc=lambda x: len(x), fill_value=0)


data = data.merge(event_type_merge, left_on='id', right_index=True)




#------------ log features--------------

log_feature = log_feature.merge(data[['id','fault_severity','source']], on='id')
log_feature_unq = pd.DataFrame(log_feature['log_feature'].value_counts())

#Determine % of training samples:
log_feature_unq['PercTrain'] = log_feature.pivot_table(values='source',index='log_feature',aggfunc=lambda x: sum(x=='train')/float(len(x)))

#Determine the mode of each:
log_feature_unq['Mode_Severity'] = log_feature.loc[log_feature['source']=='train'].pivot_table(values='fault_severity',index='log_feature', aggfunc=lambda x: mode(x).mode[0])

#define action for each:
#initialize:
log_feature_unq['preprocess'] = log_feature_unq.index.values

#remove the ones all in train
log_feature_unq['preprocess'].loc[log_feature_unq['PercTrain']==1] = np.nan


#replace the lower ones with mode:
top_unchange = 128
log_feature_unq['preprocess'].iloc[top_unchange:] = log_feature_unq['Mode_Severity'].iloc[top_unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'feature others_%d'%int(x))


#Merge preprocess into original and then into train:
log_feature = log_feature.merge(log_feature_unq[['preprocess']], left_on='log_feature',right_index=True)

log_feature_merge = log_feature.pivot_table(values='volume',index='id',columns='preprocess',aggfunc=np.sum, fill_value=0)

data = data.merge(log_feature_merge, left_on='id', right_index=True)






#--------Resource Type-------------------

resource_type = resource_type.merge(data[['id','fault_severity','source']], on='id')
resource_type_unq = pd.DataFrame(resource_type['resource_type'].value_counts())

#Determine % of training samples:
resource_type_unq['PercTrain'] = resource_type.pivot_table(values='source',index='resource_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))
resource_type_unq.head()
#Determine the mode of each:
resource_type_unq['Mode_Severity'] = resource_type.loc[resource_type['source']=='train'].pivot_table(values='fault_severity',index='resource_type', aggfunc=lambda x: mode(x).mode[0])


resource_type_merge = resource_type.pivot_table(values='source',index='id',columns='resource_type',aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(resource_type_merge, left_on='id', right_index=True)



# ---------Severity Type-------------------
severity_type = severity_type.merge(data[['id','fault_severity','source']], on='id')
severity_type_unq = pd.DataFrame(severity_type['severity_type'].value_counts())

#Determine % of training samples:
severity_type_unq['PercTrain'] = severity_type.pivot_table(values='source',index='severity_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))

#Determine the mode of each:
severity_type_unq['Mode_Severity'] = severity_type.loc[severity_type['source']=='train'].pivot_table(values='fault_severity',index='severity_type', aggfunc=lambda x: mode(x).mode[0])

severity_type_merge = severity_type.pivot_table(values='source',index='id',columns='severity_type',aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(severity_type_merge, left_on='id', right_index=True)








# riesenie location. Ku kazdemu zaznamu pridana numericka hodnota poctu lokation.

#Location Count:
location_count = data['location'].value_counts()
data['location_count'] = data['location'].apply(lambda x: location_count[x])


# Convert location to numeric:
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])


# Remove extra columns and split
data.drop(['Remove_x','Remove_y'],axis=1,inplace=True)

train_mod = data.loc[data['source']=='train']
test_mod = data.loc[data['source']=='test']

train_mod.drop('source',axis=1,inplace=True)
test_mod.drop(['source','fault_severity'],axis=1,inplace=True)

train_mod.to_csv('data/train_modified_5.csv',index=False)
test_mod.to_csv('data/test_modified_5.csv',index=False)


