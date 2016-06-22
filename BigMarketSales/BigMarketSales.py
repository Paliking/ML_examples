
# priklad kompletnej analyzy
# zdroj : http://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

# sikovne pandas techniky http://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# 1. Hypothesis Generation
# 2. Data Exploration-------------------------------------------------------------------------

#Read files:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['source']='train'
test['source']='test'
# spojenie datasetov, aby sme ich jednotne opravovali
# ignore index, aby bolo nove cislovanie indexu
data = pd.concat([train, test], ignore_index=True)

print (train.shape, test.shape, data.shape)

# vypis poctu chybajucich hodnot
print(data.apply(lambda x: sum(x.isnull())))

# basic statistic
print(data.describe())

# unique values
print('\nUnique values')
print(data.apply(lambda x: len(x.unique())))

# ideme sledovat frekvenciu kategorii
#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())



# 3. Data Cleaning ----------------------------------------------------------------------------
# ------------------------------Imputing Missing Values----------------------------------------
# Item_Weight and Outlet_Size are missing
print('--------Imputing Missing Values----------------------------')

# Item_Weight
#Determine the average weight per item: (pivot_table ma defaultnu funkciu mean)
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
print ('Final #missing: %d'% sum(data['Item_Weight'].isnull()))



# impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet

#Import mode function:
from scipy.stats import mode

#Determing the mode for each ( doplnil som dropna() aby to fungovalo )
outlet_size_mode = data.dropna().pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x: mode(x).mode[0]) )
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print ('Final missing:',sum(data['Outlet_Size'].isnull()))


# 4. Feature Engineering---------------------------------------------------------------------------------
print('--------Feature engineering----------------------')



# a)
# rozmyslame ze zlucime Outlet_Type Supermarket Type2 and Type3, preverime to porovnanim ci to je dobry napad
# vidime ze je v nich vyrazny rozdiel, takze ich nezlucime
print(data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type'))






# b) Modify Item_Visibility
# We noticed that the minimum value here is 0, which makes no practical sense.
# Lets consider it like missing information and impute it with mean visibility of that product.

#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))



# vytvorenie pomeru viditelnosti produktu vs priemerna viditelnost tohto produktu vo vsetkych obchodoch
#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1) # axis1 - iter. cez riadky
print ('\n',data['Item_Visibility_MeanRatio'].describe())






# c) Type of Item
# prve dva pismena z ID rozdeluju data na 3 kategorie FD, DR, NC
#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print('\n',data['Item_Type_Combined'].value_counts())






# d) Determine the years of operation of a store
#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
print('\n',data['Outlet_Years'].describe())






# e) Modify categories of Item_Fat_Content
#Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
print(data['Item_Fat_Content'].value_counts())





# f) Numerical and One-Hot Coding of Categorical variables
# Since scikit-learn accepts only numerical variables, I converted all categories of nominal variables into numeric types

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


#One Hot Coding: (jedna premenna s viac kategoriami je rozdelena na viac binarnych premennych )
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])

print(data.dtypes)




# g) Exporting Data
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.copy().loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)




# 5. Model Building------------------------------------------------------------------


# baseline model (iba jednoduchy odhad, v tomto pripade priemer)
#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
##base1.to_csv("alg0.csv",index=False)





# funkcia na predikovanie/export suboru na vlzoenie do sutaze
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
##    submission.to_csv(filename, index=False)





# a) Linear Regression Model

# lin. reg. vychadza inak ako v tutorialy why???

from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()

# koeficienty v lin. regression maju velke rozpatie, to naznacuje overfitting. Skusime Ridge alebo lasso.


# ridge regression. Mensia magnitude, ale skore zostava podobne. O mnoho sa to uz nezlepsi ani po tunovani parametrov.
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
plt.show()



# b) Decision Tree Model

from sklearn.tree import DecisionTreeRegressor

# CV skore naznacuje mierny overfitting
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
plt.show()


# dalo by sa este zlepsit tunovanim parametrov
predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')
plt.show()



# C) Random Forest

from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')
plt.show()


# ine parametre
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')
plt.show()



