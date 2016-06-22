# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:01:40 2016

@author: Pablo

zdroj: http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/


"""

import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

print(df.describe())

# frekvency of non-numerical values
df['Property_Area'].value_counts()


#  study distribution of various variables
df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='ApplicantIncome', by = 'Education')


df['Loan_Status'] = df['Loan_Status'].apply(lambda x: 1 if x=='Y' else 0)
