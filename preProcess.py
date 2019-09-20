# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:45:49 2019

@author: Nitin.Sharma9
"""
#%%Imports https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1
#%%

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

#%% load data
#%%

#working_dir = "C:\\Users\\nitin.sharma9\\Documents\\Nitin\\python codes\\myProject\\data\\"
#name = "application_train.csv"
#def read_file(wdir,name):
#    os.chdir(wdir)
#    df = pd.read_csv(".\\"+name, low_memory=False)
#    return(df)

#trainCsv = read_file(working_dir,name)


#%% Splitting into train test split
#%%
# input target column name
target="TARGET"
X = trainCsv.drop([target], axis = 1)
y = pd.DataFrame(columns = [target])
y[target] = trainCsv[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify= y[[target]])

#%%
#%%
#input null percentage to retain
nullPercentage = 0.99
nullDf = pd.DataFrame(X_train.isnull().sum().sort_values(ascending= False)/X_train.shape[0]).reset_index()
nullDf.columns = ["ColName","Per"]
dtype = pd.DataFrame(trainCsv.dtypes).reset_index()
dtype.columns = ["ColName","type"]
dfInfoFull = pd.merge(nullDf,dtype,on=["ColName"])
#Subselecting column list with less than nullPercentage%null
dfInfo = dfInfoFull.copy(deep=True)
dfInfo = dfInfoFull.loc[dfInfoFull['Per'] < nullPercentage]
print("-- Columns Excluded --")
print(dfInfoFull.loc[dfInfoFull['Per'] > nullPercentage, ['ColName']])


#%% detecting missing values for numerical and categorical
#%%
dfInfo['type'].value_counts()
idVars = list()
catVars = list(dfInfo.loc[dfInfo['type'] == 'object']['ColName'])
catVars = [i for i in catVars if "TARGET" not in i and "_ID" not in i]
intVars = list(dfInfo.loc[dfInfo['type'] == 'int64']['ColName'])
intVars = [i for i in intVars if "TARGET" not in i and "_ID" not in i]
floatVars = list(dfInfo.loc[dfInfo['type'] == 'float64']['ColName'])
floatVars = [i for i in floatVars if "TARGET" not in i and "_ID" not in i]
totalVars = list(dfInfo['ColName'])
idVars = [i for i in totalVars if "_ID" in i]


#%% Identifying categorical variables in form of flags or discrete values
#%% 
flagVars = list()
pseudoNumVars = list()
shapeTrain = X_train.shape[0]
for col in intVars+floatVars:
    if X_train[col].nunique()/shapeTrain < 0.001 or X_train[col].nunique() < 20:
        pseudoNumVars.append(col)
#print(len(pseudoNumVars))

#%%
#%%
intTotalVars = intVars+floatVars
intTotalVars = [i for i in intTotalVars if i not in pseudoNumVars]


#%% fillna and outlier detection
#%%
#outlier detection for numerical 
def remove_outliers(df,col):
    std = 3*(df[col].std())
    mean = df[col].mean()
    l = mean-std
    u = mean+std
    df[col] = df[col].clip(lower=l,upper=u)
    return(df[col])   
    
medDict = dict()
numDf = X_train[intTotalVars]
for i in (intTotalVars):
    medDict[i] = numDf[i].median()
    numDf[i] = remove_outliers(numDf,i)
    numDf[i] = numDf[i].fillna(medDict[i])  #fill null with median/0

#catVars
modeDict = dict()
catDf = X_train[catVars]
for i in catVars:
    modeDict[i] = catDf[i].mode()[0]
    catDf[i]= catDf[i].fillna(catDf[i].mode()[0])
#flag cols    
flagDf = X_train[pseudoNumVars]
flagDf = flagDf.loc[:,~flagDf.columns.duplicated()]
flagDf.shape
for i in pseudoNumVars:
    #print(i)
    modeDict[i] = flagDf[i].mode()[0]
    flagDf[i]= flagDf[i].fillna(flagDf[i].mode()[0])
    
    
catDfDummy = pd.get_dummies(catDf)
catColsDummies = list(catDfDummy.columns)

#combining all columns
totalDf = pd.concat([numDf,catDfDummy,flagDf], axis=1)

#%%
#%%


#%% check and filter important variables
#%%
#input variable importance - 
imp = 0.98

rfmodel = RandomForestClassifier()
rfmodel.fit(totalDf,y_train['TARGET'])

varImportance = rfmodel.feature_importances_
varImp = pd.DataFrame(columns = ['ColName','VImp'])

varImp['ColName'] = totalDf.columns
varImp['VImp'] = varImportance
varImp = varImp.sort_values('VImp', ascending = False)
varImp['CumSum'] = varImp['VImp'].cumsum()
impVar = list(varImp.loc[varImp['CumSum'] < imp]['ColName'])
catColsDummies,intVars,floatVars

catColsDummiesFil = [i for i in catColsDummies if i in impVar]
intVarsFil = [i for i in intVars if i in impVar and i not in pseudoNumVars]
floatVarsFil = [i for i in floatVars if i in impVar and i not in pseudoNumVars]
flagVarsFil = [i for i in pseudoNumVars if i in impVar]

catDfDummyFil = catDfDummy[catColsDummiesFil]

#%% normalising data and PCA 
#%%
# Treating numerical columns

numDf = totalDf[intVarsFil+floatVarsFil]
print(numDf.shape)
mms = MinMaxScaler()
numDfScaled = mms.fit_transform(numDf)
pca = PCA(n_components = 0.98)
numDfPca = pca.fit_transform(pd.DataFrame(numDfScaled))
print(numDfPca.shape)

#%%
#%%
catDfDummyFil = catDfDummyFil.reset_index()
catDfDummyFil = catDfDummyFil.drop(['index'], axis = 1)
numDfScaled = pd.DataFrame(numDfScaled).reset_index()
numDfScaled = numDfScaled.drop(['index'], axis = 1)
flagDf = totalDf[flagVarsFil].reset_index()
flagDf = flagDf.drop(['index'], axis = 1)
trainDfCom = pd.concat([catDfDummyFil,numDfScaled,flagDf], axis = 1,ignore_index = True)
print(y_train.shape)
print(trainDfCom.shape)

#%%
#%%
model = xgb.XGBClassifier(max_depth=1,learning_rate=1,n_estimators=1,subsample=0.95,
                          scale_pos_weight = 1,colsample_bytree=0.95, 
                           min_child_weight = 2,seed = 121)
#model = xgb.XGBClassifier(max_depth=7,learning_rate=0.03,n_estimators=489,reg_alpha=3,reg_lambda=4,subsample=0.95,
#                          scale_pos_weight = 1,colsample_bytree=0.95, 
#                           min_child_weight = 2,seed = 121)

#model = xgb.XGBClassifier(max_depth=7,learning_rate=0.03,n_estimators=486,reg_alpha=3,reg_lambda=4,subsample=0.95,
#                          scale_pos_weight = 1,colsample_bytree=0.95, 
#                           min_child_weight = 2,seed = 121)

#model = xgb.XGBClassifier(max_depth=7,learning_rate=0.03,n_estimators=486,reg_alpha=3,reg_lambda=4,subsample=0.95,
#                          scale_pos_weight = 1,colsample_bytree=0.93, 
#                           min_child_weight = 2,seed = 121)
#best model

model.fit(trainDfCom,y_train['TARGET'])

#%%
#%%
print("Model Built")