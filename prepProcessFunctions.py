import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def preprocessSteps(df,ar1,ls):
    preprocessFunctionList = ['std','pca','outlier','dummy']
    ar = ar1[0][0]
    
    requirementDict = dict(zip(preprocessFunctionList,ar))
    print(requirementDict)
    print(ar)

    X_train = df.copy(deep=True)
    #identifying columns start
    nullDf = pd.DataFrame(X_train.isnull().sum().sort_values(ascending= False)/X_train.shape[0]).reset_index()
    nullDf.columns = ["ColName","Per"]
    dtype = pd.DataFrame(df.dtypes).reset_index()
    dtype.columns = ["ColName","type"]
    dfInfoFull = pd.merge(nullDf,dtype,on=["ColName"])
    #Subselecting column list with less than 40%null
    dfInfo = dfInfoFull.copy(deep=True)
    #dfInfo = dfInfoFull.loc[dfInfoFull['Per'] < 0.4]


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

    flagCols = [i for i in intVars+floatVars if 'FLAG' in i]
    tempFlagCols = list()
    pseudoNumVars = list()
    shapeTrain = X_train.shape[0]
    for col in intVars+floatVars:
        if X_train[col].nunique()/shapeTrain < 0.001:
            pseudoNumVars.append(col)
    print(len(pseudoNumVars))
    for col in pseudoNumVars:
        if X_train[col].nunique() < 3:
            tempFlagCols.append(col)
    flagCols = flagCols+[i for i in tempFlagCols if i not in flagCols]  # accumulate all flag variables       
    pseudoNumVars = [i for i in pseudoNumVars if i not in flagCols]#getting remainder

    intTotalVars = intVars+floatVars
    intTotalVars = [i for i in intTotalVars if i not in flagCols]

    ordinalVars = list()
    flagVars = list()
    
    #check if columns are mentioned exclusively
    #['12,12', '1', '1']
    if len(ls) > 0 and ls != None:
        if int(ls[3]) < 1:
            intTotalVars = []
            catVars = []
            flagCols = []
            
            try:
                print("------Overriding Columns-------")
                if len([ls[0]][0].split(","))>0:
                    intTotalVars = [ls[0]][0].split(",")
                    intTotalVars = [i for i in intTotalVars if i in X_train.columns]
                
                if len([ls[1]][0].split(",")) >0:
                    catVars = [ls[1]][0].split(",")
                    catVars = [i for i in catVars if i in X_train.columns]
                if len([ls[2]][0].split(","))>0:
                    flagCols = [ls[2]][0].split(",")
                    flagCols = [i for i in flagCols if i in X_train.columns]
                
                print(flagCols)
                
            except:
                print("Wrong Column Names entered!")

    print("Manual Columns")
    
    #outlier treatment
    if requirementDict['outlier'] >0:
        print("Outlier start")
        def remove_outliers(df,col):
            std = 3*(df[col].std())
            mean = df[col].mean()
            l = mean-std
            u = mean+std
            df[col] = df[col].clip(lower=l,upper=u)
            return(df[col])

        medDict = dict()
        print(intTotalVars)
        numDf = X_train[intTotalVars]
        print(numDf.shape)
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

        flagDf = X_train[flagCols]
        flagDf = flagDf.loc[:,~flagDf.columns.duplicated()]

        for i in flagCols:
            
            modeDict[i] = flagDf[i].mode()[0]
            flagDf[i]= flagDf[i].fillna(flagDf[i].mode()[0])

        totalDf = pd.concat([numDf,catDf,flagDf], axis=1)
        totalDf.to_csv("totaldf.csv")
        print("outlier done")
    if requirementDict['std'] >0:
        numDf = totalDf[intTotalVars]
        mms = MinMaxScaler()
        numDfCol = numDf.columns
        numDf = mms.fit_transform(numDf)
        numDf = pd.DataFrame(numDf,columns = numDfCol)
        totalDf = pd.concat([numDf,catDf,flagDf], axis=1)
        totalDf.to_csv("totaldf.csv")
        print("Std done")
    
    if requirementDict['pca'] >0:
        numDf = totalDf[intTotalVars]
        pca = PCA(n_components = requirementDict['pca'])
        if requirementDict['std'] >0:
            numDf = pca.fit_transform(pd.DataFrame(numDf))
        else:
            numDf = pca.fit_transform(pd.DataFrame(numDf))
        numDf = pd.DataFrame(numDf)
        totalDf = pd.concat([numDf,catDf,flagDf], axis=1)
        totalDf.to_csv("totaldf.csv")
        print("pca done")

    if requirementDict['dummy'] >0:
        if catVars != None and len(catVars) >0:
            catDf = pd.get_dummies(catDf)
            totalDf = pd.concat([numDf,catDf,flagDf], axis=1)
            totalDf.to_csv("totaldf.csv")
            print("dummy done")

  
    print("Describe")
    print(df[intTotalVars].describe())
    return(df[intTotalVars].describe())