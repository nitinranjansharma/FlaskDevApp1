# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:16:57 2019

@author: Nitin.Sharma9
"""
#imports
import pickle
from prepProcessFunctions import preprocessSteps


def create_model(df):
    import pandas as pd
    df1 = pd.read_csv(df)
    with open('args.pickle', 'rb') as handle:
        reqArg = pickle.load(handle)
    with open('ls.pickle', 'rb') as handle:
        ls = pickle.load(handle)
    print("The List")
    print(ls)
    if len(ls) == 0 or ls == None:
        ls = [0,0,0,0]
    desc = preprocessSteps(df1,reqArg,ls)
    desc.to_csv("description.csv")
    return("Success")
    
def testFunc(*args):
    ar = [i for i in args]
    with open('args.pickle', 'wb') as handle:
        pickle.dump(ar, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(ar)
    return(ar)


    




    

    
        
        
    

    



