# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:34:33 2015

@author: efourrier

Purpose : the puropose of this modest patch is to create some new methods for
the class Dataframe in order to simplify a data scientist life 
"""
import pandas as pd 
from pandas import DataFrame
from pandas import read_csv

test = DataFrame(read_csv('lc_test.csv'))
test['na_col'] = np.nan
test['constant_col'] = 'constant'

def nacolcount(self):
    """ count the number of missing values per columns """
    Serie =  self.apply(lambda x: sum(pd.isnull(x)),axis = 0)
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.shape[0])
    return df

pd.DataFrame.nacolcount = nacolcount
test.nacolcount()
    
def narowcount(self):
    """ count the number of missing values per rows """
    Serie = self.apply(lambda x: sum(pd.isnull(x)),axis = 1 )   
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.shape[1])
    return df

pd.DataFrame.nacolcount = nacolcount
test.narowcount()

def manymissing(self,a):
    """ identify columns of a dataframe with many missing values ( >= a) """
    df = self.nacolcount()
    return df[df['Napercentage'] >= a].index
    
pd.DataFrame.manymissing = manymissing
test.manymissing(0.5)

def constantcol(self):
    """ identify constant columns """
    df = self.apply(lambda x: len(x.unique()),axis = 0 )
    return df[df == 1].index
    
pd.DataFrame.constantcol = constantcol
test.constantcol()    
    
