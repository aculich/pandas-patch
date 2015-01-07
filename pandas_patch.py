# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:34:33 2015

@author: efourrier

Purpose : the puropose of this modest patch is to create some new methods for
the class Dataframe in order to simplify a data scientist life.
The module is designed as a monkey patch so just import it before starting your 
analysis
"""
#########################################################
# Import modules 
#########################################################

import pandas as pd 
from pandas import DataFrame
from pandas import read_csv

#########################################################
# Create a test dataframe 
#########################################################

test = DataFrame(read_csv('lc_test.csv'))
test['na_col'] = np.nan
test['constant_col'] = 'constant'
test['duplicated_column'] = test.id

#########################################################
# Data cleaning and exploration helpers 
#########################################################


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

def nrow(self):
    """ return the number of rows """
    return self.shape[0]
    
def ncol(self):
    """ return the number of cols """
    return self.shape[1]

pd.DataFrame.nrow = nrow
test.nrow()
pd.DataFrame.ncol = ncol
test.ncol()

def detectkey(self):
    """ identify id or key columns """
    df = self.apply(lambda x: len(x.unique()),axis = 0 )
    return df[df == self.nrow()].index
    
pd.DataFrame.detectkey = detectkey
test.detectkey()

#def findupcol(self):
#    """ find duplicated columns and return the result as a list of tuple """
#    dup = self.T.duplicated 
#    index_dup = dup[dup == True].index
#    for col in index_dup:
#        dup =test.apply(lambda x: (x == test.col))
#        dup = dup[dup == True].T.dropna().T.columns
#        tuple =  ()



#########################################################
# Data summary helpers 
#########################################################
def dfnum(self):
    """ select columns with numeric type, the output is a list of columns  """
    return self.columns[((self.dtypes == float)|(self.dtypes == int))]

pd.DataFrame.dfnum = dfnum 
test.dfnum()
       
def detailledsummary(self):
    """ provide a more complete sumary than describe, it is using only numeric
    value """
    self = self[self.dfnum()]
    func_list = [self.count(),self.min(), self.quantile(0.25),self.quantile(0.5),self.mean(),
                 self.std(),self.mad(),self.skew(),self.kurt(),self.quantile(0.75),self.max()]
    results = [f for f in func_list]
    return DataFrame(results,index=['Count','Min','FirstQuartile',
    'Median','Mean','Std','Mad','Skewness',
    'Kurtosis','Thirdquartile','Max']).T
                                      
pd.DataFrame.detailledsummary = detailledsummary
test.detailledsummary()

def groupsummarys(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar. measurevar and 
    groupvar are list of column names """
    functions = ['count','min','mean','median','std','max']
    col = measurevar + groupvar 
    df = self[col]
    return df.groupby(groupvar).agg(functions)

pd.DataFrame.groupsummarys = groupsummarys
test1 = test.groupsummarys(['grade'],['fico_range_high'])

def groupsummaryd(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar with describe helper.
    measurevar and groupvar are list of column names """
    col = measurevar + groupvar 
    df = self[col]
    return df.groupby(groupvar).describe()

pd.DataFrame.groupsummaryd = groupsummaryd
test.groupsummaryd(['grade'],['fico_range_high'])