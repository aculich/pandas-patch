# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:34:33 2015

@author: efourrier

Purpose : the puropose of this modest patch is to create some new methods for
the class Dataframe in order to simplify a data scientist life.
The module is designed as a monkey patch so just import it before starting your 
analysis.
It is providing multiple simple methods for the class dataframe 
"""
#########################################################
# Import modules 
#########################################################

import pandas as pd 
import numpy as np 
from pandas import DataFrame
from pandas import read_csv
import scipy
from scipy.stats import t
import scikits.bootstrap as bootstrap



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

    
def narowcount(self):
    """ count the number of missing values per rows """
    Serie = self.apply(lambda x: sum(pd.isnull(x)),axis = 1 )   
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.shape[1])
    return df

pd.DataFrame.narowcount = narowcount


def manymissing(self,a):
    """ identify columns of a dataframe with many missing values ( >= a) """
    df = self.nacolcount()
    return df[df['Napercentage'] >= a].index
    
pd.DataFrame.manymissing = manymissing


def constantcol(self):
    """ identify constant columns """
    df = self.apply(lambda x: len(x.unique()),axis = 0 )
    return df[df == 1].index


    
pd.DataFrame.constantcol = constantcol


def nrow(self):
    """ return the number of rows """
    return self.shape[0]
    
def ncol(self):
    """ return the number of cols """
    return self.shape[1]

pd.DataFrame.nrow = nrow
pd.DataFrame.ncol = ncol


def detectkey(self):
    """ identify id or key columns """
    df = self.apply(lambda x: len(x.unique()),axis = 0 )
    return df[df == self.nrow()].index
    
pd.DataFrame.detectkey = detectkey


def findupcol(self):
    """ find duplicated columns and return the result as a list of list
    Function to correct , working but bad coding """
    dup_index = self.T.duplicated() 
    dup_columns = self.columns[dup_index]
    l = []
    for col in dup_columns:
        index_temp = self.apply(lambda x: (x == self[col])).apply(lambda x:sum(x) == self.nrow())
        temp = list(self.columns[index_temp])
        l.append(temp)
    return list(np.unique([col for col in l if col != [] ]))

pd.DataFrame.findupcol = findupcol


def filterdupcol(self):
    """ return a dataframe without duplicated columns """
    return self.drop(self.columns[self.T.duplicated()],axis =1)
pd.DataFrame.filterdupcol = filterdupcol


#########################################################
# Data basic summary helpers 
#########################################################
def dfnum(self):
    """ select columns with numeric type, the output is a list of columns  """
    return self.columns[((self.dtypes == float)|(self.dtypes == int))]

pd.DataFrame.dfnum = dfnum 
       
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


#########################################################
# Data grouped summary helpers 
#########################################################

def groupsummarys(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar. measurevar and 
    groupvar are list of column names. this function is optimized for speed """
    functions = ['count','min','mean','median','std','max']
    col = measurevar + groupvar 
    df = self[col]
    return df.groupby(groupvar).agg(functions)

pd.DataFrame.groupsummarys = groupsummarys



def groupsummaryd(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar with describe helper.
    measurevar and groupvar are list of column names """
    col = measurevar + groupvar 
    df = self[col]
    return df.groupby(groupvar).describe()

pd.DataFrame.groupsummaryd = groupsummaryd


def groupsummarysc(self,groupvar,measurevar,confint=0.95,id_group= True,
                   cut = False, quantile = 5,is_bucket = False,**kwarg):
    """ provide a summary of measurevar groupby groupvar with student conf interval.
    measurevar and groupvar are list of column names
    if you want bucket of equal length instead of quantile put is_bucket = True """
    def se(x):
        return x.std() / np.sqrt(len(x))
 
    def student_ci(x):
        return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwarg)[1]
    functions = ['count','min','mean',se,student_ci,'median','std','max']
    col = measurevar + groupvar 
    df = self[col]
    if cut == True:
        for var in groupvar:
            if id_group == True:
                df[var] = pd.cut(df[var],bins = quantile)
            else: 
                df[var] = pd.qcut(df[var],q = quantile)
    return df.groupby(groupvar).agg(functions)

pd.DataFrame.groupsummarysc = groupsummarysc


def groupsummarybc(self,groupvar,measurevar,confint=0.95,nsamples = 500,
                   cut = False, quantile = 5,is_bucket = False, **kwarg):
    """ provide a summary of measurevar groupby groupvar with bootstrap conf interval.
    measurevar and groupvar are list of column names. You have a cut functionnality 
    if you want to cut the groupvar
    if you want bucket of equal length instead of quantile put is_bucket = True """
    def ci_inf(x):
        return bootstrap.ci(data=x, statfunction=scipy.mean, alpha = confint,
                            n_samples = nsamples,**kwarg)[0]
    def ci_up(x):
        return bootstrap.ci(data=x, statfunction=scipy.mean, alpha = confint,
                            n_samples = nsamples,**kwarg)[1]
        
    functions = ['count','min',ci_inf,'mean',ci_up,'median','std','max']
    col = measurevar + groupvar 
    df = self[col]
    if cut == True:
        for var in groupvar:
            if is_bucket == True:
                df[var] = pd.cut(df[var],bins = quantile)
            else: 
                df[var] = pd.qcut(df[var],q = quantile)
    return df.groupby(groupvar).agg(functions)

pd.DataFrame.groupsummarybc = groupsummarybc


def groupsummaryscc(self,groupvar,measurevar,confint=0.95,
                   cut = False, quantile = 5,is_bucket = False, **kwarg):
    """ provide a more complete summary than groupsummarysc of measurevar
    groupby groupvar with student conf interval.measurevar and groupvar
    are list of column names 
    if you want bucket of equal length instead of quagit ntile put is_bucket = True """
    
    # creating the list of functions 
    def se(x):
        return x.std() / np.sqrt(len(x))
    def student_ci(x):
        return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwarg)[1]
    def quantile_25(x):
        return x.quantile(0.25)
    def quantile_75(x):
        return x.quantile(0.75)
    def skewness(x):
        return x.skew()
    def kurtosis(x):
        return x.kurt()
    functions = ['count','min', quantile_25,'mean',se,student_ci,
    'median','std','mad',skewness ,kurtosis,quantile_75,'max']
    col = measurevar + groupvar 
    df = self[col]
    # Correct the problem of unicity 
    if cut == True:
        for var in groupvar:
            if is_bucket == True:
                df[var] = pd.cut(df[var],bins = quantile)
            else: 
                df[var] = pd.qcut(df[var],q = quantile)
            
    # grouping, apply function and return results 
    return df.groupby(groupvar).agg(functions)
    
pd.DataFrame.groupsummaryscc = groupsummaryscc


def fivenum(v):
    """ Returns Tukey's five number summary 
    (minimum, lower-hinge, median, upper-hinge, maximum)
    for the input vector, a list or array of numbers 
    based on 1.5 times the interquartile distance """
    q1 = scipy.stats.scoreatpercentile(v,25)
    q3 = scipy.stats.scoreatpercentile(v,75)
    iqd = q3-q1
    md = np.median(v)
    whisker = 1.5*iqd
    return min(v), md-whisker, md, md+whisker, max(v)



def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [method for method in dir(object) if callable(getattr(object, method))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print "\n".join(["%s %s" %
                      (method.ljust(spacing),
                       processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList])
if __name__ == "__main__":
    print info.__doc__

# Test of the modules of the patch
if __name__ == "__main__":
    #########################################################
    # Create a test dataframe 
    #########################################################
    test = DataFrame(read_csv('lc_test.csv'))
    test['na_col'] = np.nan
    test['constant_col'] = 'constant'
    test['duplicated_column'] = test.id
    #########################################################
    # Testing the functions 
    #########################################################
    test.nacolcount()
    test.narowcount()
    test.manymissing(0.5)
    test.nrow()
    test.ncol()
    test.detectkey()
    test.findupcol()
    test_wd = test.filterdupcol()
    test_wd.findupcol()
    test.dfnum()
    test.detailledsummary()
    test1 = test.groupsummarys(['grade'],['fico_range_high'])
    # multiindex structure 
    test.groupsummarys(['grade','sub_grade'],['fico_range_high','dti'])
    test.groupsummaryd(['grade'],['fico_range_high'])
    test.groupsummarysc(['grade'],['fico_range_high'])
    test.groupsummarybc(['grade'],['fico_range_high'])
    kwargs = {'method': 'pi'}
    test.groupsummarybc(['grade'],['fico_range_high'])
    # or other solution for the use ok kwargs 
    del kwargs
    test.groupsummarybc(['grade'],['fico_range_high'],method = 'pi')
    test.groupsummaryscc(['grade'],['fico_range_high'])
    test.groupsummaryscc(['fico_range_high'],['dti'],cut =True)