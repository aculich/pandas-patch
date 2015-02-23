#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:34:33 2015

@author: efourrier

Purpose : the puropose of this modest patch is to create some new methods for
the class Dataframe in order to simplify a data scientist life.
The module is designed as a monkey patch so just import it before starting your 
analysis.
It is providing multiple simple methods for the class dataframe

run "python -m unittest -v test" in the module directory to run the tests 
 
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
import re 
from dateutil.parser import parse
from itertools import izip #izip faster than zip 
from numpy.random import permutation

#########################################################
# Private Helpers 
#########################################################


# Find a better way to do it ( core pandas implementation)
cserie= lambda serie: serie[serie].index
    
#########################################################
# Data cleaning and exploration helpers 
#########################################################

def sample_df(self,pct = 0.05,nr = 10):
    a = max(int(pct*float(len(self.index))),nr)
    return self.loc[permutation(self.index)[:a]]

pd.DataFrame.sample_df = sample_df
    
def nrow(self):
    """ return the number of rows
        This is faster than self.shape[0] """
    return len(self.index)
    
def ncol(self):
    """ return the number of cols """
    return self.shape[1]

pd.DataFrame.nrow = nrow
pd.DataFrame.ncol = ncol

def nacolcount(self):
    """ count the number of missing values per columns """
    Serie =  self.isnull().sum()
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.nrow())
    return df

pd.DataFrame.nacolcount = nacolcount

    
def narowcount(self):
    """ count the number of missing values per rows """
    Serie = self.isnull().sum(axis = 1)
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.ncol())
    return df

pd.DataFrame.narowcount = narowcount


def manymissing(self,a,row = False):
    """ identify columns of a dataframe with many missing values ( >= a), if 
    row = False row either
    - the output is a pandas index """
    if row:
        self = self.narowcount()
    else :
        self = self.nacolcount()
    return self[self['Napercentage'] >= a].index
    
pd.DataFrame.manymissing = manymissing

def constantcol(self):
    """ identify constant columns """
    # sample to reduce computation time 
    col_to_keep = self.sample_df().apply(lambda x: len(x.unique()) == 1,axis = 0 )
    self = self.loc[:,col_to_keep]
    return cserie(self.apply(lambda x: len(x.unique()) == 1,axis = 0 ))
    
pd.DataFrame.constantcol = constantcol



def dfnum(self):
    """ select columns with numeric type, the output is a list of columns  """
    return self.columns[((self.dtypes == float)|(self.dtypes == int))]

pd.DataFrame.dfnum = dfnum 

def detectkey(self, index_format = True, pct = 0.15):
    """ identify id or key columns as an index if index_format = True or 
    as a Serie if index_format = False """
    col_to_keep = self.sample_df(pct = 0.15).apply(lambda x: len(x.unique()) == len(x) ,axis = 0)
    if index_format:
        return cserie(self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0))
    else :
        return self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0)

    
pd.DataFrame.detectkey = detectkey

def df_len_string(self):
    """ Return a Series with the max of the length of the string of string-type columns """
    return self.drop(self.dfnum(),axis = 1).apply(lambda x : np.max(x.str.len()), axis = 0 )

pd.DataFrame.df_len_string = df_len_string

def findupcol(self):
    """ find duplicated columns and return the result as a list of list
    Function to correct , working but bad coding """
    dup_index = self.T.duplicated()
    dup_index_complet = (dup_index) | (self.T.duplicated(take_last = True))
    l = []
    for col in self.columns[dup_index]:
        index_temp = self.loc[:,dup_index_complet].apply(lambda x: (x == self[col])).sum() == self.nrow()
        temp = list(self.loc[:,dup_index_complet].columns[index_temp])
        l.append(temp)
    return l

pd.DataFrame.findupcol = findupcol

def finduprow(self,subset = []):
    """ find duplicated rows and return the result a sorted dataframe of all the
    duplicates
    subset is a list of columns to look for duplicates from this specific subset . """
    if subset:
        dup_index = (self.duplicated(subset = subset)) | (self.duplicated(subset = subset,take_last =True)) 
    else :    
        dup_index = (self.duplicated()) | (self.duplicated(take_last = True))
        
    if subset :
        return self[dup_index].sort(subset)
    else :
        return self[dup_index].sort(self.columns[0])

pd.DataFrame.finduprow = finduprow
    
    
def filterdupcol(self):
    """ return a dataframe without duplicated columns """
    return self.drop(self.columns[self.T.duplicated()], axis =1)
pd.DataFrame.filterdupcol = filterdupcol

def dfquantiles(self,nb_quantiles = 10,only_numeric = True):
    """ this function gives you a all the quantiles 
    of the numeric variables of the dataframe
    only_numeric will calculate it only for numeric variables, 
    for only_numeric = False you will get NaN value for non numeric 
    variables """
    binq = 1.0/nb_quantiles
    if only_numeric:
        self = self[self.dfnum()]
    return self.quantile([binq*i for i in xrange(nb_quantiles +1)])
    
pd.DataFrame.dfquantiles = dfquantiles

def is_date(self):
    """ to reprogram it is ugly """ 
    d = {}
    for col in self.columns:
        try :
             d[col] = False
             # i loop trough non missing value 
             # this try-except loop is bad programming 
             l = [parse(e) for e in self.loc[pd.notnull(self[col]),col]]
             len_not_na = self[pd.notnull(self[col])].nrow()
             if len(l) == len_not_na and len_not_na > 0 :          
                 d[col] = True
        except : 
            continue 
    return pd.Series(d)
            
pd.DataFrame.is_date = is_date            

def structure(self):
    """ this function will return a more complete type summary of variables type
    to reprogram it is ugly 
    """
    primary_type = self.dtypes
    is_key = self.detectkey(index_format = False)
    is_date = self.is_date()
    
    df = pd.concat([primary_type,is_key,is_date],axis = 1 )
    df.columns = ['primary_type','is_key','is_date']
    return df 

pd.DataFrame.structure = structure


def nearzerovar(self, freq_cut = 95/5, unique_cut = 10, save_metrics = False):
    """ identify predictors with near-zero variance. 
            freq_cut: cutoff ratio of frequency of most common value to second 
            most common value.
            unique_cut: cutoff percentage of unique value over total number of 
            samples.
            save_metrics: if False, print dataframe and return NON near-zero var 
            col indexes, if True, returns the whole dataframe.
    """

    percent_unique = self.apply(lambda x: 100*len(x.unique())/len(x), axis=0)
    freq_ratio = []
    for col in self.columns:
        if len(self[col].unique()) == 1:
            freq_ratio += [1]
        else:
            freq_ratio += [ float(self[col].value_counts().iloc[0])/self[col].value_counts().iloc[1] ]
    
    nzv = ((np.array(freq_ratio) >= freq_cut) & (percent_unique <= unique_cut))| (percent_unique == 0)

    if save_metrics:
        return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'nzv': nzv}, index=self.columns)
    else:
        print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'nzv': nzv}, index=self.columns))
        return nzv[nzv == True].index 

pd.DataFrame.nearzerovar = nearzerovar


def findcorr(self, cutoff=.90, method='pearson', data_frame=False):
    """
        implementation of the Recursive Pairwise Elimination.        
        The function finds the highest correlated pair and removes the most 
        highly correlated feature of the pair, then repeats the process 
        until the threshold 'cutoff' is reached.
        
        will return a dataframe is 'data_frame' is set to True, and the list
        of predictors to remove otherwise.

        Adaptation of 'findCorrelation' function in the caret package in R
    """
    res = []
    temp = self
        
    cor = temp.corr(method=method)
    # pandas doesn't give a value for diagonal cells
    for col in cor.columns:
        cor[col][col] = 0
    
    max_cor = cor.max()
    while max_cor.max() > cutoff:            
        A = max_cor.idxmax()
        B = cor[A].idxmax()
        
        if cor[A].mean() > cor[B].mean():
            temp = temp.drop(A, 1)
            res += [A]
        else:
            temp = temp.drop(B, 1)
            res += [B]
            
        cor = temp.corr(method=method)
        for col in cor.columns:
            cor[col][col] = 0
    
        max_cor = cor.max()
        
    if data_frame:
        return temp
    else:
        return res

pd.DataFrame.findcorr = findcorr

#########################################################
# Data basic summary helpers 
#########################################################

       
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
                df.loc[:,var] = pd.cut(df.loc[:,var],bins = quantile)
            else: 
                df.loc[:,var] = pd.qcut(df.loc[:,var],q = quantile)
            
    # grouping, apply function and return results 
    return df.groupby(groupvar).agg(functions)
    
pd.DataFrame.groupsummaryscc = groupsummaryscc


def group_average(self,groupvar,measurevar,avg_weight):
    """ return an weighted ( the weight are given by avg_weight) mean 
    of the variable measurevar group by groupvar """
    get_wavg = lambda df: np.average(a = df[measurevar], weights = df[avg_weight], axis = 0 )
    return self.groupby(groupvar).apply(get_wavg)

pd.DataFrame.group_average = group_average



#########################################################
# Simple outlier detection based on distance 
#########################################################

# Computing Distance Score 

# We don't take the absolute value of the score by choice 

# We let numpy take care of inf when std = 0 or iqr = 0 

iqr = lambda v: scipy.stats.scoreatpercentile(v,75) - scipy.stats.scoreatpercentile(v,25)

z_score = lambda v: (v - np.mean(v))/(np.std(v))

# More robust to outliers 
iqr_score = lambda v: (v - np.median(v))/(iqr(v))

# median(abs(v -median(v))/0.6745)
mad_score = lambda v: (v - np.median(v))/(np.median(np.absolute(v -np.median(v)/0.6745)))

def fivenum(v):
    """ Returns Tukey's five number summary 
    (minimum, lower-hinge, median, upper-hinge, maximum)
    for the input vector, a list or array of numbers 
    based on 1.5 times the interquartile distance """
    md = np.median(v)
    whisker = 1.5*iqr(v)
    return min(v), md-whisker, md, md+whisker, max(v)

def outlier_detection(self,remove_constant_col = True,
                      cutoff_zscore = 3,cutoff_iqrscore = 2,cutoff_mad = 2):
    """ Return a dictionnary with z_score,iqr_score,mad_score as keys and the 
    associate dataframe of distance as value of the dictionnnary"""
    
    self = self[self.dfnum()] # take only numeric variable 
    if remove_constant_col:
        self = self.drop(self.constantcol(), axis = 1) # remove constant variable 
    
    scores = [z_score,iqr_score,mad_score]
    keys = ['z_score','iqr_score','mad_score']
    return {key : self.apply(func) for key,func in izip(keys,scores)} #optimise with izip for fun use zip instead
    
pd.DataFrame.outliers_detection =  outlier_detection

#########################################################
# Global summary and basic cleaning function  
#########################################################

def psummary(self,manymissing_p = 0.70):
    """ This function will print you a summary of the dataset, based on function 
    designed is this package 
    - Argument : pandas.Dataframe
    - Output : python print 
    """
    print 'the columns with more than {0,2}% manymissing values:\n{1} \n'.format(100 * manymissing_p,
list(self.manymissing(manymissing_p)))
    print 'the keys of the dataset are:\n{0} \n'.format(list(self.detectkey()))
    print 'the duplicated columns of the dataset are:\n{0}\n'.format(self.findupcol())
    print 'the constant columns of the dataset are:\n{0}'.format(list(self.constantcol()))
pd.DataFrame.psummary = psummary
#########################################################
# Time Series Analysis 
#########################################################

#########################################################
# Unclassified 
#########################################################

def melt(self, id_variable, value_name = "value",variable_name = "variable"):
    """ This function is used to melt a dataframe, what means transform a 
    long dataframe into a wide dataframe (like sql table with key type)
    id_variable has to be a list of columns 
    """
    df = self.copy()
    df = df.set_index(id_variable)
    df = df.stack()
    df = df.reset_index()
    df.columns = id_variable + [variable_name] + [value_name]
    return df

pd.DataFrame.melt = melt

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [method for method in dir(object) if callable(getattr(object, method))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print "\n".join(["%s %s" %
                      (method.ljust(spacing),
                       processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList])

# Test of the modules of the patch
if __name__ == "__main__":
    #########################################################
    # Create a test dataframe 
    #########################################################

    test = DataFrame(read_csv('lc_test.csv'))
    test['na_col'] = np.nan
    test['constant_col'] = 'constant'
    test['duplicated_column'] = test.id
    # test.int_rate = test.int_rate.str.findall(re.compile(r'\d+.\d+')).str.get(0)
    # test.int_rate = test.int_rate.astype(float)
    
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
    test.finduprow
    test.dfquantiles(20)
    test.is_date()
    test_wd = test.filterdupcol()
    test_wd.findupcol()
    test.nearzerovar()
    test.findcorr()
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
    test.group_average('grade','int_rate','loan_amnt')
