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

Note : don't use inplace = True on self when you are doing monkey patching,
it is dangerous.
 
"""
#########################################################
# Import modules 
#########################################################

import pandas as pd
from utils import deprecated
import numpy as np 
from pandas import DataFrame
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
# cserie return the column name with bool = True for a Serie of boolean 
cserie = lambda serie: list(serie[serie].index)
    
#########################################################
# Data cleaning and exploration helpers 
#########################################################

def sample_df(self,pct = 0.05,nr = 10,threshold = None):
    """ sample a number of rows of a dataframe = min(max(0.05*nrow(self,nr),threshold)"""
    a = max(int(pct*float(len(self.index))),nr)
    if threshold:
        a = min(a,threshold)
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

def nacount(self,axis = 1):
    """ count the number of missing values per columns or rows of the dataframe

    - Argument : axis = 1 if you want count the na values per rows, default 0 per columns.

    """
    if axis == 1 :
        Serie =  self.isnull().sum()
        df =  DataFrame(Serie,columns = ['Nanumber'])
        df['Napercentage'] = df['Nanumber']/(self.nrow())
    if axis == 0 :
        Serie = self.isnull().sum(axis = 1)
        df =  DataFrame(Serie,columns = ['Nanumber'])
        df['Napercentage'] = df['Nanumber']/(self.ncol())
    return df 

pd.DataFrame.nacount = nacount



def nacolcount(self):
    """ count the number of missing values per columns """
    print "this function is deprecated please use nacount"
    Serie =  self.isnull().sum()
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.nrow())
    return df

pd.DataFrame.nacolcount = nacolcount

def narowcount(self):
    """ count the number of missing values per rows """
    print "this function is deprecated please use nacount"
    Serie = self.isnull().sum(axis = 1)
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.ncol())
    return df

pd.DataFrame.narowcount = narowcount


def manymissing(self,a=0.9,axis = 1):
    """ identify columns of a dataframe with many missing values ( >= a), if 
    axis = 1 row either
    - the output is a pandas index """
    if axis == 0:
        self = self.nacount(axis = 0)
    if axis == 1:
        self = self.nacount(axis = 1 )
    return self[self['Napercentage'] >= a].index
    
pd.DataFrame.manymissing = manymissing

def constantcol(self,**kwargs):
    """ identify constant columns """
    # sample to reduce computation time 
    col_to_keep = self.sample_df(**kwargs).apply(lambda x: len(x.unique()) == 1,axis = 0 )
    if len(cserie(col_to_keep)) == 0:
        return []
    return cserie(self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == 1,axis = 0 ))
    
pd.DataFrame.constantcol = constantcol

def constantcol2(self):
    def helper(x):
        unique_value = set()
        for e in x:
            if len(unique_value) > 1:
                return False
            else:
                unique_value.add(e)
        return True
    return cserie(self.apply(lambda x: helper(x)))

pd.DataFrame.constantcol2 = constantcol2

def dfnum(self):
    """ select columns with numeric type, the output is a list of columns  """
    return self.columns[((self.dtypes == float)|(self.dtypes == int))]

pd.DataFrame.dfnum = dfnum 

def detectkey(self, index_format = True, pct = 0.15,**kwargs):
    """ identify id or key columns as an index if index_format = True or 
    as a Serie if index_format = False """
    col_to_keep = self.sample_df(pct = 0.15,**kwargs).apply(lambda x: len(x.unique()) == len(x) ,axis = 0)
    if index_format:
        return cserie(self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0))
    else :
        return self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0)

    
pd.DataFrame.detectkey = detectkey

def detectkey2(self):
    def helper(x):
        unique_value = set()
        for index,e in enumerate(x):
            if len(unique_value) < index :
                return False
            else:
                unique_value.add(e)
        return True
    return cserie(self.apply(lambda x: helper(x)))

pd.DataFrame.detectkey2 = detectkey2




def df_len_string(self):
    """ Return a Series with the max of the length of the string of string-type columns """
    return self.drop(self.dfnum(),axis = 1).apply(lambda x : np.max(x.str.len()), axis = 0 )

pd.DataFrame.df_len_string = df_len_string



def findupcol(self,threshold = 100,**kwargs):
    """ find duplicated columns and return the result as a list of list """

    df_s = self.sample_df(threshold = 100,**kwargs).T
    dup_index_s = (df_s.duplicated()) | (df_s.duplicated(take_last = True))
    
    if len(cserie(dup_index_s)) == 0:
        return []

    df_t = (self.loc[:,dup_index_s]).T
    dup_index = df_t.duplicated()
    dup_index_complet = cserie((dup_index) | (df_t.duplicated(take_last = True)))

    l = []
    for col in cserie(dup_index):
        index_temp = self[dup_index_complet].apply(lambda x: (x == self[col])).sum() == self.nrow()
        temp = list(self[dup_index_complet].columns[index_temp])
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

# def is_date(self,exclude_numver):
#     """ to reprogram it is ugly """
#     d = {}
#     for col in self.columns:
#         try :
#              d[col] = False
#              # i loop trough non missing value 
#              # this try-except loop is bad programming 
#              l = [parse(e) for e in self.loc[pd.notnull(self[col]),col]]
#              len_not_na = self[pd.notnull(self[col])].nrow()
#              if len(l) == len_not_na and len_not_na > 0 :          
#                  d[col] = True
#         except : 
#             continue 
#     return pd.Series(d)
            
# pd.DataFrame.is_date = is_date            

def structure(self):
    """ this function will return a more detailled column type"""
    return self.apply(lambda x: pd.lib.infer_dtype(x.values))
    
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

    percent_unique = self.apply(lambda x: float(100*len(x.unique()))/len(x), axis=0)
    freq_ratio = []
    for col in self.columns:
        if len(self[col].unique()) == 1:
            freq_ratio += [1]
        else:
            freq_ratio += [float(self[col].value_counts().iloc[0])/self[col].value_counts().iloc[1] ]

    zerovar = self.apply(lambda x: len(x.unique()) == 1, axis = 0)
    nzv = ((np.array(freq_ratio) >= freq_cut) & (percent_unique <= unique_cut)) | (percent_unique == 0)

    if save_metrics:
        return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.columns)
    else:
        print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.columns))
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

        Adaptation of 'findCorrelation' function in the caret package in R. """
    res = []

    cor = self.corr(method=method)
    for col in cor.columns:
        cor[col][col] = 0
    
    max_cor = cor.max()
    print (max_cor.max())
    while max_cor.max() > cutoff:            
        A = max_cor.idxmax()
        B = cor[A].idxmax()
        
        if cor[A].mean() > cor[B].mean():
            cor.drop(A, 1, inplace = True)
            cor.drop(A, 0, inplace = True)
            res += [A]
        else:
            cor.drop(B, 1, inplace = True)
            cor.drop(B, 0, inplace = True)
            res += [B]
        
        max_cor = cor.max()
        print (max_cor.max())
        
    if data_frame:
        return self.drop(res, 1)
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


# Helpers for groupsummary function
def se(x):
    return x.std() / np.sqrt(len(x))
    # creating the list of functions 
def quantile_25(x):
    return x.quantile(0.25)
def quantile_75(x):
    return x.quantile(0.75)
def skewness(x):
    return x.skew()
def kurtosis(x):
    return x.kurt()

# Be careful all groupsummary return a dataframe with a MultiIndex structure

def groupsummarys(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar. measurevar and 
    groupvar are list of column names. this function is optimized for speed """
    functions = ['count','min','mean','median','std','sem','max']
    return self.groupby(groupvar)[measurevar].agg(functions)

pd.DataFrame.groupsummarys = groupsummarys



def groupsummaryd(self,groupvar,measurevar):
    """ provide a summary of measurevar groupby groupvar with describe helper.
    measurevar and groupvar are list of column names """
    return self.groupby(groupvar)[measurevar].describe()

pd.DataFrame.groupsummaryd = groupsummaryd


def groupsummarysc(self,groupvar,measurevar,confint=0.95,cut = False, 
    quantile = 5,is_bucket = False,**kwargs):
    """ provide a summary of measurevar groupby groupvar with student conf interval.
    measurevar and groupvar are list of column names
    if you want bucket of equal length instead of quantile put is_bucket = True """
    def student_ci(x):
        return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwargs)[1]
    self = self.copy()
    functions = ['count','min',('ci_low',lambda x: np.mean(x) - student_ci(x)),
     'mean',('ci_up',lambda x: np.mean(x) + student_ci(x)),'median',('se',se),'std','max']
    if cut:
        for var in groupvar:
            if is_bucket:
                self[var] = pd.cut(self[var],bins = quantile)
            else: 
                self[var] = pd.qcut(self[var],q = quantile)
    return self.groupby(groupvar)[measurevar].agg(functions)

pd.DataFrame.groupsummarysc = groupsummarysc


def groupsummarybc(self,groupvar,measurevar,confint=0.95,nsamples = 500,
                   cut = False, quantile = 5,is_bucket = False, **kwargs):
    """ provide a summary of measurevar groupby groupvar with bootstrap conf interval.
    measurevar and groupvar are list of column names. You have a cut functionnality 
    if you want to cut the groupvar
    if you want bucket of equal length instead of quantile put is_bucket = True """
    def ci_low(x):
        return bootstrap.ci(data=x, statfunction=scipy.mean, alpha = confint,
                        n_samples = nsamples,**kwargs)[0]
    def ci_up(x):
        return bootstrap.ci(data=x, statfunction=scipy.mean, alpha = confint,
                    n_samples = nsamples,**kwargs)[1]
    self = self.copy()
    functions = ['count','min',ci_low,'mean',ci_up,'median','std','max']
    if cut:
        for var in groupvar:
            if is_bucket:
                self[var] = pd.cut(self[var],bins = quantile)
            else: 
                self[var] = pd.qcut(self[var],q = quantile)
    return self.groupby(groupvar)[measurevar].agg(functions)

pd.DataFrame.groupsummarybc = groupsummarybc


def groupsummaryscc(self,groupvar,measurevar,confint=0.95,
                   cut = False, quantile = 5,is_bucket = False, **kwargs):
    """ provide a more complete summary than groupsummarysc of measurevar
    groupby groupvar with student conf interval.measurevar and groupvar
    are list of column names 
    if you want bucket of equal length instead of quantile put is_bucket = True """
    def student_ci(x):
        return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwargs)[1]
    self = self.copy()
    functions = ['count','min', quantile_25,'mean',se,student_ci,
    'median','std','mad',skewness ,kurtosis,quantile_75,'max']
    # Correct the problem of unicity 
    if cut:
        for var in groupvar:
            if is_bucket:
                self.loc[:,var] = pd.cut(self.loc[:,var],bins = quantile)
            else: 
                self.loc[:,var] = pd.qcut(self.loc[:,var],q = quantile)
            
    # grouping, apply function and return results 
    return self.groupby(groupvar)[measurevar].agg(functions)
    
pd.DataFrame.groupsummaryscc = groupsummaryscc


# This is the global function that the user will be using 
def groupsummary(self,groupvar,measurevar,confint=0.95,fast_summary =False,
                detailled_summary = False,boostrap = False,
                   cut = False, quantile = 5,is_bucket = False,group_keys = False,**kwargs):
    self = self.copy()
    if fast_summary:
        return self.groupby(groupvar)[measurevar].agg(['count','min','mean','median','std','max'])
    if functions_detailled:
        return self.groupsummaryscc(groupvar,measurevar,confint,
                   cut,is_bucket,group_keys,**kwargs)
    if boostrap:
        return self.groupsummarybc(groupvar,measurevar,confint,nsamples,
                   cut, quantile,is_bucket, **kwargs)
    return self.groupsummarysc(groupvar,measurevar,confint,cut, 
    quantile,is_bucket ,**kwargs)


pd.DataFrame.groupsummary = groupsummary

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

def psummary(self,manymissing_p = 0.70,nzv_freq_cut = 95/5, nzv_unique_cut = 10,
threshold = 100):
    """ This function will print you a summary of the dataset, based on function 
    designed is this package 
    - Argument : pandas.Dataframe
    - Output : python print 
    """
    print 'there are {0} row duplicates\n'.format(self.duplicated().sum())
    print 'these colums have mixed type :\n{0} \n'.format(list(cserie(self.structure().str.contains('mixed'))))
    print 'the columns with more than {0}% manymissing values:\n{1} \n'.format(100 * manymissing_p,
    list(self.manymissing(manymissing_p)))
    print 'the detected keys of the dataset are:\n{0} \n'.format(list(self.detectkey()))
    print 'the duplicated columns of the dataset are:\n{0}\n'.format(self.findupcol(threshold = 100))
    print 'the constant columns of the dataset are:\n{0}\n'.format(list(self.constantcol()))
    print 'the columns with nearzerovariance are:\n{0}'.format(
    list(cserie(self.nearzerovar(nzv_freq_cut,nzv_unique_cut,save_metrics =True).nzv)))
    # print 'the columns highly correlated are:\n{0}'.format(self.findcorr(data_frame = False))

pd.DataFrame.psummary = psummary

def clean_df(self,manymissing_p = 0.9,drop_col = None,filter_constantcol = True):
    """ 
    Basic cleaning of the data by deleting manymissing columns, 
    constantcol and drop_col specified by the user

    """
    col_to_remove = []
    if manymissing_p:
        col_to_remove += list(self.manymissing(manymissing_p))
    if filter_constantcol:
        col_to_remove += list(self.constantcol())
    if isinstance(drop_col,list):
        col_to_remove += drop_col
    elif isinstance(drop_col,str):
        col_to_remove += [drop_col]
    else :
        pass
    return self.drop(pd.unique(col_to_remove),axis = 1)


pd.DataFrame.clean_df = clean_df   
        
#########################################################
# Time Series Analysis 
#########################################################

#########################################################
# Unclassified 
#########################################################


def common_cols(df1,df2):
    """ Return the intersection of commun columns name """
    return list(set(df1.columns) & set(df2.columns))


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

    test = pd.read_csv('lc_test.csv')
    test['na_col'] = np.nan
    test['constant_col'] = 'constant'
    test['duplicated_column'] = test.id
    # test.int_rate = test.int_rate.str.findall(re.compile(r'\d+.\d+')).str.get(0)
    # test.int_rate = test.int_rate.astype(float)
    
    #########################################################
    # Testing the functions 
    #########################################################
    
    test.nacount()
    test.nacount(axis = 0)
    test.manymissing(0.5)
    test.nrow()
    test.ncol()
    test.detectkey()
    test.findupcol()
    test.finduprow
    test.dfquantiles(20)
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
