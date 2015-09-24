#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

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
import numpy as np 
from pandas import DataFrame
from numpy.random import permutation
from numpy.random import choice 
from .utils import bootstrap_ci

# bug of 0.16 verison

# taking the duplicated version of 0.15.2



#########################################################
# Private Helpers 
#########################################################



# Find a better way to do it ( core pandas implementation)
# cserie return the column name with bool = True for a Serie of boolean 
def cserie(serie):
    return serie[serie].index.tolist()

    
#########################################################
# Data cleaning and exploration helpers 
#########################################################


# Helpers 

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

pd.DataFrame.nrow = nrow

def ncol(self):
    """ return the number of cols """
    return len(self.columns)

pd.DataFrame.ncol = ncol

def dfnum(self,index = False):
    """ Return columns with numeric type, the output is a list of columns if index 
    = False else a pandas Series of booleans  """
    if index :
        return (self.dtypes == float)|(self.dtypes == int)
    else :
        return cserie((self.dtypes == float)|(self.dtypes == int))

pd.DataFrame.dfnum = dfnum 

def dfchar(self,index = False):
    """ Return columns with numeric type, the output is a list of columns if index 
     = True else a pandas Series of booleans """
    if index :
        return self.dtypes == object
    else :
        return cserie(self.dtypes == object)

pd.DataFrame.dfchar = dfchar


def factors(self,nb_max_levels = 10,threshold_value = None, index = False):
    """ return a list of the detected factor variable, detection is based on 
    ther percentage of unicity perc_unique = 0.05 by default.
    We follow here the definition of R factors variable considering that a 
    factor variable is a character variable that take value in a list a levels

    this is a bad implementation 


    Arguments 
    ----------
    nb_max_levels: the mac nb of levels you fix for a categorical variable
    threshold_value : the nb of of unique value in percentage of the dataframe length
    index : if you want the result as an index or a list

     """
    if threshold_value:
        max_levels = max(nb_max_levels,threshold_value * self.nrow())
    else:
        max_levels = nb_max_levels
    def helper_factor(x,num_var = self.dfnum()):
        unique_value = set()
        if x.name in num_var:
            return False
        else:
            for e in x.values:
                if len(unique_value) >= max_levels :
                    return False
                else:
                    unique_value.add(e)
            return True
    

    if index:
        return self.apply(lambda x:  helper_factor(x))
    else :
        return cserie(self.apply(lambda x:  helper_factor(x)))

pd.DataFrame.factors = factors

def nacount(self,axis = 0):
    """ count the number of missing values per columns or rows of the dataframe

    Arguments
    ----------
    - axis = 1 if you want count the na values per rows, default 0 per columns.

    """
    if axis == 0:
        Serie =  self.isnull().sum(axis = 0)
        df =  DataFrame(Serie,columns = ['Nanumber'])
        df['Napercentage'] = df['Nanumber']/(self.nrow())
    if axis == 1:
        Serie = self.isnull().sum(axis = 1)
        df =  DataFrame(Serie,columns = ['Nanumber'])
        df['Napercentage'] = df['Nanumber']/(self.ncol())
    return df 

pd.DataFrame.nacount = nacount


def manymissing(self,a=0.9,axis = 0):
    """ identify columns of a dataframe with many missing values ( >= a), if 
    axis = 1 row either
    - the output is a pandas index """
    if axis == 0:
        self = self.nacount(axis = 0)
    if axis == 1:
        self = self.nacount(axis = 1)
    return self[self['Napercentage'] >= a].index
    
pd.DataFrame.manymissing = manymissing

def constantcol(self,dropna = True,**kwargs):
    """ identify constant columns """
    # sample to reduce computation time 
    def helper_cc(x):
        l_unique = x.nunique()
        return (l_unique == 1 or l_unique == 0) 
    if dropna:
        col_to_keep = self.sample_df(**kwargs).apply(lambda x: helper_cc(x),axis = 0 )
        if len(cserie(col_to_keep)) == 0:
            return []
        return cserie(self.loc[:,col_to_keep].apply(lambda x: helper_cc(x) ,axis = 0 ))
    else :
        col_to_keep = self.sample_df(**kwargs).apply(lambda x: len(x.unique()) == 1 ,axis = 0 )
        if len(cserie(col_to_keep)) == 0:
            return []
        return cserie(self.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == 1 ,axis = 0 ))

    
pd.DataFrame.constantcol = constantcol


def count_unique(self):
    """ Return a serie with the number of unique value per columns """
    return self.apply(lambda x: x.nunique(), axis = 0)

pd.DataFrame.count_unique = count_unique

def detectkey(self, index_format = False, pct = 0.15,dropna = False,**kwargs):
    """ identify id or key columns as an index if index_format = True or 
    as a list if index_format = False """
    

    if not dropna:
        col_to_keep = self.sample_df(pct = pct,**kwargs).apply(lambda x: len(x.unique()) == len(x) ,axis = 0)
        if len(col_to_keep) == 0:
            return []
        is_key_index = col_to_keep
        is_key_index[is_key_index] == self.loc[:,is_key_index].apply(lambda x: len(x.unique()) == len(x) ,axis = 0)
        if index_format:
            return is_key_index
        else :
            return cserie(is_key_index)
    else :
        col_to_keep = self.sample_df(pct = pct,**kwargs).apply(lambda x: x.nunique() == len(x.dropna()) ,axis = 0)
        if len(col_to_keep) == 0:
            return []
        is_key_index = col_to_keep
        is_key_index[is_key_index] == self.loc[:,is_key_index].apply(lambda x: x.nunique() == len(x.dropna()),axis = 0)
        if index_format:
            return is_key_index
        else :
            return cserie(is_key_index)


    
pd.DataFrame.detectkey = detectkey



def structure(self,threshold_factor = 10):
    """ this function return a summary of the structure of the pandas DataFrame 
    data looking at the type of variables, the number of missing values, the 
    number of unique values """
    
    dtypes = self.dtypes
    nacolcount = self.nacount(axis = 0)
    nb_missing = nacolcount.Nanumber
    perc_missing = nacolcount.Napercentage
    nb_unique_values = self.count_unique()
    dtypes_r = self.apply(lambda x: "character")
    dtypes_r[self.dfnum(index = True)] = "numeric"
    dtypes_r[(dtypes_r == 'character') & (nb_unique_values <= threshold_factor)] = 'factor'
    constant_columns = (nb_unique_values == 1)
    na_columns = (perc_missing == 1)
    is_key = nb_unique_values == self.nrow()
    # is_key_na = ((nb_unique_values + nb_missing) == self.nrow()) & (~na_columns)
    dict_str = {'dtypes_r': dtypes_r,'perc_missing': perc_missing,
    'nb_missing': nb_missing,'is_key': is_key,
    'nb_unique_values': nb_unique_values,'dtypes': dtypes,
    'constant_columns': constant_columns, 'na_columns': na_columns}
    df =  pd.concat(dict_str,axis =1)
    return df.loc[:,['dtypes','dtypes_r','nb_missing','perc_missing',
    'nb_unique_values','constant_columns','na_columns','is_key']]

pd.DataFrame.structure = structure

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


def serie_quantiles(array,nb_quantiles = 10):
    binq = 1.0/nb_quantiles
    if type(array) == pd.Series:
        return array.quantile([binq*i for i in xrange(nb_quantiles +1)])
    elif type(array) == np.ndarray:
        return np.percentile(array,[binq*i for i in xrange(nb_quantiles +1)])
    else :
        raise("the type of your array is not supported")

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




def nearzerovar(self, freq_cut = 95/5, unique_cut = 10, save_metrics = False):
    """ identify predictors with near-zero variance. 
            freq_cut: cutoff ratio of frequency of most common value to second 
            most common value.
            unique_cut: cutoff percentage of unique value over total number of 
            samples.
            save_metrics: if False, print dataframe and return NON near-zero var 
            col indexes, if True, returns the whole dataframe.
    """
    nb_unique_values = self.count_unique()
    percent_unique = 100 * nb_unique_values/self.nrow()

    def helper_freq(x):
        if nb_unique_values[x.name] == 0:
            return 0.0
        elif nb_unique_values[x.name] == 1:
            return 1.0
        else:
            return float(x.value_counts().iloc[0])/x.value_counts().iloc[1] 

    freq_ratio = self.apply(helper_freq)

    zerovar = (nb_unique_values == 0) | (nb_unique_values == 1) 
    nzv = ((freq_ratio >= freq_cut) & (percent_unique <= unique_cut)) | (zerovar)

    if save_metrics:
        return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.columns)
    else:
        print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.columns))
        return nzv[nzv == True].index 

pd.DataFrame.nearzerovar = nearzerovar


def findcorr(self, cutoff=.90, method='pearson', data_frame=False, printcor = False):
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
    if printcor:
        print(max_cor.max())
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
        if printcor:
            print(max_cor.max())
        
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


# def groupsummarysc(self,groupvar,measurevar,confint=0.95,cut = False, 
#     quantile = 5,is_bucket = False,**kwargs):
#     """ provide a summary of measurevar groupby groupvar with student conf interval.
#     measurevar and groupvar are list of column names
#     if you want bucket of equal length instead of quantile put is_bucket = True """
#     def student_ci(x):
#         return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwargs)[1]
#     df = self.copy()
#     functions = ['count','min',('ci_low',lambda x: np.mean(x) - student_ci(x)),
#      'mean',('ci_up',lambda x: np.mean(x) + student_ci(x)),'median',('se',se),'std','max']
#     if cut:
#         for var in groupvar:
#             if is_bucket:
#                 df[var] = pd.cut(df[var],bins = quantile)
#             else: 
#                 df[var] = pd.qcut(df[var],q = quantile)
#     return df.groupby(groupvar)[measurevar].agg(functions)

# pd.DataFrame.groupsummarysc = groupsummarysc


def groupsummarybc(self,groupvar,measurevar,confint=0.95,nsamples = 500,
                   cut = False, quantile = 5,is_bucket = False, **kwargs):
    """ provide a summary of measurevar groupby groupvar with bootstrap conf interval.
    measurevar and groupvar are list of column names. You have a cut functionnality 
    if you want to cut the groupvar
    if you want bucket of equal length instead of quantile put is_bucket = True """
    ci_low = lambda x : bootstrap_ci(x.values,n = nsamples,ci = confint)[0]
    ci_up = lambda x : bootstrap_ci(x.values,n = nsamples,ci = confint)[1]
    df = self.copy()
    functions = ['count','min',('ci_low',lambda x : bootstrap_ci(x.values,n = nsamples,ci = confint)[0]),
    'mean',('ci_up',lambda x : bootstrap_ci(x.values,n = nsamples,ci = confint)[1]),'median','std','max']
    if cut:
        if isinstance(groupvar,list):
            for var in groupvar:
                if is_bucket:
                    df[var] = pd.cut(df[var],bins = quantile)
                else: 
                    df[var] = pd.qcut(df[var],q = quantile)
        elif isinstance(groupvar,str):
                if is_bucket:
                    df[groupvar] = pd.cut(df[groupvar],bins = quantile)
                else: 
                    df[groupvar] = pd.qcut(df[groupvar],q = quantile)
        else:
            raise('groupvar is neither a string or a list please correct the type')
    return df.groupby(groupvar)[measurevar].agg(functions)



pd.DataFrame.groupsummarybc = groupsummarybc


# def groupsummaryscc(self,groupvar,measurevar,confint=0.95,
#                    cut = False, quantile = 5,is_bucket = False,**kwargs):
#     """ provide a more complete summary than groupsummarysc of measurevar
#     groupby groupvar with student conf interval.measurevar and groupvar
#     are list of column names 
#     if you want bucket of equal length instead of quantile put is_bucket = True """
#     def student_ci(x):
#         return se(x) * scipy.stats.t.interval(confint, len(x) - 1,**kwargs)[1]
#     self = self.copy()
#     functions = ['count','min', quantile_25,'mean',se,student_ci,
#     'median','std','mad',skewness ,kurtosis,quantile_75,'max']
#     # Correct the problem of unicity 
#     if cut:
#         for var in groupvar:
#             if is_bucket:
#                 self.loc[:,var] = pd.cut(self.loc[:,var],bins = quantile)
#             else: 
#                 self.loc[:,var] = pd.qcut(self.loc[:,var],q = quantile)
            
#     # grouping, apply function and return results 
#     return self.groupby(groupvar)[measurevar].agg(functions)
    
# pd.DataFrame.groupsummaryscc = groupsummaryscc


# This is the global function that the user will be using 
# def groupsummary(self,groupvar,measurevar,confint=0.95,fast_summary =False,
#                 detailled_summary = False,boostrap = False,
#                    cut = False, quantile = 5,is_bucket = False,group_keys = False,**kwargs):
#     self = self.copy()
#     if fast_summary:
#         return self.groupby(groupvar)[measurevar].agg(['count','min','mean','median','std','max'])
#     if functions_detailled:
#         return self.groupsummaryscc(groupvar,measurevar,confint,
#                    cut,is_bucket,group_keys,**kwargs)
#     if boostrap:
#         return self.groupsummarybc(groupvar,measurevar,confint,nsamples,
#                    cut, quantile,is_bucket, **kwargs)
#     return self.groupsummarysc(groupvar,measurevar,confint,cut, 
#     quantile,is_bucket ,**kwargs)


# pd.DataFrame.groupsummary = groupsummary

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

def iqr(ndarray):
    return np.percentile(ndarray,75) - np.percentile(ndarray,25)

def z_score(ndarray):
    return (ndarray - np.mean(ndarray))/(np.std(ndarray))

# More robust to outliers 
def iqr_score(ndarray):
    return (ndarray - np.median(ndarray))/(iqr(ndarray))

# median(abs(ndarray -median(ndarray))/0.6745)
def mad_score(ndarray):
    return (ndarray - np.median(ndarray))/(np.median(np.absolute(ndarray -np.median(ndarray)))/0.6745)

def fivenum(v):
    """ Returns Tukey's five number summary 
    (minimum, lower-hinge, median, upper-hinge, maximum)
    for the input vector, a list or array of numbers 
    based on 1.5 times the interquartile distance """
    md = np.median(v)
    whisker = 1.5*iqr(v)
    return min(v), md-whisker, md, md+whisker, max(v)

def check_negative_value_serie(serie):
    """ this function will detect if there is negative value and calculate the 
    ratio negative value/postive value
    """
    if serie.dtype == "object":
        TypeError("The serie should be numeric values")
    return sum(serie < 0)/sum(serie > 0)

def outlier_detection_serie_d(serie,scores = [z_score,iqr_score,mad_score],
    cutoff_zscore = 3,cutoff_iqrscore = 2,cutoff_mad = 2):
    if serie.dtype == 'object':
        raise("The variable is not a numeric variable")
    if len(serie.unique()) == 1:
        raise("The variable has a variance equal to zero")
    keys = [str(func.__name__) for func in scores]
    df = pd.DataFrame(dict((key,func(serie)) for key,func in zip(keys,scores)))
    df['is_outlier'] = 0
    if 'z_score' in keys :
        df.loc[np.absolute(df['z_score']) >= cutoff_zscore,'is_outlier'] = 1
    if 'iqr_score' in keys :
        df.loc[np.absolute(df['iqr_score']) >= cutoff_iqrscore,'is_outlier'] = 1
    if 'mad_score' in keys :
        df.loc[np.absolute(df['mad_score']) >= cutoff_mad,'is_outlier'] = 1
    return df 

def outlier_detection_d(self,subset = None,remove_constant_col = True,
                    scores = [z_score,iqr_score,mad_score],
                      cutoff_zscore = 3,cutoff_iqrscore = 2,cutoff_mad = 2):
    """ Return a dictionnary with z_score,iqr_score,mad_score as keys and the 
    associate dataframe of distance as value of the dictionnnary"""
    df = self.copy()
    if subset:
        df = df.drop(subset,axis = 1)
    df = df[df.dfnum()] # take only numeric variable 
    if remove_constant_col:
        df = df.drop(df.constantcol(), axis = 1) # remove constant variable 
    df_outlier = pd.DataFrame()
    for col in df:
        df_temp = outlier_detection_serie_d(df[col],scores,cutoff_zscore,
            cutoff_iqrscore,cutoff_mad)
        df_temp.columns = [col + '_' + col_name for col_name in df_temp.columns]
        df_outlier = pd.concat([df_outlier,df_temp],axis = 1)
    return df_outlier

    
pd.DataFrame.outliers_detection_d =  outlier_detection_d


def mahalonobis_distance(self,subset = None):
    df = self[subset]
    if (df.dtypes == "object").any():
        raise("There is a non numeric variable in the subset")
    array_numpy = df.values
    

#########################################################
# Global summary and basic cleaning function  
#########################################################

def psummary(self,manymissing_ph = 0.70,manymissing_pl = 0.05,nzv_freq_cut = 95/5, nzv_unique_cut = 10,
threshold = 100,string_threshold = 40, dynamic = False):
    """ 
    This function will print you a summary of the dataset, based on function 
    designed is this package

    Arguments
    ---------
    manymissing_ph : sup cutoff to detect columns with manymissing values
    manymissing_ph : inf cutoff to detect columns with few missing values 
    zv_freq_cut : 
    nzv_unique_cut : 
    threshold : threshold for sampling to improve performance 
    string_threshold : threshold for the length of string of character columns to detect big strings
    dynamic : dynamic printing (print one element by one, useful for big dataset)


    Returns
    -------
    python print

    """
    nacolcount_p = self.nacount(axis = 0).Napercentage
    if dynamic:
        print('there are {0} duplicated rows\n'.format(self.duplicated().sum()))
        print('the columns with more than {0:.2%} manymissing values:\n{1} \n'.format(manymissing_ph,
        cserie((nacolcount_p > manymissing_ph))))

        print('the columns with less than {0:.2%} manymissing values are :\n{1} \n you should fill them with median or most common value \n'.format(
        manymissing_pl,cserie((nacolcount_p > 0 ) & (nacolcount_p <= manymissing_pl))))

        print('the detected keys of the dataset are:\n{0} \n'.format(self.detectkey()))
        print('the duplicated columns of the dataset are:\n{0}\n'.format(self.findupcol(threshold = 100)))
        print('the constant columns of the dataset are:\n{0}\n'.format(self.constantcol()))

        print('the columns with nearzerovariance are:\n{0}\n'.format(
        list(cserie(self.nearzerovar(nzv_freq_cut,nzv_unique_cut,save_metrics =True).nzv))))
        print('the columns highly correlated to others to remove are:\n{0}\n'.format(
        self.findcorr(data_frame = False)))
        print('these columns contains big strings :\n{0}\n'.format(
            cserie(self.df_len_string() > string_threshold)))
    else:
        dict_info = {'nb_duplicated_rows': sum(self.duplicated()),
                    'many_missing_percentage': manymissing_ph,
                    'manymissing_columns': cserie((nacolcount_p > manymissing_ph)),
                    'low_missing_percentage': manymissing_pl,
                    'lowmissing_columns': cserie((nacolcount_p > 0 ) & (nacolcount_p <= manymissing_pl)),
                    'keys_detected': self.detectkey(),
                    'dup_columns': self.findupcol(threshold = 100),
                    'constant_columns': self.constantcol(),
                    'nearzerovar_columns' : cserie(self.nearzerovar(nzv_freq_cut,nzv_unique_cut,save_metrics =True).nzv),
                    'high_correlated_col' : self.findcorr(data_frame = False),
                    'big_strings_col': cserie(self.df_len_string() > string_threshold)
                    } 

        string_info = u"""
there are {nb_duplicated_rows} duplicated rows\n
the columns with more than {many_missing_percentage:.2%} manymissing values:\n{manymissing_columns} \n
the columns with less than {low_missing_percentage:.2%}% manymissing values are :\n{lowmissing_columns} \n
you should fill them with median or most common value\n
the detected keys of the dataset are:\n{keys_detected} \n
the duplicated columns of the dataset are:\n{dup_columns}\n
the constant columns of the dataset are:\n{constant_columns}\n
the columns with nearzerovariance are:\n{nearzerovar_columns}\n
the columns highly correlated to others to remove are:\n{high_correlated_col}\n
these columns contains big strings :\n{big_strings_col}\n
        """.format(**dict_info)

        print(string_info)


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


def infertype(self):
    """ this function will try to infer the dtype of the columns"""
    return self.apply(lambda x: pd.lib.infer_dtype(x.values))

pd.DataFrame.infertype = infertype


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


#########################################################
# Deprecated
#########################################################

def nacolcount(self):
    """ count the number of missing values per columns """
    print("this function is deprecated please use nacount")
    Serie =  self.isnull().sum()
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.nrow())
    return df

pd.DataFrame.nacolcount = nacolcount

def narowcount(self):
    """ count the number of missing values per rows """
    print("this function is deprecated please use nacount")
    Serie = self.isnull().sum(axis = 1)
    df =  DataFrame(Serie,columns = ['Nanumber'])
    df['Napercentage'] = df['Nanumber']/(self.ncol())
    return df

pd.DataFrame.narowcount = narowcount

