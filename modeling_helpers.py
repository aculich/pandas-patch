#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun March 1 2015

@author: efourrier

Purpose : This is a framework for Modeling with pandas, numpy and skicit-learn.
The Goal of this module is to rely on a dataframe structure for modelling 

"""


#########################################################
# Import modules and global helpers 
#########################################################

import pandas as pd 
import numpy as np 
from numpy.random import permutation

cserie = lambda serie: serie[serie].index


#########################################################
# DataCleaner class
#########################################################

class DataCleaner(object):
	""" 
	this class focuses abd identifying bad data as duplicated columns,rows,
	manymissing columns, long text data, keys of a dataframe, string aberation detection.

    Parameters
    ----------
    data : a pandas dataframe


	* object.info() to get the a global snapchat of the different stuff detected
	* data_cleaned = object.basic_cleaning() to clean your data.
	"""


	def __init__(self,data):
		self.data = data
		self.__nrow = len(self.data.index)
		self.__ncol = len(self.data.columns)
		self._dfnum = self.data.columns[((self.data.dtypes == float)|(self.data.dtypes == int))]
		self._nacolcount = self.nacolcount()
		self._constantcol =self.constantcol(threshold = 100)

	def sample_df(self,pct = 0.05,nr = 10,threshold = None):
		""" sample a number of rows of a dataframe = min(max(0.05*nrow(self,nr),threshold)"""
		a = max(int(pct*float(len(self.data.index))),nr)
		if threshold:
			a = min(a,threshold)
		return self.data.loc[permutation(self.data.index)[:a]]

	def nacolcount(self):
		""" count the number of missing values per columns """
		Serie =  self.data.isnull().sum()
		df =  pd.DataFrame(Serie,columns = ['Nanumber'])
		df['Napercentage'] = df['Nanumber']/(self.__nrow)
		return df

	def manymissing(self,a = 0.9,row = False):
		""" identify columns of a dataframe with many missing values ( >= a), if
		row = True row either.
		- the output is a pandas index """
		if row:
			nacount = self.narowcount()
		else :
			nacount= self._nacolcount
		return nacount[nacount['Napercentage'] >= a].index

	def df_len_string(self):
		""" Return a Series with the max of the length of the string of string-type columns """
		return self.data.drop(self._dfnum,axis = 1).apply(lambda x : np.max(x.str.len()), axis = 0 )

	def detectkey(self, index_format = True, pct = 0.15,**kwargs):
		""" identify id or key columns as an index if index_format = True or 
		as a Serie if index_format = False """
		col_to_keep = self.sample_df(pct = 0.15,**kwargs).apply(lambda x: len(x.unique()) == len(x) ,axis = 0)
		if index_format:
			return cserie(self.data.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0))
		else :
			return self.data.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == len(x) ,axis = 0)

	def constantcol(self,**kwargs):
		""" identify constant columns """
		# sample to reduce computation time 
		col_to_keep = self.sample_df(**kwargs).apply(lambda x: len(x.unique()) == 1,axis = 0 )
		if len(cserie(col_to_keep)) == 0:
			return []
		return cserie(self.data.loc[:,col_to_keep].apply(lambda x: len(x.unique()) == 1,axis = 0 ))

	def findupcol(self,threshold = 100,**kwargs):
		""" find duplicated columns and return the result as a list of list """

		df_s = self.sample_df(threshold = 100,**kwargs).T
		dup_index_s = (df_s.duplicated()) | (df_s.duplicated(take_last = True))
		
		if len(cserie(dup_index_s)) == 0:
			return []

		df_t = (self.data.loc[:,dup_index_s]).T
		dup_index = df_t.duplicated()
		dup_index_complet = cserie((dup_index) | (df_t.duplicated(take_last = True)))

		l = []
		for col in cserie(dup_index):
			index_temp = self.data[dup_index_complet].apply(lambda x: (x == self.data[col])).sum() == self.__nrow
			temp = list(self.data[dup_index_complet].columns[index_temp])
			l.append(temp)
		return l


	def finduprow(self,subset = []):
		""" find duplicated rows and return the result a sorted dataframe of all the
		duplicates
		subset is a list of columns to look for duplicates from this specific subset . 
		"""
		if subset:
			dup_index = (self.data.duplicated(subset = subset)) | (self.data.duplicated(subset = subset,take_last =True)) 
		else :    
			dup_index = (self.data.duplicated()) | (self.data.duplicated(take_last = True))
			
		if subset :
			return self.data[dup_index].sort(subset)
		else :
			return self.data[dup_index].sort(self.data.columns[0])

	def nearzerovar(self, freq_cut = 95/5, unique_cut = 10, save_metrics = False):
		""" 
		identify predictors with near-zero variance. 
		freq_cut: cutoff ratio of frequency of most common value to second 
		most common value.
		unique_cut: cutoff percentage of unique value over total number of 
		samples.
		save_metrics: if False, print dataframe and return NON near-zero var 
		col indexes, if True, returns the whole dataframe.
		"""

		percent_unique = self.data.apply(lambda x: float(100*len(x.unique()))/len(x), axis=0)
		freq_ratio = []
		for col in self.data.columns:
			if len(self.data[col].unique()) == 1:
				freq_ratio += [1]
			else:
				freq_ratio += [float(self.data[col].value_counts().iloc[0])/self.data[col].value_counts().iloc[1] ]

		zerovar = self.data.apply(lambda x: len(x.unique()) == 1, axis = 0)
		nzv = ((np.array(freq_ratio) >= freq_cut) & (percent_unique <= unique_cut)) | (percent_unique == 0)

		if save_metrics:
			return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns)
		else:
			print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns))
			return nzv[nzv == True].index 



	def findcorr(self, cutoff=.90, method='pearson', data_frame=False, print_mode = False):
		"""
		implementation of the Recursive Pairwise Elimination.        
		The function finds the highest correlated pair and removes the most 
		highly correlated feature of the pair, then repeats the process 
		until the threshold 'cutoff' is reached.
		
		will return a dataframe is 'data_frame' is set to True, and the list
		of predictors to remove oth		
		Adaptation of 'findCorrelation' function in the caret package in R. 
		"""
		res = []
		df = self.data.copy(0)
		cor = df.corr(method=method)
		for col in cor.columns:
			cor[col][col] = 0
		
		max_cor = cor.max()
		if print_mode:
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
			if print_mode:
				print (max_cor.max())
			
		if data_frame:
			return df.drop(res, 1)
		else:
			return res

	def info(self,manymissing_ph = 0.70,manymissing_pl = 0.05,nzv_freq_cut = 95/5, nzv_unique_cut = 10,
	threshold = 100,string_threshold = 40):
		""" 
		This function will print you a summary of the dataset, based on function 
		designed is this package 
		- Argument : pandas.Dataframe
		- Output : python print 

		"""
		nacolcount_p = self._nacolcount.Napercentage
		print 'the columns with more than {0}% manymissing values:\n{1} \n'.format(100 * manymissing_ph,
		list(cserie((nacolcount_p > manymissing_ph))))

		print 'the columns with less than {0}% manymissing values are :\n{1} \n you should fill them with median or most common value \n'.format(
		100 * manymissing_pl,list(cserie((nacolcount_p > 0 ) & (nacolcount_p <= manymissing_pl))))

		print 'the detected keys of the dataset are:\n{0} \n'.format(list(self.detectkey()))
		print 'the duplicated columns of the dataset are:\n{0}\n'.format(self.findupcol(threshold = 100))
		print 'the constant columns of the dataset are:\n{0}\n'.format(list(self._constantcol))

		print 'the columns with nearzerovariance are:\n{0}\n'.format(
		list(cserie(self.nearzerovar(nzv_freq_cut,nzv_unique_cut,save_metrics =True).nzv)))
		print 'the columns highly correlated to others to remove are:\n{0}\n'.format(
		self.findcorr(data_frame = False))
		print 'these columns contains big strings :\n{0}\n'.format(
			list(cserie(self.df_len_string() > string_threshold)))


	def basic_cleaning(self,manymissing_p = 0.9,drop_col = None,filter_constantcol = True):
		""" Basic cleaninf of the data by deleting manymissing columns, 
		constantcol and drop_col specified by the user """
		col_to_remove = []
		if manymissing_p:
			col_to_remove += list(self.manymissing())
		if filter_constantcol:
			col_to_remove += list(self.constantcol())
		if drop_col:
			col_to_remove += list(drop_col)
		return self.data.drop(list(set(col_to_remove)), axis = 1)






