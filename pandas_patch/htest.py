"""
@author: efourrier

Purpose : This module is designed to provide custom test assertion return in True or
False so you can quickly include them into tests against your DataFrame

"""

#########################################################
# Import Packages and Constants
#########################################################

import pandas as pd 
from pandas_patch import * 
import numpy as np 




#########################################################
# Custom assert function 
#########################################################

def isna(df,index,axis = 1):
		""" This function will return True if there is na values in the index of the DataFrame

		Parameters
		-----------

		index : index can be a list of columns or a rows index

		 """
		return pd.isnull(df[index]).any().any()

def is_nacolumns(df,columns):
	""" This function will return True if at least one of the columns is composed only of missing values """
	return pd.isnull(test[columns]).all().any()

def is_positive(df,columns):
	""" Return True if all columns are positive """
	return (df[columns] > 0).all().all()

def is_compact(df,column,inf = None,sup = None ):
	""" Return True if the column meet the inf and sup criteria """

	if inf is None :
		 (df[column] <= sup).all().all()
	elif sup is None:
		 (df[column] <= sup).all().all()
	else : 
		return ((df[column] >= inf) & (df[column] <= sup)).all().all()

def is_many_missing 
def is_key(df,columns):
	return df[columns].apply(lambda col : len(pd.is_unique(col)) == len(df.index)).all()

def is_constant(df,columns):
	return df[columns].apply(lambda col : len(pd.is_unique(col)) == 1).all()

def is_numeric(df,columns):

def is_mixed_uni_str(df):
	""" Return True if there is str type (byte in python 2.7) and unicode """
	
	types = set(df.apply(lambda x :pd.lib.infer_dtype(x.values)))
	if 'unicode' in types and 'string' in types:
		return True

def is_unicode(df,columns):
	return df[columns].apply(lambda x :pd.lib.infer_dtype(x.values) == unicode).any()
