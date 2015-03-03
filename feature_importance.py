# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pandas_patch import *


class FeatureImportance:
	
	def __init__(self, df, resp):
        	self.dataframe = df
		self.response = resp
		self.predictors = pd.Series(self.dataframe.columns)
		self._rf_imp = []
		self._boosting_imp = []
		self._rpe_imp = []
		self._kbest_imp = []
		self._rf_param = []
		self._boosting_param = []
		self._rpe_param = []
		self._kbest_param = []

	def rf(self, n_estimators=500, criterion='gini', max_features='auto'):
		""" Returns the importances calculated by a random forest classifier.
		
		To make the method more effective, the result is stored in a private 
		property, so if it is used with the same parameters again, it will only
		have to print the result.

		Parameters:
		* n_estimators: number of trees in the forest
		* criterion: optimization criterion when building the trees.
			'gini' (default) for Gini impurity
			'entropy' for the information gain
		* max_features: number of features to select at each split
		
		 """
		if self._rf_param == [n_estimators, criterion, max_features]:
			return pd.DataFrame({'Predictors': self.predictors, 'RF': self._rf_imp})
		else:
			self._rf_param = [n_estimators, criterion, max_features]
			if max_features == 'auto':
				model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=int(np.sqrt(len(self.dataframe.columns))), bootstrap=True).fit(self.dataframe, self.response)
			else:
				model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features, bootstrap=True).fit(self.dataframe, self.response)

			self._rf_imp = model.feature_importances_/model.feature_importances_.max()
			return pd.DataFrame({'Predictors': self.predictors, 'RF': self._rf_imp})
		
	def boosting(self, n_estimators=2000, learning_rate=.1, max_depth=1):
		""" Returns the importance calculated by a gradient boosting classifier.

		To make the method more effective, the result is stored in a private 
		property, so if it is used with the same parameters again, it will only
		have to print the result.

		Parameters:
		* n_estimators: number of boosting stages to perform
		* learning_rate: coefficient by which shrink the contribution of each tree
		* max_depth: maximum depth of the individual regression estimators

		"""
		if self._boosting_param == [n_estimators, learning_rate, max_depth]:
			return pd.DataFrame({'Predictors': self.predictors, 'Boosting': self._boosting_imp})
		else:
			self._boosting_param = [n_estimators, learning_rate, max_depth]
			model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth).fit(self.dataframe, self.response)
			self._boosting_imp = model.feature_importances_/model.feature_importances_.max()
			return pd.DataFrame({'Predictors': self.predictors, 'Boosting': self._boosting_imp})

	def kbest(self, score_func=f_classif, k='all'):
		""" Returns the scores of the k best predictors according to the ANOVA 
		One-way F-test.

		To make the method more effective, the result is stored in a private 
		property, so if it is used with the same parameters again, it will only
		have to print the result.

		Parameters:
		* score_func: scoring function. use f_classif or chi2 for classification
		* k: number of parameters to rank. If 'all' is given, the function will
		returns the score for all predictors

		"""
		if self._kbest_param == [score_func, k]:
			return pd.DataFrame({'Predictors': self.predictors, 'KBest': self._kbest_imp})
		else:
			self._kbest_param = [score_func, k]
			kb = SelectKBest(score_func=score_func, k=k).fit(self.dataframe, self.response)
			self._kbest_imp = pd.Series(['Ranked below k']*len(self.predictors))
			self._kbest_imp[ kb.get_support() ] = kb.scores_
			return pd.DataFrame({'Predictors': self.predictors, 'KBest': self._kbest_imp})

	def rpe(self, cutoff=.90, method='pearson'):
		""" Returns a series of boolean stating whether the corresponding predictor
		remains after performing a recursive pairwise elimination.

		To make the method more effective, the result is stored in a private 
		property, so if it is used with the same parameters again, it will only
		have to print the result.

		Parameters:
		* cutoff: correlation cutoff to stop the algorithm
		* method: method to compute the correlation matrix
			'peason'
			'kendall'
			'spearman'

		"""
		if self._rpe_param == [cutoff, method]:
			return pd.DataFrame({'Predictors': self.predictors, 'RPE': self._rpe_imp})
		else:
			self._rpe_param = [cutoff, method]
			rpe_list = findcorr(self.dataframe, cutoff=cutoff, method=method)
			self._rpe_imp = self.predictors.apply(lambda x: (x not in rpe_list))
			return pd.DataFrame({'Predictors': self.predictors, 'RPE': self._rpe_imp})

	def summary(self):
		""" Returns a dataframe with the result of all the methods. """
		return self.rf().merge(self.boosting(), on='Predictors').merge(self.kbest(), on='Predictors').merge(self.rpe(), on='Predictors')

	pass

