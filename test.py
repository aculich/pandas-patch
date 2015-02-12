# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:33:16 2015

@author: efourrier

Purpose : Automated test suites with unitest
run "python -m unittest -v test" in the module directory to run the tests 
"""

#########################################################
# Import Packages and helpers 
#########################################################

import unittest2 as unittest
from pandas_patch import *
import numpy as np 


flatten_list = lambda x: [y for l in x for y in flatten_list(l)] if isinstance(x,list) else [x]
#########################################################
# Writing the tests  
#########################################################

class TestPandasPatch(unittest.TestCase):

    def setUp(self):
        """ Creating test datasets """
        test_df = pd.read_csv('lc_test.csv')
        test_df['na_col'] = np.nan
        test_df['constant_col'] = 'constant'
        test_df['duplicated_column'] = test_df.id
        test_df['many_missing_70'] = [1]*300 + [np.nan] * 700
        test_df['num_var'] = range(test_df.shape[0])
        self.df_one = test_df
        self.nacolcount = self.df_one.nacolcount()
        self.narowcount = self.df_one.narowcount()
        self.manymissing = self.df_one.manymissing(0.7)
        self.constantcol = self.df_one.constantcol()
        self.dfnum = self.df_one.dfnum()
        self.detectkey = self.df_one.detectkey()
        self.findupcol = flatten_list(self.df_one.findupcol())
    
    def test_nrow(self):
        self.assertEqual(self.df_one.nrow(),self.df_one.shape[0])
    
    def test_col(self):
        self.assertEqual(self.df_one.ncol(),self.df_one.shape[1])
        
    def test_nacolcount_capture_na(self):
        self.assertEqual(self.nacolcount.loc['na_col','Napercentage'],1.0)
        self.assertEqual(self.nacolcount.loc['many_missing_70','Napercentage'],0.7)
        
    def test_nacolcount_is_type_dataframe(self):
        self.assertIsInstance(self.nacolcount,pd.core.frame.DataFrame)
    
    def test_narowcount_capture_na(self):
        self.assertEqual(sum(self.narowcount['Nanumber'] > 0),self.df_one.nrow())
    
    def test_narowcount_is_type_dataframe(self):
        self.assertIsInstance(self.narowcount,pd.core.frame.DataFrame)
    
    def test_manymissing_capture(self):
        self.assertIn('many_missing_70',self.manymissing)
        self.assertIn('na_col',self.manymissing)
    
    def test_constant_col_capture(self):
        self.assertIn('constant_col',self.constantcol)
    
    def test_dfnum_check_col(self):
        self.assertNotIn('constant_col', self.dfnum)
        self.assertIn('num_var', self.dfnum)
        self.assertIn('many_missing_70', self.dfnum)
        
    def test_detectkey_check_col(self):
        self.assertIn('id', self.detectkey)
    
    def test_findupcol_check(self):
        self.assertIn('duplicated_column',self.findupcol)
        self.assertIn('id',self.findupcol)
        self.assertNotIn('url',self.findupcol)

    def tearDown(self):
        """ Cleaning the environnement """
        self.df_one = None


# Adding new tests sets 
#def suite():
#    suite = unittest.TestSuite()
#    suite.addTest(TestPandasPatch('test_default_size'))
#    return suite
# Other solution than calling main 

#suite = unittest.TestLoader().loadTestsFromTestCase(TestPandasPatch)
#unittest.TextTestRunner(verbosity = 1 ).run(suite)

if __name__ == "__main__":
    unittest.main()