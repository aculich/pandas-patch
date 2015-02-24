# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:33:16 2015

@author: efourrier

Purpose : Automated test suites with unitest
run "python -m unittest -v test" in the module directory to run the tests 

This is not a clean coding but it is way faster than using a setup function
"""

#########################################################
# Import Packages and helpers 
#########################################################

import unittest2 as unittest
from pandas_patch import *
import numpy as np 

test_df = pd.read_csv('lc_test.csv')
test_df['na_col'] = np.nan
test_df['constant_col'] = 'constant'
test_df['duplicated_column'] = test_df.id
test_df['many_missing_70'] = [1]*300 + [np.nan] * 700
test_df['num_var'] = range(test_df.shape[0])
flatten_list = lambda x: [y for l in x for y in flatten_list(l)] if isinstance(x,list) else [x]

#flatten_list = lambda x: [y for l in x for y in flatten_list(l)] if isinstance(x,list) else [x]
#########################################################
# Writing the tests  
#########################################################

class TestPandasPatch(unittest.TestCase):

    def test_sample_df(self):
        self.assertEqual(len(test_df.sample_df(pct = 0.061)),
                         0.061 * float(test_df.shape[0]))

    def test_nrow(self):
        self.assertEqual(test_df.nrow(),test_df.shape[0])
    
    def test_col(self):
        self.assertEqual(test_df.ncol(),test_df.shape[1])
        
    def test_nacolcount_capture_na(self):
        nacolcount = test_df.nacolcount()
        self.assertEqual(nacolcount.loc['na_col','Napercentage'],1.0)
        self.assertEqual(nacolcount.loc['many_missing_70','Napercentage'],0.7)
        
    def test_nacolcount_is_type_dataframe(self):
        self.assertIsInstance(test_df.nacolcount(),pd.core.frame.DataFrame)
    
    def test_narowcount_capture_na(self):
        narowcount = test_df.narowcount()
        self.assertEqual(sum(narowcount['Nanumber'] > 0),test_df.nrow())
    
    def test_narowcount_is_type_dataframe(self):
        narowcount = test_df.narowcount()
        self.assertIsInstance(narowcount,pd.core.frame.DataFrame)
    
    def test_manymissing_capture(self):
        manymissing = test_df.manymissing(0.7)
        self.assertIn('many_missing_70',manymissing)
        self.assertIn('na_col',manymissing)
    
    def test_constant_col_capture(self):
        constantcol = test_df.constantcol()
        self.assertIn('constant_col',constantcol)
    
    def test_dfnum_check_col(self):
        dfnum = test_df.dfnum()
        self.assertNotIn('constant_col', dfnum)
        self.assertIn('num_var', dfnum)
        self.assertIn('many_missing_70',dfnum)
        
    def test_detectkey_check_col(self):
        detectkey = test_df.detectkey()
        self.assertIn('id', detectkey)
    
    def test_findupcol_check(self):
        findupcol = test_df.findupcol()
        self.assertIn(['loan_amnt', 'funded_amnt'],findupcol)
        self.assertIn(['id', 'duplicated_column'],findupcol)
        self.assertNotIn('member_id',flatten_list(findupcol))
        
    def test_clean_df(self):
        clean_df = test_df.clean_df().columns
        self.assertTrue(all([e not in clean_df for e in ['constant_col',
                            'na_col','duplicated_column']]))
        self.assertIn('id',clean_df)


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