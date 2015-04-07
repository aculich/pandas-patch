import pandas_patch
from pandas_patch.utils import get_test_df_complete

test = get_test_df_complete()
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
#test.groupsummarys(['grade','sub_grade'],['fico_range_high','dti'])
#test.groupsummaryd(['grade'],['fico_range_high'])
#test.groupsummarysc(['grade'],['fico_range_high'])
test.groupsummarybc(['grade'],['fico_range_high'])
test.groupsummarybc(['grade'],['fico_range_high'])
# or other solution for the use ok kwargs 

# test.groupsummaryscc(['grade'],['fico_range_high'])
# test.groupsummaryscc(['fico_range_high'],['dti'],cut =True)
test.group_average('grade','int_rate','loan_amnt')