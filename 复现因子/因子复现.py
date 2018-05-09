
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Apr 23 11:25:52 2018
    
    @author: adam
    """

#--------------------------------------------------------
#import

import os
import numpy as np
import pandas as pd
import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs_fxdayu.data.dataservice import LocalDataService

import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------
#define

start = 20170101
end = 20180101
factor_list  = ['BBI','RVI','Elder','ChaikinVolatility','EPS','PE','PS','ACCA','CTOP','MA10RegressCoeff12','AR','BR','ARBR','np_parent_comp_ttm','total_share','bps']
check_factor = ','.join(factor_list)

dataview_folder = r'/Users/adam/Desktop/intern/test5/fxdayu_adam/data'
ds = LocalDataService(fp = dataview_folder)

SH_id = ds.query_index_member("000001.SH", start, end)
SZ_id = ds.query_index_member("399106.SZ", start, end)
stock_symbol = list(set(SH_id)|set(SZ_id))

dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
    'fields': check_factor,
        'freq': 1,
            "prepare_fields": True}

dv = DataView()
dv.init_from_config(dv_props, data_api=ds)
dv.prepare_data()
def alpha106():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    alpha106 = dv.add_formula('alpha106', 'close_adj-Delay(close_adj,20)', is_quarterly=False)
    return alpha106

def alpha127():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    alpha127 = dv.add_formula('alpha127', '((100*(close_adj-Ts_Max(close_adj,12))/(Ts_Max(close_adj,12)))^2)^(1/2)', is_quarterly=False)
    return alpha127

def alpha62():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    alpha62 = dv.add_formula('alpha62', 'Rank(Decay_linear(Rank(Correlation((low_adj),Ts_Mean(volume,80), 8)), 17)) * -1', is_quarterly=False)
    return alpha62

def NPParentCompanyGrowRate():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    dv.add_field('NPParentCompanyGrowRate',data_api=ds)
    return dv.get_ts('NPParentCompanyGrowRate')

def NOCFToOperatingNI():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    dv.add_field('NOCFToOperatingNI',data_api=ds)
    return dv.get_ts('NOCFToOperatingNI')

def BearPower():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    dv.add_field('BearPower',data_api = ds)
    return dv.get_ts('BearPower')

def DHILO():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    dv.add_field('DHILO',data_api=ds)
    return dv.get_ts('DHILO')

def alpha25():
    '''
        input: void
        output: pd.DataFrame,
        factor_values , Index is trade_date, columns are symbols.
        '''
    alpha25 = dv.add_formula('alpha25', '((-1 * Rank((Delay(close_adj, 7) * (1 - Rank(Decay_linear((volume / Ts_Mean(volume,20)), 9)))))) * (1+Rank(Ts_Sum(close_adj/Delay(close_adj,1)-1, 250))))', is_quarterly=False, add_data=True)
    return alpha25














