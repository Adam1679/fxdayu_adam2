

# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
from jaqs_fxdayu.util import dp
from jaqs.data.dataapi import DataApi
import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs_fxdayu.data.dataservice import LocalDataService
import os
import numpy as np
api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("18523827661",
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk'
          )
start = 20100101
end = 20180401

SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))
factor_list = []
check_factor = ','.join(factor_list)
dataview_folder = r'/Users/adam/Desktop/intern/test5/fxdayu_adam/data'
dataview_folder2 = '/Users/adam/Desktop/intern/test5/fxdayu_adam/muti_factor/'
dv = DataView()
#ds = LocalDataService(fp=dataview_folder)
data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18523827661",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk"
}
ds = RemoteDataService()
ds.init_from_config(data_config)


# In[4]:


'''
    dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
    'fields': check_factor,
    'freq': 1,
    "prepare_fields": True,
    "benchmark":'000300.SH'}
    dv.init_from_config(dv_props, ds)
    dv.prepare_data()
    '''


# In[5]:


#dv.save_dataview(dataview_folder2)
dv.load_dataview(dataview_folder2)
dv.fields


# In[6]:


sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)


# In[76]:


pm = dv.add_formula('pm','tot_profit/float_mv',is_quarterly=False,add_data=True)
ETOP = dv.add_formula('ETOP','tot_profit/total_mv',is_quarterly=False,add_data=True)
roa = dv.add_formula('roa','roa',is_quarterly=True,add_data=True)
roe = dv.add_formula('roe','roe',is_quarterly=True,add_data=True)

import alpha32_,alpha194,alpha195,alpha42_,alpha62_,alpha64_,alpha197,alpha211,alpha56_

dv.append_df(alpha32_.run_formula(dv),'alpha32_')
dv.append_df(alpha194_.run_formula(dv),'alpha194_')
dv.append_df(alpha195_.run_formula(dv),'alpha195_')
dv.append_df(alpha42_.run_formula(dv),'alpha42_')
dv.append_df(alpha62_.run_formula(dv),'alpha62_')
dv.append_df(alpha64_.run_formula(dv),'alpha64_')
dv.append_df(alpha197_.run_formula(dv),'alpha197_')
dv.append_df(alpha211_.run_formula(dv),'alpha211_')
dv.append_df(alpha56_.run_formula(dv),'alpha56_')

factor_lis = ['alpha32_','alpha42_','alpha56_','alpha62_','alpha64_','alpha194','alpha195','alpha197','alpha211','pb','pe','roa','roe','pm','ETOP']

for each in factor_lis:
    assert(each in dv.fields)

factors = {name:dv.get_ts(name) for name in factor_lis}

dv.save_dataview('muti_factor/')


# In[87]:


import pandas as pd

id_zz500 = dp.daily_index_cons(api, "000300.SH", start, end)
id_hs300 = dp.daily_index_cons(api, "000905.SH", start, end)

columns_500 = list(set(id_zz500.columns)-set(id_hs300.columns))
def limit_up_down():
    trade_status = dv.get_ts('trade_status').fillna(0)
    mask_sus = trade_status == 0
    # 涨停
    up_limit = dv.add_formula('up_limit', '(close - Delay(close, 1)) / Delay(close, 1) > 0.095', is_quarterly=False)
    # 跌停
    down_limit = dv.add_formula('down_limit', '(close - Delay(close, 1)) / Delay(close, 1) < -0.095', is_quarterly=False)
    can_enter = np.logical_and(up_limit < 1, ~mask_sus) # 未涨停未停牌
    can_exit = np.logical_and(down_limit < 1, ~mask_sus) # 未跌停未停牌
    return can_enter,can_exit

id_member = pd.concat([id_zz500[columns_500],id_hs300],axis=1)
mask = ~id_member
can_enter,can_exit = limit_up_down()

price = dv.get_ts('close_adj')
sw1 = sw1_name
can_enter = can_enter.reindex(columns=price.columns,index=price.index)
can_exit = can_exit.reindex(columns=price.columns,index=price.index)
mask = mask.reindex(columns=price.columns,index=price.index)


# In[88]:


import matplotlib.pyplot as plt
from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import analysis

def cal_obj(signal, name, period, quantile):
    price_bench = dv.data_benchmark
    obj = SignalDigger(output_folder="hs300/%s" % name,
                       output_format='pdf')
                       obj.process_signal_before_analysis(signal,
                                                          price=price,
                                                          n_quantiles=quantile,
                                                          period=period,
                                                          mask=mask,
                                                          group=sw1,
                                                          can_enter = can_enter,
                                                          can_exit = can_exit,
                                                          commission = 0.0003
                                                          )
                       obj.create_full_report()
                       return obj

def plot_pfm(signal, name, period=5, quantile=5):
    obj = cal_obj(signal, name, period, quantile)
    plt.show()
def signal_data(signal, name, period=5, quantile=5):
    print(name)
    obj = cal_obj(signal, name, period, quantile)
    return obj.signal_data


# In[89]:


from jaqs_fxdayu.research.signaldigger import multi_factor

ic=multi_factor.get_factors_ic_df(factors,
                                  price=dv.get_ts("close_adj"),
                                  high=dv.get_ts("high_adj"), # 可为空
                                  low=dv.get_ts("low_adj"),# 可为空
                                  n_quantiles=5,# quantile分类数
                                  mask=mask,# 过滤条件
                                  can_enter=can_enter,# 是否能进场
                                  can_exit=can_exit,# 是否能出场
                                  period=20
                                  )


# In[92]:


ic_mean_table = pd.Series(data=np.nan,index=factor_lis)
ic_std_table = pd.Series(data=np.nan,index=factor_lis)
ir_table = pd.Series(data=np.nan,index=factor_lis)
for signal in factor_lis:
    ic_mean_table[signal] = ic[signal].loc[:20170101].mean()
    ic_std_table[signal] = ic[signal].loc[:20170101].std()
    ir_table[signal] = ic_mean_table[signal]/ic_std_table[signal]
get_ipython().run_line_magic('matplotlib', 'inline')
ic_mean_table.plot(kind="barh",xerr=ic_std_table,figsize=(15,5))


# In[93]:


ic_mean_table


# In[94]:


from jaqs_fxdayu.research.signaldigger import process
negative = ['alpha206','alpha197','alpha195','alpha211','alpha194','alpha64_','alpha56_','alpha42_','pb','pe']

factor_dict = dict()
for name in factor_lis:
    print(name)
    if name in negative:
        signal = -1*dv.get_ts(name) # 调整符号
    else:
        signal = dv.get_ts(name)
signal = process.winsorize(factor_df=signal,alpha=0.05)#去极值
# 行业市值中性化
signal = process.neutralize(signal,
                            group=dv.get_ts("sw1"),# 行业分类标准
                            )
    signal = process.standardize(signal) #z-score标准化 保留排序信息和分布信息
    factor_dict[name] = signal



import pickle
with open('Neutralized_Postive_Data.pkl','wb') as f:
    pickle.dump(factor_dict,f)


import pickle
#导入已经经过预处理的因子数据
with open("Neutralized_Postive_Data.pkl",'rb') as f2:
    factor_dict = pickle.load(f2)


# In[29]:


props = {
    'price':dv.get_ts("close_adj"),
    'high':dv.get_ts("high_adj"), # 可为空
    'low':dv.get_ts("low_adj"),# 可为空
    'ret_type': 'return',#可选参数还有upside_ret/downside_ret 则组合因子将以优化潜在上行、下行空间为目标
    'period': 20,
    'mask': mask,
    'can_enter': can_enter,
    'can_exit': can_exit,
    'forward': True,
    'commission': 0.0008,
    "covariance_type": "shrink",  # 协方差矩阵估算方法 还可以为"simple"
    "rollback_period": 220}  # 滚动窗口天数


# In[30]:


comb_factors = dict()
for method in ["ir_weight"]:
    comb_factors[method] = multi_factor.combine_factors(factor_dict,
                                                        standardize_type="rank",
                                                        winsorization=False,
                                                        weighted_method=method,
                                                        props=props)
    print(method)


# In[31]:


ic_20  =   multi_factor.get_factors_ic_df(comb_factors,
                                          price=dv.get_ts("close_adj"),
                                          high=dv.get_ts("high_adj"), # 可为空
                                          low=dv.get_ts("low_adj"),# 可为空
                                          n_quantiles=5,# quantile分类数
                                          mask=mask,# 过滤条件
                                          can_enter=can_enter,# 是否能进场
                                          can_exit=can_exit,# 是否能出场
                                          period=20
                                          )


# In[34]:


ic_20_mean = dict()
ic_20_std = dict()
ir_20 = dict()
from datetime import datetime
for name in ic_20.columns:    
    ic_20_mean[name]=ic_20[name].loc[20170101:].mean()
    ic_20_std[name]=ic_20[name].loc[20170101:].std()
    ir_20[name] = ic_20_mean[name]/ic_20_std[name]
ir_20


# In[35]:


ic_20_mean


# In[37]:


ic_20_mean = dict()
ic_20_std = dict()
ir_20 = dict()
from datetime import datetime
for name in ic_20.columns:    
    ic_20_mean[name]=ic_20[name].mean()
    ic_20_std[name]=ic_20[name].std()
    ir_20[name] = ic_20_mean[name]/ic_20_std[name]
ic_20_mean

