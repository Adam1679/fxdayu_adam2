
# coding: utf-8

# In[1]:
#首先观察全样本的所有因子的表现，在分别观察测试集合和训练集合的因子的表现，选择两期表现都不错的因子进入多因子合成。
#对比了所有的因子合成方法，最终选定了ir_weight的方法达到最好的效果
#所有的因子都按要求经过的了数据处理。
#IC 0.97 IC_IR 1.01

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
import alpha32_,alpha42_,alpha56_,alpha62_,alpha64_,alpha194,alpha195,alpha197,Beta3
import pandas as pd
import matplotlib.pyplot as plt
from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import analysis
from jaqs_fxdayu.research.signaldigger import multi_factor
api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("18523827661", 
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk'
)
start = 20100101
end = 20180401

SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))
factor_list = ['volume','float_mv','pe','ps']
check_factor = ','.join(factor_list)
dataview_folder = '/Users/adam/Desktop/intern/test5/fxdayu_adam/data'
dataview_folder2 = 'muti_factor/'
dv = DataView()
#ds = LocalDataService(fp=dataview_folder)
data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18523827661",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk"
}
ds = RemoteDataService()
ds.init_from_config(data_config)
dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
         'fields': ','.join(factor_list),
         'freq': 1,
         "prepare_fields": True,
           "benchmark":'000300.SH'}

dv.init_from_config(dv_props, data_api=ds)
dv.prepare_data()
sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)
factor_lis = ['alpha32_','alpha42_','alpha62_','alpha64_','alpha194',              'alpha195','alpha197','Beta3','alpha211','pe','ps']


def GetResidual():
    '''
    股价与hs300指数线性回归的残差，滑动窗口50天
    '''
    from sklearn.model_selection import TimeSeriesSplit
    import pandas as pd
    close = dv.get_ts('close_adj')
    bench = dv.data_benchmark
    bench = bench.reindex(index = close.index)
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    global i
    i = 0
    def reg2(T):
        global i
        print(i)
        i+=1
        #防止全部为Nan
        if T.isnull().sum()!=T.shape[0]:
            window = 50
            tscv = TimeSeriesSplit(n_splits = T.shape[0]-window+1)
            new_dd = pd.Series(np.NAN,index=T.index)
            for train_index, test_index in tscv.split(T):
                #print("TRAIN:", train_index[-window:], "TEST:", test_index)
                X, Y = T.iloc[train_index[-window:]],bench.iloc[train_index[-window:]]
                #防止全部为Nan
                if X.isnull().sum()!=X.shape[0]:
                    X = sm.add_constant(X)
                    model = OLS(Y,X,missing='drop')
                    results = model.fit()
                    res = results.resid.iloc[-1]
                    new_dd.iloc[train_index[-1]] = res
            #计算最后一个
            X, Y = T.iloc[-window:],bench.iloc[-window:]
            #防止全部为Nan
            if X.isnull().sum()!=X.shape[0]:
                X = sm.add_constant(X)
                model = OLS(Y,X,missing='drop')    
                results = model.fit()
                res = results.resid.iloc[-1]
                new_dd.iloc[-1] = res
                return new_dd
        else:
            return T
    return close.apply(reg2,axis=0)


# In[4]:


dv.append_df(GetResidual(),'R')
dv.append_df(alpha32_.run_formula(dv),'alpha32_')
dv.append_df(alpha42_.run_formula(dv),'alpha42_')
dv.append_df(alpha56_.run_formula(dv),'alpha56_')
dv.append_df(alpha62_.run_formula(dv),'alpha62_')
dv.append_df(alpha64_.run_formula(dv),'alpha64_')
dv.append_df(alpha194.run_formula(dv),'alpha194')
dv.append_df(alpha195.run_formula(dv),'alpha195')
dv.append_df(alpha197.run_formula(dv),'alpha197')
dv.append_df(Beta3.run_formula(dv),'Beta3')


# In[6]:


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

alpha_signal = factor_lis
price = dv.get_ts('close_adj')
sw1 = sw1_name
can_enter = can_enter.reindex(columns=price.columns,index=price.index)
can_exit = can_exit.reindex(columns=price.columns,index=price.index)
mask = mask.reindex(columns=price.columns,index=price.index)


# In[77]:



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


# In[ ]:


factors = {name:dv.get_ts(name) for name in factor_lis}


# In[ ]:


factor_dict = dict()
for name in factor_lis:
    print(name)
    if name in ['alpha197','alpha195','alpha211','alpha194','alpha64_','alpha56_','alpha42_','pb','pe','ps']:    
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


# In[14]:


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
    "rollback_period": 120}  # 滚动窗口天数


# In[16]:


comb_factors = dict()
for method in ["ir_weight"]:
    comb_factors[method] = multi_factor.combine_factors(factor_dict,
                                                        standardize_type="rank",
                                                        winsorization=False,
                                                        weighted_method=method,
                                                        props=props)
    print(method)


# In[17]:


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


# In[25]:


ic_20_mean = dict()
ic_20_std = dict()
ir_20 = dict()
from datetime import datetime
for name in ic_20.columns:    
    ic_20_mean[name]=ic_20[name].loc[20170101:].mean()
    ic_20_std[name]=ic_20[name].loc[20170101:].std()
    ir_20[name] = ic_20_mean[name]/ic_20_std[name]
print(ic_20_mean)

