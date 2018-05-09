
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


# In[23]:
#导入在数据处理中已经处理好的数据
import pickle
with open("Neutralized_Postive_Data.pkl",'rb') as f2:
    factor_dict = pickle.load(f2)

api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("18523827661", 
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk'
)
start = 20100101
end = 20180401

SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))
factor_list = ['volume','float_mv','pb','pe','ps','end_bal_cash']
check_factor = ','.join(factor_list)
dataview_folder = '/Users/adam/Desktop/intern/test5/fxdayu_adam/data'
dataview_folder2 = 'muti_factor/'
dv = DataView()
#ds = LocalDataService(fp=dataview_folder)


# In[24]:


data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18523827661",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk"
}
ds = RemoteDataService()
ds.init_from_config(data_config)


# In[25]:


#dv.save_dataview(dataview_folder2)
dv.load_dataview(dataview_folder2)


# In[28]:


factor_lis = ['alpha32_','alpha42_','alpha56_','alpha62_','alpha64_','alpha194','alpha195','alpha197',              'alpha211','pb','pe','roa','roe','pm','ETOP']


# In[29]:


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

alpha_signal = factor_lis
price = dv.get_ts('close_adj')
sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)
sw1 = sw1_name
can_enter = can_enter.reindex(columns=price.columns,index=price.index)
can_exit = can_exit.reindex(columns=price.columns,index=price.index)
mask = mask.reindex(columns=price.columns,index=price.index)


# In[7]:


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
def signal_data(signal, name, period=5, quantile=5):
    print(name)
    obj = cal_obj(signal, name, period, quantile)
    return obj.signal_data


# In[34]:


import pandas as pd



# In[35]:


factor_dict.keys()


# In[36]:


signals_name = ['alpha32_','alpha42_','alpha56_','alpha62_','alpha64_','alpha194','alpha195','alpha197',              'alpha211','pb','pe','roa','roe','pm','ETOP']


# In[37]:


#当期数据
period = 20
factor_dict['ret20'] = dv.get_ts('close_adj').pct_change(period)
signals_name.append('ret20')
index_ret20 = pd.DataFrame(index = dv.get_ts('close_adj').index,columns = dv.get_ts('close_adj').columns)
for col in index_ret20.columns:
    index_ret20[col] = 100*dv.data_benchmark.pct_change(period)
#factor_dict['index_ret20'] = index_ret20
#signals_name.append('index_ret20')
X1 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X1[signal_name] = factor_dict[signal_name].stack()
    
'''
#滞后一期数据

X2 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X2[signal_name] = factor_dict[signal_name].shift(1).stack()

#之后两期数据
X3 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X3[signal_name] = factor_dict[signal_name].shift(2).stack()
X1_ = X1_.join(X3,rsuffix='_3') 

X1_ = X1.join(X2,rsuffix='_2')
'''
X1_ = X1
X = X1_


# In[50]:


Y = dv.get_ts('close_adj').pct_change(period).shift(-period)
#Y = Y-index_ret20
Y = Y.stack().reindex(index = X.index)


# In[51]:


import jaqs.util as jutil
Y_q = jutil.to_quantile(dv.get_ts('close_adj').pct_change(period).shift(-period),n_quantiles=5)
Y_q_clip = Y_q.stack().reindex(index = X.index)
Y_q_clip = Y_q_clip[np.logical_or(Y_q_clip == 1.0,Y_q_clip == 5.0)]


# In[52]:


Y_clip = Y.reindex(index = Y_q_clip.index)
Y_clip_class = pd.Series(np.where(Y_q_clip == 5.0,1,0),index = Y_q_clip.index)
X_clip = X.reindex(index = Y_q_clip.index)


# In[53]:


from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


# In[54]:


def split(X,max_train_size=5,period = 1):
    n = len(X)
    lis = []
    for i in range(1,n):
        pred_index = [n-i]
        if (n-i-max_train_size-period)>=0:
            train_index = [j for j in range(n-i-max_train_size-period,n-i-period)]
            lis.append((train_index,pred_index))
    lis.reverse()
    return lis


# In[ ]:


stock_num = dv.get_ts('close_adj').shape[1]
time_index = X.unstack().index.values
tscv = TimeSeriesSplit(max_train_size=5,n_splits=300)
pred = []
i = 0
#预处理
for train_index, pred_index in split(X.unstack().index.values,max_train_size=120,period=period):
    i+=1
    indexer = [slice(None)] * 2
    indexer[X.index.names.index('trade_date')] = time_index[train_index]
    indexer2 = [slice(None)] * 2
    indexer2[X.index.names.index('trade_date')] = time_index[pred_index]
    clf = LogisticRegression()
    X_ = X_clip.loc[tuple(indexer),:]
    X_train = X_.dropna(how = 'any', axis = 0)
    X__ = X_clip.loc[tuple(indexer2),:]
    X_pred = X__.dropna(how = 'any', axis = 0)
    if len(X_train) == 0 or len(X_pred) == 0:
        print("%d为空"%i)
        continue
    #Y_train = Y_clip.reindex(index = X_train.index).dropna()
    Y_train = Y_clip_class.reindex(index = X_train.index).dropna()
    X_train = X_train.reindex(index = Y_train.index)
    clf.fit(X_train,Y_train)
    #prediction = clf.predict(X_pred)
    prediction = clf.predict_proba(X_pred)[:,1]
    pred_ser = pd.Series(prediction,index = X_pred.index)
    pred.append(pred_ser)
    print(i)

pred_factor = pred[0].append(pred[1:])
pred_factor = pred_factor.unstack().reindex(columns = dv.get_ts('close').columns)


