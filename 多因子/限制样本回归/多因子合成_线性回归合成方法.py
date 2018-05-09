
# coding: utf-8

# In[1]:
#IC 0.085 IC_IR 0.811
'''
该因子合成方法使用了滚动预测的机制，算法如下：
1.计算未来20天的持有收益。
2.计算所需要的因子
3.对20天的收益率计算横截面的5分quantile，然后取1，5两个quantile的股票的数据来进行模型的训练。
4.每一天的预测值是向前推20天的前120天的数据，使用逻辑回归进行训练分类，训练集的因子为计算的因子值和前天的收益率，
类似于自回归的思想，使用了滞后一期的收益率来作为因子quantile为5的就是类型1，quantile为1的就是类型0.然后用该训
练出来的模型来预测当天的每一只股票属于1的概率，使用该概率来当做当前股票当天的因子。滚动预测每一天。
'''

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
sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)
factor_lis = ['alpha32_','alpha42_','alpha62_','alpha64_','alpha194',              'alpha195','alpha197','Beta3','alpha211','pe','ps']


# In[ ]:


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


# In[ ]:


signals_name = ['alpha32_','alpha42_','alpha62_','alpha64_','alpha194','alpha195','alpha197','Beta3','alpha211','pe','ps']
signals_name.append('ret20')


# In[ ]:


#当期数据
period = 20
factor_dict['ret20'] = dv.get_ts('close_adj').pct_change(period)

X1 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X1[signal_name] = factor_dict[signal_name].stack()
#滞后一期数据
'''
X2 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X2[signal_name] = factor_dict[signal_name].shift(1).stack()
#之后两期数据
X3 = pd.DataFrame(columns = signals_name)
for signal_name in signals_name:
    X3[signal_name] = factor_dict[signal_name].shift(2).stack()
X1_ = X1.join(X2,rsuffix='_2')
X1_ = X1_.join(X3,rsuffix='_3') 
'''
X1_ = X1


# In[ ]:


train_indexer = dv.get_ts('close_adj').loc[:20160101].stack().index.values
test_indexer = dv.get_ts('close_adj').loc[20160101:].stack().index.values
X = X1_
Y = dv.get_ts('close_adj').pct_change(period).shift(-period).stack().reindex(index = X.index)
import jaqs.util as jutil
Y_q = jutil.to_quantile(dv.get_ts('close_adj').pct_change(period).shift(-period),n_quantiles=7)
Y_q_clip = Y_q.stack().reindex(index = X.index)
Y_q_clip = Y_q_clip[np.logical_or(Y_q_clip == 1.0,Y_q_clip == 7.0)]
Y_clip = Y.reindex(index = Y_q_clip.index)
Y_clip_class = pd.Series(np.where(Y_q_clip == 7.0,1,0),index = Y_q_clip.index)
X_clip = X.reindex(index = Y_q_clip.index)
from sklearn.linear_model import LogisticRegression

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


stock_num = dv.get_ts('close_adj').shape[1]
time_index = X.unstack().index.values
tscv = TimeSeriesSplit(max_train_size=5,n_splits=300)
pred = []
i = 0
for train_index, pred_index in split(X.unstack().index.values,max_train_size=120,period=period):
    i+=1
    indexer = [slice(None)] * 2
    indexer[X.index.names.index('trade_date')] = time_index[train_index]
    indexer2 = [slice(None)] * 2
    indexer2[X.index.names.index('trade_date')] = time_index[pred_index]
    #clf = RFR(max_depth=3,min_samples_leaf=9,max_leaf_nodes=4)
    #clf = SVR(C = 1)
    #clf = LinearRegression()
    #clf = Ridge()
    clf = LogisticRegression()
    X_ = X_clip.loc[tuple(indexer),:]
    X_train = X_.dropna(how = 'any', axis = 0)
    X__ = X.loc[tuple(indexer2),:]
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
    


# In[ ]:


pred_factor = pred[0].append(pred[1:])
pred_factor = pred_factor.reindex(index = X.index)
signal_data(pred_factor.unstack().loc[20170101:],'Adamzzz',period=20,quantile=5)
pred_factor = pred_factor.unstack()

