
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re


# In[2]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame.rename_axis')


# In[3]:


FactorRequirement = pd.read_excel('/Users/adam/Desktop/intern/test5/HelloGit/Internship_Factor/Internship_Factor.xlsx',                                  header = 0,                                  names = ['Name','Factors'])
MyRequireFactors_ = FactorRequirement['Factors'][FactorRequirement['Name'] == '张安翔']


# In[5]:


MyRequireFactors = MyRequireFactors_.values[0]
MyRequireFactors


# In[6]:


Compile = re.compile("\'([A-Za-z0-9]*)\'")
Factors = Compile.findall(MyRequireFactors)
Factors


# In[7]:


from jaqs_fxdayu.util import dp
from jaqs.data.dataapi import DataApi



api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("18523827661", 
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjIxMTc0NDY1MzAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1MjM4Mjc2NjEifQ.AO9Rp8jG_IWc6crPrBOC-ujMP0-g1S1c5kUlTs5qwrk'
)

start = 20130101
end = 20180101
SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))


# In[8]:


factor_list = ['volume', 'pb', 'roe']
check_factor = ','.join(factor_list)


# In[9]:


import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs_fxdayu.data.dataservice import LocalDataService
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")
dataview_folder = '/Users/adam/Desktop/intern/test5/fxdayu_adam/data'
dv = DataView()
ds = LocalDataService(fp=dataview_folder)

dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
         'fields': check_factor,
         'freq': 1,
         "prepare_fields": True}

dv.init_from_config(dv_props, data_api=ds)


# In[10]:


dv.prepare_data()


# In[11]:


dv.add_field('sw1')
sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)


# # 因子算法

# In[12]:


alpha106 = dv.add_formula('alpha106', 'close_adj-Delay(close_adj,20)', is_quarterly=False, add_data=True)
alpha127 = dv.add_formula('alpha127', '((100*(close_adj-Ts_Max(close_adj,12))/(Ts_Max(close_adj,12)))^2)^(1/2)', is_quarterly=False, add_data=True)
alpha62 = dv.add_formula('alpha62', 'Rank(Decay_linear(Rank(Correlation((low_adj),Ts_Mean(volume,80), 8)), 17)) * -1', is_quarterly=False, add_data=True)
dv.add_field('NPParentCompanyGrowRate',data_api=ds)
dv.add_field('NOCFToOperatingNI',data_api=ds)
BearPower = dv.add_field('BearPower',data_api = ds)
dv.add_field('DHILO',data_api=ds)
alpha25 = dv.add_formula('alpha25', '((-1 * Rank((Delay(close_adj, 7) * (1 - Rank(Decay_linear((volume / Ts_Mean(volume,20)), 9)))))) * (1+Rank(Ts_Sum(close_adj/Delay(close_adj,1)-1, 250))))', is_quarterly=False, add_data=True)


# # 预处理

# In[13]:


id_zz500 = dp.daily_index_cons(api, "000300.SH", start, end)
id_hs300 = dp.daily_index_cons(api, "000905.SH", start, end)

columns_500 = list(set(id_zz500.columns)-set(id_hs300.columns))


# In[14]:


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


# In[39]:


#因子的名字
alpha_signal = ['alpha106','alpha127','alpha62','alpha25','NPParentCompanyGrowRate','NOCFToOperatingNI',                'BearPower','DHILO']
price = dv.get_ts('close_adj')
sw1 = sw1_name
enter = can_enter
exit =  can_exit
mask = mask


# In[40]:


from jaqs_fxdayu.research.signaldigger.process import neutralize

neutralize_dict = {a: neutralize(factor_df = dv.get_ts(a), group = dv.get_ts("sw1")) for a in alpha_signal}


# # 分析因子周期特点

# In[42]:


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
                                   can_enter = enter,
                                   can_exit = exit,
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


# In[43]:


# 客户要求,period = 20, 池子为ZZ800
signals_dict = {a:signal_data(neutralize_dict[a], a, 20) for a in alpha_signal} 


# In[44]:


ic_pn = pd.Panel({a: analysis.ic_stats(signals_dict[a]) for a in signals_dict.keys()})
alpha_performance = round(ic_pn.minor_xs('return_ic'),2)
print(alpha_performance)


# In[50]:


alpha_IR = alpha_performance.loc["Ann. IR"]
alpha_IC = alpha_performance.loc["IC Mean"]
good_alpha = alpha_IC[(abs(alpha_IC)>=0.03) & (abs(alpha_IR)>=0.25)]
good_alpha_dict = {g: float('%.2f' % good_alpha[g]) for g in good_alpha.index}
good_alpha_dict


# # 查看因子行业特点（最优周期）
# 选择最优的持有周期三年（750天）的平均行业IC，再求其平均IC，输出IC大于0.05与小于-0.05的行业

# In[51]:


signal_dict = {alpha : signal_data(dv.get_ts(alpha), alpha, period=20, quantile=5) for alpha in good_alpha.index}


# In[52]:


from jaqs.research.signaldigger import performance as pfm

def ic_length(signal, days=750):
    return signal.loc[signal.index.levels[0][-days]:]

performance_dict = {}
for alpha in good_alpha.index:
    ic = pfm.calc_signal_ic(ic_length(signal_dict[alpha]), by_group=True)
    mean_ic_by_group = pfm.mean_information_coefficient(ic, by_group=True)
    performance_dict[alpha] = round(mean_ic_by_group,2)
    
ic_industry = pd.Panel(performance_dict).minor_xs('ic')


# In[59]:


High_IC_Industry = pd.DataFrame([ic_industry[abs(ic_industry)>=0.05][alpha].dropna(how='all') for alpha in good_alpha.index]).T


# In[66]:


High_IC_Industry = np.abs(High_IC_Industry)
High_IC_Industry


# In[67]:


DHILO = pd.Series({'name':'DHILO','data': ['low_adj','volume'] ,'IC':good_alpha_dict['alpha106'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':'MIDIAM(Low-EMA(Close,{}))','parameter':[13],'description':'最低价与收盘价13天指数平滑的差值','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['DHILO'][indu]) for indu in High_IC_Industry['DHILO'].dropna().index}})
alpha62 = pd.Series({'name':'alpha62','data': ['low_adj','volume'] ,'IC':good_alpha_dict['alpha106'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':'Rank(Decay_linear(Rank(Correlation((low_adj),Ts_Mean(volume,{}), {})), {})) * -1','parameter':[80,8,17],'description':'...','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha62'][indu]) for indu in High_IC_Industry['alpha62'].dropna().index}})
alpha106 = pd.Series({'name':'alpha106','data': ['close_adj'] ,'IC':good_alpha_dict['alpha106'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':'close_adj-Delay(close_adj,{})','parameter':[20],'description':'收盘价与20天前收盘价的差','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha106'][indu]) for indu in High_IC_Industry['alpha106'].dropna().index}})


# In[68]:


save_excel = pd.concat([globals()[name] for name in High_IC_Industry.columns],axis=1,keys=High_IC_Industry.columns).T


# In[69]:


save_excel


# In[70]:


save_excel.to_excel('Finish_alpha.xlsx')

