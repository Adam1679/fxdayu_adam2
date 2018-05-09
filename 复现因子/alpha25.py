# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = {'n1':7,'n2':20,'n3':9,'n4':250} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n1':'收盘价滞后期数','n2':'成交量移动平均的周期','n3':'线性衰减的周期','n4':'时间序列求和的周期'}

def run_formula(dv, param = default_params):
    '''
        周后7期的收盘价的Rank乘以（1-成交量除以9天移动平均成交量的20天线性衰减）乘以（1+1天收益的290天求和的Rank）。
    '''
    n1 = param['n1']
    n2 = param['n2']
    n3 = param['n3']
    n4 = param['n4']
    
    return dv.add_formula('alpha25', '((-1 * Rank((Delay(close_adj, {}) * (1 - Rank(Decay_linear((volume / Ts_Mean(volume,{})), {})))))) * (1+Rank(Ts_Sum(close_adj/Delay(close_adj,1)-1, {}))))'.format(n1,n2,n3,n4), is_quarterly=False, add_data=True)

