# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = {'n':5,'m':10} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'求收益的天数','m':'滞后的期数'}

def run_formula(dv, param = default_params):
    '''
    若横截面交易量5分quantile在3以上，则取收盘价M天收益率的N天滞后，否则取M+5天收益率的N+5天滞后。
    '''
    n = param['n']
    m = param['m']

    return dv.add_formula('alpha204','If(Quantile(volume,5)>3,Delay(Return(close_adj,{}),{}),Delay(Return(close_adj,{}+5),{})+5)'.format(n,m,n,m),is_quarterly = False)

