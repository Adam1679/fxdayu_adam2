# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = {'n':20,'m':5} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'移动平均的周期','m':'线性衰减的周期'}

def run_formula(dv, param = default_params):
    '''
    成交量与流动市值N天均值的比值，滞后M期
    '''
    n = param['n']
    m = param['m']
    
    return dv.add_formula('alpha197','Delay(volume/Ts_Mean(float_mv,{}),{})'.format(n,m),is_quarterly = False)

