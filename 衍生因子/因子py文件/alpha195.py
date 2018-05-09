# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = defult_param = {'n1':200,'n2':20}
params_description = {'n1':'移动平均的周期','n2':'方差的周期'}

def run_formula(dv, param = default_params):
    '''
    成交量与N天流动市值均值的比值的M天方差
    '''
    n1 = param['n1']
    n2 = param['n2']
    
    return dv.add_formula('alpha195','StdDev(volume/Ts_Mean(float_mv,{}),{})'.format(n1,n2),is_quarterly = False)

