# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params ={'n':20} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'周期'} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
        最高价的横截面Rank和成交量的横截面Rank的N天相关系数的横截面Rank，然后在时间序列N天上求和。
    '''

    n = param['n']

    return dv.add_formula('alpha32','-1*Ts_Sum(Rank(Corr(Rank(high_adj),Rank(volume),{})),{})'.format(n,n),is_quarterly=False,overwrite=True)
