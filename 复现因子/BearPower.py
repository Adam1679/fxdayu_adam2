# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params ={'n':13} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'指数平滑的周期'} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
        最低价与收盘价13天指数平滑的差值。试计算Elder因子的中间变量。
    '''

    n = param['n']
    return  dv.add_formula('BearPower', 'low_adj - Decay_linear(close_adj,{})'.format(n), is_quarterly=False, add_data=False)
