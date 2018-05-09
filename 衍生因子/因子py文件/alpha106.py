# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params ={'n':20} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'滞后期数'} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
        当天收盘价和20天前收盘价的差价。
    '''

    n = param['n']

    return dv.add_formula('alpha106', 'close_adj-Delay(close_adj,{})'.format(n), is_quarterly=False, add_data=True)
