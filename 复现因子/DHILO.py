# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params ={'n':89} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'求中位数的周期'} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
        最高价与最低价的log差价的89天中位数。
    '''
    n = param['n']
    import pandas as pd
    diff = dv.add_formula('diff','Log(high_adj) - Log(low_adj)',is_quarterly=False,add_data=False)
    DHILO = diff.rolling(n).median()
    return  DHILO
