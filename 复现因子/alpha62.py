# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params ={'n1':80,'n2':8,'n3':13} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n1':'移动平均周期','n2':'相关系数周期','n3':'线性衰减的周期'} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
        最低价与成交量的80天移动平均的相关系数的Rank的线性衰减的Rank。
    '''

    n1 = param['n1']
    n2 = param['n2']
    n3 = param['n3']

    return  dv.add_formula('alpha62', 'Rank(Decay_linear(Rank(Correlation((low_adj),Ts_Mean(volume,{}), {})), {})) * -1'.format(n1,n2,n3), is_quarterly=False, add_data=True)
