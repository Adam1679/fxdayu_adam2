# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔"
default_params = {'n1':80,'n2':8,'n3':17}
params_description = {'n1':'移动平均的周期','n2':'相关系数的周期','n3':'线性衰减的周期'}

def run_formula(dv, param = default_params):
    '''
    最低价与成交量的移动平均的相关系数的横截面Rank的线性衰减的横截面Rank
    '''
    n1 = param['n1']
    n2 = param['n2']
    n3 = param['n3']

    return dv.add_formula('alpha62','-1*Rank(Decay_linear(Rank(Correlation(low_adj,Ts_Mean(volume,{}),{})),{}))'.format(n1,n2,n3),is_quarterly=False,overwrite=True)

