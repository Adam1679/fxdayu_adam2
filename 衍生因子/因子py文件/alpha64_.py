# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔"
default_params = {'n1':4,'n2':20,'n3':50,'n4':14}
params_description = {}

def run_formula(dv, param = default_params):
    '''
        先求vwap的横截面Rank和成交量的横截面Rank的N1天相关系数的N1天线性衰减的横截面Rank，再求收盘价的横截面Rank和成交量N2天移动平均的N1天相关系数和N3间的较大值的N4天线性衰减，最后两者去较大值为因子值
        '''
    n1 = param['n1']
    n2 = param['n2']
    n3 = param['n3']
    n4 = param['n4']
    
    return dv.add_formula('alpha64','Max(Rank(Decay_linear(Correlation(Rank(vwap),Rank(volume),{}),{})),Rank(Decay_linear(Max(Correlation(Rank(close_adj),Rank(Ts_Mean(volume,{})),{}),{}),{})))'.format(n1,n1,n2,n1,n3,n4),is_quarterly=False,overwrite=True)

