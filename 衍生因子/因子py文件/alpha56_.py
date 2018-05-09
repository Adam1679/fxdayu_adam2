# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔"
default_params ={'n1':12,'n2':19,'n3':400,'n4':19,'n5':30}
params_description ={'n1':'最小值的周期','n2':'求和的周期','n3':'均值的周期','n4':'求和的周期','n5':'相关系数的周期'}

def run_formula(dv, param = default_params):
    '''
        先求开盘价和N1天最小开盘价的差值的横截面Rank，再求最高价和最低价求平均的N2天总和与成交量的N3天移动平均的N4天总和的N5天相关系数的横截面Rank，最后两个值中去较大值为因子值。
        '''

    n1 = param['n1']
    n2 = param['n2']
    n3 = param['n3']
    n4 = param['n4']
    n5 = param['n5']

    return dv.add_formula('alpha56','Max(Rank((open - Ts_Min(open,{}))),Rank((Rank(Correlation(Ts_Sum((high+low)/2,{}),Ts_Sum(Ts_Mean(volume,{}),{}),{})))))'.format(n1,n2,n3,n4,n5),is_quarterly=False,overwrite=True)

