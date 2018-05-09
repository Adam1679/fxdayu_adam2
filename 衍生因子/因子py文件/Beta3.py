# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "xxx" # 这里填下你的名字
default_params = {'n':20} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {} # 这里填写因子参数描述信息，比如: {"t1": "并没有用上的参数"}

def run_formula(dv, param = default_params):
    '''
    收盘价与指数的时间序列秩的Beta。需要dv中存入指数数据。
    '''
    import pandas as pd

    n = param['n']
    if 'hs300' not in dv.fields:
        index_data = dv.data_benchmark
        close = dv.get_ts('close')
        hs300 = pd.DataFrame(index=close.index,columns=close.columns)
        for col in hs300:
            hs300[col] = index_data
        dv.append_df(hs300,'hs300',overwrite=True)

    return dv.add_formula('Beta2','Covariance(Ts_Rank(close_adj,{}),Ts_Rank(hs300,{}),{})/Pow(StdDev(Ts_Rank(hs300,{}),{}),2)'.format(n,n,n,n,n),is_quarterly=False)
