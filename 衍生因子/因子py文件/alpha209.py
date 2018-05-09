# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = {} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'w':'线性回归的滚动周期'}

def run_formula(dv, param = default_params):
    '''
        使用股价和指数进行回归得到残差，因子值为N天的横截面Rank
    '''
    def GetResidual():
        '''
            股价与hs300指数线性回归的残差，滑动窗口50天
        '''
        import pandas as pd
        from sklearn.model_selection import TimeSeriesSplit
        close = dv.get_ts('close_adj')
        bench = dv.data_benchmark
        bench = bench.reindex(index = close.index)
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
        global i
        i = 0
        def reg2(T):
            global i
            print(i)
            i+=1
            #防止全部为Nan
            if T.isnull().sum()!=T.shape[0]:
                window = 50
                tscv = TimeSeriesSplit(n_splits = T.shape[0]-window+1)
                new_dd = pd.Series(np.NAN,index=T.index)
                for train_index, test_index in tscv.split(T):
                    #print("TRAIN:", train_index[-window:], "TEST:", test_index)
                    X, Y = T.iloc[train_index[-window:]],bench.iloc[train_index[-window:]]
                    #防止全部为Nan
                    if X.isnull().sum()!=X.shape[0]:
                        X = sm.add_constant(X)
                        model = OLS(Y,X,missing='drop')
                        results = model.fit()
                        res = results.resid.iloc[-1]
                        new_dd.iloc[train_index[-1]] = res
                #计算最后一个
                X, Y = T.iloc[-window:],bench.iloc[-window:]
                #防止全部为Nan
                if X.isnull().sum()!=X.shape[0]:
                    X = sm.add_constant(X)
                    model = OLS(Y,X,missing='drop')
                    results = model.fit()
                    res = results.resid.iloc[-1]
                    new_dd.iloc[-1] = res
                    return new_dd
                else:
                    return T
            else:
                return T
        return close.apply(reg2,axis=0)
    
    if 'R' not in dv.fields:
        R = GetResidual()
        dv.append_df(R,'R')
    
    return dv.add_formula('alpha209','Rank(R)',is_quarterly=False,overwrite=True)
