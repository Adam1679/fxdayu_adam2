def alpha210(n = 20):
    '''
        使用股价和指数进行回归得到残差，因子值为N天的时间序列Rank
        '''
    if 'R' not in dv.fields:
        R = GetResidual()
        dv.append_df(R,'R')
    return dv.add_formula('alpha210','Ts_Rank(R,{})'.format(n),is_quarterly=False,overwrite=True)


def run_formula(dv, param = None):
    defult_param = {'n':20}
    if not param:
        param = defult_param
    n = param['n']
    def GetResidual():
        '''
            股价与hs300指数线性回归的残差，滑动窗口50天
            '''
        from sklearn.model_selection import TimeSeriesSplit
        close = dv.get_ts('close_adj')
        bench = dv.data_benchmark
        bench = bench.iloc[bench.index>20130101]
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
    
    return dv.add_formula('alpha210','Ts_Rank(R,{})'.format(n),is_quarterly=False,overwrite=True)
