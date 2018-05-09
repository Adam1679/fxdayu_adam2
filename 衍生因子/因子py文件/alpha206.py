# encoding: utf-8
# 文件需要以utf-8格式编码
# 文件名代表因子名称，需满足命名规范
__author__ = "张安翔" # 这里填下你的名字
default_params = {'n':400} # 这里填写因子参数默认值，比如: {"t1": 10}
params_description = {'n':'成交量的求和的周期'}

def run_formula(dv, param = default_params):
    '''
        N*成交量比上前N天成交量的总和
        '''
    defult_param = {'n':400}
    if not param:
        param = defult_param
    
    n = param['n']

    return dv.add_formula('alpha206','volume*{}/Ts_Sum(volume,{})'.format(n,n),is_quarterly=False,overwrite=True)
