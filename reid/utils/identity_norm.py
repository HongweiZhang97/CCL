import numpy as np

def identity_norm_train(x, gamma, beta, idn_param, momentum=0.1, eps=1e-5):
    """
    param:x    : 输入数据，设shape(B,L)
    param:gama : 缩放因子  γ
    param:beta : 平移因子  β
    param:bn_param   : batchnorm所需要的一些参数
        eps      : 接近0的数，防止分母出现0
        momentum : 动量参数，一般为0.9， 0.99， 0.999
        running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
        running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
    """
    running_mean = idn_param['running_mean']  # shape = [B]
    running_var = idn_param['running_var']    # shape = [B]

    # 计算均值方差
    x_mean=x.mean(axis=0)  # 计算x的均值
    x_var=x.var(axis=0)    # 计算方差

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    x_normalized=(x - running_mean)/np.sqrt(running_var + eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移


    #记录新的值
    idn_param['running_mean'] = running_mean
    idn_param['running_var'] = running_var

    return results, idn_param


def identity_norm_test(x, gamma, beta, idn_param, eps=1e-5):
    """
    param:x    : 输入数据，设shape(B,L)
    param:gama : 缩放因子  γ
    param:beta : 平移因子  β
    param:bn_param   : batchnorm所需要的一些参数
        eps      : 接近0的数，防止分母出现0
        momentum : 动量参数，一般为0.9， 0.99， 0.999
        running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
        running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
    """
    running_mean = idn_param['running_mean']  # shape = [B]
    running_var = idn_param['running_var']    # shape = [B]

    x_normalized = (x-running_mean) / np.sqrt(running_var + eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移

    return results