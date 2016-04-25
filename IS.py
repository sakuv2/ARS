# -*- coding: utf-8 -*-
import numpy as np

def IS(p, lnp, X0, size=1):
    """[FUNCTION] 関数説明文
    Importance Sampling(重点サンプリング)
    p: {function}サンプリングしたい分布関数(正規化してなくてもいい)
    lnp: {function}Pの対数をとった関数
    X0: {list of float}初期グリッド点集合
    size: {int}サンプルを生成する個数
    return: {list of float}サンプル集合
    """

def EnvelopeFunction(fp, Z0):
    """[FUNCTION] 関数説明文
    グリッド点と関数と密度関数の積の関数から抱絡線を構築する
    fp: {lambda x}期待値を取りたい関数と期待値をとる分布の積の関数
    Z0: {list of float}グリッド点
    return: {
    """


def IndicatorFunction(x, condition):
    """[FUNCTION] 関数説明文
    指示関数
    e.g. IndicatorFunction(range(5), lambda x: x > 1) = [ 0.  0.  1.  1.  1.]
    x: {list of float}変数
    condition: {lambda x}条件
    return: {list of float}0 or 1
    """
    List = [1.0 if condition(xi) else 0.0 for xi in x]
    return np.array(List)


import matplotlib.pyplot as plt
import scipy.special as ssp
import scipy.stats as sst
import scipy

def __test():
    L = 17000000
    X_p = sst.norm.rvs(0., 1., size=L)
    X_q = sst.norm.rvs(5., 1., size=L)

    p = sst.norm(loc=0, scale=1).pdf
    q = sst.norm(loc=5, scale=1).pdf
    f = lambda x: x >= 5

    print "S_p =", np.mean(f(X_p))
    print "var_p =", np.var(f(X_p))

    print "S_q =", np.mean(p(X_q) / q(X_q) * f(X_q))
    print "var_q =", np.var(p(X_q) / q(X_q) * f(X_q))

    print "S =", scipy.integrate.quad(sst.norm.pdf, 5.0, np.inf)[0]

    #plt.figure(figsize=(8, 6))
    #plt.tick_params(labelbottom='off')
    #plt.tick_params(labelleft='off')
    #plt.gca().get_xaxis().set_ticks_position('none')
    #plt.gca().get_yaxis().set_ticks_position('none')

    #plt.xlim([0,22])
    #plt.ylim([0,13])

    x = np.arange(0.0, 35.0, 0.05)

    #plt.show()

def __test2():
    a = 10

    f = lambda x: x >= 5
    f2 = lambda x: x
    p = sst.norm(loc=0, scale=1).pdf

    x = np.arange(-12.0, 12.55, 0.05)
    y = np.fabs(f2(x)) * p(x)
    #y = np.log(y)

    plt.plot(x, y, "b", linewidth=0.9, alpha=1.0)
    plt.show()


if __name__ == "__main__":
    __test2()
