# coding:utf-8
import numpy as np

def ARS(p, lnp, dlnp, X0, size=1):
    """[FUNCTION] Adaptive Rejection Sampling(適応的棄却サンプリング)を行う

    p: {function}サンプリングしたい分布関数(正規化してなくてもいい)
    lnp: {function}Pの対数をとった関数
    dlnp: {function}上記の微分関数
    X0: {list of float}初期グリッド点集合
    size: {int}サンプルを生成する個数
    return: {list of float}サンプル集合
    """
    X = X0[:]  # グリッド集合
    L = []  # セクションごとの包絡関数の傾き
    E = []  # 包絡関数の交点 len(E) = len(X) - 1
    E2 = []  # -inf + 交点集合 + inf
    Zqi = []  # 各セクションの指数分布の面積
    Zq = 0.  # 提案分布の正規化係数(sum(Zqi))
    pi = []  # 各セクションの全体に対する面積の割合

    Samples = []  # サンプル集合

    def section(x, E):  # xが該当するリストEの区間インデックスを返す(線形探索)
        xx = np.array(x)

        def f(xi):
            for i in xrange(len(E)):
                if xi <= E[i]:
                    return i
            return len(E)
        if xx.size == 1:
            return np.array(f(x))
        return np.array([f(i) for i in xx])

    def q(x):  # 提案分布(正規化されていない区分的指数分布)
        I = section(x, E)  # len(x) == len(i)
        if I.size == 1:
            return np.array(p(X[I]) * np.exp(L[I] * (x - X[I])))
        List = [p(X[i]) * np.exp(L[i] * (xi - X[i])) for i, xi in zip(I, x)]
        return np.array(List)

    def update():  # グローバル変数を更新する(一部分だけ更新するほうが効率的だが...)
        L = [dlnp(xi) for xi in X]
        E = []
        for i in xrange(len(X) - 1):
            E.append((L[i] * X[i] - L[i + 1] * X[i + 1] -
                      lnp(X[i]) + lnp(X[i + 1])) / float(L[i] - L[i + 1]))
        E2 = [-float("inf")] + E + [float("inf")]  # -infとinfを追加した完全な区切り位置
        Zqi = [p(xi) / -L[i] * (np.exp(L[i] * (E2[i] - xi)) -
                                np.exp(L[i] * (E2[i + 1] - xi))) for i, xi in enumerate(X)]
        Zq = sum(Zqi)
        pi = [zqi / float(Zq) for zqi in Zqi]  # 各セクションが出現する確率(面積比)
        return L, E, E2, Zqi, Zq, pi

    def randPWED(size=1):  # 区分的指数分布に従う乱数を発生する
        I = np.random.choice(range(len(X)), p=pi, size=size)  # どのセクションを選ぶか
        U = np.random.uniform(size=size)  # 0~1の乱数
        z = [1.0 / L[i] * np.log((1 - u) * np.exp(L[i] * E2[i]) +
                                 u * np.exp(L[i] * E2[i + 1])) for i, u in zip(I, U)]
        return np.array(z)

    # 初期化
    L, E, E2, Zqi, Zq, pi = update()
    # メインループ
    for i in xrange(size):
        while True:
            z = randPWED()
            u = np.random.uniform(0.0, q(z))
            if u <= p(z):
                Samples.append(z[0])
                break
            else:
                X.append(z[0])
                X.sort()
                L, E, E2, Zqi, Zq, pi = update()

    return Samples, Zq, lambda x: q(x)

import matplotlib.pyplot as plt
import scipy.special as ssp
def test():
    a = 10.0
    b = 1.0
    X0 = [4.0, 10.0, 19.0]

    p = lambda x: x ** (a - 1) * np.exp(-b * x)
    lnp = lambda x: (a - 1) * np.log(x) - b * x
    dlnp = lambda x: (a - 1) / float(x) - b

    Z, Zq, q = ARS(p, lnp, dlnp, X0, size=1)

    print "p(s=1) =", ssp.gamma(a) / b ** a / Zq

    plt.figure(figsize=(8, 6))
    plt.xlim([0, 22])

    x = np.arange(0.0, 35.0, 0.05)
    y = b ** a / ssp.gamma(a) * p(x)
    y = p(x)
    #y = np.log(y)
    plt.plot(x, y, "b", linewidth=0.9, alpha=1.0)

    y = 1. / Zq * q(x)
    y = q(x)
    #y = np.log(y)
    plt.plot(x, y, "g", linewidth=0.9, alpha=1.0)

    #plt.hist(Z, bins=60, normed=True, facecolor="r", alpha=.1)
    plt.show()

if __name__ == "__main__":
    test()

