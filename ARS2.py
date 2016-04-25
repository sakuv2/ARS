# -*- coding: utf-8 -*-
import numpy as np

def ARS2(p, lnp, X0, size=1):
    """[FUNCTION] Adaptive Rejection Sampling改(適応的棄却サンプリング)を行う

    p: {function}サンプリングしたい分布関数(正規化してなくてもいい)
    lnp: {function}Pの対数をとった関数
    X0: {list of float}初期グリッド点集合
    size: {int}サンプルを生成する個数
    return: {list of float}サンプル集合
    """
    X = X0[:]  # グリッド集合
    K = []  # 傾きのリスト len(X) - 1
    E = []  # 包絡関数の交点 len(E) = len(X) - 1
    Zq = 0.  # 提案分布の正規化係数(sum(Zqi))
    pi = []  # 各セクションの全体に対する面積の割合
    pij = []  # 各セクションの左か右かを選択する確率

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
        I = section(x, X)  # len(x) == len(i)
        L = lambda x, i: p(X[i]) * np.exp(K[i] * (x - X[i]))

        def Line(x, i):
            if i == 0:
                return L(x, 0)
            if i == len(X):
                return L(x, i - 2)
            if i == 1:
                return L(x, 1)
            if i == len(X) - 1:
                return L(x, i - 2)
            if x < E[i - 1]:
                return L(x, i - 2)
            else:
                return L(x, i)
        if I.size == 1:
            return np.array(Line(x, I))
        List = [Line(xi, i) for i, xi in zip(I, x)]
        return np.array(List)

    def update():  # グローバル変数を更新する(一部分だけ更新するほうが効率的だが...)
        K = [(lnp(X[i]) - lnp(X[i + 1])) / float(X[i] - X[i + 1])
             for i in xrange(len(X) - 1)]
        E = []
        for i in xrange(len(X) - 1):
            if i == 0:
                E.append(X[i])
                continue
            if i == len(X) - 2:
                E.append(X[i + 1])
                continue
            E.append((lnp(X[i + 1]) - lnp(X[i]) + K[i - 1] *
                      X[i] - K[i + 1] * X[i + 1]) / (K[i - 1] - K[i + 1]))
        # E2 = [-float("inf")] + E + [float("inf")] #-infとinfと交点をあわせた完全な区切り位置

        def zqij(i, j):
            ff = lambda jj, a, b: p(
                X[jj]) / K[jj] * (np.exp(K[jj] * (b - X[jj])) - np.exp(K[jj] * (a - X[jj])))
            if i == 0 and j == 0:
                return ff(0, -float("inf"), X[0])
            if i == len(X) and j == 0:
                return ff(len(X) - 2, X[-1], float("inf"))
            if i == 1 and j == 0:
                return ff(1, X[0], X[1])
            if i == len(X) - 1 and j == 0:
                return ff(len(X) - 3, X[-2], X[-1])
            if (i == 0 or i == 1 or i == len(X) or i == len(X) - 1) and j == 1:
                return 0.
            if j == 0:
                return ff(i - 2, X[i - 1], E[i - 1])
            else:
                return ff(i, E[i - 1], X[i])

        Zqij = [(zqij(i, 0), zqij(i, 1)) for i in xrange(len(X) + 1)]
        Zq = np.sum(Zqij)
        pi = [sum(Zqij[i]) / float(Zq)
              for i in xrange(len(X) + 1)]  # 各セクションが出現する確率(面積比)
        pij = [(Zqij[i][0] / sum(Zqij[i]), 1. - Zqij[i][0] / sum(Zqij[i]))
               for i in xrange(len(X) + 1)]

        return K, E, Zq, pi, pij

    def randPWED(size=1):  # 区分的指数分布に従う乱数を発生する
        I = np.random.choice(range(len(X) + 1), p=pi, size=size)  # どのセクションを選ぶか
        J = np.array([np.random.choice(range(2), p=pij[i]) for i in I])
        U = np.random.uniform(size=size)  # 0~1の乱数

        def fz(i, j, u):
            ff = lambda ii, a, b: 1.0 / \
                K[ii] * np.log((1 - u) * np.exp(K[ii] * a) +
                               u * np.exp(K[ii] * b))
            if i == 0 and j == 0:
                return ff(0, -float("inf"), X[0])
            if i == len(X) and j == 0:
                return ff(i - 2, X[-1], float("inf"))
            if i == 1 and j == 0:
                return ff(1, X[0], X[1])
            if i == len(X) - 1 and j == 0:
                return ff(i - 2, X[-2], X[-1])
            if (i == 0 or i == 1 or i == len(X) or i == len(X) - 1) and j == 1:
                return None
            if j == 0:
                return ff(i - 2, X[i - 1], E[i - 1])
            else:
                return ff(i, E[i - 1], X[i])
        z = [fz(i, j, u) for i, j, u in zip(I, J, U)]
        return np.array(z)

    # 初期化
    K, E, Zq, pi, pij = update()
    # メインループ(棄却サンプリング)
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
                K, E, Zq, pi, pij = update()

    return Samples, Zq, lambda x: q(x)

import matplotlib.pyplot as plt
import scipy.special as ssp

def __test():
    a = 10.0
    b = 1.0
    X0 = [2.0, 5.0, 9.0, 14.0]

    p = lambda x: x ** (a - 1) * np.exp(-b * x)
    lnp = lambda x: (a - 1) * np.log(x) - b * x
    dlnp = lambda x: (a - 1) / float(x) - b

    Z, Zq, q = ARS2(p, lnp, X0, size=100)

    print "p(s=1) =", ssp.gamma(a) / b ** a / Zq

    plt.figure(figsize=(8, 6))
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')

    plt.xlim([0, 22])
    # plt.ylim([0,13])

    x = np.arange(0.0, 35.0, 0.05)
    y = b ** a / ssp.gamma(a) * p(x)
    y = p(x)
    #y = np.log(y)
    plt.plot(x, y, "b", linewidth=0.9, alpha=1.0)

    y = 1. / Zq * q(x)
    y = q(x)
    #y = np.log(y)
    plt.plot(x, y, "g", linewidth=0.9, alpha=1.0)

    #plt.vlines(X0, [0], lnp(np.array(X0)), linestyles=[(0,(2,2,2,2))], colors='k')

    #plt.hist(Z, bins=60, normed=True, facecolor="r", alpha=.1)
    plt.show()

if __name__ == "__main__":
    __test()

