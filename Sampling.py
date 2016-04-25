# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
import scipy.special as ssp

R = (0.0, 30.0)
a = 10.0
b = 1.0
X = [4.0, 10.0, 19.0]
X = [4.0, 5.9150968164109567, 7.262318738878041, 10.0,
     11.851573086280592, 14.352782269930136, 19.0]
#X = [4.0, 10.0]

p = lambda x: x ** (a - 1) * np.exp(-b * x)
lnp = lambda x: (a - 1) * np.log(x) - b * x
dlnp = lambda x: (a - 1) / float(x) - b


def section(x, E):  # リストEの間を返す
    xx = np.array(x)

    def f(xi):
        for i in xrange(len(E)):
            if xi <= E[i]:
                return i
        return len(E)
    if xx.size == 1:
        return np.array(f(x))
    return np.array([f(i) for i in xx])

# 傾きの計算
L = [dlnp(xi) for xi in X]

E = []  # 包絡関数の交点 len(E) = len(X) - 1
for i in xrange(len(X) - 1):
    E.append((L[i] * X[i] - L[i + 1] * X[i + 1] - lnp(X[i]) +
              lnp(X[i + 1])) / float(L[i] - L[i + 1]))
E2 = [-float("inf")] + E + [float("inf")]  # -infとinfを追加した完全な区切り位置


def q(x):  # 提案分布(正規化されていない区分的指数分布)
    i = section(x, E)  # len(x) == len(i)
    if i.size == 1:
        return np.array(p(X[i]) * np.exp(L[i] * (x - X[i])))
    List = [p(X[ii]) * np.exp(L[ii] * (xi - X[ii])) for ii, xi in zip(i, x)]
    return np.array(List)

# 正規化係数の計算
Zqi = [p(xi) / -L[i] * (np.exp(L[i] * (E2[i] - xi)) -
                        np.exp(L[i] * (E2[i + 1] - xi))) for i, xi in enumerate(X)]
Zq = sum(Zqi)

K = [zqi / float(Zq) for zqi in Zqi]  # 各セクションが出現する確率(面積比)
print "E2 =", E2
print "Zqi =", Zqi
print "K =", K
print "p(s=1) =", ssp.gamma(a) / b ** a / Zq


def randPWED(size=1):  # 区分的指数分布に従う乱数を発生する
    I = np.random.choice(range(len(X)), p=K, size=size)  # どのセクションを選ぶか
    U = np.random.uniform(size=size)  # 0~1の乱数
    z = [1.0 / L[i] * np.log((1 - u) * np.exp(L[i] * E2[i]) +
                             u * np.exp(L[i] * E2[i + 1])) for i, u in zip(I, U)]
    return np.array(z)


def plot1():
    plt.figure(figsize=(8, 6))
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')

    x = np.arange(R[0], R[1], 0.05)
    # ガンマ分布
    y1 = p(x)
    #y1 = np.log(y1)
    plt.plot(x, y1, "b", linewidth=0.9, alpha=1.0, label="base")

    for ei in E:
        plt.axvline(x=ei, color='green')

    x = np.arange(0.0, 30.0, 0.05)
    y2 = q(x)
    #y2 = np.log(y2)
    plt.plot(x, y2, "r", linewidth=0.9, alpha=1.0)

    # plt.ylim([0,12])
    plt.xlim([0, 22])
    plt.vlines(X, [0], p(np.array(X)), linestyles=[
               (0, (2, 2, 2, 2))], colors='k')

    plt.show()


def plot2():
    plt.figure(figsize=(8, 6))

    x = np.arange(R[0], R[1], 0.05)
    # 区分的指数分布に従う乱数のヒストグラム
    Z = randPWED(size=100000)
    plt.hist(Z, bins=80, normed=True, alpha=.1)

    # 区分的指数分布(正規化)
    y3 = 1.0 / Zq * q(x)
    plt.plot(x, y3, "r", linewidth=0.9, alpha=1.0)

    for ei in E:
        plt.axvline(x=ei, color='green')

    plt.show()


def plot3():
    # 切断指数分布をプロット

    lam = 1.0
    x0 = 2.0
    A = 2.0
    B = 4.0

    I = lambda x: np.array([A <= xi <= B for xi in x])

    p = lambda x: lam * np.exp(-lam * (x - x0)) * I(x)
    p2 = lambda x: lam * np.exp(-lam * x)

    plt.figure(figsize=(8, 5))
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.gca().get_xaxis().set_ticks_position('none')
    plt.gca().get_yaxis().set_ticks_position('none')

    x = np.arange(2, 4.05, 0.05)
    y = p(x)

    plt.plot(x, y, "r", linewidth=1.3, alpha=1.0)

    x = np.arange(0, 6.05, 0.05)
    y = p2(x)
    plt.plot(x, y, "g", linewidth=1.3, alpha=1.0)

    X = [A, B]
    plt.vlines(X, [0], p(np.array(X)), linestyles=[
               (0, (2, 2, 2, 2))], colors='k')

    plt.show()


plot1()
