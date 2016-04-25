ß# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
import scipy.special as ssp

R = (-6.0, 6.0)
sp = 1.0
sq = 1.01 * sp
k = sq / sp
D = 1000

p = lambda x: sst.norm.pdf(x, loc=0, scale=sp)
q = lambda x: sst.norm.pdf(x, loc=0, scale=sq)

f = lambda x: sq / sp + np.exp(-x * x / 2. * (1. / sp / sp - 1. / sq / sq))

plt.ylim([0, 0.45])

x = np.arange(R[0], R[1], 0.05)

y1 = p(x)
plt.plot(x, y1, "b", linewidth=0.9, alpha=1.0, label="base")

y2 = k * q(x)
plt.plot(x, y2, "r", linewidth=0.9, alpha=1.0, label="base")

y3 = f(x)
#plt.plot(x, y3, "g", linewidth=0.9, alpha=1.0, label="base")

# plt.show()

print "p(s=1) =", (k) ** D
