import numpy as np
from math import ceil, exp

# ----------------------------
#
#  Simple Linear Regresion
#
# ----------------------------


def lfit(y, x, weights):
    w = np.eye(len(weights))*(weights)
    mat = (x.T.dot(w)).dot(x)
    try:
        mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        mat = np.linalg.pinv(mat)

    return((mat.dot(x.T).dot(w).dot(y)))

# ---------------------------------------------------
#
#   Regression around a point using given kernel
#
# ---------------------------------------------------


def LocalKfit(y, pos, kernel, mode=1):
    if kernel == [[], [], []]:
        print(f"-------------------BUG HERE-------------------{pos}\n")
        return np.array([y[tuple(pos)], 0, 0, 0, 0, 0])
    ind0 = np.array(kernel[0])
    ind1 = np.array(kernel[1])
    rows = y.shape[0]
    cols = y.shape[1]

    def f(a, max):
        for i in range(len(a)):
            a[i] = abs(a[i])
            a[i] = min(a[i], max - abs(max - a[i]))
        return a
    ind0 = f(ind0, rows-1)
    ind1 = f(ind1, cols-1)
    try:
        data = y[ind0, ind1]
    except IndexError:
        print(f"{ind0}----{ind1}----{kernel}")
    weights = kernel[2]
    x = np.array([kernel[0], kernel[1]]).T-pos
    if mode == 1:
        x = np.array([np.repeat(1, len(weights)), x[:, 0], x[:, 1]]).T
    elif mode == 2:
        x = np.array([np.repeat(1, len(weights)), x[:, 0], x[:, 1],
                      x[:, 0]**2, x[:, 1]**2, x[:, 0]*x[:, 1]]).T
    return(lfit(data, x, weights))

def LocalSfit(y,pos,kernel,kernels,var,mode = 0):
    smoothing =  2*var*100
    if kernel == [[], [], []]:
        print(f"-------------------BUG HERE-------------------{pos}\n")
        return np.array([y[tuple(pos)], 0, 0, 0, 0, 0])
    ind0 = np.array(kernel[0])
    ind1 = np.array(kernel[1])
    rows = y.shape[0]
    cols = y.shape[1]
    def f(a, max):
        for i in range(len(a)):
            a[i] = abs(a[i])
            a[i] = min(a[i], max - abs(max - a[i]))
        return a
    ind0 = f(ind0, rows-1)
    ind1 = f(ind1, cols-1)
    data = y[ind0,ind1]
    weights = kernel[2]
    k = kernels.eval(pos)
    k[0] = f(k[0],rows - 1)
    k[1] = f(k[1],cols - 1)
    atp = y[k[0],k[1]]
    denom = ((np.sum(atp**2))**0.5)*smoothing
    for i in range(len(weights)):
        coord = [ind0[i],ind1[i]]
        k = kernels.eval(coord)
        k[0] = f(k[0],rows - 1)
        k[1] = f(k[1],cols - 1)
        atop = y[k[0],k[1]]
        weights[i] = exp(-(np.sum((atop - atp)**2)/denom))
    estimate = np.sum(weights*data)/np.sum(weights)
    x = np.array([kernel[0], kernel[1]]).T-pos
    if mode == 1:
        x = np.array([np.repeat(1, len(weights)), x[:, 0], x[:, 1]]).T
        estimate = lfit(data, x, weights)[0]
    elif mode == 2:
        x = np.array([np.repeat(1, len(weights)), x[:, 0], x[:, 1],
                      x[:, 0]**2, x[:, 1]**2, x[:, 0]*x[:, 1]]).T
        estimate = lfit(data, x, weights)[0]
    return(estimate)

def Distance(pos1, pos2, band):
    res = (pos1-pos2)/band
    return sum(res**2) <= 1

def Epanechnikov(pos1, pos2, band):
    res = (pos1-pos2)/band
    u = sum(res**2)
    if (u>1):
        return 0
    else:
        return 3*(1-u)/4


# ------------------------------------------------------------------------
#
#     Kernel object which return value of kernel at a given point
#
# comment: This probably is the reason behind time increase in regression
#
# ------------------------------------------------------------------------


class Kernel(object):
    def DefK(self, default, band, distfunc):
        if default != 0:
            return default
        else:
            max = self.max
            pos = np.array([max, max])
            res = [list(), list(), list()]
            for i in range(2*max+1):
                for j in range(2*max+1):
                    w = distfunc(pos, np.array([i, j]), band)
                    if w != 0:
                        res[0].append(i-max)
                        res[1].append(j-max)
                        res[2].append(w)
            return res

    def eval(self, pos, filt=lambda pos, currpos: True):
        a = self.default
        b = [list(), list(), list()]
        for i in range(len(a[0])):
            res1 = a[0][i]+pos[0]
            res2 = a[1][i]+pos[1]
            if filt([res1, res2], pos):
                b[0].append(res1)
                b[1].append(res2)
                b[2].append(a[2][i])
        return b

    def __init__(self, rows, cols, band=[2, 2], distfunc= Epanechnikov , default=0,
                 maxdist=0):
        self.rows = rows
        self.cols = cols
        self.band = band
        self.max = ceil(max(maxdist, max(band)))
        self.default = Kernel.DefK(self, default, band, distfunc)
