import random
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, log, sin, cos, pi

def UniGaussianDataGenerator(n=1, mu=0, sigma=1):
    u = np.random.uniform(size=n)
    v = np.random.uniform(size=n)
    
    z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * pi * v)
    z2 = np.sqrt(-2 * np.log(u)) * np.cos(2 * pi * v)
    
    x1 = mu + z1 * sigma
    x2 = mu + z2 * sigma
    return x1, x2

def PolynomialDataGenerator(n=1, degree=2, mu=0, sigma=1, w=None, lower=-1, upper=1):
    if w == None:
        w = [i+1 for i in range(degree)]
    e, _ = UniGaussianDataGenerator(n, mu, sigma)
    X = np.random.uniform(lower, upper, size=n)
    y = np.zeros(n)
    for i in range(degree):
        y += w[i] * np.power(X, i)
    y += e
    return X, y

if __name__ == '__main__':
    X, y = PolynomialDataGenerator(n=10, degree=2, sigma=0)

    plt.scatter(X, y)
    plt.show()