import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log, sin, cos, pi
from argparse import ArgumentParser

class UniGaussianGenerator():
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        
    def data_generate(self, n):
        u = np.random.uniform(size=n)
        v = np.random.uniform(size=n)
        z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * pi * v)
        z2 = np.sqrt(-2 * np.log(u)) * np.cos(2 * pi * v)
        x1 = self.mu + z1 * np.sqrt(self.sigma)
        x2 = self.mu + z2 * np.sqrt(self.sigma)
        return x1, x2
    
    def sample(self, n):
        self.x1, self.x2 = self.data_generate(n=n)
    
    def add_sample(self, n=1):
        x1s, x2s = self.data_generate(n=n)
        self.x1 = np.append(self.x1, x1s)
    
    def mean(self):
        return(np.sum(self.x1) / len(self.x1))
        
    def variance(self):
        N = len(self.x1)
        x1power2sum = np.sum(np.power(self.x1, 2))
        x1sumpower2 = np.power(np.sum(self.x1), 2) / N
        return((x1power2sum - x1sumpower2) / N)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mu', help='the mean of data generator', default=0, type=float)
    parser.add_argument('--sigma', help='the variance of data generator', default=1, type=float)
    parser.add_argument('--threshold', help='the threshold to stop iteraction', default=0.05, type=float)
    args = parser.parse_args()
    
    mu, sigma = args.mu, args.sigma
    threshold = args.threshold
    print('Data point source function: N(%.1f, %.1f)\n'%(mu, sigma))
    gen = UniGaussianGenerator(mu=mu, sigma=sigma)
    gen.sample(10)

    iter_ = 0
    while True:
        n = 1
        gen.add_sample(n=n)
        iter_ += 1
        print(iter_, 'Add data point: ', gen.x1[-n])
        print('Mean： %f    Variance: %f'%(gen.mean(), gen.variance()))
        print()
        if (abs(gen.mean()-mu)<0.05) and (abs(gen.variance()-sigma)<threshold):
            break