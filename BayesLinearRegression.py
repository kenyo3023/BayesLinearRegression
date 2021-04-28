import matplotlib.pyplot as plt
import numpy as np
import math
from argparse import ArgumentParser

from Generator import UniGaussianDataGenerator, PolynomialDataGenerator

class BayesLinearRegression():
    def __init__(self, beta=1, degree=2, sigma=1, w=None):
        self.beta = beta
        self.degree = degree
        self.sigma = sigma
        if w == None:
            self.w = [i+1 for i in range(self.degree)]
        else:
            self.w = w
        self.iteration = 0
        self.X = []
        self.Y = []
        self.alpha = 1 / self.sigma
        self.prior_mean = np.zeros((self.degree, 1))
        self.prior_cov = (1 / self.beta) * np.identity(self.degree)
        
    def update(self, n=1):
        self.iteration += 1
        print('\n--------------------------------%d--------------------------------'%self.iteration)
        x, y = PolynomialDataGenerator(n=n, degree=self.degree, sigma=self.sigma, w=self.w)
        print('Add data point (%f, %f):'%(x, y))
        self.X.append(x)
        self.Y.append(y)
        self.x_poly = np.array([np.power(x, i) for i in range(self.degree)]).T
        
        self.post_cov = np.linalg.inv(np.linalg.inv(self.prior_cov) + self.alpha * np.dot(self.x_poly.T, self.x_poly))
        self.post_mean = np.dot(self.post_cov, (self.alpha * self.x_poly.T * y + \
                                                          np.dot(np.linalg.inv(self.prior_cov), self.prior_mean)))
        
        self.y_pred = np.dot(self.x_poly, self.post_mean)[0][0]
        self.y_var = (1 / self.alpha) + np.dot(np.dot(self.x_poly, self.post_cov), self.x_poly.T)[0][0]
        print('Predictive distribution ~ N(%f, %f)\n'%(self.y_pred, self.y_var))
        
    def fit(self, max_iter=3000, sample_size=100, save_every_iter=['10','50']):
        self.save_every_iter = save_every_iter
        self.save_meta_iter = {}
        while True:
            self.update()
            if self.iteration >= max_iter:
                w_pred = self.post_mean.reshape(self.degree,).tolist()
                cov_pred = self.post_cov
                break
            else:
                self.prior_mean = self.post_mean
                self.prior_cov = self.post_cov
                for i in self.save_every_iter:
                    i = int(i)
                    if self.iteration == i:
                        self.save_meta_iter['%d'%i] = self.save_iter(iter=i)
            
        plot_col = (1 + int(len(self.save_every_iter) / 2)) * 100
        X_simulate = np.linspace(-2, 2, sample_size)
        plt.figure(figsize=(10,8))
        plt.subplot(21+plot_col)
        self.show_result('Ground truth', X_simulate, self.w, self.sigma, False)
        plt.subplot(22+plot_col)
        self.show_result('Predict result', X_simulate, w_pred, cov_pred, True, self.X, self.Y)

        for i, iter_ in enumerate(self.save_every_iter):
            plt.subplot(20+plot_col+(i+3))
            self.show_result('After %s incomes'%iter_, X_simulate, self.save_meta_iter[iter_]['w'], self.save_meta_iter[iter_]['cov'], True, self.X[:int(iter_)], self.Y[:int(iter_)])
        plt.show()

    def save_iter(self, iter):
        iter_ = {}
        iter_['w'] = self.post_mean.reshape(self.degree,).tolist()
        iter_['cov'] = self.post_cov
        return iter_
    
    def show_result(self, title, X_simulate, w, cov, isCov, X=None, y=None):
        sample_size = len(X_simulate)
        x_poly = np.array([np.power(X_simulate, i) for i in range(self.degree)]).T
        y_pred = np.dot(x_poly, w)
        if isCov:
            x_poly = np.array(x_poly).reshape((sample_size, self.degree))
            y_var = (1 / self.alpha) + np.diagonal(np.dot(np.dot(x_poly, cov), x_poly.T))
            y_upper = y_pred + y_var
            y_lower = y_pred - y_var
        else:
            y_upper = y_pred + cov
            y_lower = y_pred - cov

        plt.plot(X_simulate, y_pred, 'k')
        plt.plot(X_simulate, y_upper, 'r')
        plt.plot(X_simulate, y_lower, 'r')
        if isCov:
            plt.scatter(X, y, color='b')
        plt.title(title)
        plt.xlim((-2, 2))
        plt.ylim((-20, 20))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--b', help='beta, the precision for initial prior w', default=1, type=int)
    parser.add_argument('--n', help='degree of input x', default=4, type=int)
    parser.add_argument('--a', help='sigma, variance of the error term', default=1, type=int)
    parser.add_argument('--w', help='weight of the x', default=None)
    parser.add_argument('--max_iter', help='maximum iteraction', default=3000, type=int)
    parser.add_argument('--save_every_iter', help='save the meta every iteraction', default='10,50')
    args = parser.parse_args()

    beta = args.b
    degree = args.n
    sigma = args.a
    w = args.w
    max_iter= args.max_iter
    save_every_iter = args.save_every_iter.split(',')

    np.random.seed(1)
    blr = BayesLinearRegression(beta=beta, degree=degree, sigma=sigma, w=w)
    blr.fit(max_iter=max_iter, save_every_iter=save_every_iter)