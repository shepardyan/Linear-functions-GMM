from typing import Union
from typing import Iterable
import numpy as np
from scipy.special import erf


def gaussian_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))


class GMM:
    def __init__(self, mu, sigma, weights):
        self.mu = np.array(mu).reshape(-1)
        self.sigma = np.array(sigma).reshape(-1)
        self.weights = np.array(weights).reshape(-1)
        assert self.mu.shape == self.sigma.shape
        assert self.mu.shape == self.weights.shape

    def __str__(self):
        return f"GMM with {len(self.mu)} components"

    def __repr__(self):
        return f"GMM with {len(self.mu)} components"

    @staticmethod
    def fit(data, num_of_gauss=3, max_iter=100):
        """
        利用EM算法生成GMM拟合数据
        :param data: array_like
        :param num_of_gauss: 高斯分量的个数
        :param max_iter: 最大迭代次数
        :return:
        """
        # initialize
        internal_data = np.array(data).reshape(1, -1)
        mu, sigma, weights = np.array([0] * num_of_gauss) + 10 * np.random.rand(
            num_of_gauss), np.array(
            [np.std(internal_data)] * num_of_gauss) + 10 * np.random.rand(num_of_gauss), np.array(
            [1 / num_of_gauss] * num_of_gauss)

        # iteration steps
        likelihood_last, likelihood_this = np.inf, 0
        step = 0
        while step <= max_iter and abs(likelihood_this - likelihood_last) > 1e-6:
            # E-step
            responses = weights.reshape(-1, 1) * np.vstack(
                [gaussian_pdf(internal_data, mu[i], sigma[i]) for i in range(num_of_gauss)])
            cs = responses.sum(axis=0)
            likelihood_last, likelihood_this = likelihood_this, np.mean(np.log(cs))
            responses /= cs  # normalize the weights
            # M-step
            rs = responses.sum(axis=1)
            weights = rs / responses.sum()  # normalize the new priors
            mu = np.sum(internal_data * responses, axis=1) / rs
            sigma = np.sqrt(np.sum((internal_data - mu.reshape(-1, 1)) ** 2 * responses, axis=1) / rs)
            print(f'End iteration {step}. Likelihood function = {likelihood_this}')
            step += 1
        return GMM(mu, sigma, weights)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if other != 0.:
                return GMM(np.array(self.mu) * other, np.sqrt(np.array(self.sigma) ** 2 * (other ** 2)),
                           weights=self.weights)
            else:
                return GMM([0], [0], [1.])
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, GMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = self.mu.reshape(-1, 1) + other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return GMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return GMM(self.mu + other, self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, GMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = self.mu.reshape(-1, 1) - other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return GMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return GMM(self.mu - other, self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __neg__(self):
        return GMM(-self.mu, self.sigma, self.weights)

    def __rsub__(self, other):
        if isinstance(other, GMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = -self.mu.reshape(-1, 1) + other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return GMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return -GMM(self.mu, self.sigma, self.weights) + other
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def pdf(self, x: Union[np.ndarray, Iterable, int, float]):
        """
        计算高斯混合模型的概率密度函数
        :param x: 需要计算的点
        :return:
        """
        if self.sigma[0] == 0:
            res = np.zeros_like(x)
            res[x == self.mu[0]] = np.inf
            return res
        else:
            pdf_data = self.weights[0] * gaussian_pdf(x, self.mu[0], self.sigma[0])
            if len(self.mu) > 1:
                for i in range(1, len(self.mu)):
                    pdf_data += self.weights[i] * gaussian_pdf(x, self.mu[i], self.sigma[i])
            return pdf_data

    def cdf(self, x: Union[np.ndarray, Iterable, int, float]):
        if self.sigma[0] == 0:
            return np.sign(x - self.mu[0])
        else:
            cdf_data = self.weights[0] * gaussian_cdf(x, self.mu[0], self.sigma[0])
            if len(self.mu) > 1:
                for i in range(1, len(self.mu)):
                    cdf_data += self.weights[i] * gaussian_cdf(x, self.mu[i], self.sigma[i])
            return cdf_data

    def sample(self, num_of_samples=1):
        """
        从高斯混合模型中采样
        :param num_of_samples: 采样数
        :return: data: 样本
        """
        if num_of_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample."
            )

        if self.sigma[0] == 0:
            return np.ones(int(num_of_samples))
        else:
            components = np.random.choice(len(self.mu), size=int(num_of_samples), p=self.weights)
            components_index = [np.nonzero(components == i)[0] for i in range(len(self.mu))]
            del components
            data = np.zeros(num_of_samples)
            for i, c in enumerate(components_index):
                data[c] = np.random.normal(loc=self.mu[i], scale=self.sigma[i], size=len(c))
            del components_index
            return data

    def normalization(self):
        if np.sum(self.weights) != 1.0:
            self.weights /= np.sum(self.weights)

    def reduce_components(self, tol=1e-6, copy=True):
        """
        根据给定阈值减少GMM的分量
        :param tol:
        :param copy:
        :return:
        """
        saved_index = np.nonzero(self.weights >= tol)[0]
        if copy:
            new_weight = self.weights[saved_index] / np.sum(self.weights[saved_index])
            return GMM(self.mu[saved_index], self.sigma[saved_index], new_weight)
        else:
            self.mu = self.mu[saved_index]
            self.sigma = self.sigma[saved_index]
            self.weights = self.weights[saved_index] / np.sum(self.weights[saved_index])

    def abs_risk(self, threshold=10.):
        return 1 - self.cdf(np.abs(threshold)) + self.cdf(-np.abs(threshold))
