import numpy as np
from scipy.special import comb
class Generator():
    def kernel_value(self, x, kernel_option):
        m, n = np.shape(x)
        p = np.zeros((m, n))
        for i in range(n):
            k = self.Normalized(x[:, i])
            p[:, i] = k
        kernel_value = self.cal_kernel_value(p, kernel_option)
        return kernel_value
    def cal_kernel_value(self, x, kernel_option):
        m, n = np.shape(x)
        if kernel_option == 1:
            kernel_value = x
        elif kernel_option == 2:
            k1 = int(comb(n, 1))
            k2 = int(comb(n + 1, 2))
            k = 0
            kernel_value = np.zeros((m, k1 + k2))
            for i in range(k1):
                kernel_value[:, i] = x[:, i]
            kernel_value[:, k1:] = np.triu(np.dot(x, x.T)).reshape(-1, k2)[:m]
        elif kernel_option == 3:
            k1 = int(comb(n, 1))
            k2 = int(comb(n + 1, 2))
            k3 = int(comb(n + 2, 3))
            k = 0
            kernel_value = np.zeros((m, k1 + k2 + k3))
            for i in range(k1):
                kernel_value[:, i] = x[:, i]
            kernel_value[:, k1:k1 + k2] = np.triu(np.dot(x, x.T)).reshape(-1, k2)[:m]
            kernel_value[:, k1 + k2:] = np.triu(np.dot(x, np.dot(x.T, x))).reshape(-1, k3)[:m]
        elif kernel_option == 4:
            k1 = int(comb(n, 1))
            k2 = int(comb(n + 1, 2))
            k3 = int(comb(n + 2, 3))
            k4 = int(comb(n + 3, 4))
            k = 0
            kernel_value = np.zeros((m, k1 + k2 + k3 + k4))
            for i in range(k1):
                kernel_value[:, i] = x[:, i]
            kernel_value[:, k1:k1 + k2] = np.triu(np.dot(x, x.T)).reshape(-1, k2)[:m]
            kernel_value[:, k1 + k2:k1 + k2 + k3] = np.triu(np.dot(x, np.dot(x.T, x))).reshape(-1, k3)[:m]
            kernel_value[:, k1 + k2 + k3:] = np.triu(np.dot(x, np.dot(x.T, np.dot(x, x.T)))).reshape(-1, k4)[:m]
        else:
            k1 = int(comb(n, 1))
            k2 = int(comb(n + 1, 2))
            k3 = int(comb(n + 2, 3))
            k4 = int(comb(n + 3, 4))
            k5 = int(comb(n + 4, 5))
            k = 0
            kernel_value = np.zeros((m, k1 + k2 + k3 + k4 + k5))
            for i in range(k1):
                kernel_value[:, i] = x[:, i]
            kernel_value[:, k1:k1 + k2] = np.triu(np.dot(x, x.T)).reshape(-1, k2)[:m]
            kernel_value[:, k1 + k2:k1 + k2 + k3] = np.triu(np.dot(x, np.dot(x.T, x))).reshape(-1, k3)[:m]
            kernel_value[:, k1 + k2 + k3:k1 + k2 + k3 + k4] = np.triu(np.dot(x, np.dot(x.T, np.dot(x, x.T)))).reshape(
                -1, k4)[:m]
            kernel_value[:, k1 + k2 + k3 + k4:] = np.triu(np.dot(x, np.dot(x.T, np.dot(x, np.dot(x.T, x))))).reshape(-1,
                                                                                                                     k5)[
                                                  :m]
        return kernel_value

    def Normalized(self, x):
        k_1 = max(x)
        k_2 = min(x)
        m = len(x)
        k = np.zeros(m)
        if (k_1 == k_2):
            pass
        else:
            for i in range(m):
                k[i] = ((x[i]-k_2)/(k_1-k_2))
        return k
    def neuron(self, k, kernel_value):
        if k >= 1:
            spikes = kernel_value
        else:
            pass
        return spikes