import numpy as np
class Classifier():
    def __init__(self, kernel_value, y ):

        m = (kernel_value.shape[1])
        n = y.shape[1]
        self.w = np.random.uniform(0.45, 0.55, size=(m, n))

    def train_classification(self, spikes , y ):
        for i in range(spikes.shape[0]):
            output = np.zeros(y.shape[1])
            input_spikes = np.dot(spikes[i, :], self.w)
            index = np.argmax(input_spikes)
            output[index] = 1

            if (output == y[i, :]).all():
                self.w= self.deta_rule(y[i, :] - output, spikes[i, :])
            else:
                self.w= self.deta_rule(y[i, :] - output, spikes[i, :])
        return self.w

    def deta_rule(self, e, x):
        # 计算误差信号和输入信号的范数
        x_norm = np.linalg.norm(x)
        # 更新自适应滤波器的权重
        if (e == np.zeros(len(e))).all():
            pass
        else:
            epsilon = 1e-8
            self.Alpha = float(self.Alpha)
            eta = self.Alpha / (epsilon + x_norm ** 2)
            for i in range(self.w.shape[1]):
                self.w[:, i] = self.w[:, i] - 0.01 * self.w[:, i] + x * e[i] * eta

        return self.w


