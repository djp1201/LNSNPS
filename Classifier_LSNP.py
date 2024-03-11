import numpy as np
class Classifier():
    def __init__(self, kernel_value, y):
        m = (kernel_value.shape[1])
        n = y.shape[1]
        self.deta = 0.05
        self.w = np.random.uniform(0.45, 0.55, size=(m, n))
        self.b = np.full((n),0)
    def train_classification(self, spikes, y):
        for i in range(spikes.shape[0]):
            output = np.zeros(y.shape[1])
            input_spikes = np.dot(spikes[i, :], self.w) + self.b
            index = np.argmax(input_spikes)
            output[index] = 1
            if (output == y[i, :]).all():
                self.w, self.b = self.deta_rule(y[i, :] - output, spikes[i, :])
            else:
                self.w, self.b = self.deta_rule(y[i, :] - output, spikes[i, :])
        return self.w, self.b
    def deta_rule(self, e, x):
        if (e == np.zeros(len(e))).all():
            pass
        else:
            for i in range(self.w.shape[1]):
                self.w[:, i] = self.w[:, i] + 2*self.deta * x * e[i]
                self.b[i] = self.b[i] + 2*self.deta * e[i]
        return self.w, self.b
