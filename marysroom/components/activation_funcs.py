import numpy as np


class Act:
    def __init__(self, ap=np):
        self.ap = ap

    def forward(self, X):
        return X

    def backprop(self, grad):
        return grad


class SIGMOID(Act):
    def forward(self, X):
        self.x = X
        return 1/(1 + self.ap.exp(-X))

    def backprop(self, grad):
        e = self.ap.exp(self.x)
        return (e / ((e + 1)**2)) * grad


class SOFTMAX(Act):
    def forward(self, X):
        self.x = X
        e = self.ap.exp(X - self.ap.max(X, axis=1, keepdims=True))
        return e / self.ap.sum(e, axis=1, keepdims=True)

    def backprop(self, grad):
        e = self.ap.exp(self.x - self.ap.max(self.x, axis=1, keepdims=True))
        esum = self.ap.sum(e, axis=1, keepdims=True)

        return ((e * esum - e**2) / esum**2) * grad
