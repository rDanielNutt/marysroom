import numpy as np


class Opt:
    def __init__(self, lr, ap=np, **opt_args):
        self.ap = ap
        self.lr = lr


class NADAM(Opt):
    def __init__(self, *, params, decayG=0.8, decayV=0.9, epsilon=1e-8, **opt_args):

        super().__init__(**opt_args)

        self.decayG = decayG
        self.decayV = decayV
        self.eps = epsilon

        self.vel = self.ap.zeros(shape=params.shape)
        self.grads = self.ap.zeros(shape=params.shape)
        self.prev_params = params

    def backprop(self, *, grad, params):
        self.vel = (self.decayV * self.vel) + ((1 - self.decayV) * grad)
        self.grads = (self.decayG * self.grads) + ((1 - self.decayG) * grad**2)

        v = self.vel/(1 - self.decayV)
        g = self.grads/(1 - self.decayG)

        next_params = self.prev_params - \
            (v * (self.lr/(self.eps + np.sqrt(g))))
        self.prev_params = params

        return next_params


class NESTMOMENTUM(Opt):
    def __init__(self, *, params, decayV=0.8, **opt_args):
        super().__init__(**opt_args)

        self.decayV = decayV
        self.vel = self.ap.zeros(shape=params.shape)
        self.prev_params = params

    def backprop(self, *, grad, params):
        self.vel = (self.vel * self.decayV) - (self.lr * grad)
        next_params = self.prev_params + self.vel
        self.prev_params = params
        return next_params


class GRADDESCENT(Opt):
    def backprop(self, *, grad, params):
        return params - (self.lr * grad)
