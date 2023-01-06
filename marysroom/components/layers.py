import numpy as np
from marysroom.components import activation_funcs, optimizers


class Layer:
    def set_optimizer(self, **na):
        pass

    def forward(self, X):
        return X

    def backprop(self, grad):
        return grad

    def reset(self):
        pass

    def get_args(self):
        return {}


class Dense(Layer):
    def __init__(self, n_ins=0, n_outs=0, *, ap=np, w_sigma=0.1, w_mu=0, b_sigma=1, b_mu=0, act='none',
                 weights=np.empty([]), biases=np.empty([]),
                 ):

        self.ap = ap
        act = act.upper().replace(' ', '').replace('-', '').replace('_', '')
        self.act = getattr(activation_funcs, act)(ap=self.ap)

        if np.any(weights):
            self.w = self.ap.array(weights)
        else:
            self.w = w_sigma * self.ap.random.randn(n_ins, n_outs) + w_mu

        if np.any(biases):
            self.b = self.ap.array(biases)
        else:
            self.b = b_sigma * self.ap.random.randn(1, n_outs) + b_mu

    def set_optimizer(self, opt, lr, **opt_args):
        opt = opt.upper().replace(' ', '').replace('-', '').replace('_', '')
        self.w_opt = getattr(optimizers, opt.upper())(
            params=self.w, lr=lr, ap=self.ap, **opt_args)
        self.b_opt = getattr(optimizers, opt.upper())(
            params=self.b, lr=lr, ap=self.ap, **opt_args)

    def forward(self, X):
        self.x = X
        return self.act.forward(self.ap.dot(X, self.w) + self.b)

    def backprop(self, grad):
        grad = self.act.backprop(grad)

        w_grad = self.ap.dot(self.x.T, grad) / self.x.shape[0]
        b_grad = self.ap.mean(grad, axis=0, keepdims=True)

        self.w = self.w_opt.backprop(grad=w_grad, params=self.w)
        self.b = self.b_opt.backprop(grad=b_grad, params=self.b)

        return self.ap.dot(grad, self.w.T)

    def get_args(self):
        return {'weights': self.w, 'biases': self.b, 'act': str(type(self.act).__name__)}


class Flatten(Layer):
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, X):
        return X.reshape([-1]+self.out_shape)

    def backprop(self, grad):
        return grad.reshape([-1]+self.in_shape)

    def get_args(self):
        return {'in_shape': self.in_shape, 'out_shape': self.out_shape}


class Fully_Recurrent(Layer):
    def __init__(self, n_ins=0, n_outs=0, n_hid=0, *, ap=np, act='none', w_sigma=0.1, w_mu=0, b_sigma=1, b_mu=0,
                 weights=np.empty([]), biases=np.empty([]),
                 ):
        self.ap = ap

        act = act.upper().replace(' ', '').replace('-', '').replace('_', '')
        self.act = getattr(activation_funcs, act)(self.ap)

        self.n_ins = n_ins
        self.n_outs = n_outs
        self.n_tot = n_ins + n_outs + n_hid

        if np.any(weights):
            self.w = self.ap.array(weights)
        else:
            self.w = w_sigma * \
                self.ap.random.randn(self.n_tot, self.n_tot) + w_mu

        if np.any(biases):
            self.b = self.ap.array(biases)
        else:
            self.b = b_sigma * self.ap.random.randn(1, self.n_tot) + b_mu

    def set_optimizer(self, opt, lr, **opt_args):
        opt = opt.upper().replace(' ', '').replace('-', '').replace('_', '')
        self.w_opt = getattr(optimizers, opt.upper())(
            params=self.w, lr=lr, **opt_args)
        self.b_opt = getattr(optimizers, opt.upper())(
            params=self.b, lr=lr, **opt_args)

    def reset(self):
        self.n *= 0
        self.grad *= 0

    def forward(self, X):
        self.n[:, :self.n_ins] = X
        self.x = self.n

        self.n = self.act.forward(self.ap.dot(self.n, self.w) + self.b)

        return self.n[:, -self.n_outs:]

    def backprop(self, grad):
        self.grad[:, -self.n_outs:] = grad

        w_grad = (self.ap.dot(self.x.T, self.grad) / self.x.shape[0])
        b_grad = self.ap.mean(self.grad, axis=0, keepdims=True)

        self.w = self.w_opt.backprop(grad=w_grad, params=self.w)
        self.b = self.b_opt.backprop(grad=b_grad, params=self.b)

        self.grad = self.ap.dot(self.grad, self.w.T)
        return self.grad[:, :self.n_ins]

    def get_args(self):
        return {'weights': self.w, 'biases': self.b, 'act': str(type(self.act).__name__)}
