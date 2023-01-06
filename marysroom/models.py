import numpy as np
import pickle

from components import layers, loss_funcs
from mlfuncs import one_hot


class NN:
    def __init__(self, ap=np):
        self.layers = []
        self.ap = np

    def save_net(self, path, filename):
        save_dict = {}
        for i, l in enumerate(self.layers):
            save_dict[i] = {'ltype': str(
                type(l).__name__), 'args': l.get_args()}

        with open(f'{path}{filename}.pickle', 'wb') as save_file:
            pickle.dump(save_dict, save_file)

    def load_net(self, path, filename):
        with open(f'{path}{filename}.pickle', 'rb') as load_file:
            load_dict = pickle.load(load_file)

        for l, layer in sorted(load_dict.items()):
            self.add_layer(getattr(layers, layer['ltype'])(
                ap=self.ap, **layer['args']))

    def add_layer(self, layer, layer_ind=None):
        if layer_ind:
            self.layers.insert(layer_ind, layer)
        else:
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backprop(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

    def predict(self, X):
        preds = self.forward(X)
        if self.ap == np:
            return preds
        else:
            return preds.get()

    def batch_data(self, X_train, y_train, batch_size):
        xt_batches = []
        yt_batches = []
        for i in range(0, X_train.shape[0]-batch_size, batch_size):
            xt_batches.append(self.ap.array(X_train[i:i+batch_size]))
            yt_batches.append(self.ap.array(y_train[i:i+batch_size]))

        if X_train.shape[0] % batch_size > (batch_size//10):
            xt_batches.append(self.ap.array(
                X_train[-(X_train.shape[0] % batch_size):]))
            yt_batches.append(self.ap.array(
                y_train[-(y_train.shape[0] % batch_size):]))

        return list(zip(xt_batches, yt_batches))

    def fit(self, X_train, y_train, *, lr, epochs=10, batch_size=1, display=False, onehot=False,
            checkpoints=None, checkpoint_path='', checkpoint_file='',
            loss='mse', opt='graddescent', **opt_args):

        if onehot:
            y_train = one_hot(y_train, self.layers[-1].w.shape[1])

        loss = loss.upper().replace(' ', '').replace('-', '').replace('_', '')
        loss_func = getattr(loss_funcs, loss)(self.ap)

        for l in self.layers:
            l.set_optimizer(opt, lr=lr, **opt_args)

        batches = self.batch_data(X_train, y_train, batch_size)

        n_dig = len(str(epochs))
        self.loss_curve = []
        for epoch in range(1, epochs+1):
            ep_loss = 0
            for b, (X_batch, y_batch) in enumerate(batches, 1):
                preds = self.forward(X_batch)

                if self.ap.isnan(preds).any() or self.ap.isinf(preds).any():
                    raise ValueError(f'Net Degraded on Epoch {epoch}')

                grad = loss_func.backprop(preds, y_batch)
                self.backprop(grad)

                ep_loss += (loss_func.calculate(preds, y_batch) - ep_loss)/b

            self.loss_curve.append(ep_loss)

            if display and (epoch % display == 0):
                print(
                    f'Epoch {epoch:<{n_dig}}: loss[{self.loss_curve[-1]:^10.4E}]')

            if checkpoints and (epoch % checkpoints == 0):
                self.save_net(path=checkpoint_path,
                              filename=f'{checkpoint_file}/{epoch}')
