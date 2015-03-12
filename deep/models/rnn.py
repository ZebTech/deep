from deep.models import NN

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


from deep.layers.base import Layer
from deep.updates import GradientDescent


class RecurrentLayer(Layer):

    def __init__(self, n_hidden, activation):
        self.n_hidden = n_hidden
        self.activation = activation

    def __call__(self, X):
        pass

    def fit(self, incoming_shape):
        size = incoming_shape[1], self.n_hidden
        self.W_hidden = self.initialize.W(size)
        self.b = self.initialize.b(self.n_hidden)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W, self.b, self.activation.params
        return self.W, self.b

    @property
    def shape(self):
        return self.W.get_value().shape


class RNN(NN):

    def softmax_layer(self, n_hidden, nout, rng):
        self.W = np.asarray(rng.normal(size=(n_hidden, nout), scale=.01, loc=0.0), dtype=theano.config.floatX)
        self.b = np.zeros((nout,), dtype=theano.config.floatX)
        self.W = theano.shared(self.W, 'W_hy')
        self.b = theano.shared(self.b, 'b_hy')

    def recurrent_layer(self, n_hidden, nin, rng):
        self.W_uh = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_hh = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.b_hh = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.W_uh = theano.shared(self.W_uh, 'W_uh')
        self.W_hh = theano.shared(self.W_hh, 'W_hh')
        self.b_hh = theano.shared(self.b_hh, 'b_hh')
        self.activ = T.nnet.sigmoid
        self.h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

    def __init__(self, nin, n_hidden, nout, update=GradientDescent()):
        self.update = update

        #: output layer
        from deep.layers import Dense
        from deep.activations import Identity
        self.output_layer = Dense(nout, Identity())
        shape = 10, n_hidden
        self.output_layer.fit(shape)

        rng = np.random.RandomState(1234)
        self.lr = T.scalar()
        x = T.matrix()
        y = T.scalar()

        self.recurrent_layer(n_hidden, nin, rng)

        cost = self._symbolic_score(x, y)
        updates = self._symbolic_updates(x, y)
        self.train_step = theano.function([x, y, self.lr], cost, updates=updates)

    @property
    def params(self):
        return [self.W_hh, self.W_uh, self.b_hh] + list(self.output_layer.params)
        #return [self.W_hh, self.W_uh, self.W, self.b_hh, self.b]

    def _symbolic_predict(self, x):
        h, _ = theano.scan(self.recurrent_fn, x, [self.h0_tm1], [self.W_hh, self.W_uh, self.b_hh])

        return self.output_layer(h[-1])
        #return T.dot(h[-1], self.W) + self.b

    def _symbolic_score(self, x, y):
        cost = ((y - self._symbolic_predict(x)) ** 2).mean(axis=0).sum()
        return cost

    def _symbolic_updates(self, x, y):
        cost = self._symbolic_score(x, y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.lr))
        return updates

    def recurrent_fn(self, u_t, h_tm1, W_hh, W_uh, b_hh):
        h_t = self.activ(T.dot(h_tm1, W_hh) + T.dot(u_t, W_uh) + b_hh)
        return h_t

    def fit(self, X, y):
        vals = []
        for i in range(10):
            for x, y_ in zip(X, y):
                c = rnn.train_step(x, y_, lr)
                vals.append(c)
            plt.plot(vals)
        plt.show()


if __name__ == '__main__':

    rnn = RNN(2, 50, 1)
    lr = 0.0001

    X = np.random.rand(100, 100, 2)
    y = np.sum(X[:, :, 0] * X[:, :, 1], axis=1)

    rnn.fit(X, y)