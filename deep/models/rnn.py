from deep.models import NN

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


from deep.layers.base import Layer
from deep.updates import GradientDescent
from deep.initialize import Normal

class RecurrentLayer(Layer):

    def __init__(self, n_hidden, activation, initialize=Normal()):
        self.n_hidden = n_hidden
        self.activation = activation
        self.initialize = initialize

    def __call__(self, X, H):
        return self.activation(T.dot(H, self.W_hidden) + T.dot(X, self.W_visible) + self.b)

    def fit(self, incoming_shape):
        size = self.n_hidden, self.n_hidden
        self.W_hidden = self.initialize.W(size)
        size = incoming_shape[1], self.n_hidden
        self.W_visible = self.initialize.W(size)
        self.b = self.initialize.b(self.n_hidden)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W_visible, self.W_hidden, self.b, self.activation.params
        return self.W_visible, self.W_hidden, self.b

    @property
    def shape(self):
        return self.W_visible.get_value().shape


class RNN(NN):

    def __init__(self, output_layer, recurent_layer, update=GradientDescent()):
        self.update = update

        #: 50 = n_hidden (should this go in recurrent layer?
        self.h0_tm1 = theano.shared(np.zeros(50, dtype=theano.config.floatX))

        self.lr = T.scalar()

        self.output_layer = output_layer
        self.recurrent_layer_ = recurent_layer


    @property
    def params(self):
        return list(self.recurrent_layer_.params) + list(self.output_layer.params)

    def _symbolic_predict(self, x):
        h, _ = theano.scan(self.recurrent_layer_, x, [self.h0_tm1])
        return self.output_layer(h[-1])

    def _symbolic_score(self, x, y):
        return ((y - self._symbolic_predict(x)) ** 2).mean(axis=0).sum()

    def _symbolic_updates(self, x, y):
        cost = self._symbolic_score(x, y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.lr))
        return updates

    def fit(self, X, y):
        x = T.matrix()
        t = T.scalar()

        self.recurrent_layer_.fit(X[0].shape)
        shape = self.recurrent_layer_.shape
        self.output_layer.fit(shape)

        cost = self._symbolic_score(x, t)
        updates = self._symbolic_updates(x, t)
        self.train_step = theano.function([x, t, self.lr], cost, updates=updates)

        vals = []
        for i in range(10):
            for x, y_ in zip(X, y):
                c = rnn.train_step(x, y_, lr)
                vals.append(c)
            plt.plot(vals)
        plt.show()


if __name__ == '__main__':
    from deep.layers import Dense
    from deep.activations import Identity, Sigmoid

    rnn = RNN(Dense(1, Identity()), RecurrentLayer(50, Sigmoid()))
    lr = 0.0001

    X = np.random.rand(100, 100, 2)
    y = np.sum(X[:, :, 0] * X[:, :, 1], axis=1)

    rnn.fit(X, y)