from deep.models import NN

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


from deep.layers.base import Layer


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

    def __init__(self, nin, n_hidden, nout):

        rng = np.random.RandomState(1234)
        lr = T.scalar()
        x = T.matrix()
        t = T.scalar()

        self.W_uh = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_hh = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.b_hh = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.W_uh = theano.shared(self.W_uh, 'W_uh')
        self.W_hh = theano.shared(self.W_hh, 'W_hh')
        self.b_hh = theano.shared(self.b_hh, 'b_hh')

        self.W_hy = np.asarray(rng.normal(size=(n_hidden, nout), scale=.01, loc=0.0), dtype=theano.config.floatX)
        self.b_hy = np.zeros((nout,), dtype=theano.config.floatX)
        self.W_hy = theano.shared(self.W_hy, 'W_hy')
        self.b_hy = theano.shared(self.b_hy, 'b_hy')

        self.activ = T.nnet.sigmoid

        h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h, _ = theano.scan(self.recurrent_fn, sequences=x,
                           outputs_info=[h0_tm1],
                           non_sequences=[self.W_hh, self.W_uh, self.W_hy, self.b_hh])

        y = T.dot(h[-1], self.W_hy) + self.b_hy


        cost = ((t - y) ** 2).mean(axis=0).sum()
        gW_hh, gW_uh, gW_hy, gb_hh, gb_hy = T.grad(cost, [self.W_hh, self.W_uh, self.W_hy, self.b_hh, self.b_hy])
        self.train_step = theano.function([x, t, lr], cost,
                                          on_unused_input='warn',
                                          updates=[(self.W_hh, self.W_hh - lr * gW_hh),
                                                   (self.W_uh, self.W_uh - lr * gW_uh),
                                                   (self.W_hy, self.W_hy - lr * gW_hy),
                                                   (self.b_hh, self.b_hh - lr * gb_hh),
                                                   (self.b_hy, self.b_hy - lr * gb_hy)],
                                          allow_input_downcast=True)

    def recurrent_fn(self, u_t, h_tm1, W_hh, W_uh, W_hy, b_hh):
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