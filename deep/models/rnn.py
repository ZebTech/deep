from deep.models import NN

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from deep.updates import GradientDescent


class RNN(NN):

    def __init__(self, output_layers, recurrent_layers,
                 learning_rate=0.0001, update=GradientDescent()):
        self.update = update
        self.learning_rate = learning_rate
        self.output_layers = output_layers
        self.recurrent_layers = recurrent_layers

        #: 50 = n_hidden (should this go in recurrent layer?)
        self.h0_tm1 = theano.shared(np.zeros(50, dtype=theano.config.floatX))

    @property
    def params(self):
        output_params = [param for layer in self.output_layers for param in layer.params]
        recurrent_params = [param for layer in self.recurrent_layers for param in layer.params]
        return recurrent_params + output_params

    def _symbolic_recurrence(self, x, h):
        for layer in self.recurrent_layers:
            h = layer(x, h)
        return h

    def _symbolic_predict(self, x):
        h, _ = theano.scan(self._symbolic_recurrence, x, [self.h0_tm1])

        x = h[-1]
        for layer in self.output_layers:
            x = layer(x)
        return x

    def _symbolic_score(self, x, y):
        return ((y - self._symbolic_predict(x)) ** 2).mean(axis=0).sum()

    def _symbolic_updates(self, x, y):
        cost = self._symbolic_score(x, y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def fit(self, X, y):
        x = T.matrix()
        t = T.scalar()

        shape = X[0].shape
        for layer in self.recurrent_layers + self.output_layers:
            layer.fit(shape)
            shape = layer.shape

        cost = self._symbolic_score(x, t)
        updates = self._symbolic_updates(x, t)

        self.train_step = theano.function([x, t], cost, updates=updates)

        vals = []
        for i in range(10):
            for x, y_ in zip(X, y):
                c = self.train_step(x, y_)
                vals.append(c)
            plt.plot(vals)
        plt.show()
