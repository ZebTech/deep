from deep.models import NN
import theano.tensor as T
_x = T.matrix()
_y = T.scalar()
_i = T.lscalar()

import theano
import numpy as np

from deep.updates import GradientDescent
from deep.fit import Iterative
from theano import shared, config


class RNN(NN):

    def __init__(self, output_layers, recurrent_layers,
                 learning_rate=0.0001, update=GradientDescent(),
                 fit_type=Iterative()):
        self.update = update
        self.learning_rate = learning_rate
        self.output_layers = output_layers
        self.recurrent_layers = recurrent_layers
        self.fit_scores = []
        self.fit_model = fit_type.fit_model

        #: 50 = n_hidden (should this go in recurrent layer?)
        self.h0_tm1 = theano.shared(np.zeros(50, dtype=theano.config.floatX))

    @property
    def params(self):
        output_params = [param for layer in self.output_layers for param in layer.params]
        recurrent_params = [param for layer in self.recurrent_layers for param in layer.params]
        return recurrent_params + output_params

    def _symbolic_updates(self, x, y):
        cost = self._symbolic_score(x, y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

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

    def score(self, X, y):
        try:
            x = T.matrix()
            t = T.scalar()
            return self._symbolic_score(x, t).eval({x: X, t: y})
        except AttributeError:
            raise AttributeError("'{}' model has not been fit yet. "
                                 "Trying calling fit() or fit_layers()."
                                 .format(self.__class__.__name__))

    def fit(self, X, y):
        self.fit_layers(X[0].shape)
        self.fit_model(self, X, y)
        return self

    def fit_function(self, X, y):
        score = self._symbolic_score(_x, _y)
        updates = self._symbolic_updates(_x, _y)
        givens = self.fit_givens(X, y)
        return theano.function([_i], score, None, updates, givens)

    def fit_givens(self, X, y):
        givens = dict()
        X = shared(np.asarray(X, dtype=config.floatX))
        y = shared(np.asarray(y, dtype='float64'))
        givens[_x] = X[_i]
        givens[_y] = y[_i]
        return givens

    def fit_layers(self, shape):
        for layer in self.recurrent_layers + self.output_layers:
            try:
                shape = layer.shape
            except AttributeError:
                layer.fit(shape)
                shape = layer.shape