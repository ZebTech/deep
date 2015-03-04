# -*- coding: utf-8 -*-
"""
    deep.models.base
    ---------------------

    Implements the feed forward neural network model.

    :references: theano deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from deep.fit.base import Iterative
from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.updates.base import GradientDescent


class NN(object):

    def __init__(self, layers, learning_rate=10, update=GradientDescent(),
                 fit=Iterative(), cost=NegativeLogLikelihood(), regularize=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.update = update
        self.fit_method = fit
        self.cost = cost
        self.regularize = regularize

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    #: can we move this to updates?
    def _symbolic_updates(self, x, y):
        cost = self.cost(self._symbolic_predict_proba(x), y)

        if self.regularize is not None:
            for param in self.params:
                cost += self.regularize(param)

        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_predict_proba(self, X):
        for layer in self.layers:
            X = layer._symbolic_transform(X)
        return X

    def _symbolic_predict(self, x):
        return T.argmax(self._symbolic_predict_proba(x), axis=1)

    def _symbolic_score(self, x, y):
        from deep.costs import NegativeLogLikelihood
        # for plankton
        #cost = NegativeLogLikelihood()
        #return cost(self._symbolic_predict_proba(x), y)

        cost = PredictionError()
        return cost(self._symbolic_predict(x), y)


    def predict_proba(self, X):
        batch_size = self.fit_method.batch_size
        n_batches = len(X) / batch_size
        score_function = compile_predict_function(self, X, batch_size)
        return np.mean(map(score_function, range(n_batches)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        X = self.predict_proba(X)
        return -np.mean(np.log(X)[np.arange(y.shape[0]), y])

    def fit(self, X_train, y_train=None, X_valid=None, y_valid=None):

        n_train_batches = len(X_train) / self.fit_method.batch_size
        n_valid_batches = len(X_valid) / self.fit_method.batch_size

        X_train = np.asarray(X_train, dtype='float32')
        batch = X_train[:1]
        for layer in self.layers:
            batch = layer.fit_transform(batch)

        from deep.fit.function import compile_batch_function
        train_batch_function = compile_batch_function(X_train, y_train, self._symbolic_score, self.fit_method.batch_size, self._symbolic_updates)
        valid_batch_function = compile_batch_function(X_valid, y_valid, self._symbolic_score, self.fit_method.batch_size)

        from deep.fit.function import BatchIterator
        train_batch_iterator = BatchIterator(train_batch_function, n_train_batches)
        valid_batch_iterator = BatchIterator(valid_batch_function, n_valid_batches)

        from deep.fit.function import EpochIterator
        train_epoch_iterator = EpochIterator(train_batch_iterator)
        #valid_epoch_iterator = EpochIterator(valid_batch_iterator)

        for train_cost in train_epoch_iterator:
            print train_cost

        #for train_cost, valid_cost in zip(train_epoch_iterator, valid_epoch_iterator):
        #    print train_cost, valid_cost

        return self

    def __str__(self):
        hyperparams = ("""
  Hyperparameters |      Value
------------------|------------------
  Learning Rate   | {:>s}
  Update Method   | {:>s}
  Fit Method      | {:>s}
  Training Cost   | {:>s}
  Regularization  | {:>s}
""").format(str(self.learning_rate), str(self.update), str(self.fit_method),
            str(self.cost), str(self.regularize))

        layers = """
  Layer | Shape | Activation | Corrupt
--------|-------|------------|--------
"""

        #: push this into layer __str__
        for layer in self.layers:

            #: how can we avoid this?
            from deep.layers import PreConv, PostConv, Pooling
            if isinstance(layer, PreConv) or isinstance(layer, PostConv):
                continue

            if isinstance(layer, Pooling):
                layers += ('  {:>s} | {:>s} | {:>s} | {:>s} \n'.format(
                    layer.__class__.__name__, layer.pool_size, None, None
                ))
                continue

            layers += ('  {:>s} | {:>s} | {:>s} | {:>s} \n'.format(
                layer.__class__.__name__, layer.shape, layer.activation, layer.corruption
            ))

        return hyperparams + layers

