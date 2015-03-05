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

from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.updates.base import GradientDescent


class NN(object):

    def __init__(self, layers, n_epochs=100, batch_size=128, learning_rate=0.1,
                 update=GradientDescent(), cost=NegativeLogLikelihood(),
                 regularize=None):
        self.layers = layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update = update
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
        n_batches = len(X) / self.batch_size
        #: compile batch function needs to be tweaked for this to work
        predict_proba_function = compile_batch_function(X, None, self._symbolic_predict_proba, self.batch_size)
        return np.mean(map(predict_proba_function, range(n_batches)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        X = self.predict_proba(X)
        return -np.mean(np.log(X)[np.arange(y.shape[0]), y])

    def fit(self, X_train, y_train=None, X_valid=None, y_valid=None, augment=None):
        if augment is not None:
            augment = augmentation_generator(X_train, augment)

        batch = X_train[:1]
        for layer in self.layers:
            batch = layer.fit_transform(batch)

        n_train_batches = len(X_train) / self.batch_size
        n_valid_batches = len(X_valid) / self.batch_size

        #: where is the best place to wrap with shared since
        #: we need access to shared to update
        from theano import shared, config
        X_train = shared(np.asarray(X_train, dtype=config.floatX))
        y_train = shared(np.asarray(y_train, dtype='int64'))

        X_valid = shared(np.asarray(X_valid, dtype=config.floatX))
        y_valid = shared(np.asarray(y_valid, dtype='int64'))

        train_batch_function = compile_batch_function(X_train, y_train, self._symbolic_score, self.batch_size, self._symbolic_updates)
        valid_batch_function = compile_batch_function(X_valid, y_valid, self._symbolic_score, self.batch_size)

        _print_header()

        from time import time
        for epoch in range(1, self.n_epochs+1):
            begin = time()
            train_cost = np.mean(map(train_batch_function, range(n_train_batches)))
            valid_cost = np.mean(map(valid_batch_function, range(n_valid_batches)))

            if augment is not None:
                X_train.set_value(next(augment))

            _print_iter(epoch, train_cost, valid_cost, time() - begin)

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


def _print_header():
    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(iteration, train_cost, valid_cost, elapsed):
    print("  {:>5} | {:>7.4f} | {:>7.4f} | {:>4.1f}s".format(
        iteration, train_cost, valid_cost, elapsed))


#: not sure where to put these yet (utils?)
def compile_batch_function(X, Y, score, batch_size=128, updates=None):
        x = T.matrix()
        y = T.lvector()
        i = T.lscalar()

        score = score(x, y)
        if updates is not None:
            updates = updates(x, y)

        batch_start = i * batch_size
        batch_end = (i+1) * batch_size

        givens = dict()
        givens[x] = X[batch_start:batch_end]
        givens[y] = Y[batch_start:batch_end]

        from theano import function
        return function([i], score, None, updates, givens)


def augmentation_generator(X, augment):
    while True:
        yield augment.fit_transform(X)
