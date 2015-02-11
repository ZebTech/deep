# -*- coding: utf-8 -*-
"""
    deep.fit.base
    -------------

    Implements various fitting schemes.

    :references: pylearn2 (cost module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import time
import numpy as np
import theano.tensor as T
from abc import abstractmethod
from theano import shared, config, function

class Fit(object):

    @abstractmethod
    def __call__(self, model, X, y):
        """"""


class Iterative(Fit):

    def __init__(self, n_iterations=10, batch_size=100):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.i = T.lscalar()

    #: it might be cleaner to pass the data into fit as well and construct
    #: the fit function directly in fit instead of in the model

    def __call__(self, model, X, y):
        n_batches = len(X) / self.batch_size

        X_shared = shared(np.asarray(X.next(), dtype=config.floatX))
        y_shared = shared(np.asarray(y, dtype='int64'))

        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        givens = {model.x: X_shared[batch_start:batch_end],
                  model.y: y_shared[batch_start:batch_end]}

        fit_function = function(inputs=[self.i],
                                outputs=model._symbolic_score(model.x, model.y),
                                updates=model.updates,
                                givens=givens)

        begin = time.time()

        for iteration in range(1, self.n_iterations + 1):

            batch_costs = [fit_function(batch) for batch in range(n_batches)]

            print("[%s] Iteration %d, costs = %.2f, time = %.2fs"
                  % (type(model).__name__, iteration, np.mean(batch_costs), time.time() - begin))

            #: cleaner way to use iterator?
            X_shared.set_value(X.next())

        return model


class EarlyStopping(Fit):

    def __init__(self, X_valid=None, y_valid=None, n_iterations=10, batch_size=100):
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    def __call__(self, model):

        raise NotImplementedError
