from base import Fit
from theano import shared, config, function

import theano.tensor as T
import numpy as np


class Iterative(Fit):

    def __init__(self, n_epochs=10, batch_size=128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def givens(self, X, y):
        givens = dict()
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        X = shared(np.asarray(X, dtype=config.floatX))
        y = shared(np.asarray(y, dtype='int64'))
        givens[self.x] = X[batch_start:batch_end]
        givens[self.y] = y[batch_start:batch_end]
        return givens

    def compile_batch_function(self, X, y, output, updates=None):
        output = output(self.x, self.y)
        givens = self.givens(X, y)
        if updates is not None:
            updates = updates(self.x, self.y)
        return function([self.i], output, None, updates, givens)

    def __call__(self, model, X, y=None, X_valid=None, y_valid=None):
        self.x = T.matrix()
        self.y = T.lvector()
        self.i = T.lscalar()
        self.train_scores = []
        self.valid_scores = []

        n_train_batches = len(X) / self.batch_size
        train_function = self.compile_batch_function(X, y, model._symbolic_score, model._symbolic_updates)

        if X_valid is not None:
            n_valid_batches = len(X_valid) / self.batch_size
            valid_function = self.compile_batch_function(X, y, model._symbolic_score)

        _print_header(self)

        from time import time
        valid_cost = np.inf
        for epoch in range(1, self.n_epochs+1):
            begin = time()
            train_cost = np.mean([train_function(i) for i in range(n_train_batches)])
            self.train_scores.append(train_cost)
            if X_valid is not None:
                valid_cost = np.mean([valid_function(i) for i in range(n_valid_batches)])
                self.valid_scores.append(valid_cost)
            _print_iter(epoch, train_cost, valid_cost, time() - begin)


def _print_header(model):

    print model

    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(iteration, train_cost, valid_cost, elapsed):
    print("  {:>5} | {:>7.4f} | {:>7.4f} | {:>4.1f}s".format(
        iteration, train_cost, valid_cost, elapsed))
