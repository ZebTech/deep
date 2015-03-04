# -*- coding: utf-8 -*-
"""
    deep.fit.base
    -------------

    Implements various fitting schemes.

    :references: nolearn

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from time import time


class Iterative(object):

    def __init__(self, n_iterations=100, batch_size=128):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.train_costs = []
        self.valid_costs = []

    def fit(self, model, X, y=None, X_valid=None, y_valid=None):
        n_train_batches = len(X) / self.batch_size
        n_valid_batches = len(X_valid) / self.batch_size
        train_function = compile_train_function(model, X, y, self.batch_size)
        score_function = compile_score_function(model, X_valid, y_valid, self.batch_size)

        print model
        _print_header()

        for iteration in range(1, self.n_iterations+1):
            begin = time()
            train_cost = np.mean(map(train_function, range(n_train_batches)))
            valid_cost = np.mean(map(score_function, range(n_valid_batches)))
            self.train_costs.append(train_cost)
            self.valid_costs.append(valid_cost)
            elapsed = time() - begin
            _print_iter(iteration, train_cost, valid_cost, elapsed)

        return model

    def __str__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.n_iterations, self.batch_size)


def _print_header():
    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(iteration, train_cost, valid_cost, elapsed):
    print("  {:>5} | {:>7.4f} | {:>7.4f} | {:>4.1f}s".format(
        iteration, train_cost, valid_cost, elapsed))

