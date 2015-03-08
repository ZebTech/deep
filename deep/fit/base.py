# -*- coding: utf-8 -*-
"""
    deep.fit.base
    -------------

    Implements various fitting schemes.

    :references: nolearn

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import time
import numpy as np
import theano.tensor as T
from theano import function, shared, config


def _print_header():
    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(iteration, train_cost, valid_cost, elapsed):
    print("  {:>5} | {:>7.4f} | {:>7.4f} | {:>4.1f}s".format(
        iteration, train_cost, valid_cost, elapsed))


#: separate X, y givens to combine these
def supervised_givens(i, x, X, y, Y, batch_size):
    batch_start = i * batch_size
    batch_end = (i+1) * batch_size
    return {x: X[batch_start:batch_end],
            y: Y[batch_start:batch_end]}


def unsupervised_givens(i, x, X, batch_size):
    batch_start = i * batch_size
    batch_end = (i+1) * batch_size
    return {x: X[batch_start:batch_end]}


class Iterative(object):

    def __init__(self, n_iterations=100, batch_size=128, augment=None, X_test=None):
        self.X_test = X_test
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.augment = augment
        self.train_scores = [np.inf]
        self.valid_scores = [np.inf]

    #: does it matter that these are class variables?
    x = T.matrix()
    y = T.lvector()
    i = T.lscalar()

    def compile_train_function(self, model, X, y):

        if y is None:
            score = model._symbolic_score(self.x)
            updates = model.updates(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)
        else:
            #: for plankton competition
            from deep.costs import NegativeLogLikelihood
            score = NegativeLogLikelihood()(model._symbolic_predict_proba(self.x), self.y)
            score = model._symbolic_score(self.x, self.y)

            updates = model.updates(self.x, self.y)
            givens = supervised_givens(self.i, self.x, X, self.y, y, self.batch_size)
        return function([self.i], score, None, updates, givens)

    def fit(self, model, X, y=None, X_valid=None, y_valid=None):

        #: passing in X_test since we need to normalize with it

        if self.augment is not None:
            begin = time.time()
            X_clean = np.copy(X)

            n_augmentations, n_samples, n_features = X_valid.shape
            X_valid = X_valid.reshape(n_augmentations*n_samples, n_features)
            X = np.vstack([self.augment.fit_transform(X_clean) for i in xrange(n_augmentations)])
            y = np.tile(y, n_augmentations)

            from sklearn.preprocessing import StandardScaler
            X = np.vstack((X, X_valid, self.X_test))
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.X_test = X[-len(self.X_test):]
            X_valid = X[-(len(X_valid) + len(self.X_test)):-len(self.X_test)]
            X = X[:-(len(X_valid) + len(self.X_test))]
            X_valid = X_valid.reshape(-1, n_samples, n_features)

            print 'augmentation took', time.time() - begin


        #: moved this here because need to fit model
        #: to post augmented data (patch changes dims)
        model._fit(X[:1], y)

        print model

        n_train_batches = len(X) / self.batch_size
        #: moved this here because need to update it
        #: for continuous augmentation
        X = shared(np.asarray(X, dtype=config.floatX))
        Y = shared(np.asarray(y, dtype='int64'))
        train_function = self.compile_train_function(model, X, Y)

        #: hack so prediction compiles without corruption
        for layer in model.layers:
            layer.corruption = None

        _print_header()

        self.best_model = None

        for iteration in range(1, self.n_iterations+1):
            begin = time.time()

            train_costs = [train_function(batch) for batch in range(n_train_batches)]
            train_cost = np.mean(train_costs)
            self.train_scores.append(train_cost)

            valid_cost = np.inf
            if X_valid is not None:
                batch_size = 100
                n_valid_samples = len(X_valid)
                n_valid_batches = n_valid_samples / batch_size
                valid_cost = []
                for batch in range(n_valid_batches+1):
                    X_valid_batch = X_valid[:, batch*batch_size:(batch+1)*batch_size]

                    print X_valid_batch.shape

                    y_valid_batch = y_valid[batch*batch_size:(batch+1)*batch_size]
                    valid_cost.append(model.score(X_valid_batch, y_valid_batch))
                valid_cost = np.mean(valid_cost)

            self.valid_scores.append(valid_cost)

            elapsed = time.time() - begin

            _print_iter(iteration, train_cost, valid_cost, elapsed)

            if self.finished:
                break

            if self.augment is not None:
                X_augmented = np.vstack([self.augment.fit_transform(X_clean) for i in range(n_augmentations)])
                X_scaled = self.scaler.transform(X_augmented)
                X.set_value(X_scaled)

        return model

    @property
    def finished(self):
        return False

    def __str__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.n_iterations, self.batch_size)


class EarlyStopping(Iterative):

    def __init__(self, patience=1, n_iterations=100, batch_size=128, augment=None, X_test=None):
        super(EarlyStopping, self).__init__(n_iterations, batch_size, augment, X_test)
        self.patience = patience

    @property
    def finished(self):
        if len(self.valid_scores) <= self.patience:
            return False
        else:
            return self.valid_scores[-1] > self.valid_scores[-(self.patience+1)]
