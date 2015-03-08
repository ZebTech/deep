from time import time
from base import Fit
from theano import shared, config, function

import theano.tensor as T
import numpy as np


class Iterative(Fit):

    def __init__(self, n_epochs=10, batch_size=128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.x = T.matrix()
        self.y = T.lvector()
        self.i = T.lscalar()

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

    def fit_model(self, model, X, y):
        score = model._symbolic_score(self.x, self.y)
        updates = model._symbolic_updates(self.x, self.y)
        givens = self.givens(X, y)
        train = function([self.i], score, None, updates, givens)
        batches = len(X) / self.batch_size

        model.train_scores = []
        for epoch in range(1, self.n_epochs+1):
            batch_scores = [train(batch) for batch in range(batches)]
            model.train_scores.append(np.mean(batch_scores))

    def fit_validate_model(self, model, X, y, X_valid, y_valid):
        raise NotImplementedError