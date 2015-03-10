from base import Fit

import theano.tensor as T
import numpy as np


class Iterative(Fit):

    def __init__(self, n_epochs=10, batch_size=128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.x = T.matrix()
        self.y = T.lvector()
        self.i = T.lscalar()

    def fit_model(self, model, X, y=None):
        train = model.fit_function(X, y)
        batches = len(X) / self.batch_size

        for epoch in range(1, self.n_epochs+1):
            batch_scores = [train(batch) for batch in range(batches)]
            model.fit_scores.append(np.mean(batch_scores))

    def fit_validate_model(self, model, X, y, X_valid, y_valid):
        raise NotImplementedError