import numpy as np
import theano.tensor as T

from deep.costs import NegativeLogLikelihood, PredictionError
from deep.updates import Momentum
from deep.fit import Iterative


class NN(object):

    def __init__(self, layers, learning_rate=0.1, update=Momentum(),
                 fit_type=Iterative(), cost=NegativeLogLikelihood()):
        self.layers = layers
        self.learning_rate = learning_rate
        self.update = update
        self.cost = cost
        self.fit_model = fit_type.fit_model
        self.fit_validate_model = fit_type.fit_validate_model

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def _symbolic_updates(self, x, y):
        cost = self.cost(self._symbolic_predict_proba(x), y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_predict_proba(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def _symbolic_predict(self, x):
        return T.argmax(self._symbolic_predict_proba(x), axis=1)

    def _symbolic_score(self, x, y):
        cost = PredictionError()
        return cost(self._symbolic_predict(x), y)

    def predict_proba(self, X):
        self._fit.compile_batch_function()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        X = self.predict_proba(X)
        return -np.mean(np.log(X)[np.arange(y.shape[0]), y])

    def fit(self, X, y):
        self.fit_layers(X.shape)
        self.fit_model(self, X, y)
        return self

    def fit_validate(self, dataset):
        raise NotImplementedError

    def fit_layers(self, shape):
        for layer in self.layers:
            layer.fit(shape)
            shape = layer.shape
