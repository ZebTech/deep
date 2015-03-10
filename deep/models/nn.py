import theano.tensor as T
_x = T.matrix()
_y = T.vector()

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
        self.fit_scores = []
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
        try:
            return self._symbolic_predict_proba(_x).eval({_x: X})
        except AttributeError:
            raise AttributeError("'{}' model has not been fit yet. "
                                 "Trying calling fit() or fit_layers()."
                                 .format(self.__class__.__name__))

    def predict(self, X):
        try:
            return self._symbolic_predict(_x).eval({_x: X})
        except AttributeError:
            raise AttributeError("'{}' model has not been fit yet. "
                                 "Trying calling fit() or fit_layers()."
                                 .format(self.__class__.__name__))

    def score(self, X, y):
        try:
            return self._symbolic_score(_x, _y).eval({_x: X, _y: y})
        except AttributeError:
            raise AttributeError("'{}' model has not been fit yet. "
                                 "Trying calling fit() or fit_layers()."
                                 .format(self.__class__.__name__))

    def fit(self, X, y):
        self.fit_layers(X.shape)
        self.fit_model(self, X, y)
        return self

    def fit_validate(self, dataset):
        raise NotImplementedError

    def fit_layers(self, shape):
        for layer in self.layers:
            try:
                shape = layer.shape
            except AttributeError:
                layer.fit(shape)
                shape = layer.shape
