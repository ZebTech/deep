from deep.fit import Iterative
from deep.costs import BinaryCrossEntropy
from deep.updates import GradientDescent


class AE(object):

    def __init__(self, encoder, decoder=None, learning_rate=10, update=GradientDescent(),
                 fit_type=Iterative(), cost=BinaryCrossEntropy()):
        self.encoder = encoder
        self.decoder = decoder or []
        self.learning_rate = learning_rate
        self.update = update
        self.cost = cost
        self.fit_scores = []
        self.fit_model = fit_type.fit_model
        self.fit_validate_model = fit_type.fit_validate_model

    @property
    def params(self):
        return [param for layer in self.encoder + self.decoder for param in layer.params]

    def _symbolic_updates(self, x):
        cost = self._symbolic_score(x)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_transform(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def _symbolic_inverse_transform(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def _symbolic_score(self, x):
        reconstruct = self._symbolic_inverse_transform(self._symbolic_transform(x))
        return self.cost(reconstruct, x)

    def transform(self, X):
        raise AttributeError("'{}' model has not been fit yet."
                             .format(self.__class__.__name__))

    def inverse_transform(self, X):
        raise AttributeError("'{}' model has not been fit yet."
                             .format(self.__class__.__name__))

    def reconstruct(self, X, y):
        raise AttributeError("'{}' model has not been fit yet."
                             .format(self.__class__.__name__))

    def score(self, X, y):
        raise AttributeError("'{}' model has not been fit yet."
                             .format(self.__class__.__name__))

    def fit(self, X):
        self.fit_layers(X.shape)
        self.fit_model(self, X)
        return self

    def fit_validate(self, dataset):
        raise NotImplementedError

    def fit_layers(self, shape):
        for layer in self.encoder + self.decoder:
            try:
                shape = layer.shape
            except AttributeError:
                layer.fit(shape)
                shape = layer.shape