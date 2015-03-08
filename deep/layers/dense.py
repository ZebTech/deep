import theano.tensor as T
from deep.layers.base import Layer
from deep.activations import Rectified
from deep.initialize import Normal


class Dense(Layer):

    def __init__(self, n_hidden=100, activation=Rectified(), initialize=Normal()):
        self.n_hidden = n_hidden
        self.activation = activation
        self.initialize = initialize

    def __call__(self, X):
        return self.activation(T.dot(X, self.W) + self.b)

    def fit(self, incoming_shape):
        size = incoming_shape[1], self.n_hidden
        self.W = self.initialize.W(size)
        self.b = self.initialize.b(self.n_hidden)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W, self.b, self.activation.params
        return self.W, self.b

    @property
    def shape(self):
        return self.W.get_value().shape
