import theano.tensor as T
from deep.layers.base import Layer
from deep.activations import Rectified
from deep.initialize import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Gaussian(Layer):

    rng = MRG_RandomStreams(1)

    def __init__(self, n_hidden=100, activation=Rectified(), initialize=Normal()):
        self.n_hidden = n_hidden
        self.activation = activation
        self.initialize = initialize

    def __call__(self, X):
        mean = self.activation(T.dot(X, self.W_mean) + self.b_mean)
        var = self.activation(T.dot(X, self.W_var) + self.b_var)
        return self.rng.normal(mean.shape, mean, var)

    def fit(self, incoming_shape):
        size = incoming_shape[1], self.n_hidden
        self.W_mean = self.initialize.W(size)
        self.W_var = self.initialize.W(size)
        self.b_mean = self.initialize.b(self.n_hidden)
        self.b_var = self.initialize.b(self.n_hidden)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W_mean, self.W_var, self.b_mean, self.b_var, self.activation.params
        return self.W_mean, self.W_var, self.b_mean, self.b_var

    @property
    def shape(self):
        return self.W_mean.get_value().shape
