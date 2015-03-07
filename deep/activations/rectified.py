import numpy as np
import theano.tensor as T

from base import Activation
from theano import config, shared


class Rectified(Activation):

    def __call__(self, X):
        return T.switch(X > 0.0, X, 0.0)


class Leaky(Activation):

    def __init__(self, slope):
        self.slope = slope

    def __call__(self, X):
        return T.switch(X > 0.0, X, X * self.slope)


class Parametrized(Activation):

    def __init__(self, n_hidden):
        self.slope = shared(np.zeros(n_hidden, dtype=config.floatX))

    def __call__(self, X):
        return T.switch(X > 0.0, X, X * self.slope)

    @property
    def params(self):
        return self.slope


class ParametrizedConv(Activation):

    def __init__(self, n_hidden):
        self.slope = shared(np.zeros(n_hidden, dtype=config.floatX))

    def __call__(self, X):
        return T.switch(X > 0.0, X, X * self.slope.dimshuffle('x', 0, 'x', 'x'))

    @property
    def params(self):
        return self.slope
