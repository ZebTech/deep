from base import Layer
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Corruption(Layer):

    rng = MRG_RandomStreams(1)

    def __init__(self, corruption=0.5):
        self.corruption = corruption

    def fit(self, shape):
        #: hack since shape needs to return shape
        self._shape = shape
        return self

    @property
    def params(self):
        return []

    @property
    def shape(self):
        return self._shape


class Binomial(Corruption):

    def __call__(self, x):
        return x * self.rng.binomial(size=x.shape, p=1-self.corruption, dtype=config.floatX)


class Dropout(Binomial):

    def __call__(self, x):
        scaler = 1.0 / (1.0 - self.corruption)
        return scaler * super(Dropout, self).__call__(x)


class Gaussian(Corruption):

    def __call__(self, x):
        return x + self.rng.normal(size=x.shape, std=self.corruption, dtype=config.floatX)


class SaltAndPepper(Corruption):

    def __call__(self, X):
        a = self.rng.binomial(size=X.shape, p=1-self.corruption, dtype=config.floatX)
        b = self.rng.binomial(size=X.shape, p=0.5, dtype=config.floatX)
        return X * a + T.eq(a, 0) * b
