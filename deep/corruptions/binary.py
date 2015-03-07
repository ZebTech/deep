import theano.tensor as T

from base import Corruption
from theano import config


class Binomial(Corruption):

    def __call__(self, x):
        return x * self.rng.binomial(size=x.shape, p=1-self.corruption_level, dtype=config.floatX)


class Dropout(Binomial):

    def __call__(self, x):
        scaler = 1.0 / (1.0 - self.corruption_level)
        return scaler * super(Dropout, self).__call__(x)

class SaltAndPepper(Corruption):

    def __call__(self, X):
        a = self.rng.binomial(size=X.shape, p=1-self.corruption_level, dtype=config.floatX)
        b = self.rng.binomial(size=X.shape, p=0.5, dtype=config.floatX)
        return X * a + T.eq(a, 0) * b
