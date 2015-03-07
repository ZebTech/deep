from base import Corruption
from theano import config


class Gaussian(Corruption):

    def __call__(self, x):
        return x + self.rng.normal(size=x.shape, std=self.corruption_level, dtype=config.floatX)
