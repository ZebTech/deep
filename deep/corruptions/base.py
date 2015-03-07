from abc import abstractmethod
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Corruption(object):

    def __init__(self, corruption_level=0.5):
        self.corruption_level = corruption_level

    rng = MRG_RandomStreams(1)

    @abstractmethod
    def __call__(self, X):
        return

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.corruption_level)
