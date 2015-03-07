import theano.tensor as T
from abc import abstractmethod


class Regularize(object):

    def __init__(self, alpha=.0005):
        self.alpha = alpha

    @abstractmethod
    def __call__(self, X):
        pass

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.alpha)


class L1(Regularize):

    def __call__(self, param):
        return self.alpha * T.sum(abs(param))


class L2(Regularize):

    def __call__(self, param):
        return self.alpha * T.sum(param ** 2)
