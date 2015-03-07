import theano.tensor as T
from abc import abstractmethod


class Activation(object):

    @abstractmethod
    def __call__(self, X):
        return

    @property
    def params(self):
        return

    def __repr__(self):
        return self.__class__.__name__


class Identity(Activation):

    def __call__(self, X):
        return X


class Sigmoid(Activation):

    def __call__(self, X):
        return T.nnet.sigmoid(X)


class Softmax(Activation):

    def __call__(self, X):
        return T.nnet.softmax(X)


class Tanh(Activation):

    def __call__(self, X):
        return T.tanh(X)
