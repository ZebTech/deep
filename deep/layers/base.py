from abc import ABCMeta, abstractmethod, abstractproperty


class Layer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, X):
        """"""

    @abstractmethod
    def fit(self, shape):
        """"""

    @abstractproperty
    def params(self):
        """"""

    @abstractproperty
    def shape(self):
        """"""