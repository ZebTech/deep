from abc import ABCMeta, abstractmethod


class Fit(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, X, y=None, X_valid=None, y_valid=None):
        """"""
