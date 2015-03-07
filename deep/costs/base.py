from abc import abstractmethod


class Cost(object):

    @abstractmethod
    def __call__(self, x, y):
        return

    def __repr__(self):
        return self.__class__.__name__

