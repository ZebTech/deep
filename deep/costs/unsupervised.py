from base import Cost
import theano.tensor as T


class SquaredError(Cost):

    def __call__(self, x, y):
        return T.mean(T.sum((x - y) ** 2, axis=-1))


class BinaryCrossEntropy(Cost):

    def __call__(self, x, y):
        return T.mean(T.nnet.binary_crossentropy(x, y))
