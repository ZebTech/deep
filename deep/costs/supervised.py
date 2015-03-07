from base import Cost
import theano.tensor as T


class NegativeLogLikelihood(Cost):

    def __call__(self, x, y):
        return T.mean(T.nnet.categorical_crossentropy(x, y))


class PredictionError(Cost):

    def __call__(self, x, y):
        return T.mean(T.neq(x, y))