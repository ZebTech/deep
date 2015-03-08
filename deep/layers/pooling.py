from base import Layer
from theano.tensor.signal.downsample import max_pool_2d


class MaxPooling(Layer):

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def __call__(self, x):
        return max_pool_2d(x, self.pool_size)

    def fit(self, incoming_shape):
        return self

    @property
    def params(self):
        return []

    @property
    def shape(self):
        return self.pool_size, self.pool_size