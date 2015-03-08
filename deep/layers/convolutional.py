from deep.layers.base import Layer
from deep.activations import Rectified
from deep.initialize import Normal
from theano.tensor.nnet import conv2d


class Convolutional(Layer):

    def __init__(self, n_filters=32, filter_size=3, activation=Rectified(), initialize=Normal()):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.activation = activation
        self.initialize = initialize

    def __call__(self, x):
        x = conv2d(x, self.W, filter_shape=self.W.get_value().shape)
        return self.activation(x + self.b.dimshuffle('x', 0, 'x', 'x'))

    def fit(self, X):
        n_channels = X.shape[1]
        size = self.n_filters, n_channels, self.filter_size, self.filter_size
        self.W = self.initialize.W(size)
        self.b = self.initialize.b(self.n_filters)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W, self.b, self.activation.params
        return self.W, self.b

    def shape(self):
        return self.W.get_value().shape
