# -*- coding: utf-8 -*-
"""
    deep.layers.base
    ----------------

    Implements different layer classes.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from theano import shared, config, function
from theano.tensor.nnet import conv2d

from deep.activations.base import Sigmoid


#: change the noisy argument to a boolean in each layer


class Layer(object):
    """An abstract class that represents a neural network layer. Is used
    implicitly by models and can also be used explicity to create
    custom architectures.

    :param shape: a tuple (input_size, output_size).
    :param activation: the activation function to apply after linear transform.
    """
    def __init__(self, n_hidden=100, activation=Sigmoid(), corruption=None, regularization=None):
        self.b = shared(np.zeros(n_hidden, dtype=config.floatX))
        self._transform_function = None
        self.activation = activation
        self.corruption = corruption
        self.regularization = regularization
        self.n_hidden = n_hidden
        self.x = T.matrix()

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W, self.b, self.activation.params
        return self.W, self.b

    @property
    def shape(self):
        return self.W.get_value().shape

    #: if we add a noisy option like _symbolic we need to either
    #: add a separate function or recompile each time
    def transform(self, X):
        """ """
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, X, noisy=True):
        if noisy and self.corruption is not None:
            X = self.corruption(X)
        return self.activation(T.dot(X, self.W) + self.b)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        #: where should the seed go?
        np.random.seed(1)
        size = X.shape[1], self.n_hidden

        self.W = np.random.normal(loc=0, scale=.01, size=size)
        self.W = shared(np.asarray(self.W, dtype=config.floatX))
        return self

    def __repr__(self):
        layer_name = str(self.activation) + self.__class__.__name__
        layer_shape = str(self.shape)
        return layer_name + '(shape=' + layer_shape + ')'


class PreConv(object):

    def fit(self, X):
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        dim = int(np.sqrt(n_features))
        return X.reshape(n_samples, 1, dim, dim)

    def _symbolic_transform(self, x, noisy=None):
        n_samples, n_features = x.shape
        dim = T.cast(T.sqrt(n_features), dtype='int64')
        return x.reshape((n_samples, 1, dim, dim))

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def params(self):
        return []


class PostConv(object):

    def fit(self, X):
        return self

    def transform(self, X):
        return X.reshape(1, -1)

    def _symbolic_transform(self, x, noisy=None):
        return x.flatten(2)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def params(self):
        return []

from theano.tensor.signal.downsample import max_pool_2d
class Pooling(object):

    def __init__(self, pool_size, stride_size):
        self.pool_size = (pool_size, pool_size)
        self.stride_size = stride_size
        self._transform_function = None
        self.x = T.tensor4()

    @property
    def params(self):
        return []

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        return self

    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, x, noisy=None):
        return max_pool_2d(x, self.pool_size)


class ConvolutionLayer(Layer):
    """An abstract class that represents a convolutional layer. This is called
    implicitely by the ConvolutionalNN class and can also be used explicitly
    to create custom architectures.

    :param filter_size: shape of the convolutional filter
                        (n_filters, n_kernals, height, width).
    :param pool_size: the size of the subsampling pool.
    :param activation: the non-linearly to apply after pooling.
    """
    def __init__(self, n_filters=10, filter_size=5, activation=Sigmoid(), corruption=None, regularization=None):
        self.b = shared(np.zeros(n_filters, dtype=config.floatX))
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.corruption = corruption
        self.x = T.tensor4()
        self.activation = activation
        self.regularization = regularization
        self._transform_function = None

    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, x, noisy=True):
        if noisy and self.corruption is not None:
            x = self.corruption(x)
        x = conv2d(x, self.W, filter_shape=self.W.get_value().shape)
        return self.activation(x + self.b.dimshuffle('x', 0, 'x', 'x'))

    def fit(self, X):

        #: where should the seed go?
        np.random.seed(1)
        n_channels = X.shape[1]
        size = self.n_filters, n_channels, self.filter_size, self.filter_size
        self.W = np.random.normal(loc=0, scale=.01, size=size)
        self.W = shared(np.asarray(self.W, dtype=config.floatX))
        return self

    def __repr__(self):
        return self.__class__.__name__
