from base import Layer
import theano.tensor as T

from deep.initialize import Normal


class Recurrent(Layer):

    def __init__(self, n_hidden, activation, initialize=Normal()):
        self.n_hidden = n_hidden
        self.activation = activation
        self.initialize = initialize

    def __call__(self, X, H):
        return self.activation(T.dot(H, self.W_hidden) + T.dot(X, self.W_visible) + self.b)

    def fit(self, incoming_shape):
        size = self.n_hidden, self.n_hidden
        self.W_hidden = self.initialize.W(size)
        size = incoming_shape[1], self.n_hidden
        self.W_visible = self.initialize.W(size)
        self.b = self.initialize.b(self.n_hidden)
        return self

    @property
    def params(self):
        if self.activation.params is not None:
            return self.W_visible, self.W_hidden, self.b, self.activation.params
        return self.W_visible, self.W_hidden, self.b

    @property
    def shape(self):
        return self.W_visible.get_value().shape

