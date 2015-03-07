import unittest
import theano.tensor as T
from deep.activations import Identity, Sigmoid, Softmax, Tanh


class TestActivations(unittest.TestCase):

    def test_activations(self):
        for corruption in [Identity, Sigmoid, Softmax, Tanh]:
            corruption = corruption()
            corruption(T.matrix())
