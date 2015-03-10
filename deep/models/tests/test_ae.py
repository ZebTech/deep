import unittest
import numpy as np
from copy import deepcopy


from deep.datasets import load_mnist
X, y = load_mnist()[1]
X_valid, y_valid = load_mnist()[2]


from deep.models import AE
from deep.layers import Dense
from deep.activations import Sigmoid
encoder = [
    Dense(100),
]
decoder = [
    Dense(784, Sigmoid()),
]


class TestAE(unittest.TestCase):

    def setUp(self):
        self.ae = AE(deepcopy(encoder), deepcopy(decoder))

    def test_transform(self):
        self.assertRaises(AttributeError, self.ae.transform, X[:1])
        self.ae.fit_layers(X.shape)
        self.ae.transform(X[:1])

    def test_inverse_transform(self):
        self.assertRaises(AttributeError, self.ae.inverse_transform, np.ones((1, 100)))
        self.ae.fit_layers(X.shape)
        self.ae.inverse_transform(np.ones((1, 100)))

    def test_reconstruct(self):
        self.assertRaises(AttributeError, self.ae.reconstruct, X[:1])
        self.ae.fit_layers(X.shape)
        self.ae.reconstruct(X[:1])

    def test_score(self):
        self.assertRaises(AttributeError, self.ae.score, X[:1])
        self.ae.fit_layers(X.shape)
        self.ae.score(X[:1])

    def test_fit(self):
        self.ae.fit(X)
        score = self.ae.fit_scores[-1]
        self.assertLess(score, 0.1)

        self.ae.fit(X)
        score = self.ae.fit_scores[-1]
        self.assertLess(score, 0.09)
