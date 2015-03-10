import unittest
from copy import deepcopy


from deep.datasets import load_mnist
X, y = load_mnist()[1]
X_valid, y_valid = load_mnist()[2]

from deep.models import NN
from deep.layers import Dense
from deep.activations import Softmax
layers = [
    Dense(100),
    Dense(10, Softmax())
]


class TestNN(unittest.TestCase):

    def setUp(self):
        self.nn = NN(deepcopy(layers))

    def test_predict(self):
        self.assertRaises(AttributeError, self.nn.predict, X[:1])
        self.nn.fit_layers(X.shape)
        self.nn.predict_proba(X[:1])

    def test_predict_proba(self):
        self.assertRaises(AttributeError, self.nn.predict_proba, X[:1])
        self.nn.fit_layers(X.shape)
        self.nn.predict_proba(X[:1])

    def test_score(self):
        self.assertRaises(AttributeError, self.nn.score, X[:1], y[:1])
        self.nn.fit_layers(X.shape)
        self.nn.score(X[:1], y[:1])

    def test_fit(self):
        self.nn.fit(X, y)
        score = self.nn.fit_scores[-1]
        self.assertLess(score, 0.02)

        self.nn.fit(X, y)
        score = self.nn.fit_scores[-1]
        self.assertEqual(score, 0)

    def test_fit_validate(self):
        pass

    def test_fit_layers(self):
        pass