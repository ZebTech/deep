import unittest
from deep.models import NN
from deep.layers import *
from deep.activations import *


class TestNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        layers = [
            Dense(100),
            Dense(10, Softmax())
        ]
        cls.nn = NN(layers)

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_score(self):
        pass

    def test_fit(self):
        from deep.datasets import load_mnist
        X, y = load_mnist()[1]
        self.nn.fit(X, y)
        self.assertLess(self.nn._fit.train_scores[-1], 1 - 0.98)
