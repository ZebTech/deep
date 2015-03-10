import unittest
from copy import deepcopy


from deep.datasets import load_mnist
X, y = load_mnist()[1]
X_valid, y_valid = load_mnist()[2]

from deep.models import RNN
layers = None

class TestRNN(unittest.TestCase):

    def setUp(self):
        self.rnn = RNN(deepcopy(layers))

    @unittest.expectedFailure
    def test_fit(self):
        self.rnn.fit(X, y)
        score = self.rnn.fit_scores[-1]
        self.assertLess(score, 0.02)
