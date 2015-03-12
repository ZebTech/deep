import unittest
from copy import deepcopy

import numpy as np
X = np.random.rand(100, 100, 2)
y = np.sum(X[:, :, 0] * X[:, :, 1], axis=1)

from deep.models import RNN
from deep.layers import Dense, Recurrent
from deep.activations import Identity, Sigmoid
from deep.fit import Iterative
recurrent_layers = [Recurrent(50, Sigmoid())]
output_layers = [Dense(1, Identity())]


class TestRNN(unittest.TestCase):

    def setUp(self):
        self.rnn = RNN(deepcopy(output_layers), deepcopy(recurrent_layers),
                       fit_type=Iterative(batch_size=1))

    def test_predict(self):
        self.assertRaises(AttributeError, self.rnn.predict, X[0])
        self.rnn.fit_layers(X[0].shape)
        self.rnn.predict(X[0])

    def test_score(self):
        self.assertRaises(AttributeError, self.rnn.score, X[0], y[0])
        self.rnn.fit_layers(X[0].shape)
        self.rnn.score(X[0], y[0])

    def test_fit(self):
        self.rnn.fit(X, y)
        print self.rnn.fit_scores
        score = self.rnn.fit_scores[-1]
