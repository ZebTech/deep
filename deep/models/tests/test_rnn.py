import unittest
from copy import deepcopy

import numpy as np
X = np.random.rand(100, 100, 2)
y = np.sum(X[:, :, 0] * X[:, :, 1], axis=1)

from deep.models import RNN
from deep.layers import Dense, Recurrent
from deep.activations import Identity, Sigmoid
recurrent_layers = Recurrent(50, Sigmoid())
output_layers = [Dense(1, Identity())]


class TestRNN(unittest.TestCase):

    def setUp(self):
        self.rnn = RNN(deepcopy(output_layers),deepcopy(recurrent_layers))

    def test_fit(self):
        self.rnn.fit(X, y)
        score = self.rnn.fit_scores[-1]
