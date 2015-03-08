import unittest
import theano.tensor as T
from deep.costs.supervised import NegativeLogLikelihood, PredictionError
from deep.costs.unsupervised import BinaryCrossEntropy, SquaredError


class TestCosts(unittest.TestCase):

    def test_costs(self):
        for cost in [NegativeLogLikelihood, PredictionError,
                           BinaryCrossEntropy, SquaredError]:
            cost = cost()
            cost(T.matrix(), T.matrix())
