import unittest
import theano.tensor as T
from deep.corruptions import Binomial, Dropout, Gaussian, SaltAndPepper


class TestCorruptions(unittest.TestCase):

    def test_corruptions(self):
        for corruption in [Binomial, Dropout, Gaussian, SaltAndPepper]:
            corruption = corruption()
            corruption(T.matrix())
