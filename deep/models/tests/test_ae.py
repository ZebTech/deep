import unittest


from deep.datasets import load_mnist
X, y = load_mnist()[1]
X_valid, y_valid = load_mnist()[2]


class TestAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from deep.models import AE
        from deep.layers import Dense
        from deep.activations import Sigmoid
        encoder = [
            Dense(100),
        ]
        decoder = [
            Dense(784, Sigmoid()),
        ]
        cls.ae = AE(encoder, decoder)

    def test_transform(self):
        pass

    def test_inverse_transform(self):
        pass

    def test_score(self):
        pass

    def test_fit(self):
        self.ae.fit(X)
        score = self.ae.fit_scores[-1]
        self.assertLess(score, 0.02)

        self.ae.fit(X)
        score = self.ae.fit_scores[-1]
        self.assertEqual(score, 0)


class TestTiedAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from deep.models import AE
        from deep.layers import Dense
        from deep.activations import Softmax
        layers = [
            Dense(100),
            Dense(10, Softmax())
        ]
        cls.ae = AE(layers)

    def test_transform(self):
        pass

    def test_inverse_transform(self):
        pass

    def test_score(self):
        pass

    def test_fit(self):
        self.ae.fit(X, y)
        score = self.ae.fit_scores[-1]
        self.assertLess(score, 0.02)

        self.ae.fit(X, y)
        score = self.ae.fit_scores[-1]
        self.assertEqual(score, 0)
