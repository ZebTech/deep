import unittest


from deep.datasets import load_mnist
X, y = load_mnist()[1]
X_valid, y_valid = load_mnist()[2]


class TestNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from deep.models import NN
        from deep.layers import Dense
        from deep.activations import Softmax
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