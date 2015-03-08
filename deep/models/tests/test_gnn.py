import unittest


from deep.datasets import load_mnist
X, y = load_mnist()[1]


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
        cls.nn.fit(X, y)

    def test_predict(self):
        self.nn.predict(X[:1])

    def test_predict_proba(self):
        pass

    def test_score(self):
        pass

    def test_fit(self):
        score = self.nn._fit.train_scores[-1]
        self.assertLess(score, 0.02)
