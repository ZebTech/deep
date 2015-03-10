from nn import NN


class RNN(NN):

    def _symbolic_predict_proba(self, X):
        raise NotImplementedError

    def _symbolic_predict(self, x):
        raise NotImplementedError

    def _symbolic_score(self, x, y):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError
