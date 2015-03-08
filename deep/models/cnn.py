class Reshape4D(Layer):

    def fit(self, X):
        return self

    def transform(self, incoming_shape):
        n_samples, n_features = X.shape
        dim = int(np.sqrt(n_features))
        return X.reshape(n_samples, 1, dim, dim)

    def _symbolic_transform(self, x):
        n_samples, n_features = x.shape
        dim = T.cast(T.sqrt(n_features), dtype='int64')
        return x.reshape((n_samples, 1, dim, dim))

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def params(self):
        return []


class Reshape2D(Layer):

    def fit(self, incoming_shape):
        return self

    def transform(self, X):
        return X.reshape(1, -1)

    def _symbolic_transform(self, x):
        return x.flatten(2)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def params(self):
        return []
