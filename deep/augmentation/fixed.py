import numpy as np
from base import Augmentation


class HorizontalReflection(Augmentation):

    def __call__(self, X):

        n_samples, n_features = X.shape
        dim = int(np.sqrt(n_features))
        X = X.reshape(-1, dim, dim)

        X_lr = np.fliplr(X).reshape(n_samples, n_features)
        X = X.reshape(n_samples, n_features)

        return X, X_lr


class Rotate(Augmentation):

    def __call__(self, X):
        n_samples, n_features = X.shape
        dim = int(np.sqrt(n_features))
        X = X.reshape(-1, dim, dim)

        X_90 = np.rot90(X.T, 1).T.reshape(n_samples, n_features)
        X_180 = np.rot90(X.T, 2).T.reshape(n_samples, n_features)
        X_270 = np.rot90(X.T, 3).T.reshape(n_samples, n_features)
        X = X.reshape(n_samples, n_features)

        return X, X_90, X_180, X_270

