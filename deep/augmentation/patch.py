import numpy as np


def random_patch_generator(X, patch_size):
    while True:
        patches = []
        for x in X:

            height, width = x.shape

            print height, width

            height_offset = np.random.randint(height - patch_size + 1)
            width_offset = np.random.randint(width - patch_size + 1)
            patches.append(x[height_offset:height_offset+patch_size,
                     width_offset:width_offset+patch_size])

        n_samples = len(X)
        patches = np.asarray(patches).reshape(n_samples, -1)
        yield patches


class PatchWrapper(object):

    def __init__(self, model, patch_size=27):
        self.model = model
        self.patch_size = patch_size

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        pass

    def fit(self, X, y, X_valid, y_valid):
        patch_gen = random_patch_generator(X, self.patch_size)
        return self.model.fit(X, y, X_valid, y_valid, patch_gen)