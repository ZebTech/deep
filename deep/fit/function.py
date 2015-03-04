import numpy as np
import theano.tensor as T

from theano import shared, config, function


def compile_batch_function(X, Y, score, batch_size=128, updates=None):
        x = T.matrix()
        y = T.lvector()
        i = T.lscalar()

        score = score(x, y)
        if updates is not None:
            updates = updates(x, y)

        batch_start = i * batch_size
        batch_end = (i+1) * batch_size

        X = shared(np.asarray(X, dtype=config.floatX))
        Y = shared(np.asarray(Y, dtype='int64'))

        givens = dict()
        givens[x] = X[batch_start:batch_end]
        givens[y] = Y[batch_start:batch_end]
        return function([i], score, None, updates, givens)


class BatchIterator():

    def __init__(self, batch_function, n_batches):
        self.batch_function = batch_function
        self.n_batches = n_batches
        self.batch_index = 0

    def next(self):
        if self.batch_index == self.n_batches:
            self.batch_index = 0
            raise StopIteration
        else:
            batch_result = self.batch_function(self.batch_index)
            self.batch_index += 1
            return batch_result

    def __iter__(self):
        return self


class EpochIterator():

    def __init__(self, batch_iterator, n_epcohs=10, data_updates=None):
        self.batch_iterator = batch_iterator
        self.n_epochs = n_epcohs
        self.data_updates = data_updates
        self.epoch_index = 0

    def next(self):
        if self.epoch_index == self.n_epochs:
            self.batch_index = 0
            raise StopIteration
        else:
            batch_results = [batch_result for batch_result in self.batch_iterator]
            self.epoch_index += 1
            return np.mean(batch_results)

    def __iter__(self):
        return self


class AugmentationIterator():

    def __init__(self, X, augmentation):
        self.X = X
        self.augmentation = augmentation

    def next(self):
        return self.augmentation(self.X)

    def __iter__(self):
        return self