import numpy as np
import theano.tensor as T

from theano import shared, config, function


def compile_train_function(model, X, Y, batch_size):
    x = T.matrix()
    y = T.lvector()
    i = T.lscalar()

    batch_start = i * batch_size
    batch_end = (i+1) * batch_size

    X = shared(np.asarray(X, dtype=config.floatX))
    Y = shared(np.asarray(Y, dtype='int64'))

    score = model._symbolic_score(x, y)
    updates = model._symbolic_updates(x, y)
    givens = {x: X[batch_start:batch_end], y: Y[batch_start:batch_end]}

    return function([i], score, None, updates, givens)


def compile_score_function(model, X, Y, batch_size):

    #: strip corruption to get noiseless prediction
    for layer in model.layers:
        layer.corruption = None

    x = T.matrix()
    y = T.lvector()
    i = T.lscalar()

    batch_start = i * batch_size
    batch_end = (i+1) * batch_size

    X = shared(np.asarray(X, dtype=config.floatX))
    Y = shared(np.asarray(Y, dtype='int64'))

    score = model._symbolic_score(x, y)
    givens = {x: X[batch_start:batch_end], y: Y[batch_start:batch_end]}

    return function([i], score, None, None, givens)


def compile_predict_function(model, X, batch_size):

    #: strip corruption to get noiseless prediction
    for layer in model.layers:
        layer.corruption = None

    x = T.matrix()
    i = T.lscalar()

    batch_start = i * batch_size
    batch_end = (i+1) * batch_size

    X = shared(np.asarray(X, dtype=config.floatX))

    score = model._symbolic_predict_proba(x)
    givens = {x: X[batch_start:batch_end]}

    return function([i], score, None, None, givens)

