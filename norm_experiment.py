import numpy as np
from deep.datasets import load_mnist
mnist = load_mnist()
X_train, y_train = mnist[0]
X_valid, y_valid = mnist[1]


def zero_through_one(X_train, X_valid):
    return X_train, X_valid


def mean_of_zero(X_train, X_valid):
    return X_train - np.mean(X_train, axis=0), X_valid - np.mean(X_valid, axis=0)


def minus_to_plus_one(X_train, X_valid):
    return (X_train-.5)*2, (X_valid-.5)*2


def standardized_separately(X_train, X_valid):
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(np.mean(X_train, axis=0), axis=0)
    X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(np.mean(X_train, axis=0), axis=0)
    return X_train, X_valid


def standardized_together(X_train, X_valid):
    X = np.vstack((X_train, X_valid))
    X = (X - np.mean(X, axis=0)) / np.std(np.mean(X, axis=0), axis=0)
    return X[:len(X_train)], X[len(X_train):]


def whitened_separately(X_train, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.fit_transform(X_valid)


def whitened_together(X_train, X_valid):
    from sklearn.preprocessing import StandardScaler
    X = np.vstack((X_train, X_valid))
    X = StandardScaler().fit_transform(X)
    return X[:len(X_train)], X[len(X_train):]


def whiten_to_validation(X_train, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_valid = scaler.fit_transform(X_valid)
    return scaler.transform(X_train), X_valid


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    normalizers = [zero_through_one, mean_of_zero, minus_to_plus_one, standardized_separately,
                   standardized_together, whitened_separately, whitened_together]

    colors = ['b', 'r', 'g', 'c', 'k', 'y', 'm']

    from deep.layers import Layer, ConvolutionLayer, PreConv, Pooling, PostConv
    from deep.activations import RectifiedLinear, Softmax
    layers = [
        Layer(1000, RectifiedLinear()),
        Layer(1000, RectifiedLinear()),
        Layer(10, Softmax())
    ]

    for color, normalize in zip(colors, normalizers):
        X_train, X_valid = normalize(X_train, X_valid)

        from deep.models import NN
        from deep.updates import Momentum
        from deep.fit import Iterative
        nn1 = NN(layers, .1, Momentum(.9), Iterative(100))
        nn1.fit(X_train, y_train, X_valid, y_valid)

        nn2 = NN(layers, .01, Momentum(.9), Iterative(100))
        nn2.fit(X_train, y_train, X_valid, y_valid)

        plt.yscale('log')
        if min(nn1.fit_method.valid_costs) < min(nn2.fit_method.valid_costs):
            plt.plot(nn1.fit_method.valid_costs, color=color, label=normalize.__name__)
            #plt.plot(nn1.fit_method.valid_costs, color=color, linestyle='--')
        else:
            plt.plot(nn2.fit_method.valid_costs, color=color, label=normalize.__name__)
            #plt.plot(nn2.fit_method.valid_costs, color=color, linestyle='--')

    plt.legend(loc=3)
    plt.show()