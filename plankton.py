from deep.datasets.load import load_plankton
X, y = load_plankton()
#X_test, y_text = load_plankton(test=True)

from skimage.transform import resize
size = (48, 48)
X = [resize(x, size) for x in X]
#X_test = [resize(x_test, size) for x_test in X_test]

import numpy as np
X = np.asarray(X, dtype='float32')
#X_test = np.asarray(X_test, dtype='float64') / 255.0

X = 1- X.reshape(-1, np.prod(size))
#X_test = X_test.reshape(-1, np.prod(size))

#X -= np.mean(X, axis=1).reshape(-1, 1)

print np.max(X), np.min(X)

from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1)

from deep.layers import Layer, PreConv, PostConv, ConvolutionLayer, Pooling
from deep.activations import RectifiedLinear, Softmax
layers = [
    Layer(1000, RectifiedLinear()),
    Layer(121, Softmax())
]

from deep.models import NN
from deep.updates import Momentum
nn = NN(layers, learning_rate=.01, update=Momentum(.9))

from affine import plankton_augment
nn.fit(X, y, X_valid, y_valid, plankton_augment())