from deep.datasets import load_mnist
mnist = load_mnist()
X, y = mnist[0]
X_valid, y_valid = mnist[1]

from deep.layers import Layer
from deep.activations import RectifiedLinear, Softmax
layers = [
    Layer(1000, RectifiedLinear()),
    Layer(1000, RectifiedLinear()),
    Layer(10, Softmax())
]

from deep.models import NN
from deep.updates import Momentum
nn = NN(layers, update=Momentum(.9))

from affine import plankton_augment
nn.fit(X, y, X_valid, y_valid, plankton_augment())