""" Load datasets
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

from os.path import dirname
from os.path import join

import gzip
import cPickle


def load_mnist():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    module_path = dirname(__file__)
    with gzip.open(join(module_path, 'mnist', 'mnist.pkl.gz')) as data_file:
        return cPickle.load(data_file)


def load_cifar_10():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError


def load_cifar_100():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError


def load_stl_10():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError


def load_svhn():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError