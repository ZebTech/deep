# -*- coding: utf-8 -*-
"""
    deep.datasets.load
    ------------------

    functions to load data.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import gzip
import cPickle

from os.path import join
from os.path import dirname
module_path = dirname(__file__)


def load_mnist():
    """Load and return the mnist digit dataset (classification).

    :reference:

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
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


def load_graham():
    with gzip.open(join(module_path, 'graham', 'graham.pkl.gz')) as data_file:
        return cPickle.load(data_file)
