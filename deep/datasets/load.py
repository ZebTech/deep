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


def load_plankton(test=False):
    """Load and return the mnist digit dataset (classification).

    :reference:

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~350
    Samples total                33000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    if test:
        path = '/home/gabrielpereyra/Desktop/plankton_test.pkl.gz'
    else:
        path = '/home/gabrielpereyra/Desktop/plankton.pkl.gz'
    with gzip.open(path) as data_file:
       return cPickle.load(data_file)


def load_paino_midi():
    """Load and return the piano midi dataset (sequential).

    :reference: http://www-etud.iro.umontreal.ca/~boulanni/icml2012

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~350
    Samples total                33000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    with gzip.open(join(module_path, 'midi', 'piano_midi.pkl.gz')) as data_file:
        #: repickle this in the mnist format instead
        #: of doing it manually
        dataset = cPickle.load(data_file)
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        return train, valid, test
