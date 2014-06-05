# system
import os
import sys
import gzip

# module path
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

# utility
import cPickle
import time

# main modules
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# convolution
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

def load_data(dataset):

    # open file and discard first line
    f = open(dataset)
    f.readline()

    # read lines into numpy array
    data = [line.split(',') for line in f.readlines()]
    data = numpy.array(data, dtype=float)

    # separate examples and labels
    data_x = data[:,1:]
    data_y = data[:,0]

    # split data parameters
    end_of_train = data.shape[0] * .7
    end_of_valid = data.shape[0] * .85

    # split data
    train_set_x = data_x[:end_of_train]
    train_set_y = data_y[:end_of_train]
    valid_set_x = data_x[end_of_train:end_of_valid]
    valid_set_y = data_y[end_of_train:end_of_valid]
    test_set_x  = data_x[end_of_valid:]
    test_set_y  = data_y[end_of_valid:]

    def shared_dataset(data_x, data_y, borrow=True):

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x , test_set_y  = shared_dataset(test_set_x , test_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def theano_load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class dA(object):

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size = (n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                                   dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):

        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1-corruption_level,
                                        dtype=theano.config.floatX) *input

    def get_hidden_values(self, input):

        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):

        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1-self.x) * T.log(1-z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = []

        for param, gparam, in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    datasets = theano_load_data(dataset)

    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=28*28, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.,
                                        learning_rate=learning_rate)

    train_da = theano.function([index], cost, updates=updates,
                               givens = {
                                   x: train_set_x[index*batch_size:(index+1)*batch_size]
                               })

    start_time = time.clock()

    # no corruption

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = PIL.Image.fromarray(
        tile_rater_images(X=da.W.get_value(borrow=True).T,
                          img_shape=(28, 28), tile_shape=(10, 10),
                          tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    # 30% corruption

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    da = dA(numpy_rng, theano_rng=theano_rng, input=x,
            n_visible=28*28, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.3,
                                         learning_rate=learning_rate)

    train_da = theano.function([index], cost, udpates=updates,
                               gives={
                                   x: train_set_x[index*batch_size:(index+1)*batch_size]
                               })

    start_time = time.clock()

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = end_time - start_time

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))

    image = PIL.Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')

    os.chdir('../')

if __name__ == '__main__':

    test_dA()