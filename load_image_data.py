import numpy as np
import struct

DATASETS_PATH = './datasets/'
TEST_DATA_PATH    = DATASETS_PATH + 't10k-images.idx3-ubyte'
TEST_LABELS_PATS  = DATASETS_PATH + 't10k-labels.idx1-ubyte'
TRAIN_DATA_PATH   = DATASETS_PATH + 'train-images.idx3-ubyte'
TRAIN_LABELS_PATH = DATASETS_PATH + 'train-labels.idx1-ubyte'

def _read_idx(filename):
    """ A function that can read MNIST's idx file format into numpy arrays.
    The MNIST data files can be downloaded from here:
    
    http://yann.lecun.com/exdb/mnist/
    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.

    Taken from: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_data():
    '''
    return a tuple of X_train, y_train, X_test, y_test where:
    X is a matrix of examples (one example one row)
    y is a column vector of labels
    '''
    x_train = _read_idx(TRAIN_DATA_PATH)
    y_train = _read_idx(TRAIN_LABELS_PATH)
    x_test = _read_idx(TEST_DATA_PATH)
    y_test = _read_idx(TEST_LABELS_PATS)
    return x_train, y_train, x_test, y_test

