__author__ = 'gabrielpereyra'

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

def create_features(x):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x -= x.mean(0)
    x = np.nan_to_num(x / x.std(0))
    return x

train_subjects = range(1,3) # load 1-15 and use 16 to validate
train_x = []
train_y = []

for subject in train_subjects:
    filename = 'train_01_16/train_subject%02d.mat' % subject
    print 'Loading', filename
    data = loadmat(filename, squeeze_me=True)
    subject_x = data['X']
    subject_y = data['y']

    subject_x = create_features(subject_x)

    train_x.append(subject_x)
    train_y.append(subject_y)

train_x = np.vstack(train_x)
train_y = np.concatenate(train_y)

data = loadmat('train_01_16/train_subject16.mat',squeeze_me=True)
valid_x = data['X']
valid_y = data['y']

valid_x = create_features(valid_x)

clf = LogisticRegression(random_state=0)
clf.fit(train_x, train_y)

predict_y = clf.predict(valid_x)

diff = valid_y - predict_y
print (len(valid_y) - np.count_nonzero(diff)) / len(valid_y)