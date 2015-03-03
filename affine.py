import numpy as np


class plankton_augment():

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        dim = int(np.sqrt(n_features))
        X = X.reshape(n_samples, dim, dim)

        augmented = []
        for x in X:
            augmented.append(augment(x))

        from theano import config
        return np.asarray(augmented, dtype=config.floatX).reshape(n_samples, n_features)


def augment(x):
    #: random reflection
    reflect = np.random.randint(2)
    if reflect:
        x = np.fliplr(x)

    #: random rotation
    rotation = np.random.randint(4)
    x = np.rot90(x, rotation)

    scale = np.random.uniform(1 / 1.2, 1.2, 2)
    #shear = np.random.uniform(-.4, .4)
    translation = np.random.uniform(-8, 8, 2)

    #: full rotations don't work very well
    #: can probably just use reflections and rot90
    #: rotation = np.random.uniform(0, 2)

    from skimage.transform import warp, AffineTransform
    transform = AffineTransform(scale=scale, shear=None, translation=translation)
    return warp(x, transform, mode='nearest')
    #return x

if __name__ == '__main__':
    from deep.datasets.load import load_plankton
    X = load_plankton()[0]

    from deep.augmentation import Reshape
    X = Reshape(48).fit_transform(X)
    X = X.reshape(-1, 48, 48)
    X /= 255.0

    #: setup plot
    from matplotlib import pyplot as plt
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0,
                top=1.0, wspace=0.0, hspace=0.0)

    #: plot augmentations
    for row in range(20):
        #: plot random example
        x = X[np.random.randint(len(X))]
        plt.setp(plt.subplot(20, 20, row*20+1), xticks=[], yticks=[])
        plt.imshow(augment(x), cmap=plt.get_cmap('gray'), interpolation='nearest')

        #: plot 19 augmentations of x
        for col in range(2, 20):
            plt.setp(plt.subplot(20, 20, row*20+col), xticks=[], yticks=[])
            plt.imshow(augment(x), cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.show()
