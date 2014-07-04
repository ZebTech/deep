import practice
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def get_max_activation(hidden_weights):

    squared = hidden_weights**2
    sum = np.sum(squared, axis=1)
    root = np.sqrt(sum)

    return hidden_weights / root.reshape(len(root), 1)

def plot_max_activations(max_activations):

    fig = plt.figure()

    for x in range(6):
        for y in range(5):

            ax = fig.add_subplot(6, 5, 6*y+x)
            image = max_activations[x+y].reshape(28,28)
            ax.matshow(image, cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()

if __name__ == "__main__":

    data = practice.load_mnist_data()
    train_x = data[0][0]

    net = practice.NeuralNetwork(784, 30, 784)

    net.train(train_x, train_x, epochs=50)

    max_activations = get_max_activation(net.hidden_w)

    plot_max_activations(max_activations)



