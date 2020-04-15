import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.


def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    any_softmax_regression(trainingImages, trainingLabels, 3e-3, 128)
    pass


def any_softmax_regression(X, y, epsilon=3e-3, batchsize=128):
    W = np.random.rand(785,10)
    # randomize_order(X, y)
    for i in range(3000):  # Number of epochs
        for j in range(0, y.shape[1], batchsize):
            end = min(j + batchsize, y.shape[1] - 1)
            Z = np.dot(W.T, X[:, j:end])
            Z = np.exp(Z)
            Z = Z / np.sum(Z, axis=0)
            gradW = X[:,j:end].T.dot(Z - y[:, j:end]) / (end - j)
            gradW[:, -1] = 0
            W = W - (epsilon * gradW)
    return W


def randomize_order(X, y, order_seed=None):
    if order_seed is None:
        order_seed = np.random.random_integers(0, y.shape[0] - 1)
    for i in range(y.shape[0] // 2):
        Ytemp = y[order_seed[2 * i]].copy()
        Xtemp = X[order_seed[2 * i]].copy()
        y[order_seed[2 * i]] = y[order_seed[(2 * i) + 1]]
        X[order_seed[2 * i]] = X[order_seed[(2 * i) + 1]]
        y[order_seed[(2 * i) + 1]] = Ytemp
        X[order_seed[(2 * i) + 1]] = Xtemp


def reshape_and_append_1s(images):
    # images = images.reshape(-1, 28, 28)
    # images = np.reshape(images, (images.shape[0] ** 2, images.shape[2]))
    ones = np.ones((images.shape[0], 1))
    images = np.hstack((images, ones))
    images = images.T
    return images


if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = reshape_and_append_1s(trainingImages)
    testingImages = reshape_and_append_1s(testingImages)
    trainingLabels = trainingLabels.T
    testingLabels = testingLabels.T


    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)
    
    # Visualize the vectors
    # ...
