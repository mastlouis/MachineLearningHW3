import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.


def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    return any_softmax_regression(trainingImages, trainingLabels, epsilon, batchSize)


def any_softmax_regression(X, y, epsilon=0.1, batchsize=100, epochs = 3000):
    W = np.random.rand(785,10)
    for i in range(epochs):  # Number of epochs
        for j in range(0, y.shape[1], batchsize):
            end = min(j + batchsize, y.shape[1] - 1)
            Yhat = normalize(W, X[:, j:end])
            gradW = np.dot(X[:, j:end], (Yhat - y[:, j:end]).T) / (end - j)
            gradW[:, -1] = 0
            W = W - (epsilon * gradW)
            if i == epochs - 1 and (y.shape[1] - j)/batchsize <= 20:  # If this is in the last 20 iterations
                print('Percent accuracy is ' + str(percent_accuracy(X, W, y)))
                print('Cross entropy loss is ' + str(cross_entropy(X, W, y)))
    return W


def normalize(W, X):
    Z = np.dot(W.T, X)
    Z = np.exp(Z)
    Z = Z / np.sum(Z, axis=0)
    return Z


def percent_accuracy(X, W, Y):
    Yhat = normalize(W, X)
    guesses = np.argmax(Yhat, axis=0)
    ground_truth = np.argmax(Y, axis=0)
    accuracy_array = guesses - ground_truth
    accuracy_array[accuracy_array != 0] = 1
    return 1 - (accuracy_array.sum() / accuracy_array.shape[0])


def cross_entropy(X, W, Y):
    Yhat = normalize(W, X)
    loss = Y * np.log(Yhat)  # Element-wise multiplication
    return 0 - (loss.sum() / Y.shape[1])


def randomize_order(X, y, rounds=1, original_seed=None):
    for _ in range(rounds):
        if original_seed is None:
            seed = np.random.random_integers(0, y.shape[0] - 1, size=(y.shape[0]))
        else:
            seed = original_seed.copy()
        for i in range(y.shape[0] // 2):
            Ytemp = y[seed[2 * i]].copy()
            Xtemp = X[seed[2 * i]].copy()
            y[seed[2 * i]] = y[seed[(2 * i) + 1]]
            X[seed[2 * i]] = X[seed[(2 * i) + 1]]
            y[seed[(2 * i) + 1]] = Ytemp
            X[seed[(2 * i) + 1]] = Xtemp


def reshape_and_append_1s(images):
    # images = images.reshape(-1, 28, 28)
    # images = np.reshape(images, (images.shape[0] ** 2, images.shape[2]))
    ones = np.ones((images.shape[0], 1))
    images = np.hstack((images, ones))
    images = images.T
    return images


if __name__ == "__main__":
    # Xt = np.arange(100)
    # Yt = np.arange(100)
    # randomize_order(Xt, Yt, 5)

    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    Xtr = trainingImages.copy()
    Ytr = trainingLabels.copy()
    Xte = testingImages.copy()
    Yte = testingLabels.copy()

    randomize_order(Xtr, Ytr, 5)
    randomize_order(Xte, Yte, 5)

    # Append a constant 1 term to each example to correspond to the bias terms
    Xtr = reshape_and_append_1s(Xtr)
    Xte = reshape_and_append_1s(Xte)
    Ytr = Ytr.T
    Yte = Yte.T


    W = softmaxRegression(Xtr, Ytr, Xte, Yte, epsilon=0.5, batchSize=100)

    print("Training percent correct accuracy: " + str(percent_accuracy(Xtr, W, Ytr)))
    print("Testing percent correct accuracy: " + str(percent_accuracy(Xte, W, Yte)))
    print("Training cross entropy: " + str(cross_entropy(Xtr, W, Ytr)))
    print("Testing cross entropy: " + str(cross_entropy(Xte, W, Yte)))
    
    # Visualize the vectors
    # ...
