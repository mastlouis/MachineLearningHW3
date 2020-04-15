import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.util import random_noise
from skimage.transform import rotate
import random


# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.


def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    return any_softmax_regression(trainingImages, trainingLabels, epsilon, batchSize)


def any_softmax_regression(X, y, epsilon=0.5, batchsize=100, epochs = 600):
    W = np.random.rand(785,10)
    for i in range(epochs):  # Number of epochs
        for j in range(0, y.shape[1], batchsize):
            end = min(j + batchsize, y.shape[1] - 1)
            Yhat = normalize(W, X[:, j:end])
            gradW = np.dot(X[:, j:end], (Yhat - y[:, j:end]).T) / (end - j)
            # gradW[:, -1] = 0  # This would be right for L2 regularization, but it doesn't belong here
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


def reshapeAndAppend1s(faces):
    faces = faces.reshape(-1, 28, 28)
    faces = faces.T
    faces = np.reshape(faces, (faces.shape[0] ** 2, faces.shape[2]))
    ones = np.ones((faces.shape[1]))
    faces = np.vstack((faces, ones))
    return faces


# Draws the two images side by side to compare
def showImages(before, after, op):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(before)
    ax[0].set_title("Original image")

    ax[1].imshow(after)
    ax[1].set_title(op + " image")
    plt.show()


# Function that produces a rescaled version of the image
# The second parameter is the scale you want to use
def scaleFunc(image, scale):
    scaledImage = rescale(image, scale)
    if scale > 1:
        y, x = image.shape
        startx = x // 2 - (28 // 2)
        starty = y // 2 - (28 // 2)
        finalScaleIm = scaledImage[starty:starty + 28, startx:startx + 28]
    else:
        A = np.zeros((28,28))
        y,x = scaledImage.shape
        randNum = random.randint(0,28 - x)
        randNum2 = random.randint(0, 28 - y)
        A[randNum:randNum+x,randNum2:randNum2 + y] = scaledImage
        finalScaleIm = A
    return finalScaleIm


# Function that produces an image with random noise
def randomNoiseFunc(image):
    noiseImage = random_noise(image)
    return noiseImage


# Function that produces an image is rotated
# degree is the amount you want it rotated
def rotationFunc(image, degree):
    rotIm = rotate(image, degree)
    return rotIm

#translates an a given image
def translationFunc(image, xshift,yshift):
    transIm = image.copy()
    transIm = np.roll(transIm, -yshift, axis=0)  # Positive y rolls up
    transIm = np.roll(transIm, xshift, axis=1)  #Positive x translates right
    return transIm

#performs a random transformation given an image
def performTransform(image):
    randNum = random.randint(1,4)
    final = image
    # translation
    if (randNum == 1):
        randNum = random.randint(-4, 4)
        final = translationFunc(image, randNum,randNum)
    # rotation
    if (randNum == 2):
        randDegree = random.randint(-15, 15)
        final = rotationFunc(image, randDegree)
    # scaling
    if (randNum == 3):
        randScale = random.uniform(0.8, 1.2)
        final = scaleFunc(image, randScale)
    # random noise
    if (randNum == 4):
        final = randomNoiseFunc(image)
    return final

def makeTransformArray(images):
    transformArray = [None] * len(images)
    for i in range(0, len(images)):
        final = performTransform(images[i,:])
        transformArray[i] = final
    np_transform_array = np.array(transformArray)
    return np_transform_array

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
    randomize_order(Xtr, Ytr, 10)
    randomize_order(Xte, Yte, 10)

    # Append a constant 1 term to each example to correspond to the bias terms
    Xtr = reshape_and_append_1s(Xtr)
    Xte = reshape_and_append_1s(Xte)
    Ytr = Ytr.T
    Yte = Yte.T

    # W = softmaxRegression(Xtr, Ytr, Xte, Yte, epsilon=0.5, batchSize=100)
    #
    # print("Training percent correct accuracy: " + str(percent_accuracy(Xtr, W, Ytr)))
    # print("Testing percent correct accuracy: " + str(percent_accuracy(Xte, W, Yte)))
    # print("Training cross entropy: " + str(cross_entropy(Xtr, W, Ytr)))
    # print("Testing cross entropy: " + str(cross_entropy(Xte, W, Yte)))

    # Operate on Augmented data
    Xaug = makeTransformArray(np.reshape(trainingImages, (-1, 28, 28)))
    Yaug = trainingLabels.copy()
    Xaug = np.reshape(Xaug, (Yaug.shape[0], -1))
    # randomize_order(Xaug, Yaug, 10)
    Xaug = reshape_and_append_1s(Xaug)
    Yaug = trainingLabels.T

    Waug = any_softmax_regression(Xaug, Yaug, epsilon=0.5, batchsize=100)

    print("Training percent correct accuracy: " + str(percent_accuracy(Xaug, Waug, Yaug)))
    print("Testing percent correct accuracy: " + str(percent_accuracy(Xte, Waug, Yte)))
    print("Training cross entropy: " + str(cross_entropy(Xaug, Waug, Yaug)))
    print("Testing cross entropy: " + str(cross_entropy(Xte, Waug, Yte)))
    
    # Visualize the vectors

    testImage = np.reshape(testingImages[1,:], (28,28))
    ##testImage = testImageArray[0,:]
    plt.show()

    #Random Transformation Test
    transformedIm = performTransform(testImage)
    showImages(testImage, transformedIm, "Random")

    # Scaling Test
    scaleIm = scaleFunc(testImage, 1.5)
    showImages(testImage, scaleIm, "Scaled")

    # Random noise Test
    randomNoiseIm = randomNoiseFunc(testImage)
    showImages(testImage, randomNoiseIm, "Random Noise")

    # Rotation Test
    rotatedIm = rotationFunc(testImage, 45)
    showImages(testImage, rotatedIm, "Rotated")

    # Translation Test
    translatedIm = translationFunc(testImage, 2,3)
    showImages(testImage, translatedIm, "Translated")
