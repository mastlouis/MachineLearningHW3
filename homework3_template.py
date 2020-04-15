import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.util import random_noise
from skimage.transform import rotate
import random


# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=None, batchSize=None):
    pass

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
        randDegree = random.randint(-45, 45)
        final = rotationFunc(image, randDegree)
    # scaling
    if (randNum == 3):
        randScale = random.uniform(0.8, 1.2)
        final = scaleFunc(image, randScale)
    # random noise
    if (randNum == 4):
        final = randomNoiseFunc(image)
    return final


if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)

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
