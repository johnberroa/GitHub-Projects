"""
Creates a difference image with two different methods.  First, simple subtraction betweent two images.  Second, a
cumulative difference image, in which the last image can be weighted to be brighter.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import misc

def getFileInfo():
    temp = plt.imread(currentDir+"\{} ({}).jpg".format(fileName, 1))
    temp = misc.imresize(temp, .077)
    M, N, z = temp.shape
    return M, N

def rgb2gray(rgb):
    #Retrieved from "http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python"
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def cumulWeighting(length, late):
    if late:
        weighting = [.5] * length
        weighting[-1] = 3
        return weighting
    else:
        weighting = [1] * length
        return weighting

def showSteps(diff, imgSet, index):
    new = simpleDifference(imgSet[index], imgSet[0])
    plt.imshow(new)
    plt.show()
    plt.imshow(diff)
    plt.show()

def simpleDifference(img1, img2):
    epsilon = 20
    M, N = img1.shape
    diffImg = np.zeros((M, N))
    for x in range(M):
        for y in range(N):
            if np.abs(img1[x, y] - img2[x, y]) <= epsilon:
                diffImg[x, y] = 0
            else:
                diffImg[x, y] = 1
    return diffImg

def cumulativeDifference(imgs):
    epsilon = 10
    weighting = cumulWeighting(len(imgs), False)
    M, N = imgs[0].shape
    diffImg = np.zeros((M, N))
    cumulImg = np.zeros((M, N))
    for i in range(len(imgs)):
        for x in range(M):
            for y in range(N):
                diffImg[x, y] += weighting[i] * np.abs(imgs[0][x, y] - imgs[i][x, y])
                #showSteps(diffImg, imgs, i) #activate if you want to visualize the steps
    return diffImg

#name of files must follow the format "name (count)" where count starts at 0
#files must be located in a folder named "images" in the directory of this file
fileName = 'dolphinc'
currentDir = os.getcwd()
currentDir = currentDir + "\images\\"
numImgs = sum(os.path.isfile(os.path.join(currentDir, f)) for f in os.listdir(currentDir))# - 1 #-1 to not include this coding file
size = getFileInfo()
images = np.zeros([numImgs, size[0], size[1]])
index = 0


for img in range(numImgs):
    loaded = plt.imread(currentDir+"\{} ({}).jpg".format(fileName, index))
    loaded = misc.imresize(loaded, .077)
    loaded = rgb2gray(loaded)
    loaded = np.flipud(loaded)
    loaded = np.fliplr(loaded)
    images[index] = loaded
    index += 1

plt.gray()
simpleDolphin = simpleDifference(images[0],images[1])
plt.imshow(simpleDolphin)
plt.show()
cumulativeDolphin = cumulativeDifference([images[i] for i in range(len(images))])
plt.imshow(cumulativeDolphin)
plt.show()