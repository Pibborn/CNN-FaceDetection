import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import sys
import skimage as skimg
import numpy as np
import random
import csv
import linecache
from PIL import Image
from scipy import misc

debug = False
create_false_examples = True
create_true_examples = True
augment_data = False

def getRandomCsvTokens(csvPath):
    csvFile = open(csvPath, "r")
    random.seed()
    i = 0
    for i in range(0, random.randint(0, 4500)):
        csvFile.readline()
    line = csvFile.readline() # skipping the CSV header
    tokens = line.split(",")
    csvFile.close()
    return tokens

def tokensToInt(tokens):
    for i in range(0, len(tokens)):
        tokens[i] = int(float(tokens[i]))
    return tokens

def augmentData(numLines, infile):
    return generateFalseExampleOverlap(numLines, infile, offset=0.2)

def normalize():
    # todo: part of the ELIVA project
    return

def generateFalseExampleOverlap(numLines, inFile, offset=0.5, mode='x'):
    if offset > 1 or offset < 0:
        raise ValueError("Invalid offset value: must be between 0 and 1")
    if mode != 'x' and mode != 'y':
        raise ValueError("Invalid mode value: must be either 'x' or 'y', found "+ str(mode))
    randomLine = random.randint(2, numLines-1)
    line = linecache.getline(inFile.name, randomLine)
    csvTokens = line.split(",")
    
    imagePath = csvTokens[0]

    # convert to pixel values
    csvTokens = tokensToInt(csvTokens[1:])
    pixelMatrix = misc.imread("originalPics/"+imagePath+".jpg")
    
    x_length = csvTokens[7] - csvTokens[3]
    y_length = csvTokens[0] - csvTokens[2]

    if mode == 'y':
        x_left = x_length * offset + csvTokens[7]
        x_right = x_length * offset + csvTokens[3]
        y_up = csvTokens[0]
        y_down = csvTokens[2]

    if mode == 'x':
        x_left = csvTokens[7]
        x_right = csvTokens[3]
        y_up = y_length * offset + csvTokens[0]
        y_down = y_length * offset + csvTokens[2]

    pixelMatrixFake = pixelMatrix[x_left:x_right, y_up:y_down]

    if debug == True:
        plt.figure(0)
        pylab.imshow(pixelMatrix)
        plt.figure(1)
        pylab.imshow(pixelMatrixFake)
        pylab.show()

    # resize to 30x30
    try:
        pixelMatrixFake = misc.imresize(pixelMatrixFake, (30, 30), interp='nearest')
    except ValueError:
        # matrix could not be resized correctly: skip this face patch
        return None
    return pixelMatrixFake

def generateFalseExampleRandom(numLines, inFile):
    randomLine = random.randint(2, numLines-1)
    line = linecache.getline(inFile.name, randomLine)
    tokens = line.split(",")
    imagePath = tokens[0]
    pixelMatrix = misc.imread("originalPics/"+imagePath+".jpg")
    pixelMatrixRandom = pixelMatrix[0:150, 0:150]
    try:
        pixelMatrixRandom = misc.imresize(pixelMatrixRandom, (30, 30), interp='nearest')
    except ValueError:
        # matrix could not be resized correctly: skip this face patch
        print "WARNING: an image was skipped during resizing, in generateFalseExampleRandom"
        return None
    return pixelMatrixRandom
    
def generateTrueExample(csvTokens):
    imagePath = csvTokens[0]
    pixelMatrix = misc.imread("originalPics/"+imagePath+".jpg")
    csvTokens = tokensToInt(csvTokens[1:])
    pixelMatrixCropped = pixelMatrix[csvTokens[7]:csvTokens[3], csvTokens[0]:csvTokens[2]]
    pixelMatrixDownsampled = [[]]
    if debug == True:
        plt.figure(0)
        pylab.imshow(pixelMatrixCropped)
        pylab.figure(1)
        pylab.imshow(pixelMatrix)
        print pixelMatrixCropped.shape
    # this is a candidate for custom re-implementation: using scipy.misc for the moment
    try:
        pixelMatrixDownsampled = misc.imresize(pixelMatrixCropped, (30, 30), interp='nearest')
    except ValueError:
        # matrix could not be resized correctly: skip this face patch
        return None
    if debug == True:
        pylab.figure(2)
        pylab.imshow(pixelMatrixDownsampled)
        pylab.show()
    return pixelMatrixDownsampled

def toCsvHeader(outFile):
    for j in range(0, 3):
        for i in range(0, 900):
            if j == 0:
                outFile.write("pixr"+str(i)+",")
            if j == 1:
                outFile.write("pixg"+str(i)+",")
            if j == 2:
                outFile.write("pixb"+str(i)+",")
    outFile.write("class\n")
    
def toCsv(pixelMatrix, outFile, label):
    if pixelMatrix.shape != (30, 30, 3):
        # should also clean up the file created by toCsvHeader(filePath)
        # raise ValueError("Wrong image matrix size: 30x30x3 required, "+str(pixelMatrix.shape)+" found")
        return
    if label != 0 and label != 1:
        raise ValueError("Invalid label: 0 or 1 required")
    lineWriter = csv.writer(outFile, delimiter=",")
    pixelFlat = np.reshape(pixelMatrix, (2700))
    # add label to flattened array
    pixelFlat = np.append(pixelFlat, label)
    lineWriter.writerow(pixelFlat)
    return

# debugging function, please ignore
def impressOnRandomImage(outFilePath):
    tokens = getRandomCsvTokens(outFilePath)
    imagePath = tokens[0]
    tokens = tokensToInt(tokens[1:])
    img = mpimg.imread("originalPics/"+imagePath+".jpg")
    imgplot = plt.imshow(img)
    for i in range(1, len(tokens)):
        if float(tokens[i]) < 0:
            tokens[i] = 0
        # rounding the pixel values to integer
        tokens[i] = int(tokens[i])
    # tokens are in this order:
    # A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y
    # matplotlib requires a list of x-axis coordinates and a list of y-axis coordinates as two separate arguments, hence the weird order
    plt.plot([tokens[0], tokens[2], tokens[4], tokens[6], tokens[0]], [tokens[1], tokens[3], tokens[5], tokens[7], tokens[1]], 'r-')
    plt.show()
    
def main(argList):
    inPath = argList[1] 
    tokens = getRandomCsvTokens(inPath)
    outFile = open(argList[2], "wb")
    inFile = open(argList[1], "r")
    inFile.readline() # skipping CSV header in input file
    toCsvHeader(outFile) # writing CSV header in output file

    if create_true_examples == True:
        while True:
            line = inFile.readline()
            if not line:
                break
            tokens = line.split(",")
            pixelMatrix = generateTrueExample(tokens)
            if pixelMatrix == None: # the square face patch was out of image bounds
                continue
            toCsv(pixelMatrix, outFile, 1)

    if augment_data == True:
        inFile.seek(0)
        numLines = sum(1 for line in inFile)
        # reset the file pointer to the beginning of file
        # number of false examples should be the same as true examples
        for j in range(0, numLines):
            pixelMatrix = augmentData(numLines, inFile)
            if pixelMatrix == None:
                continue
            toCsv(pixelMatrix, outFile, 1)

    if create_false_examples == True:
        inFile.seek(0)
        numLines = sum(1 for line in inFile)
        # reset the file pointer to the beginning of file
        # number of false examples should be the same as true examples
        for k in range(0, numLines):
            pixelMatrix = generateFalseExampleRandom(numLines, inFile)
            if pixelMatrix == None:
                continue
            toCsv(pixelMatrix, outFile, 0)

    outFile.close()
    inFile.close()

main(sys.argv)
