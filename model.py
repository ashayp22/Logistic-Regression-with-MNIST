#imports
from mnist import MNIST
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import tkinter as tk
import math
import time
import csv
import scipy
from scipy import optimize
from scipy.optimize import minimize

#load and format data
mndata = MNIST("Images")

X, y = mndata.load_training() #get x(images) and y(labels)
Xtest, ytest = mndata.load_testing()

# X = X[0:10000]
# y = y[0:10000]

#format y
temp_y = []
for p in y:
    temp_y.append([p])

y = temp_y

print("loaded")

#logistic regression functions---------------------------------------------------------------------------------

#sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#regularized cost function
def cost(theta, X, y, l): #np array theta, x(features), y(labels), along with lambda

    m = len(y)

    # makes it so theta is 2d array (size is 785 * 1)
    temp_theta = []

    for z in theta:
        temp_theta.append([z])

    theta = np.array(temp_theta)

    predictions = sigmoid(np.dot(X, theta)) #hypothesis
    error = (-y * np.log(predictions)) - ((1 - y) * np.log(1 - predictions)) #error

    cost = 1 / m * sum(error) #calculates cost
    cost += (l / (2 * m)) * sum(theta[1:len(theta)]**2) #regularizes

    return cost

#gradient function
def gradient(theta, X, y, l): #np array theta, x(features), y(labels), along with lambda
    m = len(y)

    #makes it so theta is 2d array (size is 785 * 1)
    temp_theta = []

    for z in theta:
        temp_theta.append([z])

    theta = np.array(temp_theta)

    predictions = sigmoid(np.dot(X, theta))  # hypothesis
    grad = 1 / m * np.dot(X.transpose(), (predictions - y)) #calculates gradient
    grad[1:len(grad)] += (l / m) * theta[1:len(theta)] #adds regularization

    grad = np.ndarray.flatten(grad)

    return grad


#normalizing features
def normalizeFeatures(X): #numpy array of X values
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    norm = (X - mean) / std

    return norm, mean, std #return normalized X values, the mean and standard deviation of each feature


#do training
def train(X, y, num_labels, l, max_iterations):
    m = len(y)  # number of training examples
    n = len(X[0])  # pixels per example(784) + additional 1(total 785)
    all_theta = []

    for lab in range(num_labels): #train for each of the labels
        #perform logistic regression
        #create initial theta
        initial_theta = []
        for b in range(n):  # for each of the pixels
            initial_theta.append(random.random() * 2 * 0.12 - 0.12)
        initial_theta = np.array(initial_theta)  # make numpy array
        #set up y to make it one vs all
        altered_y = []
        for z in y:
            if z == lab:  # the example is this label being checked, correct answer is 1
                altered_y.append([1])
            else:  # not want this label for the example, set to 0
                altered_y.append([0])
        altered_y = np.array(altered_y)

        print('initial: ' + str(len(initial_theta)))

        #now, finally train using advanced optimization(conjugate gradient algorithm)
        OptimizeResult = minimize(cost, initial_theta, args=(X, altered_y, l), method='BFGS', jac=gradient, options={'disp': True})
        print("here")
        print(len(OptimizeResult.x))
        all_theta.append(OptimizeResult.x)  #add flattened version of it
        print("did: " + str(lab))


    return all_theta #return at the end


#CALLS ALL THE FUNCTIONS-----------------------------------------------------------------------------------

for p in X:
    p.insert(0, 1)

#second, make into numpy array
np_X = np.array(X)
np_Y = np.array(y)


#third, normalize features
temp_X = []

#does to every example
for i in range(len(np_X)):
    temp_X.append(normalizeFeatures(np_X[i])[0])

norm_X = np.array(temp_X) #convert into numpy array

#fourth, do logistic regression
all_theta = train(norm_X, np_Y, 10, 2, 400)
all_theta = np.array(all_theta)

#now, add to csv
with open("Saved/loggradients.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(all_theta)

#finally, we see how accurate it is with the test set

for p in Xtest:
    p.insert(0, 1)

np_Xtest = np.array(Xtest)
np_ytest = np.array(ytest)

#normalizes features from the test set

norm_Xtest = []

for i in range(len(norm_Xtest)):
    norm_Xtest.append(normalizeFeatures(norm_Xtest[i])[0])

norm_Xtest = np.array(norm_Xtest) #convert into numpy array

correct = 0
iterator = 0
for example in norm_Xtest:
    values = np.dot(example, all_theta.transpose())
    number = -1
    biggest = -1000000000
    for p in range(0, len(values)):
        if values[p] > biggest:
            biggest = values[p]
            number = p

    if number == np_ytest[iterator][0]:
        correct += 1
    iterator += 1

print("number correct: " + str(correct / len(norm_Xtest)))



#ANOTHER OPTION IS TO GET SAVED THETA-------------------------------------------------------------------------

#or, we read them in
#
# all_theta = []
#
# with open('Saved/loggradients.csv', 'r') as f:
#     reader = csv.reader(f)
#
#     # read file row by row
#     row_num = 0
#     for row in reader:
#         row_num += 1 #to make sure the skipped row doesn't get added
#         if row_num % 2 == 0:
#             continue
#         #set to temp
#         temp_gradients = []
#
#         #make all floats
#         for z in row:
#             temp_gradients.append(float(z))
#
#         #now append to alltheta
#         temp_gradients = np.array(temp_gradients)
#         all_theta.append(temp_gradients)
#
# all_theta = np.array(all_theta)


#vars
mouse_pressed = False #is mouse pressed
WINDOW_SIZE = 140
pixels = []

def printPixels(): #prints the pixels
    for i in range(0, WINDOW_SIZE):
        text = ""
        for j in range(0, WINDOW_SIZE):
            if pixels[j][i] == 0:
                text += "."
            elif pixels[j][i] == 255:
                text += "$"
        print(text)


def createPixels(): #create matrix representing screen
    global pixels
    pixels = []
    for i in range(0, WINDOW_SIZE):
        pixels.append([])
        for j in range(0, WINDOW_SIZE):
            pixels[i].append(0)


def addPixels(x, y):
    global pixels
    for i in range(x-1, x+2):
        if i < 0 or i >= WINDOW_SIZE:
            continue
        for j in range(y-1, y+2):
            if j < 0 or j >= WINDOW_SIZE:
                continue
            pixels[i][j] = 255

createPixels()

#event handlers
def drawline(event):
    global pixels
    x, y = event.x, event.y
    if canvas.old_coords and mouse_pressed:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1)
        addPixels(x, y)
        addPixels(x1, y1)
        #pixels[x][y] = 255
        #pixels[x1][y1] = 255
        #print(str(x) + " " + str(y))
    canvas.old_coords = x, y



def predictDrawing(drawing, theta):
    # first, format
    # drawing
    # since
    # it is too
    # big
    features = []  # ending array

    # add empty values to features
    for i in range(28):
        t = []
        for j in range(28):
            t.append(0)
        features.append(t)

    # now, scale down image
    multiplier = int(WINDOW_SIZE / 28)
    for i in range(0, len(drawing)):
        for j in range(0, len(drawing[i])):
            features[int(j / multiplier)][int(i / multiplier)] += drawing[i][j]

    print("picture")
    for k in features:
        t = ""
        for u in k:
            if u > 0:
                t += "$"
            else:
                t += "."
        print(t)

    features = np.array(features)  # convert the features into
    features.flatten()  # make 1 dimension
    features = np.true_divide(features, multiplier ** 2)  # average out

    # now, make 2d into 1d

    features = np.insert(features, 0, 1)  # first one

    print(features)
    print(len(features))

    # now, must normalize
    norm_features, mean, std = normalizeFeatures(features)

    vals = sigmoid(np.dot(norm_features, theta.transpose()))  # now multiplies
    print(vals)
    prediction = -1
    num = -1000000000
    for p in range(0, len(vals)):
        if vals[p] > num:
            num = vals[p]
            prediction = p
    return prediction

def keydown(e):
    printPixels()
    if e.char == "c":
        canvas.delete("all")
        createPixels()
    elif e.char == "d":
        print("predicting: " + str(predictDrawing(pixels, all_theta)))

def pressed(event):
    global mouse_pressed
    mouse_pressed = True

def released(event):
    global mouse_pressed
    mouse_pressed = False

#window
root = tk.Tk()

root.geometry("" + str(WINDOW_SIZE) + "x" + str(WINDOW_SIZE))

#create canvas
canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE)
canvas.pack()
canvas.old_coords = None

#binds
root.bind('<Motion>', drawline)
root.bind("<KeyPress>", keydown)
root.bind("<Button-1>", pressed)
root.bind("<ButtonRelease-1>", released)

root.mainloop() #loop, no code after gets run
