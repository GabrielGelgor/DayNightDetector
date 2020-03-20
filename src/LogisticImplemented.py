import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Sigmoid function implementation
def sigmoid(x):
    return (1/(1+np.exp(-x)))

#Return the cost function mapping based on the current parameters
def cost_function(x, y, theta):
    m = len(y)
    h0 = sigmoid(x @ theta) #Hypothesis; weighted sum of x values by weights returns a prediction for y
    epsilon = 1e-5

    cost = (1/m) * ( ( (-y).T @ np.log(h0 + epsilon) ) - ((1-y).T @ np.log(1-h0 + epsilon))) #Determines an error value for current hypothesis
    return cost

def gradient_descent(x, y, theta, alpha, n):
    m = len(y)
    learningCurve = np.zeros((n,1))

    for i in range(n):
        #                            Provides a rate of change for every feature
        theta = theta - (alpha/m) * (x.T @ (sigmoid(x @ theta) - y)) #modify current weights by changing them in the direction of our gradient
        learningCurve[i] = (cost_function(x,y,theta))

    return (learningCurve, theta)

#returns the hypothesis value for debugging purposes
def predict(x, theta):
    return np.round(sigmoid(x @ theta))

def numpArray(filename):
    testExamples = []
    fileTestExample = open(filename, "r")
    for line in fileTestExample:
        newline = line.replace(" ", "")
        linearr = newline.split(",")
        if(linearr[-1][-1] == '\n'):
            linearr[-1] = linearr[-1][:-1]
        for i in range(len(linearr)):
            linearr[i] = int(linearr[i])
        testExamples.append(linearr[:])
    numpTestArray = np.asarray(testExamples)
    fileTestExample.close()
    return numpTestArray

#TODO: Pull in training examples/labels. Replace values once we get the actual data in

x = numpArray("TrainExamples.txt")
y = numpArray("TrainLabels.txt")
m = len(y)

x = np.hstack((np.ones((m,1)), x)) #Adding x_0 to the training example vector (1)
n = np.size(x,1)

#0.00029

alpha = 0.00001
iters = 500000
theta = np.zeros((n,1))

print("Initial Cost:", cost_function(x, y, theta),"\n")

(historic, optimal) = gradient_descent(x, y, theta, alpha, iters)
print(len(historic))

print("Optimal parameters:\n",optimal,"\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(historic)), historic)
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()


y_predict = predict(x, optimal)
score = float(sum(y_predict == y))/ float(len(y))

print("Percentage of correct guesses using the optimal parameters:",score)

#Convert arrays


'''
#TODO: Uncomment once we put this into hackerrank
#HR main function
input = input().split(" ")
grayScale = []
for i in range(len(input)):
    onePix = input[i].split(",")
    a = int((int(onePix[0]) + int(onePix[1]) + int(onePix[2]))/3)
    grayScale.append(a)
'''