import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#Implementation of the sigmoid function
def sigmoid(x: float) -> float:
    return (1/(1 + np.exp(-x)))

#Calculate the value of the cost function given a set of weights
def cost(x, y, theta):
    m = len(y)
    h0 = sigmoid(x @ theta)
    epsilon = 1e-5

    cost = (1/m) * ( ( (-y).T @  np.log(h0+epsilon) ) - ((1-y).T @ np.log(1-h0 + epsilon)))
    return cost

#Gradient descent optimization
def gradientDescent(x, y, theta, alpha, n):
    m = len(y)
    learningCurve = np.zeros((n,1))

    for i in range(n):
        theta = theta - (alpha/m) * (x.T @ (sigmoid(x @ theta) - y))
        learningCurve[i] = (cost(x,y,theta))

    return (learningCurve, theta)

#predict a value given a slice of x values and the current weights
def prediction(x, theta):
    return np.round(sigmoid(x @ theta))

x, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=14)
y = y[:,np.newaxis]
sns.set_style('white')
sns.scatterplot(x[:,0],x[:,1],hue=y.reshape(-1))
plt.show()


m = len(y)

x = np.hstack((np.ones((m,1)), x)) #introducing x_0 to the parameters
n = np.size(x,1) #number of training examples - improvement would be splitting the data up...60/20/20
params = np.zeros((n,1))

iterations = 1500
alpha = 0.03

initial_cost = cost(x,y,params)

print("initial cost is: {}\n".format(initial_cost))

(history, opt_params) = gradientDescent(x, y, params, alpha, iterations)

print("Optimal parameters are\n", opt_params, "\n")

'''
plt.figure()
sns.set_style('white')
plt.plot(range(len(history)), history)
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
'''

y_predict = prediction(x, opt_params)
score = float(sum(y_predict == y))/ float(len(y))

slope = -(opt_params[1] / opt_params[2])
intercept = -(opt_params[0]/opt_params[1])

sns.set_style('white')
sns.scatterplot(x[:,1],x[:,2],hue=y.reshape(-1))
ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k")
plt.show()