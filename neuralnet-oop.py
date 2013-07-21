import math
import random
import string
import numpy as np
from scipy import optimize

#sigmoid
def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x), dtype = np.float)

# calculate a random init for theta
def rand_init(m, n):
    epsilon_init = np.sqrt(6) / np.sqrt(m + n)
    return np.random.random((m, n)) * 2 * epsilon_init - epsilon_init

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    y = sigmoid(y)
    return y * (1 - y)

def softmax(x):
    ex = np.exp(x-x.max(1)[:,None])
    return ex/ex.sum(1)[:,np.newaxis]

class GenericNet():
    def __init__(self, input, hidden, output, layers = 2):
        """
        for MNIST---
        input: 784
        hidden: 500
        output: 10
        
        m: 42000
        Y: [10, 42000]
        X: [42000, 784]
        
        theta: [in, out] [784,10]
        
        Layer matrix shapes---
        L1(input): [42000, 784]
        L2(hidden): [42000, 500]
        L3(output): [42000, 10]
        """
        self.input = input
        self.hidden = hidden
        self.output = output
        
        layers = 2
        
        self.theta = []
        #randomly initialize the weights (thetas)
        theta1 = rand_init(hidden, input + 1)
        theta2 = rand_init(hidden, hidden + 1)
        theta3 = rand_init(output, hidden + 1)
        
        self.theta = np.append(theta1,theta2)
        self.theta = np.append(self.theta, theta3)
    def predict(self, theta):
        m = X.shape[0]
        #reshape thetas
        theta1 = self.predictions[0:((self.input + 1) * self.hidden)].reshape(self.hidden, -1)
        theta2 = self.predictions[theta1.size:theta1.size+((self.hidden+1)*self.hidden)].reshape(self.hidden, -1)
        theta3 = self.predictions[theta1.size+theta2.size:].reshape(self.output, -1)
        #input layer
        a1 = X.T
        a1 = np.vstack((np.ones((1, m)), a1))

        z2 = np.dot(theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.vstack((np.ones((1, m)), a2))

        z3 = np.dot(theta2, a2)
        a3 = sigmoid(z3)
        a3 = np.vstack((np.ones((1, m)), a3))
        
        z4 = np.dot(theta3, a3)
        a4 = sigmoid(z4)

        return np.argmax(a4, axis = 0)
        
    def costfunc(self, X, y, theta, Y,lamb, m):
        #feed forward
        #reshape thetas
        theta1 = self.theta[0:((self.input + 1) * self.hidden)].reshape(self.hidden, -1)
        theta2 = self.theta[theta1.size:theta1.size+((self.hidden+1)*self.hidden)].reshape(self.hidden, -1)
        theta3 = self.theta[theta1.size+theta2.size:].reshape(self.output, -1)
        #input layer
        a1 = X.T
        a1 = np.vstack((np.ones((1, m)), a1))
        #hidden layer 
        print theta1.shape
        print a1.shape
        z2 = np.dot(theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.vstack((np.ones((1,m)), a2))
        #hidden #2 layer
        z3 = np.dot(theta2, a2)
        a3 = sigmoid(z3)
        a3 = np.vstack((np.ones((1,m)), a3))
        #output layer
        z4 = np.dot(theta3, a3)
        a4 = sigmoid(z4)
        
        #return a1,a2,a3,a4,z2,z3
    #def cost(self, a4, m, Y, lamb):
        #compute cost
        J = - np.sum(Y * np.log(a4) + (1 - Y) * np.log(1 - a4)) / m
        J = J + lamb / (2.0 * m) * np.sum(self.theta1[:, 1:] ** 2)
        J = J + lamb / (2.0 * m) * np.sum(self.theta2[:, 1:] ** 2)
        J = J + lamb / (2.0 * m) * np.sum(self.theta3[:, 1:] ** 2)
        
        #return J
    #def backprop(self,cost,Y, lamb):
        #returns gradient
        delta4 = a4 - Y
        delta3 = np.dot(theta3.T[1:, :], delta4) * dsigmoid(z3)
        delta2 = np.dot(theta2.T[1:, :], delta3) * dsigmoid(z2)

        gradient1 = np.dot(delta2, a1.T) / (m * 1.0)
        gradient2 = np.dot(delta3, a2.T) / (m * 1.0)
        gradient3 = np.dot(delta4, a3.T) / (m * 1.0)
    
        temp_theta1 = theta1.copy()
        temp_theta1[:, 0:1] = 0
        temp_theta2 = theta2.copy()
        temp_theta2[:, 0:1] = 0
        temp_theta3 = theta3.copy()
        temp_theta3[:, 0:1] = 0

        gradient1 = gradient1 + lamb * 1.0 / m * temp_theta1
        gradient2 = gradient2 + lamb * 1.0 / m * temp_theta2
        gradient3 = gradient3 + lamb * 1.0 / m * temp_theta3
        gradient = np.append(gradient1,gradient2)

        return np.append(gradient, gradient3)
   
    def train(self, X, y, lamb):
        m = y.size # number of training examples
        #create array for correct labels for supervised learning
        Y = np.zeros((self.output, m))
        for i, j in enumerate(y):
            Y[j, i] = 1
        
        result = optimize.minimize(
            lambda t: self.costfunc( X, y, t, Y,lamb, m),
            x0 = self.theta, method = 'CG', jac = True,
            options = {'maxiter': 500, 'disp': True})
        self.optimweights = result.x
        return self.optimweighst
    def test(self, prediction, labels):
        print 'Accuracy is %.2f%%' % (
            np.count_nonzero(prediction == labels.flatten()) * 1.0 / m * 100)
        return
    
class Layer:
    def __init__(self, input,hidden, output, function):
        return
   
    
def main():
    mat_contents= np.genfromtxt('train.csv', delimiter=',')
    #y = correct label
    y = mat_contents[1:,0]
    #X = data inputs for each sample
    X = mat_contents[1:,1:785]
    y = y.astype(np.int)
    y[y == 10] = 0
    y.shape = -1, 1
    X.shape = 42000, -1
    # Y = label array for supervised classification
    #Y = np.zeros((output, m))
    #for i, j in enumerate(y):
    #    Y[j, i] = 1   
        
    net = GenericNet(784,100,10,layers = 2)
    net.train(X, y, 0)
    predictions = net.predict(net.predictions)
    net.test(self.optimweights, y)
    
    
if __name__ == '__main__':
    main()   
        