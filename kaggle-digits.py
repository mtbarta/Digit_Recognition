import math
import random
import string
import numpy as np
from scipy import optimize
from sklearn import decomposition


#make matrix
def makeMatrix(i,j):
    matrix = np.zeros((i,j))
    return matrix

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

def main(): 
    input = 784
    hidden = 800
    output = 10

    inputLayers = input    
    hiddenLayers = hidden  
    outputLayers = output  
    
    #test data
    Xtest_content = np.genfromtxt('test.csv', delimiter=',')
    Xtest = Xtest_content[1:,:]
    
    
    #input source (each image is a 1x400 matrix) 
    #5000 examples gives us a 5000x400 matrix
    mat_contents= np.genfromtxt('train.csv', delimiter=',')
    y = mat_contents[1:,0]
    X = mat_contents[1:,1:785]
    y = y.astype(np.int)
    y[y == 10] = 0
    m = y.size # number of training examples
    y.shape = -1, 1
    X.shape = 42000, -1

    #zero mean and unity variance
    X=X-np.mean(X)
    X=X/np.std(X)
    
    #PCA preprocessing
    #pca = decomposition.PCA(n_components= 5, copy=True, whiten= True)
    #pca.fit(X)
    #X = pca.transform(X)
    
    #pca_test = decomposition.PCA(n_components= 50, copy=True, whiten= True)
    #pca_test.fit(Xtest)
    #Xtest = pca_test.transform(Xtest)

    Y = np.zeros((output, m))
    for i, j in enumerate(y):
        Y[j, i] = 1
    
    #randomly initialize the weights (thetas)
    
    theta1 = rand_init(hidden, input + 1)
    theta2 = rand_init(hidden, hidden + 1)
    theta3 = rand_init(output, hidden + 1)
    theta = np.append(theta1,theta2)

    theta = np.append(theta, theta3)
    #regularization lambda
    lamb = .002
    #optimize
    result = optimize.minimize(
            lambda t: costf(X, Y, t, lamb, input, hidden, output),
            x0 = theta, method = 'CG', jac = True,
            options = {'maxiter': 500, 'disp': True})
    
    f = open('digit_prediction.txt', 'r+')
    for x in test_predict(Xtest, result.x, input, hidden, output):
        f.write(str(x))
        f.write('\n')

    pre_result = predict(X, result.x, input, hidden, output)
    print 'Accuracy is %.2f%%' % (
            np.count_nonzero(pre_result == y.flatten()) * 1.0 / m * 100)
  
def costf(X,Y,theta,lamb,input, hidden, output): #compute the cost function
        m = Y.shape[1]
        #reshape thetas
        #theta1 = 100x785
        theta1 = theta[0:((input + 1) * hidden)].reshape(hidden, -1)
        #theta2 = 0
        theta2 = theta[theta1.size:theta1.size+((hidden+1)*hidden)].reshape(hidden, -1)
        theta3 = theta[theta1.size+theta2.size:].reshape(output, -1)
       #input layer
        a1 = X.T
        a1 = np.vstack((np.ones((1, m)), a1))
       #hidden layer 
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
    
        
        #compute cost
        J = - np.sum(Y * np.log(a4) + (1 - Y) * np.log(1 - a4)) / m
        J = J + lamb / (2.0 * m) * np.sum(theta1[:, 1:] ** 2)
        J = J + lamb / (2.0 * m) * np.sum(theta2[:, 1:] ** 2)
        J = J + lamb / (2.0 * m) * np.sum(theta3[:, 1:] ** 2)
        
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

        return J, np.append(gradient, gradient3)

def predict(X, theta, input, hidden, output):
        m = X.shape[0]
        theta1 = theta[0:((input + 1) * hidden)].reshape(hidden, -1)
        theta2 = theta[theta1.size:theta1.size+((hidden+1)*hidden)].reshape(hidden, -1)
        theta3 = theta[theta1.size+theta2.size:].reshape(output, -1)
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
        
def test_predict(Xtest, theta, input, hidden, output):
        m = Xtest.shape[0]
        #theta1 = 100x785
        theta1 = theta[0:((input + 1) * hidden)].reshape(hidden, -1)
        #theta2 = 0
        theta2 = theta[theta1.size:theta1.size+((hidden+1)*hidden)].reshape(hidden, -1)
        theta3 = theta[theta1.size+theta2.size:].reshape(output, -1)
        a1 = Xtest.T
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



if __name__ == '__main__':
    main()