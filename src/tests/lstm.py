# A modified copy of lstm.py from pylstm
# -*- coding: utf-8 -*-

import numpy as np


class Weights:
    def __init__(self,dim_s, dim_x):
        self.dim_x = dim_x
        self.dim_s = dim_s
        
        self.wg = 0.1 * np.random.randn(dim_s, dim_x + dim_s)
        self.wi = 0.1 * np.random.randn(dim_s, dim_x + dim_s)
        self.wo = 0.1 * np.random.randn(dim_s, dim_x + dim_s)
        
        self.dwg = np.zeros((dim_s, dim_x + dim_s))
        self.dwi = np.zeros((dim_s, dim_x + dim_s))
        self.dwo = np.zeros((dim_s, dim_x + dim_s))
        
    def descend_gradient(self, learning_rate=0.3):
        self.wg -= learning_rate * self.dwg
        self.wi -= learning_rate * self.dwi
        self.wo -= learning_rate * self.dwo
        
        self.dwg = np.zeros((self.dim_s, self.dim_x + self.dim_s))
        self.dwi = np.zeros((self.dim_s, self.dim_x + self.dim_s))
        self.dwo = np.zeros((self.dim_s, self.dim_x + self.dim_s))
        
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cost_function(x,y):
    return np.norm2(x-y)

def cost_function_derivative(x,y):
    return 2*(x-y)

class LstmCell:
    def __init__(self, weights):
        self.weights = weights
    
    def propagate(self, previous_s, previous_h, x):
        self.x = x
        self.x_concatenated = np.concatenate((previous_h, x), axis=0)
        self.g = sigmoid(np.dot(self.weights.wg, self.x_concatenated))
        self.i = np.tanh(np.dot(self.weights.wi, self.x_concatenated))
        self.s = self.g * self.i + previous_s
        self.o = sigmoid(np.dot(self.weights.wo, self.x_concatenated))
        self.l = np.tanh(self.s)
        self.h = self.l * self.o
    
    def backpropagate(self, ds, dh, y_true):
        self.dJ = cost_function_derivative(self.h, y_true)
        self.dh = dh + self.dJ

        self.weights.dwo += np.dot(self.dh*self.l*self.o*(1-self.o), self.x_concatenated.T)

        self.ds = ds + self.dh*self.o*(1-self.l**2)

        self.weights.dwi += np.dot(self.ds*self.g*(1-self.i**2), self.x_concatenated.T)
        self.weights.dwg += np.dot(self.ds*self.i*self.g*(1-self.g), self.x_concatenated.T)

        self.dh = np.dot(self.weights.wi.T, self.ds*self.g*(1-self.i**2)) + \
                np.dot(self.weights.wg.T, self.ds*self.i*self.g*(1-self.g)) + \
                np.dot(self.weights.wo.T, self.dh*self.l*self.o*(1-self.o))
        self.dh = self.dh[:self.weights.dim_s]

        return self.ds, self.dh


class LstmNetwork:
    def __init__(self, weights, length):
        self.weights = weights
        self.cells = [LstmCell(self.weights) for _ in range(length)]
        self.length = length

    def propagate(self, X, previous_s=None, previous_h=None):
        previous_s = previous_s or np.zeros((self.weights.dim_s, 1))
        previous_h = previous_h or np.zeros((self.weights.dim_s, 1))
        self.cells[0].propagate(previous_s, previous_h, X[0])

        for i in range(1,self.length):
            self.cells[i].propagate(self.cells[i-1].s, self.cells[i-1].h, X[i])

        return [self.cells[i].h for i in range(self.length)]

    def propagate_self_feeding(self, x, previous_s=None, previous_h=None):
        previous_s = previous_s or np.zeros((self.weights.dim_s, 1))
        previous_h = previous_h or np.zeros((self.weights.dim_s, 1))
        self.cells[0].propagate(previous_s, previous_h, x)

        for i in range(1,self.length):
            self.cells[i].propagate(self.cells[i-1].s, self.cells[i-1].h, self.cells[i-1].h)

        return [self.cells[i].h for i in range(self.length)]

    def backpropagate(self, Y):
        self.cells[-1].backpropagate(np.zeros((self.weights.dim_s, 1)), np.zeros((self.weights.dim_s, 1)), Y[-1])
        for i in range(self.length-2, -1, -1):
            self.cells[i].backpropagate(self.cells[i+1].ds, self.cells[i+1].dh, Y[i])

    def descend_gradient(self, learning_rate):
        self.weights.descend_gradient(learning_rate)