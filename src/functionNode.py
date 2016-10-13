from node import *
import numpy as np

class FunctionNode(Node):

    def evaluate(self):
        """Evaluate the output of the neuron with the f function implemented in the sub-classes"""
        if self.y is None:
            
            self.x = self.parents[0].evaluate()
            self.y = self.f(self.parents[0].evaluate())
        return self.y

    def f(self,x):
        """Apply the activation function, defined for each sub-classes"""

        raise NotImplementedError()

class SigmoidNode(FunctionNode):
    """The sigmoid function node"""

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_gradient(self, dJdy):
        return [dJdy * self.y*(1-self.y)]




class TanhNode(FunctionNode):
    """The thanh function node"""

    def f(self, x):
        return np.tanh(x)

    def compute_gradient(self, dJdy):
        return [dJdy * (1-np.square(self.y))]


class ReluNode(FunctionNode):
    """The rectified linear unit node"""

    def f(self, x):
        return np.maximum(0,x)

    def compute_gradient(self, dJdy):
        return [dJdy * (self.evaluate() > 0)]


class SoftMaxNode(FunctionNode):
    """The softmax function node"""
    

    def f(self, x):
        #print(x)
        exp_x = np.exp(x)
        #print(x[0])
        #print('exp',exp_x.shape,self, exp_x[0,0])
        sums = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
        #print(sums[0,0])
        #print('sortie soft',(exp_x / sums))
        return (exp_x / sums)

    def compute_gradient(self, dJdy):
        def delta(j, k):
            return 1 if j == k else 0

        dJdx = np.zeros((self.parents[0].evaluate().shape[0], self.parents[0].evaluate().shape[1]))
        #print('y', self.y.shape)
        #print('djdyshape',dJdy.shape)
        #print(dJdy)
        for i in range(dJdx.shape[0]):
            for j in range(dJdx.shape[1]):
                dJdx[i, j] = np.sum(dJdy[i, k] * (delta(j, k) - self.y[i, k]) * self.y[i, j] for k in range(self.y.shape[1]))
        #print('gradsoft',dJdx)
        return [dJdx]


class Norm2Node(FunctionNode):

    """The norm two function node"""

    def f(self, x):
        return np.sum(np.square(x))

    def compute_gradient(self, dJdy):
        return [dJdy *2 * self.parents[0].evaluate()]

class ScalarMultiplicationNode(FunctionNode):

    """The scalar multiplication node, needed for balancing some weights"""

    def __init__(self,parent,scalar):

        FunctionNode.__init__(self,parent)
        self.scalar = scalar


    def f(self,x):
        return self.scalar*x

    def compute_gradient(self, dJdy):
        return [self.scalar * dJdy]