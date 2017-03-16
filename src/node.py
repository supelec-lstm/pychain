import numpy as np

class Node:
    def __init__(self, parents=None):
        # Parents for each input
        self.set_parents(parents or [])
        # Children for each output
        self.children = []

        self.x = None
        self.y = None
        self.dJdx = None

        # Dirty flags
        self.output_dirty = True
        self.gradient_dirty = True

    def set_parents(self, parents):
        self.parents = []
        for i, pair in enumerate(parents):
            parent = None
            i_parent_output = 0
            # Check if an output is given
            if type(pair) == tuple:
                parent, i_parent_output = pair
            # Else only a node is given
            else:
                parent = pair
            self.parents.append((parent, i_parent_output))
            parent.add_child(self, i, i_parent_output)
        
    def add_child(self, child, i_child_input, i_output):
        if(i_output >= len(self.children)):
            self.children += [[] for _ in range(i_output - len(self.children) + 1)]
        self.children[i_output].append((child, i_child_input))

    def reset_memoization(self):
        # Reset flags
        self.output_dirty = True
        self.gradient_dirty = True

    def get_output(self, i_output=0):
        if self.output_dirty:
            self.x = [parent.get_output(i_parent_output) for (parent, i_parent_output) in self.parents]
            self.y = self.compute_output()
            self.output_dirty = False
        return self.y[i_output]

    def compute_output(self):
        raise NotImplementedError()

    def get_gradient(self, i_input=0):
        # Get gradient with respect to the i_inputth input
        if self.gradient_dirty:
            self.dJdy = [np.sum(child.get_gradient(i) for child, i in children) for children in self.children]
            self.dJdx = self.compute_gradient()
            self.gradient_dirty = False
        return self.dJdx[i_input]

    def compute_gradient(self):
        raise NotImplementedError()

    def clone(self):
        return type(self)()

class InputNode(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        self.value = value

    def set_value(self, value):
        self.value = value

    def get_output(self, i_output=0):
        return self.value

    def compute_gradient(self):
        return self.dJdy

class GradientInputNode(Node):
    def __init__(self, parents, value=1):
        Node.__init__(self, parents)
        self.value = value

    def set_value(self, value):
        self.value = value

    def compute_output(self):
        return self.x

    def get_gradient(self, i_child_input=0):
        return self.value

class LearnableNode(Node):
    def __init__(self, w, acc_dJdw=None):
        Node.__init__(self)
        self.w = w
        self.acc_dJdw = np.zeros(self.w.shape) if acc_dJdw is None else acc_dJdw
        #self.rms = np.zeros(self.w.shape)

    def compute_output(self):
        return [self.w]

    def compute_gradient(self):
        self.acc_dJdw += self.dJdy[0]
        return self.dJdy

    def descend_gradient(self, learning_rate, batch_size):
        #self.rms = 0.9*self.rms + 0.1*np.square(self.acc_dJdw)
        #self.acc_dJdw /= np.sqrt(self.rms + 0.0001)
        #self.acc_dJdw.clip(-5, 5)
        self.w -= (learning_rate/batch_size)*self.acc_dJdw
        #self.w -= (learning_rate/batch_size)*np.clip(self.acc_dJdw, -5, 5)
        #self.w -= (learning_rate/batch_size) / np.sqrt(self.rms + 0.0001) * self.acc_dJdw

    def reset_accumulator(self):
        self.acc_dJdw.fill(0)

    def clone(self):
        # The weights are always shared between the clones
        return LearnableNode(self.w, self.acc_dJdw)

class AddBiasNode(Node):
    def compute_output(self):
        return [np.concatenate((np.ones((self.x[0].shape[0], 1)), self.x[0]), axis=1)]

    def compute_gradient(self):
        return [self.dJdy[0][:,1:]]

class IdentityNode(Node):
    def compute_output(self):
        return [self.x[0]]

    def compute_gradient(self):
        return [self.dJdy[0]]

class SigmoidNode(Node):
    def compute_output(self):
        return [1 / (1 + np.exp(-self.x[0]))]

    def compute_gradient(self):
        return [self.dJdy[0] * (self.y[0]*(1 - self.y[0]))]

class TanhNode(Node):
    def compute_output(self):
        return [np.tanh(self.x[0])]

    def compute_gradient(self):
        return [self.dJdy[0] * (1-np.square(self.y[0]))]

class ReluNode(Node):
    def compute_output(self):
        return [np.maximum(0, self.x[0])]

    def compute_gradient(self):
        return [self.dJdy[0] * (self.x[0] >= 0)]

class SoftmaxNode(Node):
    def compute_output(self):
        exp_x = np.exp(self.x[0])
        sums = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
        return [(exp_x / sums)]

    def compute_gradient(self):
        dJdx = np.zeros((self.x[0].shape))
        for i in range(dJdx.shape[0]):
            y = np.array([self.y[0][i]])
            dydx = -np.dot(y.T, y) + np.diag(self.y[0][i])
            dJdx[i,:] = np.dot(dydx, self.dJdy[0][i])
        return [dJdx]

class ScalarMultiplicationNode(Node):
    def __init__(self, parents, scalar=1):
        Node.__init__(self, parents)
        self.scalar = scalar

    def compute_output(self):
        return [self.scalar * self.x[0]]

    def compute_gradient(self):
        return [self.scalar * self.dJdy[0]]

    def clone(self):
        return ScalarMultiplicationNode(scalar=self.scalar)

class Norm2Node(Node):
    def compute_output(self):
        return [np.sum(np.square(self.x[0]))]

    def compute_gradient(self):
        return [2 * self.x[0] * self.dJdy[0]]

class SelectionNode(Node):
    def __init__(self, parents, start=0, end=0):
        Node.__init__(self, parents)
        self.start = start
        self.end = end

    def compute_output(self):
        return [self.x[0][:,self.start:self.end]]

    def compute_gradient(self):
        gradient = np.zeros(self.x[0].shape)
        gradient[:,self.start:self.end] = self.dJdy[0]
        return [gradient]

    def clone(self):
        return SelectionNode(start=self.start, end=self.end)

class AdditionNode(Node):
    def compute_output(self):
        return [self.x[0] + self.x[1]]

    def compute_gradient(self):
        return [self.dJdy[0], self.dJdy[0]]

class SubstractionNode(Node):
    def compute_output(self):
        return [self.x[0] - self.x[1]]
    
    def compute_gradient(self):
        return [self.dJdy[0], -self.dJdy[0]]

class MultiplicationNode(Node):
    def compute_output(self):
        return [np.dot(self.x[0], self.x[1])]

    def compute_gradient(self):
        return [np.dot(self.dJdy[0], self.x[1].T), np.dot(self.x[0].T, self.dJdy[0])]

# Element wise multiplication
class EWMultiplicationNode(Node):
    def compute_output(self):
        return [self.x[0] * self.x[1]]

    def compute_gradient(self):
        return [self.dJdy[0]*self.x[1], self.dJdy[0]*self.x[0]]

class  ConcatenationNode(Node):
    def compute_output(self):
        return [np.concatenate((self.x[0], self.x[1]), axis=1)]

    def compute_gradient(self):
        return [self.dJdy[0][:,:self.x[0].shape[1]], self.dJdy[0][:,self.x[0].shape[1]:]]

class SoftmaxCrossEntropyNode(Node):
    def compute_output(self):
        return [-np.sum(self.x[0]*np.log(self.x[1]))]

    def compute_gradient(self):
        return [-self.dJdy[0]*np.log(self.x[1]), -self.dJdy[0]*(self.x[0]/self.x[1])]

class SigmoidCrossEntropyNode(Node):
    def compute_output(self):
        return [-np.sum((self.x[0]*np.log(self.x[1]) + (1-self.x[0])*np.log(1-self.x[1])))]

    def compute_gradient(self):
        return [-self.dJdy[0]*(np.log(self.x[1]/(1-self.x[1]))), \
            -self.dJdy[0]*(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))]

class SumNode(Node):
    def compute_output(self):
        return [np.sum(self.x, axis=0)]

    def compute_gradient(self):
        return [self.dJdy[0] for _ in self.parents]