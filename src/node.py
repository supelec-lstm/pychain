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
        self.x = None
        self.y = None
        self.dJdx = None

    def evaluate(self, i_output=0):
        if self.y is None:
            self.x = [parent.evaluate(i_parent_output) for (parent, i_parent_output) in self.parents]
            self.y = self.compute_output()
        return self.y[i_output]

    def compute_output(self):
        raise NotImplementedError()

    def get_gradient(self, i_child_input=0):
        if self.dJdx is None:
            self.dJdy = [np.sum(child.get_gradient(i) for child, i in children) for children in self.children]
            self.dJdx = self.compute_gradient()
        return self.dJdx[i_child_input]

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

    def evaluate(self, i_output=0):
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
    def __init__(self, w):
        Node.__init__(self)
        self.w = w
        self.acc_dJdw = np.zeros(self.w.shape)

    def compute_output(self):
        return [self.w]

    def compute_gradient(self):
        self.acc_dJdw += self.dJdy[0]
        return self.dJdy

    def descend_gradient(self, learning_rate, batch_size):
        self.w -= (learning_rate/batch_size)*self.acc_dJdw

    def reset_accumulator(self):
        self.acc_dJdw = np.zeros(self.w.shape)

    def clone(self):
        # The weights are always shared between the clones
        return LearnableNode(self.w)
        
class FunctionNode(Node):
    def __init__(self, parent=None):
        if parent:
            Node.__init__(self, [parent])
        else:
            Node.__init__(self)

    def evaluate(self, i_output=0):
        if self.y is None:
            self.x = self.parents[0][0].evaluate(self.parents[0][1])
            self.y = self.compute_output()
        return self.y

    def get_gradient(self, i_child_input=0):
        if self.dJdx is None:
            self.dJdy = np.sum(child.get_gradient(i) for child, i in self.children[0])
            self.dJdx = self.compute_gradient()
        return self.dJdx

class AddBiasNode(FunctionNode):
    def compute_output(self):
        return np.concatenate((np.ones((self.x.shape[0], 1)), self.x), axis=1)

    def compute_gradient(self):
        return self.dJdy[:,1:]

class IdentityNode(FunctionNode):
    def compute_output(self):
        return self.x

    def compute_gradient(self):
        return self.dJdy

class SigmoidNode(FunctionNode):
    def compute_output(self):
        return 1 / (1 + np.exp(-self.x))

    def compute_gradient(self):
        return self.dJdy * (self.y*(1 - self.y))

class TanhNode(FunctionNode):
    def compute_output(self):
        return np.tanh(self.x)

    def compute_gradient(self):
        return self.dJdy * (1-np.square(self.y))

class ReluNode(FunctionNode):
    def compute_output(self):
        return np.maximum(0, self.x)

    def compute_gradient(self):
        return self.dJdy * (self.x >= 0)

class SoftmaxNode(FunctionNode):
    def compute_output(self):
        exp_x = np.exp(self.x)
        sums = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
        return (exp_x / sums)

    def compute_gradient(self):
        dJdx = np.zeros((self.x.shape[0], self.x.shape[1]))
        for i in range(dJdx.shape[0]):
            y = np.array([self.y[i]])
            dydx = -np.dot(y.T, y) + np.diag(self.y[i])
            dJdx[i,:] = np.dot(dydx, self.dJdy[i])
        return dJdx

class ScalarMultiplicationNode(FunctionNode):
    def __init__(self, parent=None, scalar=1):
        FunctionNode.__init__(self,parent)
        self.scalar = scalar

    def compute_output(self):
        return self.scalar * self.x

    def compute_gradient(self):
        return self.scalar * self.dJdy

    def clone(self):
        return ScalarMultiplicationNode(scalar=self.scalar)

class Norm2Node(FunctionNode):
    def compute_output(self):
        return np.sum(np.square(self.x))

    def compute_gradient(self):
        return 2 * self.x * self.dJdy

class SelectionNode(FunctionNode):
    def __init__(self, parent=None, start=0, end=0):
        FunctionNode.__init__(self, parent)
        self.start = start
        self.end = end

    def compute_output(self):
        return self.x[:,self.start:self.end]

    def compute_gradient(self):
        gradient = np.zeros(self.x.shape)
        gradient[:,self.start:self.end] = self.dJdy
        return gradient

    def clone(self):
        return SelectionNode(start=self.start, end=self.end)

class BinaryOpNode(Node):
    def __init__(self, parent1=None, parent2=None):
        if parent1 is None and parent2 is None:
            Node.__init__(self)
        else:
            Node.__init__(self, [parent1, parent2])

    def evaluate(self, i_output=0):
        if self.y is None:
            self.x = [parent.evaluate(i_parent_output) for (parent, i_parent_output) in self.parents]
            self.y = self.compute_output()
        return self.y

    def get_gradient(self, i_child_input=0):
        if self.dJdx is None:
            """print(self)
            print(self.children[0])
            print(self.children[0][0][0].get_gradient())
            print(np.sum(child.get_gradient(i) for child, i in self.children[0]))
            print()"""
            self.dJdy = np.sum(child.get_gradient(i) for child, i in self.children[0])
            self.dJdx = self.compute_gradient()
        return self.dJdx[i_child_input]

class AdditionNode(BinaryOpNode):
    def compute_output(self):
        return self.x[0] + self.x[1]

    def compute_gradient(self):
        return [self.dJdy, self.dJdy]

class SubstractionNode(BinaryOpNode):
    def compute_output(self):
        return self.x[0] - self.x[1]
    
    def compute_gradient(self):
        return [self.dJdy, -self.dJdy]

class MultiplicationNode(BinaryOpNode):
    def compute_output(self):
        return np.dot(self.x[0], self.x[1])

    def compute_gradient(self):
        """print(self.dJdy)
        print(self.x)
        print(self.children)"""
        return [np.dot(self.dJdy, self.x[1].T), np.dot(self.x[0].T, self.dJdy)]

# Element wise multiplication
class EWMultiplicationNode(BinaryOpNode):
    def compute_output(self):
        return self.x[0] * self.x[1]

    def compute_gradient(self):
        print(self.x)
        print(self.dJdy)
        print(self.children)
        print()
        return [self.dJdy*self.x[1], self.dJdy*self.x[0]]

class  ConcatenationNode(BinaryOpNode):
    def compute_output(self):
        return np.concatenate((self.x[0], self.x[1]), axis=1)

    def compute_gradient(self):
        return [self.dJdy[:,:self.x[0].shape[1]], self.dJdy[:,self.x[0].shape[1]:]]

class SoftmaxCrossEntropyNode(BinaryOpNode):
    def compute_output(self):
        return -np.sum(self.x[0]*np.log(self.x[1]))

    def compute_gradient(self):
        return [-self.dJdy*np.log(self.x[1]), -self.dJdy*(self.x[0]/self.x[1])]

class SigmoidCrossEntropyNode(BinaryOpNode):
    def compute_output(self):
        return -np.sum((self.x[0]*np.log(self.x[1]) + (1-self.x[0])*np.log(1-self.x[1])))

    def compute_gradient(self):
        return [-self.dJdy*(np.log(self.x[1]/(1-self.x[1]))), -self.dJdy*(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))]

class SumNode(Node):
    def compute_output(self):
        return [np.sum(self.x, axis=0)]

    def compute_gradient(self):
        return [self.dJdy[0] for _ in self.parents]