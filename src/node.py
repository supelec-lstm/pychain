import numpy as np

class Node:
    """Create a node associated with an elementary operation and capable of backpropagation."""

    def __init__(self, parents = None):

        self.parents = parents
        self.children = []

        for i,parent in enumerate(parents):
            parent.add_child(self, i)

        #memoization variables
        self.x = None
        self.y = None
        self.dJdx = None
      

    def evaluate(self):
        raise NotImplementedError()

    def add_child(self, parent, i):
        """Add a child to the list, along with the indice the parent has in the child referential."""

        if not self.children:
            self.children = []
        self.children.append((parent,i))

    def reset_memoization(self):
        """Reset all memoization variables"""

        self.x = None
        self.y = None
        self.dJdx = None  #list of the gradient with respect to each parent

    def get_gradient(self, i_child):
        if self.dJdx is None:
            dJdy = np.sum(child.get_gradient(i) for child,i in self.children)
            self.dJdx = self.compute_gradient(dJdy)
        print(self)
        return self.dJdx[i_child]





class InputNode(Node):
    """Create the nodes for the inputs of the graph"""
     
    def __init__(self, value = None):
        self.value = None
        self.children = []
         
    def set_value(self, value):
        self.value = value
        
         
    def evaluate(self):
        
        return self.value
        
        
class LearnableNode(Node):
    """A node which contains the parameters we want to evaluate"""
    def __init__(self, shape, init_function = None):
        self.shape = shape
        if not init_function:
            self.w = np.random.randn(self.shape[0],self.shape[1])*2-np.ones(self.shape)
        else:
            self.w = init_function(self.shape)
        self.dJdx = None
        self.acc_dJdw = np.zeros(self.shape)
        self.children = []
        
    def descend_gradient(self, learning_rate, batch_size):
        self.w -= (learning_rate/batch_size) * self.acc_dJdw
        
    def evaluate(self):
        return self.w

    def reset_accumulator(self):
        self.acc_dJdw = np.zeros(self.w.shape)


    def compute_gradient(self,dJdy):
        print(dJdy, "y")
        print(self.acc_dJdw)
        self.acc_dJdw += dJdy
        return [dJdy]

 
        
class AdditionNode(Node):
    
    def evaluate(self):
        if self.y is None:
            self.y = self.parents[0].evaluate() + self.parents[1].evaluate()
        return self.y
        
    def compute_gradient(self,dJdy):
        return [dJdy,dJdy]
        
class SubstractionNode(Node):
 
    def evaluate(self):
        if self.y is None:
            self.y = self.parents[0].evaluate() - self.parents[1].evaluate()
        return self.y

    def compute_gradient(self, dJdy):
        return [dJdy,-dJdy]

class MultiplicationNode(Node):
 
    def evaluate(self):
        if self.y is None:
            self.y = np.dot(self.parents[0].evaluate(), self.parents[1].evaluate())
        return self.y

    def compute_gradient(self, dJdy):
        return [np.dot(dJdy, self.parents[1].evaluate().T), np.dot(self.parents[0].evaluate().T, dJdy)]

class ConstantGradientNode(Node):

    def evaluate(self):
        if self.y is None:
            self.y = self.parents[0].evaluate()
        return self.y

    def compute_gradient(self, dJdy):
        return [1]
         
class SoftmaxCrossEntropyNode(Node):

    def evaluate(self):
        return (-1*np.sum(self.parents[0].evaluate()*np.log(self.parents[1].evaluate())))

    def compute_gradient(self, dJdy):
        return [-dJdy * np.log(self.parents[1].evaluate()), -dJdy * (self.parents[0].evaluate() / self.parents[1].evaluate())]


class SigmoidCrossEnropyNode(Node):

    def evaluate(self):
        return -np.sum((self.parents[0].evaluate() * np.log(self.parents[1].evaluate()) + (1 - self.parents[0].evaluate()) * np.log(1 - self.parents[1].evaluate())))

    def compute_gradient(self, dJdy):

        return [-dJdy * (np.log(self.parents[1].evaluate() / (1 - self.parents[1].evaluate()))),-dJdy * (self.parents[0].evaluate() / self.parents[1].evaluate() - (1 - self.parents[0].evaluate()) / (1 - self.parents[1].evaluate()))]


