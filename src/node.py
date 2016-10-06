import numpy as np

class Node:
    """Create a node associated with an elementary operation and capable of backpropagation."""

    def __init__(self, parents = []):

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

    def get_gradient(self, i_child):

        raise NotImplementedError()

    def add_child(self, parent, i):
        """Add a child to the list, along with the indice the parent has in the child referential."""

        self.children.append((parent,i))

    def reset_memoization(self):
        """Reset all memoization variables"""

        self.x = None
        self.y = None
        self.dJdx = None  #list of the gradient with respect to each parent
          




class InputNode(Node):
    """Create the nodes for the inputs of the graph"""
     
    def __init__(self, value = None):
        self.value = value
        self.children = []
         
    def set_value(self, value):
        self.value = value
        print(self.value)
         
    def evaluate(self):
        print(self)
        return self.value
        
        
class LearnableNode(Node):
    """A node which contains the parameters we want to evaluate"""
    
    def __init__(self, shape, init_function = None):
        i, j = shape
        if not init_function:
            self.w = np.random.randn(i,j)*2-np.ones((i,j))
        else:
            self.w = init_function(shape)
        self.acc_dJdw = np.zeros([i,j])
        self.children = []
        
    def descend_gradient(self, learning_rate, batch_size):
        """descend the gradient and reset the accumulator"""

        self.w = self.w - (learning_rate/batch_size) * self.acc_dJdw
        self.acc_dJdw = np.zeros(self.w.shape)
        
    def evaluate(self):
        print(self)
        return self.w

        
    def get_gradient(self, i_child = None):
        if self.dJdx:
            return self.dJdx
        self.dJdx=[]
        i=0
        parent_indice = self.children[i][1]
        gradchildren = np.zeros(self.children[0][0].get_gradient(parent_indice).shape)
        for i in (0, len(self.children)):
            parent_indice = self.children[i][1]
            gradchildren = gradchildren + self.children[i][0].get_gradient(parent_indice)
        self.dJdx.append(gradchildren)
        self.acc_dJdw+=self.dJdx
        return self.dJdx
        



class BinaryOpNode(Node):
    """These nodes are used for the different operations"""
  
    def evaluate(self):
        raise NotImplementedError()
        
    def get_gradient(self):
        raise NotImplementedError()
 
        
class AdditionNode(BinaryOpNode):
    
    def evaluate(self): 
        print(self)
        if not self.y: 
            self.y = self.parents[0].evaluate() + self.parents[1].evaluate()
        return self.y
        
    def get_gradient(self, i_child):
        if self.dJdx:
            return self.dJdx
        self.dJdx=[]
        i=0
        parent_indice = self.children[i][1]
        gradchildren = np.zeros(self.children[0][0].get_gradient(parent_indice).shape)
        for i in (0, len(self.children)):
            parent_indice = self.children[i][1]
            gradchildren = gradchildren + self.children[i][0].get_gradient(parent_indice)
        self.dJdx.append(gradchildren) #list of the gradient for the 2 parents
        self.dJdx.append(gradchildren)
        return self.dJdx[i_child]
 
        
class SubstractionNode(BinaryOpNode):
 
    def evaluate(self):
        print(self)
        if self.y == None:
            self.y = self.parents[0].evaluate() - self.parents[1].evaluate()
        return self.y
        
    def get_gradient(self, i_child):
        if self.dJdx:
            return self.dJdx
        self.dJdx=[]
        i=0
        parent_indice = self.children[i][1]
        gradchildren = np.zeros(self.children[0][0].get_gradient(parent_indice).shape)
        for i in (0, len(self.children)):
            parent_indice = self.children[i][1]
            gradchildren = gradchildren + self.children[i][0].get_gradient(parent_indice)
        self.dJdx.append(gradchildren) #list of the gradient for the 2 parents
        self.dJdx.append(-gradchildren)
        return self.dJdx[i_child]


class MultiplicationNode(BinaryOpNode):
 
    def evaluate(self): 
        """Multpiplication with matrix, parent1*parent2""" 
        print(self)
        if not self.y: 
            print(self.parents)
            self.y = np.dot(self.parents[0].evaluate().T, self.parents[1].evaluate())
        return self.y
        
    def get_gradient(self, i_child):
        if self.dJdx:
            return self.dJdx
        self.dJdx=[]
        i=0
        parent_indice = self.children[i][1]
        gradchildren = np.zeros(self.children[0][0].get_gradient(parent_indice).shape)
        for i in (0, len(self.children)):
            parent_indice=self.children[i][1]
            gradchildren = gradchildren + self.children[i][0].get_gradient(parent_indice)
        self.dJdx.append(gradchildren * self.children[1][0].evaluate()) #list of the gradient for the 2 parents
        self.dJdx.append(gradchildren * self.children[0][0].evaluate())
        return self.dJdx[i_child]

class ConstantGradientNode(Node):

    def evaluate(self):
        print(self)
        if not self.y:
            self.y = self.parents[0].evaluate()

        return self.y

    def get_gradient(self, i_child):
        return np.array([1])
         
class SoftmaxCrossEntropyNode(BinaryOpNode): 
    
    pass
        
 

 
class SigmoidCrossEnropyNode(BinaryOpNode): 
    pass
 
