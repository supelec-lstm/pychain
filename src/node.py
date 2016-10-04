import numpy as np

class Node:
	"""Create a node associated with an elementary operation and capable of backpropagation."""

	def __init__(self, parents = []):

		self.parents = parents
		self.children = []

		for i,parent in enumerate(parents):
			parent.add_child(parent, i) 

		#memoization variables
		self.x = None
		self.y = None
		self.dJdx = None
      

	def evaluate(self):
		raise NotImplementedError()

	def get_gradient(self, i_child):

		raise NotImplementedError()

	def add_child(self, parent, i):
		"""add a child to the list, along with the indice the parent has in the child referential."""

		self.children.append((parent,i))

	def reset_memoization(self):
		"""Reset all memoization variables"""

		self.x = None
		self.y = None
		self.dJdx = None
          




class InputNode(Node):
     """Create the nodes for the inputs of the graph"""
     
     def __init__(self, value = None):
         self.value = value
         
     def set_value(self, value):
         self.value = value
         
     def evaluate(self):
         return self.value
         
        
class LearnableNode(Node):
    """A node which contains the parameters we want to evaluate"""
    
    def __init__(self, shape, init_function = None):
        if not init_function:
            i,j = shape
            self.w = np.random.randn(i,j)
        else:
            self.w = init_function(shape)
        self.acc_dJdw = np.zeros([i,j])
        
    def descend_gradient(self, learning_rate, batch_size):
        self.w = self.w - (learning_rate/batch_size) * self.acc_dJdw
        self.acc_dJdw = np.zeros(self.w.shape)
        
    def evaluate(self):
        return self.w
        
    def get_gradient(self):
        if self.dJdx:
            return self.dJdx
        gradchildren = np.zeros(self.children[0].get_gradient().shape)
        for child in self.children:
            gradchildren = gradchildren + child.get_gradient()
        self.dJdx.append(gradchildren)
        self.acc_dJdw+=self.dJdw
        return self.dJdx
        



class BinaryOpNode(Node):
    """These nodes are used for the different operations"""

    def __init__(self, parent1, parent2): 
        self.parent1 = parent1 
        self.parent2 = parent2
  
    def evaluate(self):
        raise NotImplementedError()
        
    def get_gradient(self):
        raise NotImplementedError()
 
        
class AdditionNode(BinaryOpNode):
    
    def evaluate(self): 
        if not self.y: 
            self.y = self.parent1.evaluate() + self.parent2.evaluate() 
        return self.y
        
    def get_gradient(self):
        if self.dJdx:
            return self.dJdx
        gradchildren = np.zeros(self.children[0].get_gradient().shape)
        for child in self.children:
            gradchildren = gradchildren + child.get_gradient()
        self.dJdx.append(gradchildren)
        self.dJdx.append(gradchildren)
        return self.dJdx
 
        
class SubstractionNode(BinaryOpNode):
 
    def evaluate(self):
        if not self.y:
            self.y = self.parent1.evaluate() - self.parent2.evaluate()
        return self.y
        
    def get_gradient(self):
        if self.dJdx:
            return self.dJdx
        gradchildren = np.zeros(self.children[0].get_gradient().shape)
        for child in self.children:
            gradchildren = gradchildren + child.get_gradient()
        self.dJdx.append(gradchildren)
        self.dJdx.append(-gradchildren)
        return self.dJdx


class MultiplicationNode(BinaryOpNode):
 
    def evaluate(self): 
        """multpiplication with matrix, parent1*parent2""" 
        if not self.y: 
            self.y = np.dot(self.parent1.evaluate(), self.parent2.evaluate()) 
        return self.y
        
    def get_gradient(self):
        if self.dJdx:
            return self.dJdx
        gradchildren = np.zeros(self.children[0].get_gradient().shape)
        for child in self.children:
            gradchildren = gradchildren + child.get_gradient()
        self.dJdx.append(gradchildren*self.parent2.evaluate())
        self.dJdx.append(gradchildren*self.parent1.evaluate())
        return self.dJdx
 
         
class SoftmaxCrossEntropyNode(BinaryOpNode): 
    
    pass
        
 

 
class SigmoidCrossEnropyNode(BinaryOpNode): 
    pass
 
