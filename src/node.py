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
		self.dJdx = []

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
		self.dJdx = []




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
    
    def __init__(self, shape, init_function):
        self.w = np.random.randn(shape)
        
    def descend_gradient(self, learning_rate, batch_size):
        self.w = self.w - (learning_rate/batch_size)*self.acc_dJdw
        
    def evaluate(self):
        return self.w
        
    
    
class FunctionNode(Node):
    """A node which apply an activation function"""
    
    def __init__(self,parent):
        """parent is a Node"""
        
        self.parent = parent

	def evaluate(self):
		if not self.y:
			self.y = self.f(self.parents[0].evaluate())
		return self.y

    def f(self):
        """apply the activation function
        the function is defined in the sub-classes"""
        
        raise NotImplementedError()

        
    def gradient_f(self):
        
        raise NotImplementedError()


class SigmoidNode(FunctionNode):

	def f(self,x):
		return 1/(1+np.exp(x))

    def gradient_f(self,x):
        return self.f(x)*(1-self.f(x))
        
class BinaryOpNode(Node):
    """These nodes are used for the different operations"""
    
    def __init__(self, parent1, parent2):
        self.parent1 = parent1
        self.parent2 = parent2
        
    def evaluate(self):
        raise NotImplementedError()
        
        
class AdditionNode(BinaryOpNode):
    
    def evaluate(self):
        if not self.y:
            self.y = self.parent1.evaluate() + self.parent2.evaluate()
        return self.y
        
class SubstractionNode(BinaryOpNode):
    
    def evaluate(self):
        if not self.y:
            self.y = self.parent1.evaluate() - self.parent2.evaluate()
        return self.y
        
class MultiplicationNode(BinaryOpNode):
    
    def evaluate(self):
        """multpiplication with matrix, parent1*parent2"""
        if not self.y:
            self.y = np.dot(self.parent1.evaluate(), self.parent2.evaluate())
        return self.y
        
class SoftmaxCrossEntropyNode(BinaryOpNode):
    pass

class SigmoidCrossEnropyNode(BinaryOpNode):
    pass
        