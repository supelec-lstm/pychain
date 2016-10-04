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
         
        
class LearnableNode(Node):
    """A node which contains the parameters we want to evaluate"""
    
    def __init__(shape, init_function):
        self.w = 
        
    def descend_gradient(learning_rate, batch_size):
        self.w = self.w - (learning_rate/batch_size)*self.acc_dJdw
        
    
    
class FunctionNode(Node):
    """A node which apply an activation function"""
    
    def __init__(parent):
        """parent is a Node"""
        
        self.parent = parent
        
    def f():
        """apply the activation function
        the function is defined in the sub-classes"""
        
        raise NotImplementedError
        
    def gradient_f():
        
        raise NotImplementedError
        

class SigmoidNode(FunctionNode):
    
    def f():
        return f(self.parent.evaluate())
        
    def gradient_f():
        return f(self.parent.evaluate())*(1-self.parent.evaluate())
        
    