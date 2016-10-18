import numpy as np

class Node:
	def __init__(self, parents=None):
		self.parents = parents or []
		self.children = []
		for i, parent in enumerate(self.parents):
			parent.add_child(self, i)

		self.x = None
		self.y = None
		self.dJdx = None

	def add_child(self, child, i_child):
		self.children.append((child, i_child))

	def reset_memoization(self):
		self.x = None
		self.y = None
		self.dJdx = None

	def evaluate(self):
		if self.y is None:
			self.x = np.array([parent.evaluate() for parent in self.parents])
			self.y = self.compute_output()
		return self.y

	def compute_output(self):
		raise NotImplementedError()

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = np.sum(child.get_gradient(i) for child, i in self.children)
			self.dJdx = self.compute_gradient(dJdy)
		return self.dJdx[i_child]

	def compute_gradient(self, dJdy):
		raise NotImplementedError()

class InputNode(Node):
	def __init__(self, value=None):
		Node.__init__(self)
		self.value = value

	def set_value(self,value):
		self.value = value

	def evaluate(self):
		return self.value

class LearnableNode(Node):
	def __init__(self, shape, init_function):
		Node.__init__(self)
		self.shape = shape
		self.w = init_function(self.shape)
		self.acc_dJdw = np.zeros(self.shape)

	def compute_output(self):
		return self.w

	def compute_gradient(self, dJdy):
		self.acc_dJdw += dJdy
		return dJdy

	def descend_gradient(self, learning_rate, batch_size):
		self.w -= (learning_rate/batch_size)*self.acc_dJdw

	def reset_accumulator(self):
		self.acc_dJdw = np.zeros(self.shape)

class ConstantGradientNode(Node):
	def __init__(self, parents):
		Node.__init__(self, parents)

	def get_gradient(self, i_child):
		return 1

class FunctionNode(Node):
	def __init__(self, parent):
		Node.__init__(self, [parent])

	def evaluate(self):
		if self.y is None:
			self.x = self.parents[0].evaluate()
			self.y = self.compute_output()
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = np.sum(child.get_gradient(i) for child, i in self.children)
			self.dJdx = self.compute_gradient(dJdy)
		return self.dJdx

class AddBiasNode(FunctionNode):
	def compute_output(self):
		return np.concatenate((np.ones((self.x.shape[0], 1)), self.x), axis=1)

	def compute_gradient(self, dJdy):
		return dJdy[:,1:]

class SigmoidNode(FunctionNode):
	def compute_output(self):
		return 1 / (1 + np.exp(-self.x))

	def compute_gradient(self, dJdy):
		return dJdy * (self.y*(1 - self.y))

class TanhNode(FunctionNode):
	def compute_output(self):
		return np.tanh(self.x)

	def compute_gradient(self, dJdy):
		return dJdy * (1-np.square(self.y))

class ReluNode(FunctionNode):
	def compute_output(self):
		return np.maximum(0, self.x)

	def compute_gradient(self, dJdy):
		return dJdy * (self.x >= 0)

class SoftmaxNode(FunctionNode):
	def compute_output(self):
		exp_x = np.exp(self.x)
		sums = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
		return (exp_x / sums)

	def compute_gradient(self, dJdy):
		def delta(j, k):
			return 1 if j == k else 0

		dJdx = np.zeros((self.x.shape[0], self.x.shape[1]))
		for i in range(dJdx.shape[0]):
			for j in range(dJdx.shape[1]):
				dJdx[i,j] = np.sum(dJdy[i,k]*(delta(j, k)-self.y[i,k])*self.y[i,j] for k in range(self.y.shape[1]))
		return dJdx

class ScalarMultiplicationNode(FunctionNode):
	def __init__(self, parent, scalar):
		FunctionNode.__init__(parent)
		self.scalar = scalar

	def compute_output(self):
		return self.scalar * self.x

	def compute_gradient(self, dJdy):
		return self.scalar * dJdy

class Norm2Node(FunctionNode):
	def compute_output(self):
		return np.sum(np.square(self.x))

	def compute_gradient(self, dJdy):
		return 2*self.x*dJdy

class BinaryOpNode(Node):
	def __init__(self, parent1, parent2):
		Node.__init__(self, [parent1, parent2])

class AdditionNode(BinaryOpNode):
	def compute_output(self):
		return self.x[0] + self.x[1]

	def compute_gradient(self, dJdy):
		return [dJdy, dJdy]

class SubstractionNode(BinaryOpNode):
	def compute_output(self):
		return self.x[0] - self.x[1]

	def compute_gradient(self, dJdy):
		return [dJdy, -dJdy]

class MultiplicationNode(BinaryOpNode):
	def compute_output(self):
		return np.dot(self.x[0], self.x[1])

	def compute_gradient(self, dJdy):
		return [np.dot(dJdy, self.x[1].T), np.dot(self.x[0].T, dJdy)]

class SoftmaxCrossEntropyNode(BinaryOpNode):
	def compute_output(self):
		return -np.sum(self.x[0]*np.log(self.x[1]))

	def compute_gradient(self, dJdy):
		return [-dJdy*np.log(self.x[1]), -dJdy*(self.x[0]/self.x[1])]

class SigmoidCrossEntropyNode(BinaryOpNode):
	def compute_output(self):
		return -np.sum((self.x[0]*np.log(self.x[1]) + (1-self.x[0])*np.log(1-self.x[1])))

	def compute_gradient(self, dJdy):
		return [-dJdy*(np.log(self.x[1]/(1-self.x[1]))), -dJdy*(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))]