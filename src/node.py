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
		raise NotImplementedError()

	def get_gradient(self):
		raise NotImplementedError()

	def retrieve_output_gradient(self):
		return np.sum(child.get_gradient(i) for child, i in self.children)

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
		self.acc_dJdw = np.zeros(self.shape).T

	def evaluate(self):
		return self.w

	def get_gradient(self, i_child):
		if self.dJdx is None:
			self.dJdx = self.retrieve_output_gradient()
			self.acc_dJdw += self.dJdx
		return self.dJdx

	def descend_gradient(self, learning_rate, batch_size):
		self.w -= (learning_rate/batch_size)*self.acc_dJdw.T

	def reset_accumulator(self):
		self.acc_dJdw = np.zeros(self.shape).T

class ConstantGradientNode(Node):
	def __init__(self, parents):
		Node.__init__(self, parents)

	def get_gradient(self, i_child):
		return 1

class FunctionNode(Node):
	def __init__(self, parent):
		Node.__init__(self, [parent])

	def f(self):
		raise NotImplementedError()

	def gradient_f(self):
		raise NotImplementedError()

	def evaluate(self):
		if self.x is None:
			self.x = self.parents[0].evaluate()
			self.y = self.f()
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = self.retrieve_output_gradient()
			self.dJdx = np.dot(dJdy, self.gradient_f())
		return self.dJdx

class AddBiasNode(FunctionNode):
	def f(self):
		return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

	def gradient_f(self):
		if self.dJdx is None:
			dJdy = self.retrieve_output_gradient()
			self.dJdx = dJdy[1:,:]
		return self.dJdx

class SigmoidNode(FunctionNode):
	def f(self):
		return 1 / (1 + np.exp(-self.x))

	def gradient_f(self):
		return self.y * (1 - self.y)

class TanhNode(FunctionNode):
	def f(self):
		return np.tanh(self.x)

	def gradient_f(self):
		return 1 - np.square(self.y)

class ReluNode(FunctionNode):
	def f(self):
		return np.maximum(0, self.x)

	def gradient_f(self):
		return (self.y >= 0).astype(float)

class SoftmaxNode(FunctionNode):
	def f(self):
		exp_x = np.exp(self.x)
		total = np.sum(exp_x)
		return exp_x / total

	def gradient_f(self):
		df = np.zeros((len(self.y), len(self.x)))
		for i in range(len(self.y)):
			for j in range(len(self.x)):
				df[i, j] = -self.y[i]*self.y[j] if i != j else self.y[i]*(1-self.y[i])
		return df

def ScalarMultiplicationNode(FunctionNode):
	def __init__(self, parent, scalar):
		FunctionNode.__init__(parent)
		self.scalar = scalar

	def f(self):
		return self.scalar * self.x

	def gradient_f(self):
		return self.scalar * np.eye(len(self.x))

class Norm2Node(FunctionNode):
	def f(self):
		return np.sum(np.square(self.x))

	def gradient_f(self):
		return 2*self.x.T

class BinaryOpNode(Node):
	def __init__(self, parent1, parent2):
		Node.__init__(self, [parent1, parent2])

class AdditionNode(BinaryOpNode):
	def evaluate(self):
		if self.y is None:
			self.x = [self.parents[0].evaluate(), self.parents[1].evaluate()]
			self.y = self.x[0] + self.x[1]
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			self.dJdx = self.retrieve_output_gradient()
		return self.dJdx

class SubstractionNode(BinaryOpNode):
	def evaluate(self):
		if self.y is None:
			self.x = [self.parents[0].evaluate(), self.parents[1].evaluate()]
			self.y = self.x[0] - self.x[1]
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			self.dJdx = self.retrieve_output_gradient()
		return self.dJdx if i_child == 0 else -self.dJdx

class MultiplicationNode(BinaryOpNode):
	def evaluate(self):
		if self.y is None:
			self.x = [self.parents[0].evaluate(), self.parents[1].evaluate()]
			self.y = np.dot(self.x[0], self.x[1])
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = self.retrieve_output_gradient()
			self.dJdx = [np.dot(self.x[1], dJdy), np.dot(dJdy, self.x[0])]
		return self.dJdx[i_child]

class SoftmaxCrossEntropyNode(BinaryOpNode):
	def evaluate(self):
		if self.y is None:
			self.x = [self.parents[0].evaluate(), self.parents[1].evaluate()]
			self.y = -np.sum(self.x[0]*np.log(self.x[1]))
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = self.retrieve_output_gradient()
			self.dJdx = [-np.log(self.x[1].T),-(self.x[0]/self.x[1]).T]
		return self.dJdx[i_child]

class SigmoidCrossEntropyNode(BinaryOpNode):
	def evaluate(self):
		if self.y is None:
			self.x = [self.parents[0].evaluate(), self.parents[1].evaluate()]
			self.y = -(self.x[0]*np.log(x[1]) + (1-self.x[0])*np.log(1-x[1]))
		return self.y

	def get_gradient(self, i_child):
		if self.dJdx is None:
			dJdy = self.retrieve_output_gradient()
			self.dJdx = [-(np.log(self.x[1]/(1-self.x[1]))), \
				-(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))]
		return self.dJdx[i_child]