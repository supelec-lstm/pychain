import numpy as np
from node import LearnableNode

class OptimizationAlgorithm:
	def __init__(self, learnable_nodes, learning_rate, learning_rate_decay=None, \
		clipping=None, momentum=None):
		self.learnable_nodes = learnable_nodes

		# Parameters
		self.learning_rate = learning_rate
		#self.learning_rate_decay = learning_rate_decay
		self.clipping = clipping
		#self.momentum = momentum

	def optimize(self, batch_size=1):
		for i, node in enumerate(self.learnable_nodes):
			direction = self.compute_direction(i, node.acc_dJdw / batch_size)
			if self.clipping:
				direction = direction.clip(-self.clipping, self.clipping)
			node.w -= self.learning_rate * direction
		self.reset_accumulators()

	def compute_direction(self, i, grad):
		raise NotImplementedError()

	def reset_accumulators(self):
		for node in self.learnable_nodes:
			node.reset_accumulator()

class GradientDescent(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, learning_rate_decay=None, clipping=None, \
		momentum=None):
		OptimizationAlgorithm.__init__(self, learnable_nodes, learning_rate, learning_rate_decay, clipping, momentum)

	def compute_direction(self, i, grad):
		return grad

class RMSProp(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, decay_rate=0.95, delta=1e-6, \
		learning_rate_decay=None, clipping=None, momentum=None):
		OptimizationAlgorithm.__init__(self, learnable_nodes, learning_rate, learning_rate_decay, clipping, momentum)
		# Follow the description given in the Deep Learning Book
		self.decay_rate = decay_rate
		# Mean squared gradient
		self.ms_grad = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.delta = delta

	def compute_direction(self, i, grad):
		self.ms_grad[i] = self.decay_rate*self.ms_grad[i] + (1-self.decay_rate)*(grad**2)
		return grad / np.sqrt(self.ms_grad[i]+self.delta)

class AdaGrad(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, epsilon=1e-6, \
		learning_rate_decay=None, clipping=None, momentum=None):
		OptimizationAlgorithm.__init__(self, learnable_nodes, learning_rate, learning_rate_decay, clipping, momentum)
		self.epsilon = epsilon
		# Accumulator of squared gradient
		self.acc_s_grad = [np.zeros(node.w.shape) for node in self.learnable_nodes]

	def compute_direction(self, i, grad):
		self.acc_s_grad[i] += grad**2
		return grad / (self.epsilon + np.sqrt(self.acc_s_grad[i]))

class AdaDelta(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate=1.0, gamma=0.95, epsilon=1e-6, \
		learning_rate_decay=None, clipping=None, momentum=None):
		OptimizationAlgorithm.__init__(self, learnable_nodes, learning_rate, learning_rate_decay, clipping, momentum)
		# We strongly advise to keep the learning_rate to 1.0
		self.gamma = gamma
		self.epsilon = epsilon
		# Mean squared gradient
		self.ms_grad = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		# Mean squared dw
		self.ms_dw = [np.zeros(node.w.shape) for node in self.learnable_nodes]

	def compute_direction(self, i, grad):
		self.ms_grad[i] = self.gamma*self.ms_grad[i] + (1-self.gamma)*(grad**2)
		dw = grad * np.sqrt(self.ms_dw[i]+self.epsilon) / np.sqrt(self.ms_grad[i]+self.epsilon)
		self.ms_dw[i] = self.gamma*self.ms_dw[i] + (1-self.gamma)*(dw**2)
		return dw

class Adam(OptimizationAlgorithm):
	# References: https://arxiv.org/pdf/1412.6980v8.pdf and the Deep Learning Book
	def __init__(self, learnable_nodes, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, \
		learning_rate_decay=None, clipping=None, momentum=None):
		OptimizationAlgorithm.__init__(self, learnable_nodes, learning_rate, learning_rate_decay, clipping, momentum)
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.m = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.v = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.t = 0

	def compute_direction(self, i, grad):
		self.t += 1
		self.m[i] = (self.beta_1*self.m[i]) + (1 - self.beta_1)*grad
		self.v[i] = (self.beta_2*self.v[i]) + (1 - self.beta_2)*(grad**2)
		m_corrected = self.m[i] / (1-self.beta_1**self.t)
		v_corrected = self.v[i] / (1-self.beta_2**self.t)
		return m_corrected / (np.sqrt(v_corrected)+self.epsilon)
