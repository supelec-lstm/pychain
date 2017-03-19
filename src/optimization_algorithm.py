import numpy as np
from node import LearnableNode

class OptimizationAlgorithm:
	def __init__(self, learnable_nodes):
		self.learnable_nodes = learnable_nodes

	def optimize(self):
		raise NotImplementedError()

	def reset_accumulators(self):
		for node in self.learnable_nodes:
			node.reset_accumulator()

class GradientDescent(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate):
		OptimizationAlgorithm.__init__(self, learnable_nodes)
		self.learning_rate = learning_rate

	def optimize(self, batch_size=1):
		for node in self.learnable_nodes:
			node.w -= (self.learning_rate/batch_size)*node.acc_dJdw
		self.reset_accumulators()

class RMSProp(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, decay_rate, delta=1e-6):
		OptimizationAlgorithm.__init__(self, learnable_nodes)
		# Follow the description given in the Deep Learning Book
		self.learning_rate = learning_rate
		self.decay_rate = decay_rate
		self.r = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.delta = delta

	def optimize(self, batch_size=1):
		for i, node in enumerate(self.learnable_nodes):
			grad = node.acc_dJdw/batch_size
			# Update r
			self.r[i] = self.decay_rate*self.r[i] + (1-self.decay_rate)*(grad**2)
			# Update weights
			node.w -= self.learning_rate * grad / np.sqrt(self.r[i] + self.delta)
		self.reset_accumulators()

class AdaGrad(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, epsilon=1e-6):
		OptimizationAlgorithm.__init__(self, learnable_nodes)
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.new_accumulator = [np.zeros(node.w.shape) for node in self.learnable_nodes]

	def optimize(self, batch_size=1):
		for i, node in enumerate(self.learnable_nodes):
			grad = node.acc_dJdw/batch_size
			self.new_accumulator[i] += grad**2
			node.w -= self.learning_rate * grad/(self.epsilon + np.sqrt(self.new_accumulator[i]))
		self.reset_accumulators()

class AdaDelta(OptimizationAlgorithm):
	def __init__(self, learnable_nodes, learning_rate, gamma=0.95, epsilon=1e-6):
		OptimizationAlgorithm.__init__(self, learnable_nodes)
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.new_accumulator = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.delta_accumulator = [np.zeros(node.w.shape) for node in self.learnable_nodes]

	def optimize(self, batch_size=1):
		for i, node in enumerate(self.learnable_nodes):
			grad = node.acc_dJdw/batch_size
			self.new_accumulator[i] = self.gamma * self.new_accumulator[i] + (1-self.gamma)*(grad**2)
			update = grad * np.sqrt(self.delta_accumulator[i] + self.epsilon)/np.sqrt(self.new_accumulator[i] + self.epsilon)
			node.w -= self.learning_rate * update
			self.delta_accumulator[i] = self.gamma * self.delta_accumulator[i] + (1-self.gamma)*(update**2)
		self.reset_accumulators()

class Adam(OptimizationAlgorithm):
	#sources : https://arxiv.org/pdf/1412.6980v8.pdf et deep learning book
	def __init__(self, learnable_nodes, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
		OptimizationAlgorithm.__init__(self, learnable_nodes)
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.m = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.v = [np.zeros(node.w.shape) for node in self.learnable_nodes]
		self.t = 0

	def optimize(self, batch_size=1):
		for i, node in enumerate(self.learnable_nodes):
			grad = node.acc_dJdw / batch_size
			self.t += 1
			self.m[i] = (self.beta_1*self.m[i]) + (1 - self.beta_1) * grad
			self.v[i] = (self.beta_2 * self.v[i]) + (1 - self.beta_2) * (grad ** 2)
			m_corrected = self.m[i] / (1-self.beta_1**self.t)
			v_corrected = self.v[i] / (1-self.beta_2**self.t)
			node.w -= self.learning_rate*m_corrected/(np.sqrt(v_corrected)+self.epsilon)
		self.reset_accumulators()
