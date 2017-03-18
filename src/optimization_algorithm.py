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
