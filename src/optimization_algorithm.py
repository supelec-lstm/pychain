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