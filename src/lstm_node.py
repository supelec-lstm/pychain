from node import *
from composite_node import *

class LSTMNode(CompositeNode):
	def __init__(self, dim_x, dim_s, parents=None):
		# Save the dimensions
		self.dim_x = dim_x
		self.dim_s = dim_s

		# Create input nodes
		self.x = InputNode()
		self.h_in = InputNode()
		self.s_in = InputNode()

		# Define c the concatenation of h and x
		self.c = ConcatenationNode(self.h_in, self.x)
		self.dim_c = self.dim_s + self.dim_x

		# g
		self.wg = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mg = MultiplicationNode(self.c, self.wg)
		self.g = SigmoidNode(self.mg)

		# i
		self.wi = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mi = MultiplicationNode(self.c, self.wi)
		self.i = TanhNode(self.mi)

		# Define a what we add to s
		self.a = EWMultiplicationNode(self.g, self.i)
		self.s_out = AdditionNode(self.s_in, self.a)

		# l
		self.l = TanhNode(self.s_out)

		# o
		self.wo = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mo = MultiplicationNode(self.c, self.wo)
		self.o = SigmoidNode(self.mo)

		# Compute h_out
		self.h_out = EWMultiplicationNode(self.l, self.o)

		# Call the constructor of CompositeNode
		nodes = [self.x, self.h_in, self.s_in, self.c, self.wg, self.mg, self.g, self.wi, self.mi, \
			self.i, self.a, self.s_out, self.l, self.wo, self.mo, self.o, self.h_out]
		CompositeNode.__init__(self, nodes, [self.x, self.h_in, self.s_in], [self.h_out, self.s_out], \
			[self.wg, self.wi, self.wo], parents)

	def clone(self):
		# Very practical for debugging
		node = CompositeNode.clone(self)
		nodes = node.nodes
		node.x = nodes[0]
		node.h_in = nodes[1]
		node.s_in = nodes[2]
		node.c = nodes[3]
		node.wg = nodes[4]
		node.mg = nodes[5]
		node.g = nodes[6]
		node.wi = nodes[7]
		node.mi = nodes[8]
		node.i = nodes[9]
		node.a = nodes[10]
		node.s_out = nodes[11]
		node.l = nodes[12]
		node.wo = nodes[13]
		node.mo = nodes[14]
		node.o = nodes[15]
		node.h_out = nodes[16]
		return node

# LSTM With Forget Gate
class LSTMWFGNode(CompositeNode):
	def __init__(self, dim_x, dim_s, parents=None):
		# Save the dimensions
		self.dim_x = dim_x
		self.dim_s = dim_s

		# Create input nodes
		self.x = InputNode()
		self.h_in = InputNode()
		self.s_in = InputNode()

		# Define c the concatenation of h and x
		self.c = ConcatenationNode(self.h_in, self.x)
		self.dim_c = self.dim_s + self.dim_x

		# g
		self.wg = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mg = MultiplicationNode(self.c, self.wg)
		self.g = SigmoidNode(self.mg)

		# i
		self.wi = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mi = MultiplicationNode(self.c, self.wi)
		self.i = TanhNode(self.mi)

		# f
		self.wf = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mf = MultiplicationNode(self.c, self.wf)
		self.f = SigmoidNode(self.mf)

		# Define k what we keep from s
		self.k = EWMultiplicationNode(self.f, self.s_in)

		# Define a what we add to k
		self.a = EWMultiplicationNode(self.g, self.i)
		self.s_out = AdditionNode(self.k, self.a)

		# l
		self.l = TanhNode(self.s_out)

		# o
		self.wo = LearnableNode(0.1 * np.random.randn(self.dim_c, self.dim_s))
		self.mo = MultiplicationNode(self.c, self.wo)
		self.o = SigmoidNode(self.mo)

		# Compute h_out
		self.h_out = EWMultiplicationNode(self.l, self.o)

		# Call the constructor of CompositeNode
		nodes = [self.x, self.h_in, self.s_in, self.c, self.wg, self.mg, self.g, self.wi, self.mi, \
			self.i, self.wf, self.mf, self.f, self.k, self.a, self.s_out, self.l, self.wo, self.mo, \
			self.o, self.h_out]
		CompositeNode.__init__(self, nodes, [self.x, self.h_in, self.s_in], [self.h_out, self.s_out], \
			[self.wg, self.wi, self.wo, self.wf], parents)

	def clone(self):
		# Very practical for debugging
		node = CompositeNode.clone(self)
		nodes = node.nodes
		node.x = nodes[0]
		node.h_in = nodes[1]
		node.s_in = nodes[2]
		node.c = nodes[3]
		node.wg = nodes[4]
		node.mg = nodes[5]
		node.g = nodes[6]
		node.wi = nodes[7]
		node.mi = nodes[8]
		node.i = nodes[9]
		node.wf = nodes[10]
		node.mf = nodes[11]
		node.f = nodes[12]
		node.k = nodes[13]
		node.a = nodes[14]
		node.s_out = nodes[15]
		node.l = nodes[16]
		node.wo = nodes[17]
		node.mo = nodes[18]
		node.o = nodes[19]
		node.h_out = nodes[20]
		return node




