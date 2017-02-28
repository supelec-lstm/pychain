from node import *
from composite_node import *

class LSTMNode(CompositeNode):
	def __init__(self, dim_x, dim_s):
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
		self.wg = LearnableNode(np.random.randn(self.dim_c, self.dim_s))
		self.mg = MultiplicationNode(self.c, self.wg)
		self.g = SigmoidNode(self.mg)

		# i
		self.wi = LearnableNode(np.random.rand(self.dim_c, self.dim_s))
		self.mi = MultiplicationNode(self.c, self.wi)
		self.i = TanhNode(self.mi)

		# Define a what we add to s
		self.a = EWMultiplicationNode(self.g, self.i)
		self.s_out = AdditionNode(self.s_in, self.a)

		# l
		self.l = TanhNode(self.s_out)

		# o
		self.wo = LearnableNode(np.random.rand(self.dim_c, self.dim_s))
		self.mo = MultiplicationNode(self.c, self.wo)
		self.o = SigmoidNode(self.mo)

		# Compute h_out
		self.h_out = EWMultiplicationNode(self.l, self.o)

		# Call the constructor of CompositeNode
		nodes = [self.x, self.h_in, self.s_in, self.c, self.wg, self.mg, self.g, self.wi, self.mi, \
			self.i, self.a, self.s_out, self.l, self.wo, self.mo, self.o, self.h_out]
		CompositeNode.__init__(self, nodes, [self.x, self.h_in, self.s_in], [self.h_out, self.s_out], \
			[self.wg, self.wi, self.wo])
