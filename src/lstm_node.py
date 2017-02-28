from node import *
from composite_node import *

class LSTMNode(CompositeNode):
	def __init__(self, dim_s, dim_y):
		# Create input nodes
		x = InputNode()
		h_in = InputNode()
		s_in = InputNode()

		# Define y the concatenation of h and x
		y = ConcatenationNode(h, x)
		dim_y = dim_s + dim_x

		# g
		wg = LearnableNode(np.random.randn(dim_y, dim_s))
		mg = MultiplicationNode(y, wg)
		g = SigmoidNode(mg)

		# i
		wi = LearnableNode(np.random.rand(dim_y, dim_s))
		mi = MultiplicationNode(y, wi)
		i = TanhNode(mi)

		# Define a what we add to s
		a = EWMultiplicationNode(g, i)
		s_out = AdditionNode(s_in, a)

		# l
		l = TanhNode(s_out)

		# o
		wo = LearnableNode(np.random.rand(dim_y, dim_s))
		mo = MultiplicationNode(y, wo)
		o = SigmoidNode(mo)

		# Compute h_out
		h_out = EWMultiplicationNode(l, o)

		# Call the constructor of CompositeNode
		nodes = [x, h_in, s_in, y, wg, mg, g, wi, mi, i, a, s_out, l, wo, mo, o, h_out]
		CompositeNode(nodes, [x, h_in, s_in], [h_out, s_out], [wg, wi, wo])
