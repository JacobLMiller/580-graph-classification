import graph_tool.all as gt
import numpy as np
import math
from random import randint
import scipy.stats as sp

def prob(a, b):

   if a == b:

       return 0.999

   else:

       return 0.001


g, bm = gt.random_graph(100, lambda: np.random.poisson(10), directed=False,

                        model="blockmodel",

                        block_membership=lambda: randint(1,10),

                        edge_probs=prob)

gt.graph_draw(g, vertex_fill_color=bm, edge_color="black", output="blockmodel.pdf")
#g.save('block_graph.dot')
print(len(list(g.vertices())))
