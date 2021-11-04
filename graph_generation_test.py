import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from SGD_MDS import myMDS, all_pairs_shortest_path, output_euclidean

G = nx.complete_graph(5)

d = np.array(all_pairs_shortest_path(G))
Y = myMDS(d)
Y.solve()

pos = {}
count = 0
for x in G.nodes():
    pos[x] = Y.X[count]
    count += 1
nx.draw(G,pos=pos)
plt.show()
plt.clf()


#Move random nodes to create a bad drawing
