import networkx as nx
import graph_tool.all as gt
import pickle
import numpy as np

with open('tsnet_block_grid.pkl', 'rb') as myfile:
    myObj = pickle.load(myfile)
print(myObj)

myObj[0].save('tsnet_block_grid.dot')


G = nx.random_partition_graph([20,20,20,20,20], 0.8, 0.01)
nx.drawing.nx_agraph.write_dot(G, "output.dot")
