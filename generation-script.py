#import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import all_pairs_shortest_path, myMDS,output_euclidean

#For i in range(n):
    #Generate graph,

    #Find a drawing | Good or bad
        #Good: Good algorithm
        #Bad: Bad drawing

    #Compute features
    #Obj: Graph, drawing, features, true label

#Save object

n = 1
for i in range(n):
    G = nx.random_tree(25)

    d = all_pairs_shortest_path(G)
    d = np.array(d)

    Y = myMDS(d,weighted=False)
    Y.solve(15)
    #print(Y.X)
    #output_euclidean(G,Y.X)

    #Compute all features
    stress = Y.calc_stress()

    #Save
