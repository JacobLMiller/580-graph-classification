import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix
from collections import namedtuple

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
    G = gt.load_graph("test_graph.dot")
    for i in G.edges():
        print(i)

    d = get_distance_matrix(G)
    #d = np.array(d)

    Y = myMDS(d,weighted=False)
    Y.solve(15)

    #print(Y.X)
    #output_euclidean(G,Y.X)

    #Compute all features
    stress = calc_stress(Y.X,d)
    neighborhood = calc_neighborhood(Y.X,d)
    edge_crossings = calc_edge_crossings(G.edges(), Y.X)
    angular_resolution = calc_angular_resolution(G,Y.X)
    avg_edge_length = calc_edge_lengths(G,Y.X)

    #Save
