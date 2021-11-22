import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix
<<<<<<< HEAD
import random
=======
from collections import namedtuple
>>>>>>> 232bd27d74bb344b83e3955f290fec34eb776b5a

#For i in range(n):
    #Generate graph,

    #Find a drawing | Good or bad
        #Good: Good algorithm
        #Bad: Bad drawing

    #Compute features
    #Obj: Graph, drawing, features, true label

#Save object

def random_init(d):
    X = [[random.uniform(-1,1),random.uniform(-1,1)] for i in range(len(d))]
    return np.asarray(X)

def draw_graph(G,X):
    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)

    # Show layout on the screen
    gt.graph_draw(G, pos=pos)

def layout_graph(d,verbose=False):

    if random.random() < 0.8:
        Y = myMDS(d,weighted=False)
        Y.solve(15)
        X = Y.X
    else:
        X = random_init(d)

    if verbose:
        draw_graph(G,X)
    return X


n = 1
for i in range(n):
    G = gt.load_graph("test_graph.dot")


    d = get_distance_matrix(G)
    #d = np.array(d)

    X = layout_graph(d,verbose=True)

    #Compute all features
    stress = calc_stress(X,d)
    neighborhood = calc_neighborhood(X,d)
    edge_crossings = calc_edge_crossings(X)
    angular_resolution = calc_angular_resolution(G,X)
    avg_edge_length = calc_edge_lengths(G.edges(),X)



    #Save
