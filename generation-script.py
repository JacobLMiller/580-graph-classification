import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix

#For i in range(n):
    #Generate graph,

    #Find a drawing | Good or bad
        #Good: Good algorithm
        #Bad: Bad drawing

    #Compute features
    #Obj: Graph, drawing, features, true label

#Save object

def draw_graph(G,X):
    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)

    # Show layout on the screen
    gt.graph_draw(G, pos=pos)



n = 1
for i in range(n):
    G = gt.load_graph("test_graph.dot")


    d = get_distance_matrix(G)
    #d = np.array(d)

    Y = myMDS(d,weighted=False)
    Y.solve(15)
    draw_graph(G,Y.X)

    #Compute all features
    stress = calc_stress(Y.X,d)
    neighborhood = calc_neighborhood(Y.X,d)
    edge_crossings = calc_edge_crossings(Y.X)
    angular_resolution = calc_angular_resolution(G,Y.X)
    avg_edge_length = calc_edge_lengths(G.edges(),Y.X)

    #Save
