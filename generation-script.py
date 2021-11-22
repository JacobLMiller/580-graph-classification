import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix
import random
import pickle



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

def draw_graph(G,X,fname=None):
    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    if fname:
        gt.graph_draw(G, pos=pos,output='drawings/' + fname)
    else:
        # Show layout on the screen
        gt.graph_draw(G, pos=pos)


def layout_graph(d,verbose=False):

    if random.random() < 0.6:
        label = 1
        Y = myMDS(d,weighted=False)
        Y.solve(15)
        X = Y.X
    else:
        label = 0
        X = random_init(d)

    if verbose:
        draw_graph(G,X)

    return X,label

def generate_graph():
    rand = random.random()
    if rand < 0.3:
        n = 10
        k = random.randint(1,9)
        G = gt.lattice([n-k,k])
    elif rand < 0.6:
        points = np.random.rand(50,2)
        #G,pos = gt.geometric_graph(points, 0.3)
        G,pos = gt.triangulation(np.random.rand(50,2)*4,type="delaunay")
    else:
        G = gt.circular_graph(50,2)
    return G


n = 25
graphs = [0 for i in range(n)]
for i in range(n):
    G = generate_graph()
    print("New graph: |V|:" + str(G.num_vertices()) + " |E|: " + str(G.num_edges()))

    d = get_distance_matrix(G)
    #d = np.array(d)

    X,label = layout_graph(d)

    draw_graph(G,X,"test_data_" + str(i) + ".png")

    #Compute all features
    Features = {
        'label': label,
        'stress': calc_stress(X,d),
        'neighborhood': calc_neighborhood(X,d),
        'edge_crossings': calc_edge_crossings(G.edges(),X),
        'angular_resolution': calc_angular_resolution(G,X),
        'avg_edge_length': calc_edge_lengths(G.edges(),X),
        '|V|': G.num_vertices(),
        '|E|': G.num_edges()
    }

    graphs.append((G,X,Features))

with open('data/test_collection.pkl', 'wb') as myfile:
    pickle.dump(graphs,myfile)
