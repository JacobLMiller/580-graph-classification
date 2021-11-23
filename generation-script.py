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
    rand = random.randint
    X = [[rand(-10,10),rand(-10,10)] for i in range(len(d))]
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

    if random.random() < 0.65:
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
    int_gen = lambda x: random.randint(1,x)
    rand = random.random()

    if rand < 0.25:
        type = 'lattice'
        G = gt.lattice([random.randint(2,10),random.randint(2,10)])

    elif rand < 0.5:
        type = 'triangulation'
        G,pos = gt.triangulation(np.random.rand(int_gen(100),2)*int_gen(8),type="delaunay")

    elif rand < 0.75:
        type = 'tree'
        H = nx.random_tree(int_gen(100))
        nx.write_graphml(H,'temp.xml')
        G = gt.load_graph('temp.xml')

    else:
        type = 'ring'
        G = gt.circular_graph(random.randint(5,100),2)

    return G,type

def generate_data(n,verbose=False,outfile="data/test_collection.pkl"):
    graphs = [0 for i in range(n)]
    for i in range(n):
        G,type = generate_graph()
        print("New graph: " + str(i) + " Type: " + type + " |V|: " + str(G.num_vertices()) + " |E|: " + str(G.num_edges()))

        d = get_distance_matrix(G,verbose=False)
        #d = np.array(d)

        X,label = layout_graph(d)

        if verbose:
            draw_graph(G,X,"test_data_" + str(i) + ".png")

        #Compute all features
        Features = {
            'label': label,
            'type': type,
            'stress': calc_stress(X,d),
            'neighborhood': calc_neighborhood(X,d),
            #'edge_crossings': calc_edge_crossings(G.edges(),X),
            'angular_resolution': calc_angular_resolution(G,X),
            'avg_edge_length': calc_edge_lengths(G.edges(),X),
            '|V|': G.num_vertices(),
            '|E|': G.num_edges()
        }

        graphs.append((G,X,Features))

    with open(outfile, 'wb') as myfile:
        pickle.dump(graphs,myfile)

generate_data(100,verbose=True)
