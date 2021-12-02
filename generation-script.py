import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix
import random
import pickle
import time

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

#For i in range(n):
    #Generate graph,

    #Find a drawing | Good or bad
        #Good: Good algorithm
        #Bad: Bad drawing

    #Compute features
    #Obj: Graph, drawing, features, true label

#Save object

def get_tsnet_layout(G,d):
    n = 2000
    momentum = 0.5
    tolerance = 1e-7
    window_size = 40

    # Cost function parameters
    r_eps = 0.05

    # Phase 2 cost function parameters
    lambdas_2 = [1, 1.2, 0]

    # Phase 3 cost function parameters
    lambdas_3 = [1, 0.01, 0.6]

    # Read input graph
    g = G

    print('(|V|, |E|) = ({0}, {1})'.format(g.num_vertices(), g.num_edges()))

    # Load the PivotMDS layout for initial placement
    Y_init = None

    # Time the method including SPDM calculations
    start_time = time.time()

    # Compute the shortest-path distance matrix.
    X = d

    #sigma = 100 if graph_name == 'jazz.vna' or graph_name == 'bigger_block.dot' else 40

    # The actual optimization is done in the thesne module.
    Y = thesne.tsnet(
        X, output_dims=2, random_state=1, perplexity=40, n_epochs=n,
        Y=Y_init,
        initial_lr=50, final_lr=50, lr_switch=n // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=True
    )

    Y = layout_io.normalize_layout(Y)

    end_time = time.time()
    comp_time = end_time - start_time
    print('tsNET took {0:.2f} s.'.format(comp_time))

    # Convert layout to vertex property
    pos = g.new_vp('vector<float>')
    pos.set_2d_array(Y.T)

    # Show layout on the screen
    #gt.graph_draw(g, pos=pos)
    return Y

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


def layout_graph(G,d,verbose=False):

    if random.random() < 0.75:
        if random.random() < 0.5 and G.num_vertices() > 10:
            label = 2
            X = get_tsnet_layout(G,d)
        else:
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
    int_gen = lambda x: random.randint(2,x)
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

def generate_data(n,verbose=False,outfile="data/train_collection1.pkl"):
    graphs = [0 for i in range(n)]
    for i in range(n):
        G,type = generate_graph()
        print("New graph: " + str(i) + " Type: " + type + " |V|: " + str(G.num_vertices()) + " |E|: " + str(G.num_edges()))

        d = get_distance_matrix(G,verbose=False)
        #d = np.array(d)

        X,label = layout_graph(G,d)

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

generate_data(1000,verbose=False)
