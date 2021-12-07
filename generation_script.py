import graph_tool.all as gt
import networkx as nx
import numpy as np
from SGD_MDS import myMDS
from feature_computation import calc_stress, calc_neighborhood, calc_edge_crossings, calc_angular_resolution, calc_edge_lengths
from graph_functions import get_distance_matrix,get_tsnet_layout
import random
import pickle
from collections import namedtuple
import math

#random.seed(12)

feature = namedtuple("feature", ['label',
                                'type',
                                'stress',
                                'neighbor',
                                'edge_crossings',
                                'angular_resolution',
                                'avg_edge_length',
                                'V',
                                'E',
                                'degree',
                                'degree_std',
                                'clusters',
                                'cluster_std'
                                ])

#For i in range(n):
    #Generate graph,

    #Find a drawing | Good or bad
        #Good: Good algorithm
        #Bad: Bad drawing

    #Compute features
    #Obj: Graph, drawing, features, true label

#Save object

def random_init(d):
    rand = random.uniform
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
    #0: ~20%
    #1: ~40%
    #2: ~40%

    if random.random() < 0.8:
        if random.random() < 0.5 and G.num_vertices() > 20:
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
    int_gen = lambda x: random.randint(1,x)
    rand = random.random()

    if rand < 0.2:
        type = 'lattice'
        G = gt.lattice([random.randint(2,10),random.randint(2,10)])

    elif rand < 0.4:
        type = 'triangulation'
        G,pos = gt.triangulation(np.random.rand(random.randint(5,100),2)*int_gen(8),type="delaunay")

    elif rand < 0.6:
        type = 'tree'
        H = nx.random_tree(random.randint(5,100))
        nx.write_graphml(H,'temp.xml')
        G = gt.load_graph('temp.xml')

    elif rand < 0.8:
        type = 'block'
        H = nx.random_partition_graph([int_gen(20),int_gen(20),int_gen(20),int_gen(20),int_gen(20),int_gen(20)],0.8,0.01)
        G = gt.Graph(directed=False)
        G.add_vertex(n=len(H.nodes()))
        for e in H.edges():
            G.add_edge(e[0],e[1])

    else:
        type = 'ring'
        G = gt.circular_graph(random.randint(5,100),2)

    return G,type



def generate_data(n,verbose=False,add_noise=False,outfile="data/test_collection.pkl"):
    graphs = [0 for i in range(n)]
    i = 0
    while i < n:
        G,type = generate_graph()
        print("New graph: " + str(i) + " Type: " + type + " |V|: " + str(G.num_vertices()) + " |E|: " + str(G.num_edges()))

        d = get_distance_matrix(G,verbose=False)
        #d = np.array(d)

        X,label = layout_graph(G,d)
        spread = np.random.rand(1)[0]*1
        if add_noise:
            X += np.random.normal(size=X.shape,loc=0.0,scale=spread)

        if verbose or i % int(math.sqrt(n)) == 0:
            draw_graph(G,X,"test_data_" + str(i) + ".png")

        #Compute all features

        cluster,cluster_std = gt.global_clustering(G)
        degree,degree_std = gt.vertex_average(G,deg='total')
        neighborhood = calc_neighborhood(X,d,rg=1)
        if neighborhood == -1:
            i -= 1
            continue

        Features = feature(label,
                            type,
                            calc_stress(X,d),
                            neighborhood,
                            calc_edge_crossings(G.edges(),X),
                            calc_angular_resolution(G,X),
                            calc_edge_lengths(G.edges(),X),
                            G.num_vertices(),
                            G.num_edges(),
                            degree,
                            degree_std,
                            cluster,
                            cluster_std
                        )

        graphs[i] = (G,X,Features)
        i += 1
        with open(outfile, 'wb') as myfile:
            pickle.dump(graphs,myfile)


    with open(outfile, 'wb') as myfile:
        pickle.dump(graphs,myfile)

if __name__ == "__main__":
    #generate_data(1000,verbose=False,outfile='data/training3.pkl')
    generate_data(200,verbose=False,outfile='data/test2_2.pkl',add_noise=False)
    # G = gt.lattice([random.randint(2,10),random.randint(2,10)])
    # d = get_distance_matrix(G,verbose=False)
    # Y = myMDS(d,weighted=False)
    # Y.solve()
    # Y.X += np.random.normal(size=Y.X.shape,loc=0.0,scale=0.1)
    # draw_graph(G,Y.X)


