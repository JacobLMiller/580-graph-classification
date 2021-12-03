import pickle
import networkx as nx
import graph_tool.all as gt
import numpy as np
from SGD_MDS import all_pairs_shortest_path,myMDS
from graph_functions import get_distance_matrix,get_tsnet_layout

with open('data/test_collection.pkl', 'rb') as myfile:

    myobj = pickle.load(myfile)

def draw_graph(G,X,fname=None):
    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    if fname:
        gt.graph_draw(G, pos=pos,output='drawings/' + fname)
    else:
        # Show layout on the screen
        gt.graph_draw(G, pos=pos)


G = nx.random_partition_graph([20,20,20,20,20],0.8,0.01)
d = np.array(all_pairs_shortest_path(G))
H = gt.Graph(directed=False)
H.add_vertex(n=len(G.nodes()))
for e in G.edges():
    H.add_edge(e[0],e[1])

print(gt.global_clustering(H))
#nx.write_graphml(G,"test.xml")
