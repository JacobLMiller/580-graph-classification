import pickle
import networkx as nx
import graph_tool.all as gt
import numpy as np
from SGD_MDS import all_pairs_shortest_path,myMDS
from graph_functions import get_distance_matrix,get_tsnet_layout

import os

path = 'data/noise'
for filename in os.listdir(path):
    print(filename)
