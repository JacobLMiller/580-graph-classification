import pickle
import networkx as nx
import graph_tool.all as gt
import numpy as np
from SGD_MDS import all_pairs_shortest_path,myMDS
from graph_functions import get_distance_matrix,get_tsnet_layout

import os

path = 'data/noise'
noisy_tests = []
for filename in os.listdir(path):
    noisy_tests.append((int(filename.split("t")[2].split('.')[0]),filename))

noisy_tests.sort()
noisy_tests = [noisy_tests[i][1] for i in range(len(noisy_tests))]
