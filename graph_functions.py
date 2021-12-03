import graph_tool.all as gt
import numpy as np

import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne

#From tsNET implementation

def get_shortest_path_distance_matrix(g, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.

    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

    return X


# Return the distance matrix of g, with the specified metric.
def get_distance_matrix(g, verbose=True, weights=None):
    if verbose:
        print('[distance_matrix] Computing distance matrix')

    X = get_shortest_path_distance_matrix(g, weights=weights)

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    if verbose:
        print('[distance_matrix] Done!')

    return X




def get_tsnet_layout(G,d):
    n = 1000
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

    #print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, g.num_vertices(), g.num_edges()))

    # Load the PivotMDS layout for initial placement
    Y_init = None

    # Time the method including SPDM calculations

    # Compute the shortest-path distance matrix.
    X = d

    # The actual optimization is done in the thesne module.
    Y = thesne.tsnet(
        X, output_dims=2, random_state=1, perplexity=100, n_epochs=n,
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


    # Convert layout to vertex property
    pos = g.new_vp('vector<float>')
    pos.set_2d_array(Y.T)

    # Show layout on the screen
    #gt.graph_draw(g, pos=pos)
    return Y
