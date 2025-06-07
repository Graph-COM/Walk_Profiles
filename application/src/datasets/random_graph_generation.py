import numpy as np
import random
import networkx as nx

def scale_free_graph(n, alpha, beta, gamma):
    G=nx.generators.directed.scale_free_graph(n=n, alpha=alpha, beta=beta, gamma=gamma)
    G=nx.DiGraph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G))) # remove self loops
    return np.array(list(G.edges)).T


def constant_depth_tree(depth, degree_prob, return_sink_nodes=False):
    # depth: int, degree: list of int
    np.random.seed(42)
    edges=[]
    node_idx_at_depth = [[0]]
    num_nodes = 1
    reachable_path = [[0]]
    for d in range(depth-1):
        root_nodes = node_idx_at_depth[d]
        new_node_at_depth_d = []
        for root_node_idx in root_nodes:
            num_children = np.random.choice([i for i in range(1, len(degree_prob)+1)], p=degree_prob)
            new_nodes = [num_nodes + i for i in range(num_children)]
            edges += [[root_node_idx, new_node_idx] for new_node_idx in new_nodes]
            reachable_path += [reachable_path[root_node_idx]+[new_node_idx] for new_node_idx in new_nodes]
            num_nodes += num_children
            new_node_at_depth_d += new_nodes
        node_idx_at_depth.append(new_node_at_depth_d)
    if return_sink_nodes:
        return edges, num_nodes, reachable_path, node_idx_at_depth[-1]
    else:
        return edges, num_nodes, reachable_path

def constant_depth_tree_with_loops(depth, degree_prob):
    np.random.seed(42)
    edges, num_nodes, reachable_path, sink_nodes = constant_depth_tree(depth, degree_prob, return_sink_nodes=True)
    cycle_node_labels = np.zeros([num_nodes, 2])
    edges = list(edges)
    for s in sink_nodes:
        coin = np.random.choice([0, 1])
        if coin == 0:
            edges.append([s, 0])
            cycle_node_labels[reachable_path[s], 0] += 1
        else:
            edges.append([0, s])
            cycle_node_labels[reachable_path[s], 1] += 1
    return edges, num_nodes, cycle_node_labels


#depth=30
#degree_prob = [0.3, 0.7]
#degree_prob = [0.7, 0.3] # for depth 30
#degree_prob = [0.85, 0.15] # for depth 50
#edges, num_nodes, y = constant_depth_tree_with_loops(depth, degree_prob)
#edges = np.array(edges).T
#np.save("../../dataset/tree_graph/"+ "tree-loop_depth%d_max-degree_%d.npy" % (depth, 2), edges)
#np.save("../../dataset/tree_graph/"+ "y_tree-loop_depth%d_max-degree_%d.npy" % (depth, 2), y[:, 1])

n=3000
beta=0.2
#gamma=1e-9
gamma = 0.05
alpha = 1-beta-gamma
edges = scale_free_graph(n, alpha, beta, gamma)
np.save("../../dataset/sf_graph/"+'sf_n%d_a%.4f_b%.4f_c%.4f.npy' % (n, alpha, beta, gamma), edges)

