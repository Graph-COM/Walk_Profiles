import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
def graph_stat(data):
    try:
        edge_index = torch.tensor(data.edge_index)
    except:
        edge_index = torch.tensor(data['edge_index'])
    if edge_index.size(0) != 2:
        edge_index = edge_index.T
    num_nodes = edge_index.max() + 1
    num_edges = edge_index.size(-1)

    # in degree and out degree
    src = edge_index[0]
    dst = edge_index[1]
    out_degree = degree(src, num_nodes=num_nodes)
    in_degree = degree(dst, num_nodes=num_nodes)
    ave_in_degree = in_degree.mean()
    ave_out_degree = out_degree.mean()

    # distribution of shortest path distance
    #symmetric_edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=-1)
    #G = nx.from_edgelist([(i.item(), j.item()) for i, j in symmetric_edge_index.transpose(0, 1)])
    #spd = nx.floyd_warshall_numpy(G)
    #hist_spd = spd.flatten()
    #hist_spd = hist_spd[hist_spd!=np.inf]

    #return num_nodes, num_edges, hist_spd
    return num_nodes, num_edges, ave_in_degree, ave_out_degree

    #print('Num of nodes|edges: %d, %d' % (num_nodes, num_edges))
    #print('Shortest path distance stat: %.3f += %.3f' % (hist_spd.mean(), hist_spd.std()))
    #plt.hist(hist_spd)
    #plt.show()
