import numpy as np
import torch
import time
from torch_geometric.utils import degree as pyg_degree

def construct_edge_adjacency_matrix(edge_index, num_nodes, nbt=False, normalize=False, self_loop=False):
    E = edge_index.size(-1) * 2 # 2 for forward+reverse edges
    n = num_nodes
    rev_edge_index = edge_index[[1, 0]]
    # construct a neighborhood dict
    neighbor_out_dict = {i: edge_index[1, torch.where(edge_index[0, :]==i)[0]] for i in range(n)}
    neighbor_in_dict = {i: rev_edge_index[1, torch.where(rev_edge_index[0, :] == i)[0]] for i in range(n)}
    node_degree = {i: len(neighbor_in_dict[i]) + len(neighbor_out_dict[i]) for i in range(n)}
    all_edge_index = torch.cat([edge_index, rev_edge_index], dim=-1)
    edge_to_index = {'%d-%d' % (e[0], e[1]): i for i, e in enumerate(all_edge_index.T)}
    hyper_edge_pos = [] # a hyper edge is a 4-tuple (u,v,x,y) where (u,v),(x,y) are edges,  v=x but u is not y
    hyper_edge_neg = []
    hyper_edge_B_pos = [] # a B hyper edge is a 3-tuple (w, u, v) where (u,v) is an edge and w=u
    hyper_edge_B_neg = []
    hyper_edge_weight_B_pos = []
    hyper_edge_weight_B_neg = []
    hyper_edge_C = [] # a C hyper edge is a 3-tuple (u, v, w) where (u,v) is an edge and v=w

    for i, e1 in enumerate(all_edge_index.T):
        u, v = e1[0], e1[1]
        # construct hyper edges
        for y in neighbor_out_dict[v.item()]:
            if y == u and nbt:
                continue
            hyper_edge_pos.append([i, edge_to_index['%d-%d'%(v, y)]])
        for y in neighbor_in_dict[v.item()]:
            if y == u and nbt:
                continue
            hyper_edge_neg.append([i, edge_to_index['%d-%d'%(v, y)]])
        # construct B and C
        if i < E / 2: # forward edges
            hyper_edge_B_pos.append([u.item(), i])
            edge_weight = 1. if not normalize else 1 / node_degree[u.item()]
            hyper_edge_weight_B_pos.append(edge_weight)
            #B_pos2[u, i] += 1 if not normalize else 1 / node_degree[u.item()]
        else: # backward edges
            hyper_edge_B_neg.append([u.item(), i])
            edge_weight = 1. if not normalize else 1 / node_degree[u.item()]
            hyper_edge_weight_B_neg.append(edge_weight)
            #B_neg2[u, i] += 1 if not normalize else 1 / node_degree[u.item()]
        #C2[i, v] += 1
        hyper_edge_C.append([i, v.item()])

    hyper_edge_pos = torch.tensor(hyper_edge_pos).T
    hyper_edge_neg = torch.tensor(hyper_edge_neg).T
    hyper_edge_B_pos = torch.tensor(hyper_edge_B_pos).T
    hyper_edge_B_neg = torch.tensor(hyper_edge_B_neg).T
    hyper_edge_weight_B_pos = torch.tensor(hyper_edge_weight_B_pos)
    hyper_edge_weight_B_neg = torch.tensor(hyper_edge_weight_B_neg)
    hyper_edge_C = torch.tensor(hyper_edge_C).T
    B_pos = torch.sparse_coo_tensor(hyper_edge_B_pos, hyper_edge_weight_B_pos, (n, E))
    B_neg = torch.sparse_coo_tensor(hyper_edge_B_neg, hyper_edge_weight_B_neg, (n, E))
    C = torch.sparse_coo_tensor(hyper_edge_C, torch.ones(hyper_edge_C.size(-1)), (E, n))
    return hyper_edge_pos, hyper_edge_neg, B_pos, B_neg, C, node_degree



def add_self_loop_to_low_degree_nodes(edge_index, num_nodes, threshold=1):
    num_neighbors = torch.tensor([len(set(edge_index[1, torch.where(edge_index[0, :]==i)[0]].tolist() \
                          +edge_index[0, torch.where(edge_index[1, :]==i)[0]].tolist())) for i in range(num_nodes)])
    self_loop_index = torch.where(num_neighbors < threshold)[0]
    self_loop_index = torch.cat([self_loop_index[None], self_loop_index[None]], dim=0)
    return torch.cat([edge_index, self_loop_index], dim=-1)
