import torch
#from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import (
    to_dense_adj
)
from utils.edge_markov_process import construct_edge_adjacency_matrix, add_self_loop_to_low_degree_nodes
from utils.message_passing import message_passing, bidirectional_message_passing
from torch_geometric.utils import degree as pyg_degree
import time



def get_magnetic_walks(edge_index, num_nodes, q, walk_length, source_node=None, normalize=False, history=False):
    # calculate magnetic walks induced on magnetic graph A_q
    # edge_index: [2, #edges]; q: an array (1-d tesnor) of q values
    #A = to_dense_adj(edge_index)[0] # [N, N]
    # use sparse representation
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(-1)), (num_nodes, num_nodes))
    At = A.T
    Q = q.size(0)
    q = q.unsqueeze(-1).unsqueeze(-1)
    # entries at source nodes
    if source_node is not None:
        A = torch.cat([A[i][None] for i in source_node], dim=0)
        At = torch.cat([At[i][None] for i in source_node], dim=0)

    Aq = A.to_dense() * torch.exp(1j * 2 * torch.pi * q) + At.to_dense() * torch.exp(-1j * 2 * torch.pi * q)
    #Aq = torch.cat(
        #[(A * torch.exp(1j * 2 * torch.pi * q[i]) + At * torch.exp(-1j * 2 * torch.pi * q[i]))[None] for i in range(Q)],
        #dim=0)
    del A, At

    # normalize if needed
    degree = None
    if normalize:
        degree = pyg_degree(edge_index[0], num_nodes=num_nodes) + pyg_degree(edge_index[1], num_nodes=num_nodes)
        #D = torch.diag((degree + 1e-6) ** (-1))
        # A = A @ D
        # At = At @ D
        #Aq = (D * (1. + 0 * 1j)) @ Aq
        inv_degree = 1./(degree[None, :, None]+1e-6) if source_node is None else 1./(degree[None, source_node, None]+1e-6)
        Aq = inv_degree * Aq
        del inv_degree

    Aq = Aq.to_dense() # switch to dense repr.
    # compute magnetic walks recursively
    curr_walks = Aq
    history_walks = [curr_walks[:, None]] if history else None

    for r in range(1, walk_length):
        # compute magnetic walks at walk length r+1
        print('computing magnetic walk at length %d' % (r + 1), flush=True)
        time1 = time.time()

        m_st, m_ts = bidirectional_message_passing(curr_walks, edge_index, num_nodes, degree, q)
        #curr_walks = m_st * torch.exp(1j * 2 * torch.pi * q) + m_ts * torch.exp(-1j * 2 * torch.pi * q)
        curr_walks = m_st + m_ts

        if history:
            history_walks.append(curr_walks[:, None])
        time2 = time.time()
        print('computed magnetic walk at length %d with time %.3f' % (r + 1, time2-time1), flush=True)
    #if history:
        #return torch.cat(history_walks, dim=1)
    #else:
        #return curr_walks
    return curr_walks if not history else torch.cat(history_walks, dim=1)




def get_magnetic_walks_edge(edge_index, num_nodes, q, walk_length, source_node=None, normalize=False, nbt=False, history=False):
    # add self-loop to nodes with only one neighbor
    if normalize and nbt:
        edge_index = add_self_loop_to_low_degree_nodes(edge_index, num_nodes, 2)

    hyper_edge_index_pos, hyper_edge_index_neg, B_pos, B_neg, C, node_degree = \
        construct_edge_adjacency_matrix(edge_index, num_nodes, nbt=nbt, normalize=normalize)
    q = q.unsqueeze(-1).unsqueeze(-1)  # [Q, 1, 1]
    Q = q.size(0)
    E = B_pos.size(-1)
    #n = num_nodes

    # source nodes have bugs: mind if we also do this to the degree normalization
    if source_node is not None:
        B_pos = torch.cat([B_pos[i][None] for i in source_node], dim=0)
        B_neg = torch.cat([B_neg[i][None] for i in source_node], dim=0)

    mag_walks = B_pos.to_dense() * torch.exp(1j * 2 * torch.pi * q) + B_neg.to_dense() * torch.exp(-1j * 2 * torch.pi * q)
    del B_pos, B_neg
    #mag_walks = torch.cat(
        #[(B_pos * torch.exp(1j * 2 * torch.pi * q[i]) + B_neg * torch.exp(-1j * 2 * torch.pi * q[i]))[None] for i in range(Q)],
        #dim=0)
    #mag_walks = mag_walks.to_dense() # switch to dense repr., size: [Q, N_sample, E]


    if normalize:
        all_edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        #node_degree = {i: len(edge_index[1, torch.where(edge_index[0, :]==i)[0]]) + \
                       #len(edge_index[0, torch.where(edge_index[1, :]==i)[0]]) for i in range(n)}
        edge_degree = torch.tensor([node_degree[all_edge_index[1, i].item()] for i in range(E)])
        if nbt:
            # subtract number of edges that can backtrack
            _, inverse, counts = torch.unique(all_edge_index, dim=-1, return_inverse=True, return_counts=True)
            edge_degree = edge_degree - counts[inverse]
    else:
        edge_degree = None


    history_walks = [mag_walks[:, None]] if history else None

    for r in range(1, walk_length):
        # compute magnetic walks at walk length r+1
        mag_walks = message_passing(mag_walks, hyper_edge_index_pos, E, degree=edge_degree) * torch.exp(1j * 2 * torch.pi * q) + \
                    message_passing(mag_walks, hyper_edge_index_neg, E, degree=edge_degree) * torch.exp(-1j * 2 * torch.pi * q)
        if history:
            history_walks.append(mag_walks[:, None])

    #return curr_walks if not history else torch.cat(history_walks, dim=1)
    return mag_walks @ (C * (1+0*1j)) if not history else torch.cat([h @ (C * (1+0*1j)) for h in history_walks], dim=1)



def compute_magnetic_walks(edge_index, num_nodes, q, walk_length, source_node=None, nbt=False, normalize=False):
    # ultimate interface for computing magnetic walks
    #if nbt and normalize:
    if nbt:
        return get_magnetic_walks_edge(edge_index, num_nodes, q, walk_length, source_node=source_node, normalize=normalize, nbt=nbt)
    else:
        return get_magnetic_walks(edge_index, num_nodes, q, walk_length, source_node=source_node, normalize=normalize)


