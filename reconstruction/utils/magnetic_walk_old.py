import torch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import (
    to_dense_adj
)
from utils.edge_markov_process import construct_edge_adjacency_matrix
from utils.message_passing import message_passing, bidirectional_message_passing
from torch_geometric.utils import degree as pyg_degree
import time


def get_magnetic_walks(edge_index, q, walk_length, source_node=None, normalize=False, nbt=False): # TO DO: source nodes
    # calculate magnetic walks induced on magnetic graph A_q
    # edge_index: [2, #edges]; q: an array (1-d tesnor) of q values
    num_nodes = edge_index.max() + 1
    #A = to_dense_adj(edge_index)[0] # [N, N]
    # use sparse representation
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(-1)), (num_nodes, num_nodes))
    At = A.T
    Q = q.size(0)
    #Aq = A.unsqueeze(0) * torch.exp(1j * 2 * torch.pi * q) + A.T.unsqueeze(0) * torch.exp(-1j * 2 * torch.pi * q)
    # entries at source nodes
    if source_node is not None and len(source_node) < num_nodes:
        A = torch.cat([A[i][None] for i in source_node], dim=0)
        At = torch.cat([At[i][None] for i in source_node], dim=0)
    Aq = torch.cat(
        [(A * torch.exp(1j * 2 * torch.pi * q[i]) + At * torch.exp(-1j * 2 * torch.pi * q[i]))[None] for i in range(Q)],
        dim=0)
    del A, At
    # normalize if needed
    degree = None
    if normalize and not nbt:
        degree = pyg_degree(edge_index[0], num_nodes=num_nodes) + pyg_degree(edge_index[1], num_nodes=num_nodes)
        #D = torch.diag((degree + 1e-6) ** (-1))
        # A = A @ D
        # At = At @ D
        #Aq = (D * (1. + 0 * 1j)) @ Aq
        inv_degree = 1./(degree[None, :, None]+1e-6) if source_node is None else 1./(degree[None, source_node, None])
        Aq = inv_degree * Aq
        del inv_degree

    Aq = Aq.to_dense() # switch to dense repr.
    # compute magnetic walks recursively
    recursive_calculator = magnetic_walks_recursive_calculator(edge_index, q, num_nodes, normalize, nbt, degree)
    if nbt:
        prev_walks = Aq  # walk len = 1
        curr_walks = torch.bmm(Aq, Aq)  # walk len = 2
        curr_walks = curr_walks - torch.diag_embed(torch.diagonal(curr_walks, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
    else:
        prev_walks = None  # walk len = 1
        curr_walks = Aq
    start_index = 2 if nbt else 1
    for r in range(start_index, walk_length):
        # compute magnetic walks at walk length r+1
        print('computing magnetic walk at length %d' % (r + 1))
        time1 = time.time()
        curr_walks, prev_walks = recursive_calculator.magnetic_walks_recrusive_update(curr_walks, prev_walks)
        time2 = time.time()
        print('computed magnetic walk at length %d with time %.3f' % (r + 1, time2-time1))

    return curr_walks

class magnetic_walks_recursive_calculator:
    def __init__(self, edge_index, q, num_nodes, normalize=False, nbt=False, degree=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.nbt = nbt
        self.degree = degree # used for normalization
        self.q = q.unsqueeze(-1).unsqueeze(-1)  # [Q, 1, 1]
        Q = q.size(0)
        if nbt:
            raise Exception('not working for nbt now')
            #A = to_dense_adj(edge_index)[0]  # [N, N]
            #Aq = A.unsqueeze(0).tile([Q, 1, 1]) * torch.exp(1j * 2 * torch.pi * q) + A.T.unsqueeze(0).tile(
                #[Q, 1, 1]) * torch.exp(-1j * 2 * torch.pi * q)
            #self.Aq_square_diag = torch.diagonal(torch.bmm(Aq, Aq), dim1=-2, dim2=-1).unsqueeze(1)

    def magnetic_walks_recrusive_update(self, curr_walks, prev_walks):
        # next_walks = curr_walks @ Aq
        m_st, m_ts = bidirectional_message_passing(curr_walks, self.edge_index, self.num_nodes, self.degree)
        next_walks = m_st * torch.exp(1j * 2 * torch.pi * self.q) + m_ts * torch.exp(-1j * 2 * torch.pi * self.q)
        if not self.nbt:
            return next_walks, None
        else:
            return next_walks - prev_walks * (self.Aq_square_diag - 1), curr_walks



def get_magnetic_walks_edge(edge_index, q, walk_length, source_node=None, normalize=False, nbt=False):
    # TO DO: source nodes and normalize
    hyper_edge_index_pos, hyper_edge_index_neg, B_pos, B_neg, C = construct_edge_adjacency_matrix(edge_index, nbt=nbt, normalize=normalize)
    q = q.unsqueeze(-1).unsqueeze(-1)  # [Q, 1, 1]
    Q = q.size(0)
    E = B_pos.size(-1)
    n = B_pos.size(0)
    #A_q = A_pos[None].tile([Q, 1, 1]) * torch.exp(1j * 2 * torch.pi * q) + \
          #A_neg[None].tile([Q, 1, 1]) * torch.exp(-1j * 2 * torch.pi * q)
    B_q = B_pos[None].tile([Q, 1, 1]) * torch.exp(1j * 2 * torch.pi * q) + \
          B_neg[None].tile([Q, 1, 1]) * torch.exp(-1j * 2 * torch.pi * q)
    mag_walks = B_q if source_node is None else B_q[:, source_node, :]

    if normalize:
        all_edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        node_degree = {i: len(edge_index[1, torch.where(edge_index[0, :]==i)[0]]) + \
                       len(edge_index[0, torch.where(edge_index[1, :]==i)[0]]) for i in range(n)}
        edge_degree = torch.tensor([node_degree[all_edge_index[1, i].item()] for i in range(E)])
        if nbt:
            edge_degree = edge_degree - 1
    else:
        edge_degree = None

    for r in range(1, walk_length):
        # compute magnetic walks at walk length r+1
        mag_walks = message_passing(mag_walks, hyper_edge_index_pos, E, degree=edge_degree) * torch.exp(1j * 2 * torch.pi * q) + \
                    message_passing(mag_walks, hyper_edge_index_neg, E, degree=edge_degree) * torch.exp(-1j * 2 * torch.pi * q)
    return mag_walks @ (C * (1+0*1j))



def compute_magnetic_walks(edge_index, q, walk_length, source_node=None, nbt=False, normalize=False):
    # ultimate interface for computing magnetic walks
    if nbt and normalize:
        return get_magnetic_walks_edge(edge_index, q, walk_length, source_node=source_node, normalize=normalize, nbt=nbt)
    else:
        return get_magnetic_walks(edge_index, q, walk_length, source_node=source_node, normalize=normalize, nbt=nbt)
