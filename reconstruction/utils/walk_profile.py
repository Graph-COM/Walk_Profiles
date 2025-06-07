import torch
import numpy as np
from torch_geometric.utils import remove_self_loops
#from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import (
    to_dense_adj
)
from utils.edge_markov_process import construct_edge_adjacency_matrix, add_self_loop_to_low_degree_nodes
from utils.message_passing import message_passing
from torch_geometric.utils import degree as pyg_degree
#import sparse
import scipy.sparse as ss
#from utils.misc import timer
#from numba import jit
#from joblib import Parallel, delayed
from utils.sparsity import relative_sparsity
import matplotlib.pyplot as plt
import os.path as osp
import time

def message_passing_walk_profile_calculator(edge_index, M, source_node=None, normalize=False, sparse=False):
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    At = A.T
    if normalize:
        A = A.float()
        At = At.float()
        degree = torch.sum(A, dim=1) + torch.sum(A, dim=0)
        D = torch.diag((degree + 1e-6) ** (-1))
        A = A @ D
        At = At @ D
    else:
        degree = None

    if sparse:
        A = A.to_sparse()
        At = At.to_sparse()

    N = A.size(0)
    walk_profile = []
    #walk_profile_float = []
    edge_index_reverse = edge_index[[1, 0]]
    if source_node == None:  # calculate walk profile for all node pairs
        walk_profile.append(torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0))
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            if sparse:
                wp = walk_profile[m-1].to_dense()
                wp_m = message_passing(wp[:m], edge_index, N, degree).to_sparse() \
                       + message_passing(wp[1:m+1], edge_index_reverse, N, degree).to_sparse()
                Am = message_passing(wp[m:m+1], edge_index, N, degree).to_sparse()
                Atm = message_passing(wp[0:1], edge_index_reverse, N, degree).to_sparse()
                del wp
            else:
                wp_m = message_passing(walk_profile[m-1][:m], edge_index, N, degree) + message_passing(walk_profile[m-1][1:m+1],
                                                                                        edge_index_reverse, N, degree)
                Am = message_passing(walk_profile[m-1][m:m+1], edge_index, N, degree)
                Atm = message_passing(walk_profile[m-1][0:1], edge_index_reverse, N, degree)
            walk_profile.append(torch.cat([Atm, wp_m, Am], dim=0))
            #walk_profile_float.append(walk_profile[-1].float())
    return walk_profile



def message_passing_walk_profile_calculation_and_analysis(edge_index, M, source_node=None, normalize=False, sparse=False,
                                                      analysor=None, all_zero_filter=False):
    # this version only saves the latest step walk profile, and does analysis on the fly
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    At = A.T

    if all_zero_filter:
        filter_matrix = (A + At).float()
        filter_matrix = filter_matrix[source_node] if source_node is not None else filter_matrix
        filter_matrix = filter_matrix.to_sparse()
        A_sym = (A + At).float().to_sparse()

    if normalize:
        A = A.float()
        At = At.float()
        degree = torch.sum(A, dim=1) + torch.sum(A, dim=0)
        D = torch.diag((degree + 1e-6) ** (-1))
        #A = A @ D
        #At = At @ D
        A = D @ A
        At = D @ At
    else:
        degree = None

    if sparse:
        A = A.to_sparse()
        At = A.to_sparse()

    N = A.size(0)
    #walk_profile_float = []
    edge_index_reverse = edge_index[[1, 0]]

    # initialize walk profile at length = 1
    if source_node is None:  # calculate walk profile for all node pairs
        walk_profile = torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0)
    else:
        # a specific number of source nodes
        walk_profile = torch.cat([At[source_node].unsqueeze(0), A[source_node].unsqueeze(0)], dim=0)
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()


    for m in range(1, M):
        print('computing walk profile at length %d' % (m + 1))
        time1 = time.time()
        # computing walk profile at length = m+1
        if sparse:
            wp = walk_profile.to_dense()
            wp_m = message_passing(wp[:m], edge_index, N, degree).to_sparse() \
                   + message_passing(wp[1:m+1], edge_index_reverse, N, degree).to_sparse()
            Am = message_passing(wp[m:m+1], edge_index, N, degree).to_sparse()
            Atm = message_passing(wp[0:1], edge_index_reverse, N, degree).to_sparse()
            del wp
        else:
            wp_m = message_passing(walk_profile[:m], edge_index, N, degree) + message_passing(walk_profile[1:m+1],
                                                                                                   edge_index_reverse, N, degree)
            Am = message_passing(walk_profile[m:m+1], edge_index, N, degree)
            Atm = message_passing(walk_profile[0:1], edge_index_reverse, N, degree)
        walk_profile = torch.cat([Atm, wp_m, Am], dim=0)
        time2 = time.time()
        # if need for filtering node pairs
        if all_zero_filter:
            filter_matrix = torch.sparse.mm(filter_matrix, A_sym)
            analysor.compute_sparsity_histogram(walk_profile.float(), node_pair_to_eval=filter_matrix.to_dense()>0)
        else:
            analysor.compute_sparsity_histogram(walk_profile.float())
        time3 = time.time()
        print('computed walk profile at length %d, using time %.3f' % (m + 1, time2-time1))
        print('Analysed walk profile at length %d, using time %.3f' % (m + 1, time3-time2))
            #walk_profile_float.append(walk_profile[-1].float())

                # analysis
#                if analysis is not None:
#                    sparsity_m = relative_sparsity(walk_profile.flatten(1), energy_error).cpu().numpy()
#                    energy_m = (walk_profile.flatten(1) ** 2 / (torch.linalg.norm(walk_profile.flatten(1), dim=0).unsqueeze(0) + 1e-9) ** 2).cpu().numpy()
#                    # worst case
#                    sparsity_worst = sparsity_m.max()
#                    # average case
#                    sparsity_ave = sparsity_m.mean()
#                    sparsity_std = sparsity_m.std()
#                    # percentile
#                    percentile = np.array([50.0, 68.0, 95.0, 99.0])
#                    sparsity_per = np.percentile(sparsity_m, percentile)
#                    print('-----Walk length %d-----' % (m+1), file=file)
#                    print('Worst sparsity: %.3f' % sparsity_worst, file=file)
#                    print('Average sparsity: %.3f+-%.3f' % (sparsity_ave, sparsity_std), file=file)
#                    for p, s in zip(percentile, sparsity_per):
#                        print('%.2f percentile sparsity: %.3f' % (p, s), file=file)
#                    print('Energy distribution: ', energy_m.mean(-1), file=file)
#
#                    # save figures
#                    plt.hist(sparsity_m, density=True, range=(0., 1.), bins=m + 2)
#                    plt.title('Walk profile sparsity distribution over node pairs')
#                    plt.xlabel('Sparsity')
#                    plt.ylabel('PDF')
#                    save_path = osp.join(output_path, 'sparsity_m%d.png' % m)
#                    plt.savefig(save_path)
#                    plt.clf()
#    file.close()




def message_passing_nbt_walk_profile_calculation_and_analysis(edge_index, M, source_node=None, normalize=False, sparse=False,
                                                          analysor=None, all_zero_filter=False):
    # this version only saves the latest step walk profile, and does analysis on the fly
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    N = A.size(0)
    n = len(source_node) if source_node is not None else N
    At = A.T
    S = A * A.T
    flag_multi_edge = S.sum() > 0 # if graph is multi-edge, the computation is more complicated
    P = A - S
    D_u = torch.diag(torch.diag(S @ S.T))
    # D = np.diag(np.diag(A @ A.T+A.T@A)) - D_u
    D_out = torch.diag(torch.diag(P @ P.T))
    D_in = torch.diag(torch.diag(P.T @ P))
    D = 2 * D_u + D_in + D_out
    d_u = torch.diag(D_u)
    d = torch.diag(D)

    if all_zero_filter:
        # old version for bt walk profile, not working
        filter_matrix = (A + At).float()
        filter_matrix = filter_matrix[source_node] if source_node is not None else filter_matrix
        filter_matrix = filter_matrix.to_sparse()
        A_sym = (A + At).float().to_sparse()

    if normalize:
        # old version for bt walk profile, not working
        A = A.float()
        At = At.float()
        degree = torch.sum(A, dim=1) + torch.sum(A, dim=0)
        D = torch.diag((degree + 1e-6) ** (-1))
        #A = A @ D
        #At = At @ D
        A = D @ A
        At = D @ At
    else:
        degree = None

    if sparse:
        A = A.to_sparse()
        At = A.to_sparse()

    #walk_profile_float = []
    # construct edge index to be used
    edge_index_reverse = edge_index[[1, 0]]
    if flag_multi_edge:
        edge_index_u = torch.cat([torch.where(S != 0)[0][None], torch.where(S != 0)[1][None]], dim=0)

    # initialize walk profile at length = 1, 2, with zero paddings to both sides for computational convenience
    #walk_profile = torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0)
    walk_profile_list = [torch.cat([A.T[None], A[None]], dim=0),
                         torch.cat([(A.T @ A.T - D_u)[None], (A @ A.T + A.T @ A - D)[None], (A @ A - D_u)[None]], dim=0)]
    # zero paddings to both sides
    walk_profile_list = [torch.cat([torch.zeros([2, N, N]), walk_profile_list[0], torch.zeros([2, N, N])], dim=0),
                         torch.cat([torch.zeros([1, N, N]), walk_profile_list[1], torch.zeros([1, N, N])], dim=0)]
    if flag_multi_edge:
        walk_profile_u_list = [torch.cat([S.T[None], S[None]], dim=0),
                           torch.cat([(A.T @ S - D_u)[None], (A @ S + A.T @ S - 2 * D_u)[None], (A @ S - D_u)[None]], dim=0)]
        walk_profile_u_list = [torch.cat([torch.zeros([2, N, N]), walk_profile_u_list[0], torch.zeros([2, N, N])], dim=0),
                             torch.cat([torch.zeros([1, N, N]), walk_profile_u_list[1], torch.zeros([1, N, N])], dim=0)]
        #walk_profile_u_list = [torch.cat([torch.zeros([2, n, n]), wp, torch.zeros([2, n, n])]) for wp in walk_profile_u_list]
    if source_node is not None:
        # a specific number of source nodes
        walk_profile_list = [wp[:, source_node] for wp in walk_profile_list]
        if flag_multi_edge:
            walk_profile_u_list = [wp[:, source_node] for wp in walk_profile_u_list]
        #walk_profile = torch.cat([At[source_node].unsqueeze(0), A[source_node].unsqueeze(0)], dim=0)
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()


    for m in range(2, M):
        print('computing walk profile at length %d' % (m + 1))
        time1 = time.time()
        # computing walk profile at length = m+1
        # recursive formula
        wp_m_1, wp_m_2 = walk_profile_list[-1], walk_profile_list[-2]
        # grab bt walk profile
        wp_m = message_passing(wp_m_1[0:-1], edge_index, N, degree) + message_passing(wp_m_1[1:],
                                                                                   edge_index_reverse, N, degree)
        # subtract bt walks
        wp_m = wp_m - wp_m_2[1:-1] * (d - 1) - (wp_m_2[0:-2] + wp_m_2[2:]) * d_u

        if flag_multi_edge: # need to further subtract multi-edge bt walks
            wp_u_m_1, wp_u_m_2 = walk_profile_u_list[-1], walk_profile_u_list[-2]
            wp_u_m = message_passing(wp_m_1[0:-1] + wp_m_1[1:], edge_index_u, N, degree)
            wp_u_m = wp_u_m - (2 * wp_m_2[1:-1] + wp_m_2[0:-2] + wp_m_2[2:]) * d_u + 2 * wp_u_m_2[1:-1] + wp_u_m_2[0:-2] + wp_u_m_2[2:]
            wp_m = wp_m + wp_u_m_2[1:-1] + wp_u_m_2[0:-2] + wp_u_m_2[2:]

        time2 = time.time()
        # if need for filtering node pairs
        if all_zero_filter:
            filter_matrix = torch.sparse.mm(filter_matrix, A_sym)
            analysor.compute_sparsity_histogram(wp_m.float(), node_pair_to_eval=filter_matrix.to_dense()>0)
        else:
            #analysor.compute_sparsity_histogram(wp_m.float())
            analysor.verify_expectation(wp_m.float())
        time3 = time.time()

        # update for next iteration
        walk_profile_list[0] = torch.cat([torch.zeros([1, n, N]), walk_profile_list[1], torch.zeros([1, n, N])], dim=0)
        walk_profile_list[1] = torch.cat([torch.zeros([1, n, N]), wp_m, torch.zeros([1, n, N])], dim=0)
        if flag_multi_edge:
            walk_profile_u_list[0] = torch.cat([torch.zeros([1, n, N]), walk_profile_u_list[1], torch.zeros([1, n, N])],
                                 dim=0)
            walk_profile_u_list[1] = torch.cat([torch.zeros([1, n, N]), wp_u_m, torch.zeros([1, n, N])], dim=0)

        print('computed walk profile at length %d, using time %.3f' % (m + 1, time2-time1))
        print('Analysed walk profile at length %d, using time %.3f' % (m + 1, time3-time2))


def get_nbt_walk_profile(edge_index, num_nodes, M, source_node=None):
    #A = to_dense_adj(edge_index)[0].type(torch.int64)
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(-1)), (num_nodes, num_nodes))
    N = num_nodes # TO DO: the size of A and num_nodes could mismatch
    n = min(len(source_node), N) if source_node is not None else N
    #At = A.T
    S = A * A.T
    flag_multi_edge = S.sum() > 0 # if graph is multi-edge, the computation is more complicated
    P = A - S
    D_u = torch.diag(torch.diag(S @ S.T))
    # D = np.diag(np.diag(A @ A.T+A.T@A)) - D_u
    D_out = torch.diag(torch.diag(P @ P.T))
    D_in = torch.diag(torch.diag(P.T @ P))
    D = 2 * D_u + D_in + D_out
    d_u = torch.diag(D_u)
    d = torch.diag(D)

    # construct edge index to be used
    edge_index_reverse = edge_index[[1, 0]]
    if flag_multi_edge:
        edge_index_u = torch.cat([torch.where(S != 0)[0][None], torch.where(S != 0)[1][None]], dim=0)

    # initialize walk profile at length = 1, 2, with zero paddings to both sides for computational convenience
    #walk_profile = torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0)
    walk_profile_list = [torch.cat([A.T[None], A[None]], dim=0),
                         torch.cat([(A.T @ A.T - D_u)[None], (A @ A.T + A.T @ A - D)[None], (A @ A - D_u)[None]], dim=0)]
    # zero paddings to both sides
    walk_profile_list = [torch.cat([torch.zeros([2, N, N]), walk_profile_list[0], torch.zeros([2, N, N])], dim=0),
                         torch.cat([torch.zeros([1, N, N]), walk_profile_list[1], torch.zeros([1, N, N])], dim=0)]
    if flag_multi_edge:
        walk_profile_u_list = [torch.cat([S.T[None], S[None]], dim=0),
                               torch.cat([(A.T @ S - D_u)[None], (A @ S + A.T @ S - 2 * D_u)[None], (A @ S - D_u)[None]], dim=0)]
        walk_profile_u_list = [torch.cat([torch.zeros([2, N, N]), walk_profile_u_list[0], torch.zeros([2, N, N])], dim=0),
                               torch.cat([torch.zeros([1, N, N]), walk_profile_u_list[1], torch.zeros([1, N, N])], dim=0)]
        #walk_profile_u_list = [torch.cat([torch.zeros([2, n, n]), wp, torch.zeros([2, n, n])]) for wp in walk_profile_u_list]
    if source_node is not None:
        # a specific number of source nodes
        walk_profile_list = [wp[:, source_node] for wp in walk_profile_list]
        if flag_multi_edge:
            walk_profile_u_list = [wp[:, source_node] for wp in walk_profile_u_list]
        #walk_profile = torch.cat([At[source_node].unsqueeze(0), A[source_node].unsqueeze(0)], dim=0)
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()


    for m in range(2, M):
        #print('computing walk profile at length %d' % (m + 1))
        # computing walk profile at length = m+1
        # recursive formula
        wp_m_1, wp_m_2 = walk_profile_list[-1], walk_profile_list[-2]
        # grab bt walk profile
        wp_m = message_passing(wp_m_1[0:-1], edge_index, N) + message_passing(wp_m_1[1:],
                                                                                      edge_index_reverse, N)
        # subtract bt walks
        wp_m = wp_m - wp_m_2[1:-1] * (d - 1) - (wp_m_2[0:-2] + wp_m_2[2:]) * d_u

        if flag_multi_edge: # need to further subtract multi-edge bt walks
            wp_u_m_1, wp_u_m_2 = walk_profile_u_list[-1], walk_profile_u_list[-2]
            wp_u_m = message_passing(wp_m_1[0:-1] + wp_m_1[1:], edge_index_u, N)
            wp_u_m = wp_u_m - (2 * wp_m_2[1:-1] + wp_m_2[0:-2] + wp_m_2[2:]) * d_u + 2 * wp_u_m_2[1:-1] + wp_u_m_2[0:-2] + wp_u_m_2[2:]
            wp_m = wp_m + wp_u_m_2[1:-1] + wp_u_m_2[0:-2] + wp_u_m_2[2:]

        # update for next iteration
        walk_profile_list[0] = torch.cat([torch.zeros([1, n, N]), walk_profile_list[1], torch.zeros([1, n, N])], dim=0)
        walk_profile_list[1] = torch.cat([torch.zeros([1, n, N]), wp_m, torch.zeros([1, n, N])], dim=0)
        if flag_multi_edge:
            walk_profile_u_list[0] = torch.cat([torch.zeros([1, n, N]), walk_profile_u_list[1], torch.zeros([1, n, N])],
                                               dim=0)
            walk_profile_u_list[1] = torch.cat([torch.zeros([1, n, N]), wp_u_m, torch.zeros([1, n, N])], dim=0)

    return walk_profile_list[-1][1:-1]

def get_walk_profile(edge_index, num_nodes, M, source_node=None, normalize=False, history=False):
    # this version only saves the latest step walk profile, and does analysis on the fly

    # use sparse representation
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(-1)), (num_nodes, num_nodes))
    At = A.T
    #A = to_dense_adj(edge_index)[0].type(torch.int64)
    #At = A.T

    if source_node is not None:
        A = torch.cat([A[i][None] for i in source_node], dim=0)
        At = torch.cat([At[i][None] for i in source_node], dim=0)

    if normalize:
        degree = pyg_degree(edge_index[0], num_nodes=num_nodes) + pyg_degree(edge_index[1], num_nodes=num_nodes)
        inv_degree = 1. / (degree[:, None] + 1e-6) if source_node is None else 1. / (
        degree[source_node, None]+1e-6)
        A = inv_degree * A
        At = inv_degree * At
    else:
        degree = None


    edge_index_reverse = edge_index[[1, 0]]

    # initialize walk profile at length = 1
    #if source_node is not None and len(source_node) < A.size(-1):  # calculate walk profile for all node pairs
        #walk_profile = torch.cat([At[source_node].unsqueeze(0), A[source_node].unsqueeze(0)], dim=0)
    #else:
        # a specific number of source nodes
    walk_profile = torch.cat([At.to_dense()[None], A.to_dense()[None]], dim=0)
    history_walks = [walk_profile] if history else None
    del A, At
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()


    for m in range(1, M):
        # TO DO: compare boundary computation and zero padding unified computation
        print('computing walk profile at length %d' % (m + 1),flush=True)
        time1 = time.time()
        wp_m = message_passing(walk_profile[:m], edge_index, num_nodes, degree) + message_passing(walk_profile[1:m+1],
                                                                                          edge_index_reverse, num_nodes, degree)
        Am = message_passing(walk_profile[m:m+1], edge_index, num_nodes, degree)
        Atm = message_passing(walk_profile[0:1], edge_index_reverse, num_nodes, degree)
        walk_profile = torch.cat([Atm, wp_m, Am], dim=0)
        time2 = time.time()
        print('computed walk profile at length %d with time %.3f' % (m + 1, time2 - time1),flush=True)
        if history:
            history_walks.append(walk_profile)

    return walk_profile if not history else torch.cat(history_walks, dim=0)



def get_walk_profile_edge(edge_index, num_nodes, M, source_node=None, nbt=False, normalize=False, history=False):
    # add self loops to nodes with only one neighbor
    if normalize and nbt:
        edge_index = add_self_loop_to_low_degree_nodes(edge_index, num_nodes, 2)

    hyper_edge_index_pos, hyper_edge_index_neg, B_pos, B_neg, C, node_degree = \
        construct_edge_adjacency_matrix(edge_index, num_nodes, nbt=nbt, normalize=normalize)
    E = B_pos.size(-1)
    n = num_nodes

    if source_node is not None:
        B_pos = torch.cat([B_pos[i][None] for i in source_node], dim=0)
        B_neg = torch.cat([B_neg[i][None] for i in source_node], dim=0)


    # walk profile init
    walk_profile = torch.cat([B_neg.to_dense()[None], B_pos.to_dense()[None]], dim=0)
    history_walks = [walk_profile] if history else None
    del B_pos, B_neg
    if normalize:
        all_edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)
        edge_degree = torch.tensor([node_degree[all_edge_index[1, i].item()] for i in range(E)])
        if nbt:
            # subtract number of edges that can backtrack
            _, inverse, counts = torch.unique(all_edge_index, dim=-1, return_inverse=True, return_counts=True)
            edge_degree = edge_degree - counts[inverse]
    else:
        edge_degree = None

    for m in range(1, M):
        print('computing walk profile at length %d' % (m + 1))
        wp_m = message_passing(walk_profile[:m], hyper_edge_index_pos, E, degree=edge_degree) + message_passing(walk_profile[1:m + 1],
                                                                                          hyper_edge_index_neg, E, degree=edge_degree)
        Am = message_passing(walk_profile[m:m + 1], hyper_edge_index_pos, E, degree=edge_degree)
        Atm = message_passing(walk_profile[0:1], hyper_edge_index_neg, E, degree=edge_degree)
        walk_profile = torch.cat([Atm, wp_m, Am], dim=0)
        if history:
            history_walks.append(walk_profile)

    # contraction to node
    return walk_profile @ C if not history else torch.cat([h @ C for h in history_walks], dim=0)



def compute_walk_profile(edge_index, num_nodes, M, source_node=None, nbt=False, normalize=False):
    # ultimate interface for computing walk profile
    if not nbt:
        return get_walk_profile(edge_index, num_nodes, M, source_node=source_node, normalize=normalize)
    #elif nbt and not normalize:
        #return get_nbt_walk_profile(edge_index, num_nodes, M, source_node=source_node)
    else:
        return get_walk_profile_edge(edge_index, num_nodes, M, source_node=source_node, nbt=nbt, normalize=normalize)