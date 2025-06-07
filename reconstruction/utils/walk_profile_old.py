import torch
import numpy as np
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import (
    to_dense_adj
)
#import sparse
import scipy.sparse as ss
#from utils.misc import timer
#from numba import jit
#from joblib import Parallel, delayed
from utils.sparsity import relative_sparsity
import matplotlib.pyplot as plt
import os.path as osp
import time

def get_walk_profile(edge_index, M, output='spd'):
    # input: (directed) edge index
    # output: walk profile matrix up to M steps

    # step 1: construct walk sequence (A_q, A^2_q, A^3_q, ..., A^M_q)
    edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0] # [N, N]
    A_q = torch.tile(torch.unsqueeze(A, 0), (M+1, 1, 1)) # [M+1, N, N]
    phase_factor = torch.exp(1j * 2 * torch.pi * torch.tensor([j / 2 / (M+1) for j in range(0, M+1)])).to(edge_index.device)
    phase_factor = phase_factor.unsqueeze(-1).unsqueeze(-1) # [M+1, 1, 1]
    A_q = A_q * phase_factor # [M+1, N, N]
    A_q = A_q + torch.conj(torch.transpose(A_q, 1, 2)) # finished computation of Magnetic Adjacency matrix
    A_q = A_q * phase_factor
    #A_q_m_power = torch.tile(torch.eye(A_q.size(-1), A_q.size(-1)).unsqueeze(0), (M+1, 1, 1)).to(edge_index.device) # [M+1, N, N]
    #A_q_m_power = torch.complex(A_q_m_power, torch.zeros_like(A_q_m_power))
    #A_q_all_power = []
    A_q_all_power = [A_q.unsqueeze(1)]
    for m in range(1, M+1):
        #A_q_m_power = torch.bmm(A_q_m_power, A_q) * (phase_factor) # [M+1, N, N]
        A_q_m_power = torch.bmm(A_q_all_power[-1][:, 0, :, :], A_q)
        A_q_all_power.append(A_q_m_power.unsqueeze(1))

    A_q_all_power = torch.cat(A_q_all_power, dim=1) # [M+1, M, N, N]
    qs = torch.tensor([j / 2 / (M+1) for j in range(0, M+1)]).to(edge_index.device)
    inverse_mat = torch.cat([torch.exp(-1j * 4 * torch.pi * k * qs).unsqueeze(0) for k in range(M+1)], dim=0).to(edge_index.device) # [M+1, M+1]
    inverse_mat = inverse_mat / (M+1)


    walk_profile = torch.einsum('kq, qlnm->klnm', inverse_mat, A_q_all_power).real.round() # [direction_idx, length_idx, N, N]


    # output
    if output=='profile':
        return [walk_profile[:m+2, m, :, :] for m in range(M)]
    elif output=='spd':
        idx = torch.arange(0, M).to(edge_index.device)
        wp = walk_profile[idx+1, idx,:, :]
        spd = torch.argmax((wp > 0.).float(), dim=0) + 1.
        mask = wp.sum(dim=0) == 0.
        spd[mask] = 0.
        return spd
        #spd_matrix = torch.ones_like(A) * torch.inf
        #for m in range(M):
        #    w = walk_profile[m][-1] # number of directed walks
        #    indices = torch.where(w > 0)
        #    spd_at_indices = spd_matrix[indices]
        #    replace_indices = torch.where(spd_at_indices > m+1)
        #    spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m+1
        #return spd_matrix
    elif output == 'lpd':
        lpd_matrix = torch.ones_like(A_q[0].real) * -1
        for m in reversed(range(M)):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            lpd_at_indices = lpd_matrix[indices]
            replace_indices = torch.where(lpd_at_indices < m + 1)
            lpd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        lpd_matrix[torch.where(lpd_matrix == -1)] = torch.inf
        return lpd_matrix
    elif output=='uspd':
        spd_matrix = torch.ones_like(A) * torch.inf
        for m in range(M):
            w = torch.max(walk_profile[m], dim=0)[0]  # number of undirected walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix


def get_walk_profile_v2(edge_index, M, output='spd'):
    # input: (directed) edge index
    # output: walk profile matrix up to M steps

    # step 1: construct walk sequence (A_q, A^2_q, A^3_q, ..., A^M_q)
    edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0]
    A_q = torch.tile(torch.unsqueeze(A, 0), (M+1, 1, 1))
    phase_factor = torch.exp(1j * 2 * torch.pi * torch.tensor([j / 2 / (M+1) for j in range(0, M+1)])).to(edge_index.device)
    A_q = A_q * phase_factor.unsqueeze(-1).unsqueeze(-1) # [M+1, N, N]
    A_q = A_q + torch.conj(torch.transpose(A_q, 1, 2))
    A_q_m_power = torch.tile(torch.eye(A_q.size(-1), A_q.size(-1)).unsqueeze(0), (M+1, 1, 1)).to(edge_index.device) # [M+1, N, N]
    A_q_m_power = torch.complex(A_q_m_power, torch.zeros_like(A_q_m_power))
    A_q_all_power = []
    for m in range(1, M+1):
        A_q_m_power = torch.bmm(A_q_m_power, A_q) * (phase_factor.unsqueeze(-1).unsqueeze(-1)) # [M+1, N, N]
        A_q_all_power.append(A_q_m_power.unsqueeze(1))

    A_q_all_power = torch.cat(A_q_all_power, dim=1) # [M+1, M, N, N]
    qs = torch.tensor([j / 2 / (M+1) for j in range(0, M+1)]).to(edge_index.device)
    inverse_mat = torch.cat([torch.exp(-1j * 4 * torch.pi * k * qs).unsqueeze(0) for k in range(M+1)], dim=0).to(edge_index.device) # [M+1, M+1]
    inverse_mat = inverse_mat / (M+1)


    walk_profile = torch.einsum('kq, qlnm->klnm', inverse_mat, A_q_all_power).real.round() # [direction_idx, length_idx, N, N]


    # output
    if output=='profile':
        return [walk_profile[:m+2, m, :, :] for m in range(M)]
    elif output=='spd':
        idx = torch.arange(0, M).to(edge_index.device)
        wp = walk_profile[idx+1, idx,:, :]
        spd = torch.argmax((wp > 0.).float(), dim=0) + 1.
        mask = wp.sum(dim=0) == 0.
        spd[mask] = 0.
        return spd
        #spd_matrix = torch.ones_like(A) * torch.inf
        #for m in range(M):
        #    w = walk_profile[m][-1] # number of directed walks
        #    indices = torch.where(w > 0)
        #    spd_at_indices = spd_matrix[indices]
        #    replace_indices = torch.where(spd_at_indices > m+1)
        #    spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m+1
        #return spd_matrix
    elif output == 'lpd':
        lpd_matrix = torch.ones_like(A_q[0].real) * -1
        for m in reversed(range(M)):
            w = walk_profile[m][-1]  # number of directed walks
            indices = torch.where(w > 0)
            lpd_at_indices = lpd_matrix[indices]
            replace_indices = torch.where(lpd_at_indices < m + 1)
            lpd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        lpd_matrix[torch.where(lpd_matrix == -1)] = torch.inf
        return lpd_matrix
    elif output=='uspd':
        spd_matrix = torch.ones_like(A) * torch.inf
        for m in range(M):
            w = torch.max(walk_profile[m], dim=0)[0]  # number of undirected walks
            indices = torch.where(w > 0)
            spd_at_indices = spd_matrix[indices]
            replace_indices = torch.where(spd_at_indices > m + 1)
            spd_matrix[indices[0][replace_indices], indices[1][replace_indices]] = m + 1
        return spd_matrix



def naive_walk_profile_calculator(edge_index, M):
    #edge_index = remove_self_loops(edge_index)[0]
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    all_walks = naive_walk_profile_recursive(A, M)
    walk_profile = [torch.zeros([m+2, len(A), len(A)]).type(torch.int64) for m in range(M)]
    for key in all_walks.keys():
        walk_profile[len(key) - 1][key.count('1')] += all_walks[key]
    return walk_profile



def naive_walk_profile_recursive(A, m):
    # compute from naive_walk_profile_recursive(A, m-1)
    if m < 1:
        raise Exception("invalid m")
    if m == 1:
        return {'0': A.T, '1': A}
    else:
        prev = naive_walk_profile_recursive(A, m-1)
        new_res = {}
        for key in prev:
            new_res[key] = prev[key]
            new_res[key+'0'] = prev[key] @ A.T
            new_res[key + '1'] = prev[key] @ A
        return new_res


def improved_walk_profile_calculator(edge_index, M, source_node=None):
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    At = A.T

    #D_in = torch.diag((torch.sum(A, dim=0) + 1e-6)**(-1))
    #D_out = torch.diag((torch.sum(A, dim=1) + 1e-6)**(-1))
    #D = torch.diag((torch.sum(A, dim=1)+torch.sum(A, dim=0) + 1e-6)**(-1))
    #A = A @ D
    #At = At @ D
    #A = D_out @ A
    #At = D_in @ At

    walk_profile = []
    walk_profile_float = []
    if source_node == None: # calculate walk profile for all node pairs
        walk_profile.append(torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0))
        walk_profile_float.append(walk_profile[-1].float())
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            # computing walk profile phi(m+1, *)
            #wp_m = torch.einsum('qik, kj->qij', walk_profile[m - 1][:m], A) + \
                   #torch.einsum('qik, kj->qij', walk_profile[m - 1][1:m+1], At)
            wp_m = walk_profile[m - 1][:m] @ A + walk_profile[m - 1][1:m+1] @ At
            Am = walk_profile[m - 1][m] @ A
            Atm = walk_profile[m - 1][0] @ At
            walk_profile.append(torch.cat([Atm.unsqueeze(0), wp_m, Am.unsqueeze(0)], dim=0))
            walk_profile_float.append(walk_profile[-1].float())
    return walk_profile_float


def normalized_walk_profile_calculator(edge_index, M, source_node=None):
    A = to_dense_adj(edge_index)[0]
    At = A.T

    #D_in = torch.diag((torch.sum(A, dim=0) + 1e-6)**(-1))
    #D_out = torch.diag((torch.sum(A, dim=1) + 1e-6)**(-1))
    D = torch.diag((torch.sum(A, dim=1)+torch.sum(A, dim=0) + 1e-6)**(-1))
    A = A @ D
    At = At @ D
    #A = D_out @ A
    #At = D_in @ At

    walk_profile = []
    if source_node == None: # calculate walk profile for all node pairs
        walk_profile.append(torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0))
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            # computing walk profile phi(m+1, *)
            #wp_m = torch.einsum('qik, kj->qij', walk_profile[m - 1][:m], A) + \
            #torch.einsum('qik, kj->qij', walk_profile[m - 1][1:m+1], At)
            wp_m = walk_profile[m - 1][:m] @ A + walk_profile[m - 1][1:m+1] @ At
            Am = walk_profile[m - 1][m] @ A
            Atm = walk_profile[m - 1][0] @ At
            walk_profile.append(torch.cat([Atm.unsqueeze(0), wp_m, Am.unsqueeze(0)], dim=0))
    return walk_profile


def sparse_walk_profile_calculator_v1(edge_index, M, source_node=None, normalize=False):
    # sparse implementation using scipy.sparse
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    A = np.array(A)
    A = ss.csr_array(A)
    At = A.T

    if normalize:
        A = A.astype(np.float32)
        At = At.astype(np.float32)
        Dinv = np.diag((np.sum(A, axis=1) + np.sum(A, axis=0) + 1e-6) ** (-1))
        Dinv = ss.csr_array(Dinv)
        A = A.dot(Dinv)
        At = At.dot(Dinv)

    walk_profile = []
    if source_node == None:  # calculate walk profile for all node pairs
        #walk_profile.append(np.concatenate((At[None], A[None]), axis=0))
        walk_profile.append([At, A])
        clock = timer()
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            # computing walk profile phi(m+1, *)
            # wp_m = torch.einsum('qik, kj->qij', walk_profile[m - 1][:m], A) + \
            # torch.einsum('qik, kj->qij', walk_profile[m - 1][1:m+1], At)
            wp_m = [walk_profile[m-1][0].dot(At)]
            clock.start()
            for k in range(1, m+1):
                wp_m.append(walk_profile[m-1][k-1].dot(A) + walk_profile[m-1][k].dot(At))
            #wp_m = wp_m + loop_mm(walk_profile, A, At, m)
            #wp_m = wp_m + Parallel(n_jobs=-1)(delayed(customized_mm)(walk_profile[m-1][k-1], walk_profile[m-1][k], A, At) for k in range(1, m+1))
            print(clock.end())
            wp_m.append(walk_profile[m-1][m].dot(A))
            walk_profile.append(wp_m)
    return walk_profile


#@jit(nopython=True)
def loop_mm(walk_profile, A, At, m):
    temp = []
    for k in range(1, m + 1):
        temp.append(walk_profile[m-1][k-1].dot(A) + walk_profile[m-1][k].dot(At))
    return temp
def customized_mm(w1, w2, A, At):
    return w1.dot(A) + w2.dot(At)



def sparse_walk_profile_calculator_v2(edge_index, M, source_node=None, normalize=False):
    # sparse implementation using scipy.sparse
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    A = np.array(A)
    n = A.shape[0]
    A = ss.csr_array(A)
    At = A.T

    if normalize:
        A = A.astype(np.float32)
        At = At.astype(np.float32)
        Dinv = np.diag((np.sum(A, axis=1) + np.sum(A, axis=0) + 1e-6) ** (-1))
        Dinv = ss.csr_array(Dinv)
        A = A.dot(Dinv)
        At = At.dot(Dinv)

    walk_profile = []
    clock = timer()
    if source_node == None:  # calculate walk profile for all node pairs
        #walk_profile.append(np.concatenate((At[None], A[None]), axis=0))
        walk_profile.append([At, A])
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            # computing walk profile phi(m+1, *)
            # wp_m = torch.einsum('qik, kj->qij', walk_profile[m - 1][:m], A) + \
            # torch.einsum('qik, kj->qij', walk_profile[m - 1][1:m+1], At)
            wp_0m = ss.block_diag(walk_profile[m-1][:m])
            wp_1m1 = ss.block_diag(walk_profile[m-1][1:m+1])
            I = ss.csr_array(np.eye(m))
            wp_m = [walk_profile[m-1][0].dot(At)]
            #for k in range(1, m+1):
                #wp_m.append(walk_profile[m-1][k-1].dot(A) + walk_profile[m-1][k].dot(At))
            clock.start()
            temp = wp_0m.dot(ss.kron(I, A)) + wp_1m1.dot(ss.kron(I, At))
            #print(clock.end())
            for i in range(m):
                wp_m.append(temp[i*n:(i+1)*n, i*n:(i+1)*n])
            print(clock.end())
            wp_m.append(walk_profile[m-1][m].dot(A))
            walk_profile.append(wp_m)
    return walk_profile


def sparse_walk_profile_calculator_v0(edge_index, M, source_node=None, normalize=False):
    A = to_dense_adj(edge_index)[0].type(torch.int64)
    At = A.T

    if normalize:
        A = A.float()
        At = At.float()
        Dinv = torch.diag((torch.sum(A, dim=1) + torch.sum(A, dim=0) + 1e-6) ** (-1))
        A = A @ Dinv
        At = At @ Dinv



    walk_profile = []
    if source_node == None:  # calculate walk profile for all node pairs
        walk_profile.append(torch.cat([At[None].to_sparse(), A[None].to_sparse()], dim=0))
        for m in range(1, M):
            print('computing walk profile at length %d' % m)
            # computing walk profile phi(m+1, *)
            # wp_m = torch.einsum('qik, kj->qij', walk_profile[m - 1][:m], A) + \
            # torch.einsum('qik, kj->qij', walk_profile[m - 1][1:m+1], At)
            #wp_m = walk_profile[m - 1][:m].dot(A) + walk_profile[m - 1][1:m + 1].dot(At)
            # slicing
            wp_0m = walk_profile[m-1].to_dense()[:m].to_sparse()
            wp_1m1 = walk_profile[m-1].to_dense()[1:m+1].to_sparse()
            wp_m = torch.bmm(wp_0m, A[None].tile([m, 1, 1])) + torch.bmm(wp_1m1, At[None].tile([m ,1, 1]))
            wp_m = wp_m.to_sparse()
            Am = torch.mm(walk_profile[m - 1][m], A).to_sparse()
            Atm = torch.mm(walk_profile[m - 1][0], At).to_sparse()
            walk_profile.append(torch.concatenate([Atm[None], wp_m, Am[None]], dim=0))
    return walk_profile


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

def message_passing(x, edge_index, num_nodes, degree=None):
    # x: [*, N], edge_index: [2, E]
    sources = edge_index[0]
    targets = edge_index[1]
    x_s = x[..., sources]
    if degree is not None:
        x_s = x_s / (degree[None, None, sources] + 1e-8)
    x_t = scatter_add(x_s, targets, dim=-1, dim_size=num_nodes)
    #if degree is not None:
        #x_t = x_t / (degree.unsqueeze(0).unsqueeze(0)+1e-8)
    #x_t = scatter_mean(x_s, targets, dim=-1, dim_size=num_nodes)
    return x_t



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


def sparse_matrix_walk_profile_calculation_and_analysis(edge_index, M, source_node=None, normalize=False,
                                                          analysor=None):
    # this version only save latest step walk profile, and does analysis on the fly
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

    N = A.size(0)
    #walk_profile_float = []
    edge_index_reverse = edge_index[[1, 0]]

    # initialize walk profile at length = 1
    if source_node is None:  # calculate walk profile for all node pairs
        #walk_profile = torch.cat([At.unsqueeze(0), A.unsqueeze(0)], dim=0)
        walk_profile = torch.cat([At[None].to_sparse(), A[None].to_sparse()], dim=0)
    else:
        # a specific number of source nodes
        #walk_profile = torch.cat([At[source_node].unsqueeze(0), A[source_node].unsqueeze(0)], dim=0)
        walk_profile = torch.cat([At[None, source_node].to_sparse(), A[None, source_node].to_sparse()], dim=0)
        #walk_profile_float.append(walk_profile[-1].float())
        #clock = timer()


    A = A.to_sparse()
    At = A.to_sparse()


    for m in range(1, M):
        print('computing walk profile at length %d' % (m + 1))
        time1 = time.time()
        Am = walk_profile[-1] @ A
        Atm = walk_profile[0] @ At
        for k in range(1, m+1):
            temp = walk_profile[k-1] @ A + walk_profile[k] @ At
            walk_profile[k-1] = wp_m
            wp_m = temp
        temp = walk_profile[-1] @ A
        walk_profile[-1] = wp_m
        walk_profile.append(temp)
        del temp, wp_m
        # computing walk profile at length = m+1
        time2 = time.time()
        analysor.compute_sparsity_histogram(torch.cat(walk_profile.float()))
        time3 = time.time()
        print('computed walk profile at length %d, using time %.3f' % (m + 1, time2-time1))
        print('Analysed walk profile at length %d, using time %.3f' % (m + 1, time3-time2))


