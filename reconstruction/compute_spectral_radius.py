import os.path as osp
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
#import torch_geometric_signed_directed
#from torch_geometric_signed_directed.data import load_directed_real_data
from data_utils.data_loader import load_dataset
from data_utils.dataset_stat import graph_stat
from utils.walk_profile import compute_walk_profile
from utils.magnetic_walk import compute_magnetic_walks
from utils.mag_inverse_problem import setup_inverse_problem, q_sampling
import time
import random
from math import ceil
import scipy.sparse as sp
import networkx as nx

from torch_geometric.utils import degree as pyg_degree

#np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)
random.seed(42)


def take_largest_connected_components(edge_index):
    import networkx as nx
    G = nx.DiGraph()
    edges = edge_index.t().tolist()  # Convert to list of tuples
    G.add_edges_from(edges)
    connected_components = list(nx.weakly_connected_components(G))
    I = torch.tensor(list(connected_components[0]))
    # Create a mapping from old node indices to new indices
    old_to_new = -torch.ones(edge_index.max().item() + 1, dtype=torch.long)
    old_to_new[I] = torch.arange(I.size(0))

    # Mask to keep only edges where both source and target are in I
    src, dst = edge_index
    mask = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)

    # Apply the mask and remap the indices
    new_src = old_to_new[src[mask]]
    new_dst = old_to_new[dst[mask]]
    new_edge_index = torch.stack([new_src, new_dst], dim=0)
    return new_edge_index


def largest_eigenvalues(dataset, walk_length, largest_cpn, normalize, gpu, second_largest=False):
    data = load_dataset(dataset)
    try:
        edge_index = torch.tensor(data.edge_index)
    except:
        edge_index = torch.tensor(data['edge_index'])
    # check format
    if edge_index.size(0) != 2:
        edge_index = edge_index.T
    if largest_cpn:
        edge_index = take_largest_connected_components(edge_index)
    num_nodes = edge_index.max()+1
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(-1)), (num_nodes, num_nodes))
    #At = A.T
    Q = ceil(walk_length / 2) + 1
    q = torch.tensor([i / 2 / (walk_length + 1) for i in range(Q)])
    #q = q.unsqueeze(-1).unsqueeze(-1)
    #Aq = A.to_dense() * torch.exp(1j * 2 * torch.pi * q) + At.to_dense() * torch.exp(-1j * 2 * torch.pi * q)
    #Aq = torch.cat(
        #[(A * torch.exp(1j * 2 * torch.pi * q[i]) + At * torch.exp(-1j * 2 * torch.pi * q[i]))[None] for i in range(Q)],
        #dim=0)
    #del A, At

    # normalize if needed
    degree = None
    if normalize:
        degree = pyg_degree(edge_index[0], num_nodes=num_nodes) + pyg_degree(edge_index[1], num_nodes=num_nodes)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # Handle division by zero if any

        # Extract indices and values from sparse A
        indices = A.coalesce().indices()  # [2, E]
        values = A.coalesce().values()  # [E]

        # Normalize values: value_ij -> value_ij * d^{-1/2}_i * d^{-1/2}_j
        row, col = indices[0], indices[1]
        new_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]

        # Create the new normalized sparse matrix
        A = torch.sparse_coo_tensor(indices, new_values, A.size())

        #D = torch.diag((degree + 1e-6) ** (-1))
        # A = A @ D
        # At = At @ D
        #Aq = (D * (1. + 0 * 1j)) @ Aq
        #inv_degree = 1./(degree[None, :, None]+1e-6)
        #Aq = inv_degree * Aq
        #del inv_degree

    Emax = []
    E2max = []
    for i in range(Q):
        qi = q[i]
        Aq = A * torch.exp(1j * 2 * torch.pi * qi) + A.T * torch.exp(-1j * 2 * torch.pi * qi)
        A_coalesced = Aq.coalesce()
        indices = A_coalesced.indices()
        values = A_coalesced.values()
        N = Aq.size(0)
        Aq_scipy = sp.coo_matrix(
            (values.cpu().numpy(), (indices[0].cpu().numpy(), indices[1].cpu().numpy())),
            shape=(N, N)
        )
        E, _ = sp.linalg.eigsh(Aq_scipy, k=2, which='LA')  # 'LA' = Largest Algebraic
        #E, _ = torch.linalg.eigh(Aqi)
        #E = E.real
        #E = E.sort()[0]
        Emax.append(E[0])
        E2max.append(E[1])
    if second_largest:
        return np.array(torch.tensor(Emax)), np.array(torch.tensor(E2max))
    else:
        return np.array(torch.tensor(Emax))


if __name__ == '__main__':
    import argparse
    torch.set_num_threads(10)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--nbt', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--walk_length', type=int, default=50)
    parser.add_argument('--largest_cpn', action='store_true', default=False)
    args = parser.parse_args()

    normalize = args.normalize
    gpu = args.gpu
    #dataset_list = ['cornell', 'texas', 'wisconsin', 'telegram', 'blog',
    #'citeseer', 'cora_ml']

    # tpugraphs
    #dataset_list = ['wikics']
    #for dataset in dataset_list:
    #print('Computing dataset '+dataset+'...')
    #main(walk_length, energy_error, dataset, M_hop_filter, normalize)

    # cpu threads control
    torch.set_num_threads(8)
    import os
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    os.environ["TBB_NUM_THREADS"] = "8"

    # erdos-renyi graph
    #degree_list = [1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 128.0, 256.0, 512.0, 999.0]
    degree_list = [5000]
    #ks = [1, 4, 64, 256]
    #dataset_list = ['er_n1000_p_1-n_c_%.4f_seed_0' % degree for degree in degree_list]
    #dataset_list = ['er_n5000_p_1-n_c_%.4f_seed_0' % degree for degree in degree_list]
    #dataset_list = ['scdg_n1000_k_%d_shape_1.000_scale_3.000_seed_0' % k for k in ks]
    #dataset_list = ['clique_n1000']
    #dataset_list = ['CELE', 'FIG', 'bio_drug-drug', 'bio_drug-target', 'bio_function-function', 'bio_protein-protein']
    dataset_list = ['CELE']
    #dataset_list = ['ADO', 'ATC', 'EMA', 'HIG', 'USA', 'PB', 'snap-cite-hep', 'snap-epinion', 'snap-wiki-vote',
                    #'code2-9k', 'resnet', 'transformer']
    #dataset_list = ['ogbn-arxiv']
    #dataset_list = ['growing_code2_%d'%i for i in [0, 2, 5, 7]]
    #dataset_list = ['er_n5000_p_1-n_c_1.0000_seed_0',
                    #'er_n5000_p_1-n_c_2.0000_seed_0',
                    #'er_n5000_p_1-n_c_4.0000_seed_0', 'er_n5000_p_1-n_c_8.0000_seed_0',
                    #'er_n5000_p_1-n_c_16.0000_seed_0',
                    #'citeseer', 'cora_ml', 'ogbn-arxiv', 'bert', 'mask_rcnn', 'alexnet', 'code2-20k', 'code2-36k']
    #dataset_list = ['code2']
    Emax = []
    E2max = []

    Q = ceil(args.walk_length / 2) + 1
    q = torch.tensor([i / 2 / (args.walk_length + 1) for i in range(Q)])


    for dataset in dataset_list:
        print('calculating eigenvalues of %s' % dataset)
        m, m2 = largest_eigenvalues(dataset, args.walk_length, normalize, args.largest_cpn, gpu, second_largest=True)
        Emax.append(m)
        E2max.append(m2)
        #Emax.append(largest_eigenvalues(dataset, args.walk_length, normalize, args.largest_cpn, gpu))

        # save
        save_path = './results/spectrum_%s_nbt%d_norm%d_lcpn%d' % (dataset, int(args.nbt),
                                                                 int(args.normalize), int(args.largest_cpn))
        if not osp.exists(save_path):
            os.makedirs(save_path)
        np.save(osp.join(save_path, 'largest_eigenvalues.npy'), np.array(m))
        np.save(osp.join(save_path, 'largest2_eigenvalues.npy'), np.array(m2))
        np.save(osp.join(save_path, 'frequency.npy'), np.array(q))

    #import matplotlib.pyplot as plt




    #for i, degree in enumerate(degree_list):
        #plt.plot(q, Emax[i], marker='o', label='ER degree=%.1f' % degree)
        #plt.plot(q, E2max[i], marker='o', label='ER degree=%.1f' % degree)
    #for i, k in enumerate(ks):
        #plt.plot(q, Emax[i], marker='o', label='SCDG #clusters=%d' % k)
    #plt.plot(q, Emax[0], marker='o')
    #plt.yscale("log")
    #plt.legend()
    #plt.xlabel('potential q')
    #plt.ylabel('Largest eigenvalue')
    #plt.title('Largest eigenvalue of D^{-1}Aq for different q')
    #plt.savefig('./figs/spectrum_er.png')
    #plt.savefig('./figs/spectrum_er_full.png')
