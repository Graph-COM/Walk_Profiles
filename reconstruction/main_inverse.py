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

np.set_printoptions(precision=3, suppress=True)
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

def relative_error(x, xhat, ignore_all_zeros=True):
    # x, xhat: [L, N, N]
    if ignore_all_zeros: # all-zero vectors can be easily identified
        nonzeros_ind = torch.where(x.sum(0)!=0)
        #x_nonzeros = x[:, nonzeros_ind[0], nonzeros_ind[1]]
        #xhat_nonzeros = xhat[:, nonzeros_ind[0], nonzeros_ind[1]]
        x_nonzeros = x[:, nonzeros_ind[0]]
        xhat_nonzeros = xhat[:, nonzeros_ind[0]]
        x_norm = torch.sqrt((x_nonzeros).square().sum(dim=0))
        xhat_norm = torch.sqrt((xhat_nonzeros).square().sum(dim=0))
        #x_norm_mean = (x_norm + xhat_norm)/2
        x_norm_max = torch.maximum(x_norm, xhat_norm)
        return torch.sqrt((x_nonzeros - xhat_nonzeros).square().sum(dim=0)) / (x_norm_max+1e-12)

def batch_data(source_nodes, batch_size, num_nodes):
    if batch_size is None and source_nodes is not None:
        return [source_nodes]
    elif batch_size is None and source_nodes is None:
        return [[i for i in range(num_nodes)]]
    else:
        source_nodes = source_nodes if source_nodes is not None else [i for i in range(num_nodes)]
        num_source_nodes = len(source_nodes)
        num_batch = len(source_nodes) // batch_size
        source_nodes_batched = [np.array([source_nodes[i + batch_size * j] for i in range(batch_size)]) for j in range(num_batch)]
        if batch_size * num_batch < num_source_nodes:
            source_nodes_batched.append(np.array([source_nodes[i] for i in range(batch_size * num_batch, num_source_nodes)]))
        return source_nodes_batched





def main(walk_length, dataset, node_pair_filter, normalize, gpu, **kwargs):
    # parameters
    #walk_length = walk_length
    #energy_error = energy_error
    #M_hop_filter = M_hop_filter
    #dataset_name = 'code2'
    #dataset_name = 'cornell'
    dataset_name = dataset
    #dataset_name = 'tpugraphs'
    #parameters_name = '_len_%d_delta_%6f'%(walk_length, energy_error, )
    parameters_name = '_m_%d'%walk_length
    parameters_name += '_nbt' if args.nbt else ''
    #parameters_name += '_ns_%d' % (args.num_sn) if args.num_sn is not None else ''
    #parameters_name += '_filter' if node_pair_filter else ''
    parameters_name += '_norm' if normalize else ''
    output_log = osp.join('./output', dataset_name+parameters_name+'.txt')
    #if not osp.exists(output_path):
        #os.mkdir(output_path)
    #output_log = osp.join(output_path, dataset_name+parameters_name+'.txt')


    # walk profile computation and sparsity analysis
    data = load_dataset(dataset_name)
    # read edge index
    try:
        edge_index = torch.tensor(data.edge_index)
    except:
        edge_index = torch.tensor(data['edge_index'])
    # check format
    if edge_index.size(0) != 2:
        edge_index = edge_index.T

    # take the largest connected component
    if args.largest_cpn:
        edge_index = take_largest_connected_components(edge_index)
        try:
            data.edge_index = edge_index
        except:
            data['edge_index'] = np.array(edge_index)


    # compute dataset stat
    #num_nodes, num_edges, hist_spd = graph_stat(data)  # TO DO: use another efficient method to compute spd
    num_nodes, num_edges, _, _ = graph_stat(data)


    # sample source nodes
    if args.num_sn is not None:
        assert args.num_sn <= num_nodes
        source_nodes = np.array(random.sample([i for i in range(num_nodes)], args.num_sn))
    else:
        source_nodes = None

    # mini-batching for source nodes
    source_nodes = batch_data(source_nodes, args.batch_size, num_nodes)

    # recovery
    half_length = ceil(args.walk_length / 2) + 1
    #half_length = args.walk_length
    error = [[] for _ in range(2, half_length+1)]
    for i, source_node_idx in enumerate(source_nodes):
        print('mini-batch %d/%d (%d nodes)...' % (i+1, len(source_nodes), len(source_node_idx)))
        # compute magnetic walks
        #q_full = torch.tensor([i / 2 / (args.walk_length + 1) for i in range(args.walk_length + 1)])
        q_full = torch.tensor([i / 2 / (args.walk_length + 1) for i in range(half_length)])
        # compute magnetic walks
        time1 = time.time()
        mag_walks = compute_magnetic_walks(edge_index, num_nodes, q_full, walk_length, source_node=source_node_idx, nbt=args.nbt,
                                       normalize=args.normalize)
        time2 = time.time()
        time_mag_walks = time2 - time1
        mag_walks = mag_walks.flatten(1)
        # compute walk profile
        time1 = time.time()
        walk_profile = compute_walk_profile(edge_index, num_nodes, walk_length, source_node=source_node_idx, nbt=args.nbt, normalize=args.normalize)
        time2 = time.time()
        time_wp = time2 - time1
        #prob_mass_left = walk_profile.sum([0, -1])
        #with open(output_log, 'a') as file:
            #print('probability mass left is: %.3f+-%.3f' % (prob_mass_left.mean(), prob_mass_left.std()), file=file)
        walk_profile = walk_profile.flatten(1)

        # inference
        #error = []
        #time_ip = []
        for Q in range(2, half_length+1):
            #print('Q=%d' % Q)
            # subsampling of q
            q, sample_ind = q_sampling(q_full, Q, q_sampling_strategy='top-k')
            mag_walks_sampled = mag_walks[sample_ind]
            inverse_problem = setup_inverse_problem(args.walk_length, q, normalize=args.normalize)
            # walk_profile_hat = inverse_problem.solve_pseudo_inverse(mag_walks_sampled)
            #time1 = time.time()
            if args.solver == 'inverse':
                walk_profile_hat = inverse_problem.solve_pseudo_inverse(mag_walks_sampled)
            elif args.solver == 'lp':
                walk_profile_hat = inverse_problem.solve_linear_programming_with_positive_constraints(mag_walks_sampled)
            #time2 = time.time()
            #time_ip.append(time2 - time1)
            e = relative_error(walk_profile, walk_profile_hat)
            if np.array(e).mean() <= 0.05:
                wpe = walk_profile_hat
        #print('error: %.5f+-%.5f' % (e.mean(),  e.std()))
            error[Q-2] += list(e.tolist())

    #plt.plot([Q for Q in range(2, half_length+1)], np.array(error)[:, 0])
    # draw single error line
    plt.plot([Q for Q in range(2, half_length+1)], np.array(error).mean(-1), marker='o')
    plt.axhline(0.05, linestyle='--', label='error=0.05')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Number of qs')
    plt.ylabel('Relative error')
    plt.title('walk length=%d, nbt=%d, normalize=%d' % (args.walk_length, int(args.nbt), int(args.normalize)))
    #plt.show()
    plt.savefig('./figs/ip_%s_m%d_nbt%d_norm%d.png' % (dataset_name, args.walk_length, int(args.nbt), int(args.normalize)))
    # save computation results
    save_path = './results/ip_%s_m%d_nbt%d_norm%d_lcpn%d' % (dataset_name, args.walk_length, int(args.nbt),
                                                             int(args.normalize), int(args.largest_cpn))
    if not osp.exists(save_path):
        os.makedirs(save_path)
    # save (partial )data
    walk_profile = walk_profile.view([walk_profile.size(0), -1, num_nodes])[:, 0:10, :]
    wpe = wpe.view([wpe.size(0), -1, num_nodes])[:, 0:10, :]
    spectrum_distribution = (torch.abs(mag_walks)**2).mean(dim=-1)
    spectrum_distribution_std = (torch.abs(mag_walks)**2).std(dim=-1)
    mag_walks = mag_walks.view([mag_walks.size(0), -1, num_nodes])[:, 0:10, :]
    #sp.save_npz(osp.join(save_path, 'wp'), sp.csc_array(np.array(walk_profile)))
    #sp.save_npz(osp.join(save_path, 'wpe'), sp.csc_array(np.array(wpe)))
    #sp.save_npz(osp.join(save_path, 'mw'), sp.csc_array(np.array(mag_walks)))
    np.save(osp.join(save_path, 'wp.npy'), np.array(walk_profile))
    np.save(osp.join(save_path, 'wpe.npy'), np.array(wpe))
    np.save(osp.join(save_path, 'mw.npy'), np.array(mag_walks))
    np.save(osp.join(save_path, 'spectrum_dist.npy'), np.array(spectrum_distribution))
    np.save(osp.join(save_path, 'spectrum_dist_std.npy'), np.array(spectrum_distribution_std))
    if args.dataset.startswith('scdg'):
        error = np.array(error).reshape([len(error), args.num_sn, -1])
        groups = data.x
        in_groups_mask = torch.cat([(groups == groups[source_nodes[0][i]])[None] for i in range(args.num_sn)], dim=0)
        in_groups_mask = np.array(in_groups_mask)
        error_in_groups = (error * in_groups_mask).sum(axis=(1,2)) / in_groups_mask.sum()
        error_out_groups = (error * ~in_groups_mask).sum(axis=(1,2)) / (~in_groups_mask).sum()
        np.save(osp.join(save_path, 'error_in.npy'), error_in_groups)
        np.save(osp.join(save_path, 'error_out.npy'), error_out_groups)
        np.save(osp.join(save_path, 'error.npy'), error.mean(axis=(1,2)))
    else:
        np.save(osp.join(save_path, 'error.npy'), np.array(error).mean(-1))


    # save runtime
    #with open(output_log, 'a') as file:
        #print('runtime of computing %d-dim magnetic walks: %.3f' % (len(q_full), time_mag_walks), file=file)
        #print('runtime of computing %d-len walk profile: %.3f' % (args.walk_length, time_wp), file=file)
        #print('runtime of computing inverse problem: %.3f' % time_ip[-1], file=file)



if __name__ == '__main__':
    import argparse
    torch.set_num_threads(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--nbt', action='store_true', default=False)
    parser.add_argument('--solver', type=str, default='inverse')
    parser.add_argument('--no_filter', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--num_sn', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--largest_cpn', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    walk_length = args.walk_length
    node_pair_filter = not args.no_filter
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
    main(walk_length, args.dataset, node_pair_filter, normalize, gpu)
