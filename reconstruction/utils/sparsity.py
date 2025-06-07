import numpy as np
import torch
import networkx as nx
#from utils.check import walk_profile_check_reachablility
import os.path as osp
import matplotlib.pyplot as plt
import time
from math import comb


def calculate_node_pair_idx_by_spd(edge_index, cutoff):
    symmetric_edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=-1)
    G = nx.Graph()
    G.add_nodes_from([i for i in range(edge_index.max())])
    G.add_edges_from([(i.item(), j.item()) for i, j in symmetric_edge_index.transpose(0, 1)])
    # G = nx.from_edgelist([(i.item(), j.item()) for i, j in symmetric_edge_index.transpose(0, 1)])
    # spd = nx.floyd_warshall_numpy(G)
    spd = dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))
    node_pair_by_spd = {}
    for i in spd.keys():
        for j in spd[i].keys():
            if node_pair_by_spd.get(spd[i][j]) is None:
                node_pair_by_spd[spd[i][j]] = [[i, j]]
            else:
                node_pair_by_spd[spd[i][j]].append([i, j])
    # self.reachable_node_pairs = torch.tensor(node_pair_idx).transpose(0, 1)
    return node_pair_by_spd


class SparsityAnalysis:
    def __init__(self, walk_profile, edge_index):
        self.walk_profile = walk_profile
        self.edge_index = edge_index
        self.walk_length = len(self.walk_profile)
        self.node_pair_by_spd = None

    def sparsity_histogram(self, energy_error=1e-6, plot=False, M_hop_filter=False):
        assert energy_error <= 1.0 and energy_error >= 0.0
        wp = self.walk_profile
        histogram = {}

        if M_hop_filter and self.node_pair_by_spd is None:
            self.node_pair_by_spd = calculate_node_pair_idx_by_spd(self.edge_index, self.walk_length)
            node_pair_idx_within_spd_m = self.node_pair_by_spd[0]
        # sanity check
        #walk_profile_check_reachablility(wp, self.reachable_node_pairs)
        for m in range(1, self.walk_length+1):
            wp_m = wp[m-1]
            if wp_m.is_sparse:
                wp_m = wp_m.to_dense()
            if M_hop_filter:
                #node_pair_idx = torch.where(torch.tensor(spd) <= m)
                if self.node_pair_by_spd.get(m) is not None:
                    node_pair_idx_within_spd_m = node_pair_idx_within_spd_m + self.node_pair_by_spd[m]
                wp_m = wp_m[:, torch.tensor(node_pair_idx_within_spd_m)[:, 0],
                       torch.tensor(node_pair_idx_within_spd_m)[:, 1]]
            else:
                wp_m = wp_m.flatten(1)
            if energy_error == 0.0:
                sparsity = absolute_sparsity(wp_m)
            else:
                sparsity = relative_sparsity(wp_m, energy_error)
            histogram[m] = sparsity.cpu().numpy()
        if plot:
            pass
        return histogram

    def energy_distribution(self, M_hop_filter=False):
        # the energy distribution over entries
        wp = self.walk_profile
        energy_distribution = {}
        if M_hop_filter and self.node_pair_by_spd is None:
            self.node_pair_by_spd = calculate_node_pair_idx_by_spd(self.edge_index, self.walk_length)
            node_pair_idx_within_spd_m = self.node_pair_by_spd[0]

        for m in range(1, self.walk_length+1):
            wp_m = wp[m-1]
            if wp_m.is_sparse:
                wp_m = wp_m.to_dense()
            if M_hop_filter:
                # node_pair_idx = torch.where(torch.tensor(spd) <= m)
                if self.node_pair_by_spd.get(m) is not None:
                    node_pair_idx_within_spd_m = node_pair_idx_within_spd_m + self.node_pair_by_spd[m]
                wp_m = wp_m[:, torch.tensor(node_pair_idx_within_spd_m)[:, 0],
                       torch.tensor(node_pair_idx_within_spd_m)[:, 1]]
            else:
                wp_m = wp_m.flatten(1)
            energy_distribution[m] = (wp_m**2 / (torch.linalg.norm(wp_m, dim=0).unsqueeze(0)+1e-9)**2).cpu().numpy()
        return energy_distribution


class SparsityAnalysisOnFly:
    def __init__(self, edge_index, energy_error=1e-3, save_path=None, log_path=None):
        self.edge_index = edge_index
        self.save_path = save_path
        self.log_path = log_path
        self.energy_error = energy_error

        # clear the previous log
        with open(log_path, 'w') as file:
            #print('Dataset and parameters: %s. Energy error %.9f' % (log_path, energy_error), file=file)
            #print('Num of nodes|edges: %d, %d' % (num_nodes, num_edges), file=file)
            pass
        file.close()


    def compute_sparsity_histogram(self, walk_profile, node_pair_to_eval=None):
        m = walk_profile.shape[0] - 1
        energy_error = self.energy_error

        # choose node pairs to evaluate walk profiles
        if node_pair_to_eval is not None:
            # sanity check
            #assert walk_profile[:, ~node_pair_to_eval].abs().sum() == 0.
            #assert walk_profile[:, node_pair_to_eval].abs().sum(0).min() > 0.

            walk_profile = walk_profile[:, node_pair_to_eval]

        else:
            walk_profile = walk_profile.flatten(1)

        # evaluate sparsity
        time1 = time.time()
        if energy_error == 0.0:
            sparsity = absolute_sparsity(walk_profile)
        else:
            sparsity = relative_sparsity(walk_profile, energy_error)
        sparsity = sparsity.cpu().numpy()
        # worst case
        sparsity_worst = sparsity.max()
        # average case
        sparsity_ave = sparsity.mean()
        sparsity_std = sparsity.std()
        # percentile
        percentile = np.array([50.0, 68.0, 95.0, 99.0])
        sparsity_per = np.percentile(sparsity, percentile)
        time2 = time.time()

        # energy distribution
        energy = (walk_profile ** 2 / (torch.linalg.norm(walk_profile, dim=0).unsqueeze(0) + 1e-9) ** 2).cpu().numpy()
        time3 = time.time()
        # save log
        with open(self.log_path, 'a') as file:
            print('-----Walk length %d-----' % m, file=file)
            print('Worst sparsity: %.3f' % sparsity_worst, file=file)
            print('Average sparsity: %.3f+-%.3f' % (sparsity_ave, sparsity_std), file=file)
            for p, s in zip(percentile, sparsity_per):
                print('%.2f percentile sparsity: %.3f' % (p, s), file=file)
            print('Energy distribution: ', energy.mean(-1), file=file)
        file.close()
        time4 = time.time()



        # save figures
        plt.hist(sparsity, density=True, range=(0., 1.), bins=m + 2)
        plt.title('Walk profile sparsity distribution over node pairs')
        plt.xlabel('Sparsity')
        plt.ylabel('PDF')
        save_path = osp.join(self.save_path, 'sparsity_m%d.png' % m)
        plt.savefig(save_path)
        plt.clf()
        time5 = time.time()
        print(time2-time1, time3-time2, time4-time3, time5-time4)

        return sparsity_ave


    def verify_expectation(self, walk_profile):
        m = walk_profile.size(0) - 1
        n = walk_profile.size(-1)
        d = self.edge_index.size(-1) / n
        expectation = np.array([comb(m, k) * d ** m / n for k in range(m+1)])
        mean = np.array(walk_profile.mean(dim=[1,2]))
        with open(self.log_path, 'a') as file:
            print('-----Walk length %d-----' % m, file=file)
            print('Expectation of nbt walk profile:', expectation, file=file)
            print('Sample mean of nbt walk profile:', mean, file=file)





def relative_sparsity(x, energy_error, p=1):
    M = x.size(0)
    #E = torch.unsqueeze(torch.linalg.norm(x, dim=0) ** 2, dim=0)
    E = torch.linalg.vector_norm(x, dim=0, ord=p, keepdim=True)
    E_threshold = energy_error / M
    return ((torch.abs(x) / E) >= E_threshold).float().mean(dim=0)

def absolute_sparsity(x):
    #return (x >= 0.99).float().mean(dim=0) # x only takes integer values
    return (x >= 1e-9).float().mean(dim=0)

