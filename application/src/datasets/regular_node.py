import os
from time import time

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected
#from torch_sparse import coalesce
import scipy.sparse as ssp
#import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.heuristics import RA, PPR
from src.utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
#from src.hashing import ElphHashes
from src.labelling_tricks import drnl_node_labeling, de_node_labeling, de_plus_node_labeling, cn_node_labeling, \
    cn_plus_node_labeling, rw_node_labeling, rw_plus_node_labeling, mw_node_labeling, wp_node_labeling


class RegularDataset(Dataset):
    def __init__(
            self, root, data, args):
        self.root = root
        self.args = args
        self.node_label = args.node_label
        self.x = data.x
        self.y = data.y
        super(RegularDataset, self).__init__(root)
        self.z = torch.load(self.processed_paths[0])


    def len(self):
        return len(self.x)

    def get(self, idx):
        y = self.y[idx]
        node_features = self.x[idx]
        node_labels = self.z[idx]
        return node_features, node_labels, y

    def process(self):
        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        z = construct_node_labels(A, self.args)



def get_regular_train_val_test_datasets(dataset, args):
    node_label_name = args.node_label
    # extra hyperparameters for magnetic walks
    if args.node_label == 'mw':
        node_label_name += '_%dq-%.3f' % (args.q_dim, args.q) if args.q is not None else '_%dq' % args.q_dim
    # node_label_name += '_%d-q-%.3f' % (args.q_dim, args.q) if args.node_label == 'mw' else ''
    node_label_name += '-' + args.entry if args.node_label in ['mw', 'rw', 'rw+', 'wp'] else ''
    node_label_name += '-nbt' if args.nbt else ''
    node_label_name += '-norm' if args.norm else ''
    # path = f'{dataset.root}_seal_{sample}_hops_{args.num_hops}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}{args.data_appendix}'
    path = f'{dataset.root}_node_label_{node_label_name}'
    train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask
    print('constructing dataset object')
    dataset = RegularDataset(path, dataset, args)
    return dataset


def construct_node_label(adj, args):
    node_label = args.node_label
    mw_params = (args.q, args.q_dim, args.nbt, args.norm, 'full')
    max_dist = args.max_dist
    if node_label == 'degree':  # this is technically not a valid labeling trick
        # z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z = torch.tensor(adj.sum(axis=0)).squeeze(0) + torch.tensor(adj.sum(axis=1)).squeeze(1)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'degree+':
        z = torch.cat(
            [torch.tensor(adj.sum(axis=0)).squeeze(0)[:, None], torch.tensor(adj.sum(axis=1)).squeeze(1)[:, None]],
            dim=-1)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'rw':
        z = rw_node_labeling(adj, 0, 1, max_dist,
                             mw_params)  # for walk-based method, max_dist refers to the walk length to consider
    elif node_label == 'rw+':
        z = rw_plus_node_labeling(adj, 0, 1, max_dist, mw_params)
    elif node_label == 'mw':
        z = mw_node_labeling(adj, 0, 1, max_dist, mw_params)
    elif node_label == 'wp':
        z = wp_node_labeling(adj, 0, 1, max_dist, mw_params)
    else:
        raise Exception('Unrecognized node labeling during preprocessing!')

    if args.entry == 'ss':
        z = torch.diagonal(z, dim)

