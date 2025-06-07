"""
Code based on
https://github.com/facebookresearch/SEAL_OGB
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SEAL reformulates link prediction as a subgraph classification problem. To do this subgraph datasets must first be constructed
"""

from math import inf
import random
import os
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops)
#from torch_sparse import coalesce
from tqdm import tqdm
import scipy.sparse as ssp
import time
from src.utils import get_src_dst_degree, neighbors, get_pos_neg_edges
from src.labelling_tricks import drnl_node_labeling, de_node_labeling, de_plus_node_labeling, cn_node_labeling, \
    cn_plus_node_labeling, rw_node_labeling, rw_plus_node_labeling, mw_node_labeling, wp_node_labeling
from src.datasets.motif import reciprocal_edges, common_parents, common_children, feedforward_loop, three_walks, cycles,cycles_networkx


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, num_hops, node_label='drnl', ratio_per_hop=1.,
                 max_nodes_per_hop=None, max_dist=1000,
                 save_no_subgraph=False, mw_params=None, target=None, binarize_target=False):
        self.root = root
        self.data = data
        #self.data.num_nodes = self.data.x.size(-1) if 'x' in self.data else self.data.num_nodes # used later
        self.num_hops = num_hops
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.save_no_subgraph = save_no_subgraph
        self.mw_params = mw_params
        self.directed = True
        super(SEALDataset, self).__init__(root)

        self.target_save_dir = root[:root.find('_node')]
        if self.save_no_subgraph:
            # inherit the data of dataset
            self.z = torch.load(self.processed_paths[0])
            self.x = data.x
            if target is not None:
                print('target is %s. Original data labels are ignored!' % target)
                self.y = self.construct_target(data, target) # construct target if not saved in disk otherwise loading from disk
            else:
                self.y = data.y
            if binarize_target:
                self.y[self.y>1e-3] = 1
                self.y[self.y<=1e-3] = 0
                self.y = self.y.int()
                print("%.3f nodes have y=1 binary labels." % (self.y==1).float().mean())

            # re-split based on labels y
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
            self.test_mask = data.test_mask
            if binarize_target:
                # ensure that train,val,test set all has positive and negative labels
                idx_1 = np.where(self.y == 1)[0]
                idx_0 = np.where(self.y == 0)[0]

                np.random.shuffle(idx_1)
                np.random.shuffle(idx_0)

                train_rate = sum(self.train_mask) / len(self.train_mask)
                val_rate = sum(self.val_mask) / len(self.val_mask)
                test_rate = 1 - train_rate - val_rate

                def split_indices(indices):
                    n = len(indices)
                    n_train = int(n * train_rate)
                    n_val = int(n * val_rate)
                    n_test = n - n_train - n_val  # remaining
                    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]

                train_1, val_1, test_1 = split_indices(idx_1)
                train_0, val_0, test_0 = split_indices(idx_0)

                train_idx = np.concatenate([train_1, train_0])
                val_idx = np.concatenate([val_1, val_0])
                test_idx = np.concatenate([test_1, test_0])
                self.train_mask[:] = False
                self.val_mask[:] = False
                self.test_mask[:] = False
                self.train_mask[train_idx] = True
                self.val_mask[val_idx] = True
                self.test_mask[test_idx] = True
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.x)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def construct_target(self, data, target):
        if target is None:
            return data.y
        # see if labels are cached
        target_save_path = os.path.join(self.target_save_dir, 'y_'+target+'.pt')
        if os.path.exists(target_save_path):
            print('Target %s had been computed and is loaded...' % target)
            return torch.load(target_save_path)
        # otherwise, compute and save targets
        print('Calculating target %s...' % target)
        if 'edge_weight' in data:
            edge_weight = data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (data.edge_index[0], data.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )
        time1 = time.time()
        if target == 're':
            y = reciprocal_edges(A)
        elif target == 'cp':
            y = common_parents(A)
        elif target == 'common':
            y = common_parents(A) + common_children(A)
        elif target == 'walk_3':
            y = three_walks(A)
        elif target == 'fwl':
            y = feedforward_loop(A)
        elif target.startswith('cycle'):
            #y = cycles_networkx(A, target)
            y = cycles(A, target)
        time2=time.time()
        print('Target is pre-computed and being saved. Time used %.2f s' % (time2-time1))
        if not os.path.exists(self.target_save_dir):
            os.makedirs(self.target_save_dir)
        torch.save(y, target_save_path)
        return y

    def process(self):

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # directly compute node labellings on full graph
        z = assign_node_label_full_graph(A, self.node_label, self.max_dist, self.mw_params)
        z = torch.tensor(np.array(z)).float() # for some reasons the raw tensor z is very large maybe due to extra meta data
        torch.save(z, self.processed_paths[0])

        # Extract enclosing subgraphs for pos and neg edges
        #subgraph_list = extract_enclosing_subgraphs(
            #A, self.data.x, 1, self.num_hops, self.node_label,
            #self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc,
            #save_no_subgraph=self.save_no_subgraph, mw_params=self.mw_params)
        #if self.save_no_subgraph:
            # only save node features in disk
        #    torch.save(torch.cat([z[None] for z in subgraph_list], dim=0), self.processed_paths[0])
        #else:
            # save all rooted subgraphs in disk
        #    raise Exception('not implemented!')


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., use_coalesce=False, node_label='drnl',
                 ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 **kwargs):
        self.data = data
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        super(SEALDynamicDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0).tolist()
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, self.max_nodes_per_hop)
        if self.sign:
            x = [self.data.x]
            x += [self.data[f'x{i}'] for i in range(1, self.k + 1)]
        else:
            x = self.data.x
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label, self.max_dist, src_degree, dst_degree)

        return data


def sample_data(data, sample_arg):
    if sample_arg <= 1:
        samples = int(sample_arg * len(data))
    elif sample_arg != inf:
        samples = int(sample_arg)
    else:
        samples = len(data)
    if samples != inf:
        sample_indices = torch.randperm(len(data))[:samples]
    return data[sample_indices]


def get_train_val_test_datasets(dataset, args):
    node_label_name = args.node_label
    # extra hyperparameters for magnetic walks
    if args.node_label == 'mw':
        node_label_name += '_%dq-%.3f' % (args.q_dim, args.q) if args.q is not None else '_%dq' % args.q_dim
    # node_label_name += '_%d-q-%.3f' % (args.q_dim, args.q) if args.node_label == 'mw' else ''
    if args.node_label in ['mw', 'rw', 'rw+', 'wp']:
        node_label_name += '-'+args.entry
        node_label_name += '-nbt' if args.nbt else ''
        node_label_name += '-norm' if args.norm else ''
        node_label_name += '-compact' if args.compact_q and args.node_label=='mw' else ''
    # path = f'{dataset.root}_seal_{sample}_hops_{args.num_hops}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}{args.data_appendix}'
    #path = f'{dataset.root}_node_hops_{args.num_hops}_label_{node_label_name}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}'
    path = f'{dataset.root}_node_label_{node_label_name}_maxdist_{args.max_dist}'
    #path += '_drop_subgraph' if args.save_no_subgraph else ''
    print(f'seal data path: {path}')
    dataset_class = 'SEALDataset'
    dataset = eval(dataset_class)(
        path,
        dataset._data,
        num_hops=args.num_hops,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        save_no_subgraph=args.save_no_subgraph,
        mw_params=(args.q, args.q_dim, args.nbt, args.norm, args.entry, args.compact_q),
        target=args.target,
        binarize_target=args.binarize_target,
    )
    return dataset


def get_seal_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def k_hop_subgraph(src, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    """
    Extract the k-hop enclosing subgraph around link (src, dst) from A.
    it permutes the node indices so the returned subgraphs are not immediately recognisable as subgraphs and it is not
    parallelised.
    For directed graphs it adds both incoming and outgoing edges in the BFS equally and then for the target edge src->dst
    it will also delete any dst->src edge, it's unclear if this is a feature or a bug.
    :param src: source node for the edge
    :param dst: destination node for the edge
    :param num_hops:
    :param A:
    :param sample_ratio: This will sample down the total number of neighbours (from both src and dst) at each hop
    :param max_nodes_per_hop: This will sample down the total number of neighbours (from both src and dst) at each hop
                            can be used in conjunction with sample_ratio
    :param node_features:
    :param y:
    :param directed:
    :param A_csc:
    :return:
    """
    nodes = [src]
    dists = [0]
    visited = set([src])
    fringe = set([src])
    for hop in range(1, num_hops + 1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [hop] * len(fringe)
    # this will permute the rows and columns of the input graph and so the features must also be permuted
    subgraph = A[nodes, :][:, nodes]


    if isinstance(node_features, list):
        node_features = [feat[nodes] for feat in node_features]
    elif node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', max_dist=1000, save_no_subgraph=False, mw_params=None):
    """
    Constructs a pyg graph for this subgraph and adds an attribute z containing the node_label
    @param node_ids: list of node IDs in the subgraph
    @param adj: scipy sparse CSR adjacency matrix
    @param dists: an n_nodes list containing shortest distance (in hops) to the src or dst node
    @param node_features: The input node features corresponding to nodes in node_ids
    @param y: scalar, 1 if positive edges, 0 if negative edges
    @param node_label: method to add the z attribute to nodes
    @return:
    """
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'degree':  # this is technically not a valid labeling trick
        #z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z = torch.tensor(adj.sum(axis=0)).squeeze(0) + torch.tensor(adj.sum(axis=1)).squeeze(1)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'degree+':
        z = torch.cat([torch.tensor(adj.sum(axis=0)).squeeze(0)[:, None], torch.tensor(adj.sum(axis=1)).squeeze(1)[:, None]], dim=-1)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'cn': # common neighbor w/o direction
        z = cn_node_labeling(adj, 0, 0)
    elif node_label == 'cn+':  # common neighbor w/ direction
        z = cn_plus_node_labeling(adj, 0, 0)
    elif node_label == 'rw':
        z = rw_node_labeling(adj, 0, 0, max_dist, mw_params) # for walk-based method, max_dist refers to the walk length to consider
    elif node_label == 'rw+':
        z = rw_plus_node_labeling(adj, 0, 0, max_dist, mw_params)
    elif node_label == 'mw':
        z = mw_node_labeling(adj, 0, 0, max_dist, mw_params)
    elif node_label == 'wp':
        z = wp_node_labeling(adj, 0, 0, max_dist, mw_params)
    else:
        raise Exception('Unrecognized node labeling during preprocessing!')

    if save_no_subgraph:
        # when we only cares about the labeling of center nodes: just return center node labels
        #node_features = node_features[0:2]
        z = z[0:1]
        #edge_index = torch.tensor([])
        #edge_weight = torch.tensor([])
        #node_ids = node_ids[0:2]
        #num_nodes = 2
        data = z.flatten(0) # to just save link features, no need to keep node-level things
    else:
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                    node_id=node_ids, num_nodes=num_nodes, src_degree=src_degree, dst_degree=dst_degree)
    return data


def extract_enclosing_subgraphs(A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000,
                                directed=False, A_csc=None, save_no_subgraph=False, mw_params=None):
    """
    Extract a num_hops subgraph around every edge in the link index
    @param link_index: positive or negative supervision edges from train, val or test
    @param A: A scipy sparse CSR matrix containing the message passing edge
    @param x: features on the data
    @param y: 1 for positive edges, 0 for negative edges
    @param num_hops: the number of hops from the src or dst node to expand the subgraph to
    @param node_label:
    @param ratio_per_hop:
    @param max_nodes_per_hop:
    @param directed:
    @param A_csc: None if undirected, otherwise converts to a CSC matrix
    @return:
    """
    data_list = []
    for src in tqdm(range(x.size(0))):
        tmp = k_hop_subgraph(src, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label, max_dist, save_no_subgraph, mw_params)
        data_list.append(data)

    return data_list


from src.labelling_tricks import rw_node_labeling_full_graph, rw_plus_node_labeling_full_graph, \
    wp_node_labeling_full_graph,mw_node_labeling_full_graph
def assign_node_label_full_graph(A, node_label, walk_length, mw_params):
    if node_label == 'rw':
        return rw_node_labeling_full_graph(A, walk_length, mw_params)
    elif node_label == 'rw+':
        return rw_plus_node_labeling_full_graph(A, walk_length, mw_params)
    elif node_label == 'wp':
        return wp_node_labeling_full_graph(A, walk_length, mw_params)
    elif node_label == 'mw':
        return mw_node_labeling_full_graph(A, walk_length, mw_params)