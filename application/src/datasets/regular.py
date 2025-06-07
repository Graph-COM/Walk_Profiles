import os
from time import time

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected
#from torch_sparse import coalesce
import scipy.sparse as ssp
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.heuristics import RA, PPR
from src.utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
#from src.hashing import ElphHashes


class RegularDataset(Dataset):
    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, use_coalesce=False,
            directed=False, **kwargs):
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.node_label = args.node_label
        super(RegularDataset, self).__init__(root)

        self.x = data.x
        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

        self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

        #self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

        if self.node_label == 'ppr':
            self.z = PPR(self.A, self.links)
        elif self.node_label == 'ppr+':
            self.z = PPR(self.A, self.links, bidirection=True)
        else:
            raise Exception('Non-subgraph Node label is not implemented!')
            #self.RA = RA(self.A, self.links, batch_size=2000000)[0]



    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        #if self.args.use_struct_feature:
            #subgraph_features = self.subgraph_features[idx]
        #else:
        #subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        #if self.use_RA:
        #    RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
        #    RA = torch.tensor([RA], dtype=torch.float)
        #else:
        #    RA = -1
        #src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return node_features, y


def get_regular_train_val_test_datasets(dataset, train_data, val_data, test_data, args, directed=False):
    root = f'{dataset.root}/elph_'
    print(f'data path: {root}')
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    print('constructing training dataset object')
    train_dataset = RegularDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                use_coalesce=use_coalesce, directed=directed)
    print('constructing validation dataset object')
    val_dataset = RegularDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                              use_coalesce=use_coalesce, directed=directed)
    print('constructing test dataset object')
    test_dataset = RegularDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                               use_coalesce=use_coalesce, directed=directed)
    return train_dataset, val_dataset, test_dataset


