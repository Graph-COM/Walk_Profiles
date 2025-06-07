import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import remove_self_loops
import os.path as osp
import csv

class Airport(InMemoryDataset):
    def __init__(self, path):
        super(Airport, self).__init__()
        file = open(path+'/edges.txt', 'r')
        edges = []
        for line in file.readlines():
            line = line.split(' ')
            edges.append([int(line[0]), int(line[1])])
        file.close()
        file = open(path + '/labels.txt', 'r')
        labels = []
        for line in file.readlines():
            line = line.split(' ')
            labels.append(int(line[1]))
        file.close()
        edges = torch.tensor(edges).transpose(0, 1)
        labels = torch.tensor(labels)
        self.root = path
        self.data = Data(x=torch.zeros([labels.size(0), 1]), edge_index=edges, y=labels)
        
        
class ERGraph(InMemoryDataset):
    def __init__(self, path, n, ave_degree):
        super(ERGraph, self).__init__()
        file_name = "er_n%d_p_1-n_c_%.4f.npy" % (n, ave_degree)
        edges = np.load(osp.join(path, file_name))
        self.root = path+'_n%d_c_%.4f' % (n, ave_degree)
        self.data = Data(x=torch.zeros([n, 1]), edge_index=torch.tensor(edges))


class NPZ(InMemoryDataset):
    def __init__(self, path, graph_name):
        super(NPZ, self).__init__()
        file_name = {'alexnet': 'alexnet_train_batch_32.npz',
                      'resnet': 'resnet50.8x8.fp16.npz',
                      'mask_rcnn': 'mask_rcnn_batch_16_bf16_img1024.npz',
                      'mnastnet': 'mnasnet_a1_batch_128.npz',
                      'retinanet': 'retinanet.4x4.bf16.performance.npz',
                      'shapemask': 'shapemask.4x4.fp32.npz',
                      'transformer': 'transformer_tf2_dynamic_shape.npz',
                      'bert': 'bert_pretraining.2x2.fp16.npz',
                      'efficientnet': 'efficientnet_b7_eval_batch_1.npz',
                      }
        file_name = file_name[graph_name]
        dataset = dict(np.load(osp.join(path, 'layout/xla/default/train/' + file_name)))
        self.data = Data(x=torch.tensor(dataset['node_feat']), edge_index=torch.tensor(dataset['edge_index']).T)
        self.root = path+'-'+graph_name


class TreeLoopGraph(InMemoryDataset):
    def __init__(self, path, depth, max_degree):
        super(TreeLoopGraph, self).__init__()
        file_name = "tree-loop_depth%d_max-degree_%d.npy" % (depth, max_degree)
        edges = np.load(osp.join(path, file_name))
        labels = np.load(osp.join(path, 'y_'+file_name))
        n = edges.max()+1
        self.root = path+'_depth%d_d%d' % (depth, max_degree)
        self.data = Data(x=torch.zeros([n, 1]), edge_index=torch.tensor(edges), y=torch.tensor(labels))


class ScaleFreeGraph(InMemoryDataset):
    def __init__(self, path, n, alpha, beta, gamma):
        super(ScaleFreeGraph, self).__init__()
        file_name = "sf_n%d_a%.4f_b%.4f_c%.4f.npy" % (n, alpha, beta, gamma)
        edges = np.load(osp.join(path, file_name))
        self.root = path + '_n%d_a%.4f_b%.4f_c%.4f' % (n, alpha, beta, gamma)
        edges = torch.tensor(edges)
        # remove self loops if any
        edges = remove_self_loops(edges)[0]
        self.data = Data(x=torch.zeros([n, 1]), edge_index=edges)


class FIG(InMemoryDataset):
    def __init__(self, root):
        super(FIG, self).__init__()
        self.root = root
        data_name = root[root.find('dataset')+len('dataset')+1:]
        #data_name = root.split('\\')[-1]
        file_path = osp.join(root, data_name+'.txt')
        # read raw data
        edges = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line = line.split(' ')
                src, dst = int(line[0]), int(line[1])
                edges.append([src, dst])
        edges = torch.tensor(edges)
        _, edges = torch.unique(edges, return_inverse=True)
        n = edges.max()+1
        edges = edges.T
        # remove self loops if any
        edges = remove_self_loops(edges)[0]
        self.data = Data(x=torch.zeros([n, 1]), edge_index=edges)


class BIO(InMemoryDataset):
    def __init__(self, root, name):
        super(BIO, self).__init__()
        file_name = {'bio_drug-target': 'ChG-Miner_miner-chem-gene.tsv',
                     'bio_drug-target2': 'ChG-TargetDecagon_targets.csv',
                     'bio_disease-gene': 'DG-AssocMiner_miner-disease-gene.tsv',
                     'bio_protein-tissue-function': 'TFG-Ohmnet_tissue-function-gene.tsv',
                     'bio_drug-drug': 'ChCh-Miner_durgbank-chem-chem.tsv',
                     'bio_function-function': 'FF-Miner_miner-func-func.tsv',
                     'bio_protein-protein': 'PP-Pathways_ppi.csv',
                     'bio_protein-protein2': 'PP-Decagon_ppi.csv',
                     'snap-wiki-vote': 'Wiki-Vote.txt',
                     'snap-cite-hep': 'Cit-HepTh.txt',
                     'snap-epinion': 'soc-Epinions1.txt',
                     'snap-peer': 'p2p-Gnutella04.txt'}
        file_name = file_name[name]
        self.root = root
        file_path = osp.join(root, file_name)
        edges = []
        entity2node = {}
        entity_count = 0
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if '#' in row[0]:
                    continue # skip title row
                if len(row) == 1: # in case where edges are not separated
                    row = row[0].split(',') if ',' in row[0] else row[0].split(' ')
                for entity in [row[0], row[-1]]: # row[1:-1] are considered as edge features
                    if entity not in entity2node.keys():
                        entity2node[entity] = entity_count
                        entity_count += 1
                edges.append([entity2node[row[0]], entity2node[row[-1]]])
        edges = torch.tensor(edges)
        _, edges = torch.unique(edges, return_inverse=True)
        n = edges.max() + 1
        edges = edges.T
        # remove self loops if any
        edges = remove_self_loops(edges)[0]
        self.data = Data(x=torch.zeros([n, 1]), edge_index=edges)


class SR(InMemoryDataset):
    def __init__(self, root):
        super(SR, self).__init__()
        self.root = root
        file_path = osp.join(root, 'edge.csv')
        edges = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line = line.split(',')
                src, dst = int(line[0]), int(line[1])
                edges.append([src, dst])
        edges = torch.tensor(edges)
        _, edges = torch.unique(edges, return_inverse=True)
        n = edges.max() + 1
        edges = edges.T
        # remove self loops if any
        edges = remove_self_loops(edges)[0]
        self.data = Data(x=torch.zeros([n, 1]), edge_index=edges)







def random_er_graph(nb_nodes, p=0.5, directed=True, acyclic=False,
                    weighted=False, low=0.1, high=1.0, type='er'):
    """Random Erdos-Renyi graph."""
    if type == 'er':
        mat = np.random.binomial(1, p, size=(nb_nodes, nb_nodes))
    elif type == 'er_single':
        mat = np.random.binomial(1, p, size=(nb_nodes, nb_nodes))
        flip = np.random.binomial(1, 0.5, size=(nb_nodes, nb_nodes))
        flip_u, flip_l =  np.triu(flip), np.tril(((flip - 1)*-1).T)
        flip = flip_u+flip_l
        mat = mat * flip
    # remove self loop
    mat[np.arange(nb_nodes), np.arange(nb_nodes)] = 0.
    if not directed:
        mat *= np.transpose(mat)
    elif acyclic:
        mat = np.triu(mat, k=1)
        p = np.random.permutation(nb_nodes)  # To allow nontrivial solutions
        mat = mat[p, :][:, p]
    if weighted:
        weights = np.random.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
        if not directed:
            weights *= np.transpose(weights)
            weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
        mat = mat.astype(float) * weights
    return mat


