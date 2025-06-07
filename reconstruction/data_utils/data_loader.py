import numpy as np
import torch
from torch_geometric_signed_directed.data import load_directed_real_data
from data_utils.code2_dataset import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
from torch_geometric.data import Data
from data_utils.dataset import FIG, BIO

def load_dataset(name, path='./data'):
    # random graphs
    if name.startswith('er') or name.startswith('tree') or name.startswith('cycle') or name.startswith('clique'):
        edge_index = np.load(osp.join(path, 'random_graphs/'+name+'.npy'))
        dataset = Data(edge_index=torch.tensor(edge_index))
    elif name.startswith('grow'):
        graph_id = int(name[-1])
        edge_index = np.load(osp.join(path, name[:-2]+'/graph_%d.npy' % graph_id))
        dataset = Data(edge_index=torch.tensor(edge_index))
    elif name.startswith('scdg'):
        loaded_data = np.load(osp.join(path, 'spatial_graphs/' + name + '.npz'))
        edge_index = loaded_data['edge_index']
        groups = loaded_data['groups']
        dataset = Data(edge_index=torch.tensor(edge_index), x=torch.tensor(groups))
    # social networks
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = load_directed_real_data(dataset='webkb',
                                root=path, name=name)
    #elif name in ['cora_ml', 'citeseer', 'wikics', 'blog', 'wikitalk', 'migration', 'telegram', 'Slashdot']:
    elif name in ['cora_ml', 'citeseer']:
        path = osp.join(path, name)
        dataset = load_directed_real_data(dataset=name, root=path)
    # program graphs
    elif name.startswith('code2'):
        dataset = PygGraphPropPredDataset(name="ogbg-code2", root=path)
        #dataset = dataset[10] # over 200 nodes and edges
        #dataset = dataset[6851] # over 2k nodes and edges
        if name == 'code2-9k':
            dataset = dataset[261575]
        elif name == 'code2-20k':
            dataset = dataset[267936]
        elif name == 'code2-36k':
            dataset = dataset[84806]
        else:
            dataset = dataset[10] # over 200 nodes and edges
    elif name.startswith('ogb'):
        dataset = PygNodePropPredDataset(name=name, root=path)

    # computational graphs
    elif name in ['alexnet', 'resnet', 'mask_rcnn', 'transformer', 'bert', 'retinanet', 'efficientnet', 'mnastnet',
                  'shapemask']:
        #model_name = 'transformer.2x2.fp32.npz'
        model_name = {'alexnet':'alexnet_train_batch_32.npz',
                    'resnet':'resnet50.8x8.fp16.npz',
                    'mask_rcnn':'mask_rcnn_batch_16_bf16_img1024.npz',
                    'mnastnet': 'mnasnet_a1_batch_128.npz',
                    'retinanet': 'retinanet.4x4.bf16.performance.npz',
                    'shapemask': 'shapemask.4x4.fp32.npz',
                    'transformer':'transformer_tf2_dynamic_shape.npz',
                    'bert': 'bert_pretraining.2x2.fp16.npz',
                    'efficientnet': 'efficientnet_b7_eval_batch_1.npz',
                    }
        model_name = model_name[name]
        dataset = dict(np.load(osp.join(path, 'npz/layout/xla/default/train/'+model_name)))

    elif name in ['ADO', 'ATC', 'CELE', 'EMA', 'FIG', 'HIG', 'PB', 'USA']:
        dataset = FIG(root=osp.join(path, name))
    elif name.startswith('bio') or name.startswith('snap'):
        dataset = BIO(root=path, name=name)
    else:
        raise Exception("Unknown dataset name: %s!" % name)
    return dataset