import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import degree
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg
import pickle


def get_degree(data):
   data.degree = 1. / torch.sqrt(1 + degree(data.edge_index[0], data.num_nodes))
   return data

def symmetrize_transform(data):
    data.edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=-1)
    return data

def bidirect_transform(data):
    num_edges = data.edge_index.size(1)
    data.edge_attr = torch.cat([torch.zeros([num_edges]), torch.ones([num_edges])], dim=0).to(data.edge_index.device).int()
    data.edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=-1)
    return data

def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''
    # print(data)
    # print(data_new)
    ##### AST edge
    data.edge_index_origin = data.edge_index
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim = 0)
    edge_attr_ast_inverse = torch.cat([torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim = 1)


    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1,) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)


    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))


    data.edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim = 1)
    data.edge_attr = torch.cat([edge_attr_ast,   edge_attr_ast_inverse, edge_attr_nextoken,  edge_attr_nextoken_inverse], dim = 0)

    return data



class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, pre_filter=None,
                 meta_dict=None, processed_suffix=''):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  ## original name, e.g., ogbg-molhiv
        self.raw = os.path.join(root, '_'.join(name.split('-')))
        self.processed_suffix = processed_suffix
        self.pre_filter = pre_filter

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0,
                                 keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
        not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'


        #full_id = self.get_idx_split()
        #self.train_id, self.val_id, self.test_id = full_id['train'].tolist(), full_id['valid'].tolist(), full_id['test'].tolist()

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform, pre_filter=pre_filter)

        # load train/val/test id
        all_id = self.get_idx_split()
        train_id, val_id, test_id = all_id['train'], all_id['valid'], all_id['test']
        if pre_filter is not None:
            with open(osp.join(self.processed_dir, 'split.pickle'), 'rb') as handle:
                filter_id = pickle.load(handle)
            filter_id = torch.tensor(filter_id)
            id_mapping = []
            count = 0
            for flag in filter_id:
                if flag:
                    id_mapping.append(count)
                    count += 1
                else:
                    id_mapping.append(-1)
            id_mapping = torch.tensor(id_mapping)
            self.train_id = id_mapping[train_id]
            self.val_id = id_mapping[val_id]
            self.test_id = id_mapping[test_id]
            self.train_id = self.train_id[self.train_id != -1]
            self.val_id = self.val_id[self.val_id != -1]
            self.test_id = self.test_id[self.test_id != -1]
        else:
            self.train_id, self.val_id, self.test_id = train_id, val_id, test_id

        self.data, self.slices = torch.load(self.processed_paths[0])


    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_dir(self):
        return os.path.join(self.raw, 'processed' + self.processed_suffix)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                   additional_node_files=additional_node_files,
                                   additional_edge_files=additional_edge_files, binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                                header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                          header=None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_filter is not None:
            print('pre-filtering...')
            data_list_filter = []
            id_filter = []
            count = 0
            for i, data in enumerate(data_list):
                if i % 5000 == 0:
                    print('pre-filtering: %d/%d' % (i, len(data_list)))
                if self.pre_filter(data):
                    data_list_filter.append(data)
                    id_filter.append(True)
                    count += 1
                else:
                    id_filter.append(False)
            data_list = data_list_filter
            #data_list = [data for data in data_list if self.pre_filter(data)]
            print('pre-filtering finished, num of data left %d' % len(data_list))

            # save filtered idx
            with open(osp.join(self.processed_dir, 'split.pickle'), 'wb') as handle:
                pickle.dump(id_filter, handle)

        if self.pre_transform is not None:
            print('pre-transforming dataset...')
            data_list_new = []
            for i, data in enumerate(data_list):
                if i % 5000 == 0:
                    print('pre-transforming: %d/%d' % (i, len(data_list)))
                data_list_new.append(self.pre_transform(data))
            #data_list = [self.pre_transform(data) for data in data_list]
            data_list = data_list_new

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # pyg_dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba')
    # print(pyg_dataset.num_classes)
    # split_index = pyg_dataset.get_idx_split()
    # print(pyg_dataset)
    # print(pyg_dataset[0])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].y.dtype)
    # print(pyg_dataset[0].edge_index)
    # print(pyg_dataset[split_index['train']])
    # print(pyg_dataset[split_index['valid']])
    # print(pyg_dataset[split_index['test']])

    pyg_dataset = PygGraphPropPredDataset(name='ogbg-code2')
    print(pyg_dataset.num_classes)
    split_index = pyg_dataset.get_idx_split()
    print(pyg_dataset[0])
    # print(pyg_dataset[0].node_is_attributed)
    print([pyg_dataset[i].x[1] for i in range(100)])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].edge_index)
    print(pyg_dataset[split_index['train']])
    print(pyg_dataset[split_index['valid']])
    print(pyg_dataset[split_index['test']])

    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(pyg_dataset, batch_size=32, shuffle=True)
    # for batch in loader:
    #     print(batch)
    #     print(batch.y)
    #     print(len(batch.y))

    #     exit(-1)


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data
    '''

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]],
                        dtype=torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''

    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1,
                                 as_tuple=False)  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


