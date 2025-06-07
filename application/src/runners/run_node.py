"""
main module
"""
import argparse
import time
import warnings
from math import inf
import sys
import random

sys.path.insert(0, '..')

import numpy as np
import torch
from ogb.linkproppred import Evaluator

torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data_node import get_data, get_loaders
from src.wandb_setup import initialise_wandb
from src.runners.train_node import get_train_func, linear_regression_solver, logistic_regression_solver
from src.runners.inference_node import test, test_linear_regression, test_logistic_regression


def print_results_list(results_list):
    for idx, res in enumerate(results_list):
        print(f'repetition {idx}: test {res[0]:.2f}, val {res[1]:.2f}, train {res[2]:.2f}')

def set_seed(seed):
    """
    setting a random seed for reproducibility and in accordance with OGB rules
    @param seed: an integer seed
    @return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def logistic_regression(args):
    args = initialise_wandb(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    average_acc = [[], [], []]
    for rep in range(args.reps):
        print(f'running repetition {rep}')
        args.seed = rep
        # args.seed = 233
        set_seed(rep)
        dataset = get_data(args)
        eval_metric = args.eval_metric
        train_loader = get_loaders(args, dataset)
        # transform wp to mw
        if args.wp2mw:
            from test.test_mw2wp import realwp2realmw, realmw2realwp
            mw = realwp2realmw((train_loader.dataset.z).to(torch.float64), args.max_dist)
            if args.q_dim == 1:
                mw = mw[..., 1:2]
            else:
                mw = mw[..., :args.q_dim]
            train_loader.dataset.z = mw
        if args.mw2wp:
            from test.test_mw2wp import realwp2realmw, realmw2realwp, realmw2realwp_reconstruct
            wp = realmw2realwp_reconstruct((train_loader.dataset.z).to(torch.float64), args.max_dist, normalize=args.norm)
            train_loader.dataset.z = wp

        # feature normalization stage
        if args.x_norm:
            z_train = train_loader.dataset.z[train_loader.dataset.train_mask]
            #mw2wp(z_train, args.max_dist)
            #mean, std = z_train.float().mean(axis=0), z_train.float().std(axis=0)
            mean, std = z_train.mean(axis=0), z_train.std(axis=0)
            #maximum = z_train.abs().max(axis=0)[0]
            ind = (std!=0) # only apply to non-constant features
            train_loader.dataset.z[:, ind] = (train_loader.dataset.z - mean)[:, ind]/std[ind]
            #train_loader.dataset.z[:, ind] = (train_loader.dataset.z)/maximum
        # logistic regression
        model, loss = logistic_regression_solver(train_loader, args, device)
        results = test_logistic_regression(model, eval_metric, train_loader, args, eval_metric)
        for key, result in results.items():
            train_res, tmp_val_res, tmp_test_res = result
            res_dic = {f'rep{rep}_Loss':loss,f'rep{rep}_Train' + key: 100 * train_res,
                       f'rep{rep}_Val' + key: 100 * tmp_val_res,
                       f'rep{rep}_Test' + key: 100 * tmp_test_res,}
            average_acc[0].append(res_dic['rep%d_TestAUC' % rep])
            average_acc[1].append(res_dic['rep%d_Loss' % rep])
            if args.wandb:
                wandb.log(res_dic)
            else:
                for key, value in res_dic.items():
                    print(key + ':%.4f' % value)

        #average_acc[0].append(results['ACC'][0])
        #average_acc[1].append(results['ACC'][1])
        #average_acc[2].append(results['ACC'][2])
    #print(np.array(average_acc[0]).mean(), np.array(average_acc[0]).std())
    #print(np.array(average_acc[0]).mean(), np.array(average_acc[0]).std())



if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    #parser.add_argument('--dataset_name', type=str, default='Cora',
                        #choices=['Cora', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 #'ogbl-citation2'])
    #parser.add_argument('--dataset_name', type=str, default='Cora_ml', choices=['Cora_ml', 'Citeseer', 'WikiCS',
                                                                                #'Bitcoin_alpha', 'Bitcoin_OTC',
                                                                                #'Squirrel', 'Chameleon','Blog','ADO', 'ATC',
                                                                                #'CELE', 'EMA', 'FIG', 'HIG',
                                                                                #'PB', 'USA', 'bio_drug-drug',
                                                                                #'bio_drug-target', 'bio_drug-target2',
                                                                                #'bio_disease-gene', 'bio_protein-tissue-function',
                                                                                #'bio_function-function', 'bio_protein-protein',
                                                                                #'bio_protein-protein2', 'snap-wiki-vote',
                                                                                #'snap-cite-hep', 'snap-epinion', 'snap-peer'])
    parser.add_argument('--dataset_name')
    parser.add_argument('--val_pct', type=float, default=0.1)
    parser.add_argument('--test_pct', type=float, default=0.2)
    parser.add_argument('--train_samples', type=float, default=inf)
    parser.add_argument('--val_samples', type=float, default=inf)
    parser.add_argument('--test_samples', type=float, default=inf)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--directed', action='store_true', default=True)
    parser.add_argument('--target', type=str, default=None, help="re, common, cp, fwl, walk_3, cycle101")
    parser.add_argument('--task', default='node', choices=['node'])
    parser.add_argument('--binarize_target', action='store_true', default=False, help='for motif counting: if do detection task instead of counting')
    parser.add_argument('--random_n', default=3000, type=int, help='size of random graph')
    parser.add_argument('--random_deg', default=5.0000, type=float, help='ave degree of random graph')
    parser.add_argument('--random_depth', default=30, type=int, help='depth of random tree')
    parser.add_argument('--alpha', default=0.75, type=float, help='parameter of scale-free random graphs')
    parser.add_argument('--beta', default=0.2, type=float, help='parameter of scale-free random graphs')
    parser.add_argument('--gamma', default=0.05, type=float, help='parameter of scale-free random graphs')
    # GNN settings
    parser.add_argument('--model', type=str, default='LINEAR')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl',
                        choices=['zero','degree', 'degree+', 'cn', 'cn+', 'de', 'drnl', 'rw', 'rw+', 'mw', 'wp',
                                 'katz', 'katz+', 'ppr', 'ppr+'])
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--use_no_subgraph', action='store_true', default=False, help='set False if using subgraph-based label/model')
    parser.add_argument('--save_no_subgraph', action='store_true', default=True, help='set True to not save subgraph for subgraph-based label/model')
    # magnetic walks settings
    parser.add_argument('--q', type=float, default=None)
    parser.add_argument('--q_dim', type=int, default=None)
    parser.add_argument('--entry', type=str, default='ss', choices=['ss', 's'])
    parser.add_argument('--nbt', action='store_true', default=False)
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--compact_q', action='store_true', default=False)
    # Training settings
    parser.add_argument('--penalty', default='none', choices=['none','l2', 'l1'])
    parser.add_argument('--penalty_c', default=0.01, type=float)
    parser.add_argument('--wp_heuristic', action='store_true', default=False)
    parser.add_argument('--x_norm', action='store_true', default=False)
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_metric', type=str, default='rmse')

    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="wp-detection", type=str)
    parser.add_argument('--wandb_project', default="wp-detection", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
                        help='list of epochs to log gradient flow')
    parser.add_argument('--log_features', action='store_true', help="log feature importance")
    parser.add_argument('--wp2mw', action='store_true')
    parser.add_argument('--mw2wp', action='store_true')
    args = parser.parse_args()
    print(args)

    # cpu threads control
    torch.set_num_threads(10)
    import os

    torch.set_default_dtype(torch.float64)
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10"
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["TBB_NUM_THREADS"] = "10"


    logistic_regression(args)
