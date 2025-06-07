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

torch.set_printoptions(precision=4)
import wandb
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_loaders
from src.wandb_setup import initialise_wandb
from src.runners.train import get_train_func, logistic_regression_solver
from src.runners.inference import test, test_logistic_regression

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
    for rep in range(args.reps):
        print(f'running repetition {rep}')
        args.seed = rep
        # args.seed = 233
        set_seed(rep)
        dataset, splits, directed, eval_metric = get_data(args)
        eval_metric = args.eval_metric
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        # logistic regression
        model, loss = logistic_regression_solver(train_loader, args, device)
        results = test_logistic_regression(model, eval_metric, train_eval_loader, val_loader, test_loader, args, eval_metric)
        for key, result in results.items():
            train_res, tmp_val_res, tmp_test_res = result
            res_dic = {f'rep{rep}_Loss':loss,f'rep{rep}_Train' + key: 100 * train_res,
                       f'rep{rep}_Val' + key: 100 * tmp_val_res,
                       f'rep{rep}_Test' + key: 100 * tmp_test_res,}
            if args.wandb:
                wandb.log(res_dic)
            print(key)
    if args.wandb:
        wandb.finish()



if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Cora_ml', choices=['Cora_ml', 'Citeseer', 'WikiCS',
                                                                                'Bitcoin_alpha', 'Bitcoin_OTC',
                                                                                'Squirrel', 'Chameleon','Blog','ADO', 'ATC',
                                                                                'CELE', 'EMA', 'FIG', 'HIG',
                                                                                'PB', 'USA', 'bio_drug-drug',
                                                                                'bio_drug-target', 'bio_drug-target2',
                                                                                'bio_disease-gene', 'bio_protein-tissue-function',
                                                                                'bio_function-function', 'bio_protein-protein',
                                                                                'bio_protein-protein2', 'snap-wiki-vote',
                                                                                'snap-cite-hep', 'snap-epinion', 'snap-peer'])
    parser.add_argument('--val_pct', type=float, default=0.1)
    parser.add_argument('--test_pct', type=float, default=0.2)
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--directed', action='store_true', default=True)
    parser.add_argument('--neg_sample', type=str, default='biased-structured_st-third', help='apply which type of negative sampling',
                        choices=['biased', 'biased-random-half', 'biased-random', 'random', 'biased-structured_s',
                                 'biased-structured_t', 'biased-structured_st', 'biased-structured_t-half',
                                 'biased-structured_st-third'])
    parser.add_argument('--task', default='link', choices=['link'])
    # GNN settings
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=100)
    parser.add_argument('--node_label', type=str, default='wp',
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
    parser.add_argument('--entry', type=str, default='st', choices=['st', 'st-ts', 'ss-st-ts-tt', 'ss-tt'])
    parser.add_argument('--nbt', action='store_true', default=False)
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--compact_q', action='store_true', default=False)
    # Training settings
    parser.add_argument('--penalty', default=None, choices=['l2'])
    parser.add_argument('--penalty_c', default=0.01, type=float)
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_metric', type=str, default='auc,acc')
    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="link-prediction", type=str)
    parser.add_argument('--wandb_project', default="link-prediction", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
                        help='list of epochs to log gradient flow')
    parser.add_argument('--log_features', action='store_true', help="log feature importance")
    args = parser.parse_args()
    print(args)
    logistic_regression(args)

