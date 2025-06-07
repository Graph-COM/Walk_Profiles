"""
training functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.utils import get_num_samples


def get_train_func(args):
    if args.use_no_subgraph or args.save_no_subgraph:
        train_func = train
    else:
        train_func = train_subgraph
    return train_func

class heuristic_model:
    def __init__(self, walk_length, node_label):
        self.walk_length = walk_length
        self.node_label = node_label

    def predict_proba(self, z):
        if self.node_label == 'wp':
            z0 = z[:, -self.walk_length:-self.walk_length+1]  # take the entry of one-backward-edge walks
            pred_proba = (z0 == 0).astype(np.float_)
        elif self.node_label == 'mw':
            z0 = z[:, 0:1]
            pred_proba = (z0 <= 0).astype(np.float_)
        elif self.node_label == 'rw':
            pred_proba = (z == 0).astype(np.float_)
        elif self.node_label == 'rw+':
            z0 = z[:, 0:1]
            pred_proba = (z0 == 0).astype(np.float_)
        return np.concatenate((pred_proba, 1-pred_proba), axis=-1)


def linear_regression_solver(train_loader, args):
    data = train_loader.dataset
    # training
    train_mask = data.train_mask
    z = data.z[train_mask]
    z = np.array(z)
    #z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
    #node_features = data.x[train_mask]
    y = np.array(data.y[train_mask])
    model = LinearRegression().fit(z, y)
    #optimal_weight = torch.linalg.pinv(z.T @ z) @ z.T @ y.float()
    return model

def logistic_regression_solver(train_loader, args, device, emb=None):
    data = train_loader.dataset
    # training
    train_mask = data.train_mask
    z = data.z[train_mask]
    #z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
    z = np.array(z)
    # hard-coded normalization
    #if z.shape[-1] > 1:
        #nonzero_entry = np.linalg.norm(z, axis=-1) !=0
        #z[nonzero_entry] = z[nonzero_entry] / np.linalg.norm(z, axis=-1)[nonzero_entry, None]
    # node_features = data.x[train_mask]
    y = np.array(data.y[train_mask])

    if args.wp_heuristic:
        clf = heuristic_model(args.max_dist, args.node_label)
    else:
        solver = 'lbfgs' if not args.penalty=='l1' else 'liblinear'
        clf = LogisticRegression(solver=solver,penalty=args.penalty, max_iter=10000, C=args.penalty_c).fit(z, y)

    pred = clf.predict_proba(z)
    n_classes = y.max() + 1
    delta = 1e-9
    loss = [np.log(pred[y==i][:, i]+delta).sum() for i in range(n_classes)]
    loss = -sum(loss) / pred.shape[0]
    return clf, loss



def train(model, optimizer, train_loader, args, device, emb=None):
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    labels = torch.tensor(data.y)
    # find training node ids
    node_ids = torch.tensor([i for i in range(data.x.size(0))])
    train_mask = data.train_mask
    train_nodes = node_ids[train_mask]

    batch_processing_times = []
    loader = DataLoader(train_nodes, args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        node_features = data.x[indices].to(device)
        if args.node_label is not None:
            z = data.z[indices].to(device)
        else:
            z = None
        start_time = time.time()
        optimizer.zero_grad()
        pred = model(node_features, z)
        loss = get_loss(args.loss)(pred, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)

    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def train_subgraph(model, optimizer, train_loader, args, device, emb=None):
    """
    Adapted version of the SEAL training function
    :param model:
    :param optimizer:
    :param train_loader:
    :param args:
    :param device:
    :param emb:
    :return:
    """

    print('starting training')
    t0 = time.time()
    model.train()
    if args.dynamic_train:
        train_samples = get_num_samples(args.train_samples, len(train_loader.dataset))
    else:
        train_samples = inf
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    for batch_count, data in enumerate(pbar):
        start_time = time.time()
        optimizer.zero_grad()
        # todo this loop should no longer be hit as this function isn't called for BUDDY
        #if args.model == 'BUDDY':
            #data_dev = [elem.squeeze().to(device) for elem in data]
            #logits = model(*data_dev[:-1])  # everything but the labels
            #loss = get_loss(args.loss)(logits, data[-1].squeeze(0).to(device))
        #else:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.src_degree,
                       data.dst_degree)
        loss = get_loss(args.loss)(logits, data.y)
        #if args.l1 > 0:
            #l1_reg = torch.tensor(0, dtype=torch.float)
            #lin_params = torch.cat([x.view(-1) for x in model.lin.parameters()])
            #for param in lin_params:
                #l1_reg += torch.norm(param, 1) ** 2
            #loss = loss + args.l1 * l1_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        del data
        torch.cuda.empty_cache()
        batch_processing_times.append(time.time() - start_time)
        if (batch_count + 1) * args.batch_size > train_samples:
            break
    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')
    if args.model in {'linear', 'pmi', 'ra', 'aa', 'one_layer'}:
        model.print_params()

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def auc_loss(logits, y, num_neg=1):
    pos_out = logits[y == 1]
    neg_out = logits[y == 0]
    # hack, should really pair negative and positives in the training set
    if len(neg_out) <= len(pos_out):
        pos_out = pos_out[:len(neg_out)]
    else:
        neg_out = neg_out[:len(pos_out)]
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def bce_loss(logits, y, num_neg=1):
    return BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))


def mse_loss(pred, y):
    return MSELoss()(pred.view(-1), y.to(torch.float))


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    elif loss_str == 'l2':
        loss = mse_loss
    else:
        raise NotImplementedError
    return loss
