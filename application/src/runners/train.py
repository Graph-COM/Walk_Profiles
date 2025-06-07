"""
training functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import get_num_samples


def get_train_func(args):
    if args.use_no_subgraph or args.save_no_subgraph:
        train_func = train
    else:
        train_func = train_subgraph
    return train_func


def logistic_regression_solver(train_loader, args, device, emb=None):
    data = train_loader.dataset
    # training
    z = data.z
    #z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
    z = np.array(z)
    # node_features = data.x[train_mask]
    y = np.array(data.labels)
    clf = LogisticRegression(penalty=args.penalty, max_iter=3000, C=args.penalty_c).fit(z, y)

    pred = clf.predict_proba(z)
    n_classes = y.max() + 1
    delta = 1e-9
    loss = [np.log(pred[y == i][:, i] + delta).sum() for i in range(n_classes)]
    loss = -sum(loss) / pred.shape[0]

    return clf, loss


def linear_regression_solver(train_loader, train_eval_loader, val_loader, test_loader, args, device, emb=None):
    data = train_loader.dataset
    # training
    z = data.z
    z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
    #node_features = data.x[train_mask]
    y = torch.tensor(data.labels)
    optimal_weight = torch.linalg.pinv(z.T @ z) @ z.T @ y.float()
    # test
    rmse = []
    for loader in [train_eval_loader, val_loader, test_loader]:
        data = loader.dataset
        z = data.z
        z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
        y = torch.tensor(data.labels)
        yhat = z @ optimal_weight
        rmse.append(((yhat - y) ** 2).mean().sqrt().item())

    return rmse


def train(model, optimizer, train_loader, args, device, emb=None):
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        curr_links = links[indices]
        node_features = data.x[curr_links].to(device)
        node_features = node_features.flatten(1) # link_repr = (src_repr, dst_repr)
        if args.node_label is not None:
            z_indices = sample_indices[indices]
            z = data.z[z_indices].to(device)
        else:
            z = None
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(node_features, z)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

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


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    else:
        raise NotImplementedError
    return loss
