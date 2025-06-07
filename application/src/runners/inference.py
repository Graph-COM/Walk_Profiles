"""
testing / inference functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from src.evaluation import evaluate_auc, evaluate_hits, evaluate_mrr, evaluate_acc
from src.utils import get_num_samples


def get_test_func(use_link_features_only):
    if use_link_features_only:
        return get_preds
    else:
        return get_preds_subgraph

@torch.no_grad()
def test(model, evaluator, train_loader, val_loader, test_loader, args, device, emb=None, eval_metric='hits'):
    print('starting testing')
    t0 = time.time()
    model.eval()
    print("get train predictions")
    test_func = get_test_func(args.use_no_subgraph or args.save_no_subgraph)
    pos_train_pred, neg_train_pred, train_pred, train_true = test_func(model, train_loader, device, args, split='train')
    print("get val predictions")
    pos_val_pred, neg_val_pred, val_pred, val_true = test_func(model, val_loader, device, args, split='val')
    print("get test predictions")
    pos_test_pred, neg_test_pred, test_pred, test_true = test_func(model, test_loader, device, args, split='test')

    results = {}
    eval_metric = eval_metric.split(',')
    for key in eval_metric:
        if key == 'hits':
            res = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                    neg_test_pred, Ks=[args.K])
        elif key == 'mrr':
            res = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                   neg_test_pred)
        elif key == 'auc':
            res = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
        elif key == 'acc':
            res = evaluate_acc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
        results.update(res)

    print(f'testing ran in {time.time() - t0}')

    return results


@torch.no_grad()
def get_preds_subgraph(model, loader, device, args, emb=None, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    y_pred, y_true = [], []
    pbar = tqdm(loader, ncols=70)
    if args.wandb:
        wandb.log({f"inference_{split}_total_batches": len(loader)})
    batch_processing_times = []
    t0 = time.time()
    for batch_count, data in enumerate(pbar):
        start_time = time.time()
        # todo this should not get hit, refactor out the if statement
        if args.model == 'BUDDY':
            data_dev = [elem.squeeze().to(device) for elem in data]
            logits = model(*data_dev[:-1])
            y_true.append(data[-1].view(-1).cpu().to(torch.float))
        else:
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.src_degree,
                           data.dst_degree)
            y_true.append(data.y.view(-1).cpu().to(torch.float))
        y_pred.append(logits.view(-1).cpu())
        batch_processing_times.append(time.time() - start_time)
        if (batch_count + 1) * args.batch_size > n_samples:
            del data
            torch.cuda.empty_cache()
            break
        del data
        torch.cuda.empty_cache()
    if args.wandb:
        wandb.log({f"inference_{split}_batch_time": np.mean(batch_processing_times)})
        wandb.log({f"inference_{split}_epoch_time": time.time() - t0})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = pred[true == 1]
    neg_pred = pred[true == 0]
    samples_used = len(loader.dataset) if n_samples > len(loader.dataset) else n_samples
    print(f'{len(pos_pred)} positives and {len(neg_pred)} negatives for sample of {samples_used} edges')
    return pos_pred, neg_pred, pred, true


@torch.no_grad()
def get_preds(model, loader, device, args, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    for batch_count, indices in enumerate(tqdm(loader)):
        curr_links = links[indices]
        node_features = data.x[curr_links].to(device)
        node_features = node_features.flatten(1) # link_repr = (src_repr, dst_repr)
        if args.node_label is not None:
            z = data.z[indices].to(device)
        else:
            z = None
        logits = model(node_features, z)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    if args.wandb:
        wandb.log({f"inference_{split}_epoch_time": time.time() - t0})
    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred, pred, labels


def get_split_samples(split, args, dataset_len):
    """
    get the
    :param split: train, val, test
    :param args: Namespace object
    :param dataset_len: total size of dataset
    :return:
    """
    samples = dataset_len
    if split == 'train':
        if args.dynamic_train:
            samples = get_num_samples(args.train_samples, dataset_len)
    elif split in {'val', 'valid'}:
        if args.dynamic_val:
            samples = get_num_samples(args.val_samples, dataset_len)
    elif split == 'test':
        if args.dynamic_test:
            samples = get_num_samples(args.test_samples, dataset_len)
    else:
        raise NotImplementedError(f'split: {split} is not a valid split')
    return samples


@torch.no_grad()
def get_elph_preds(model, loader, device, args, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(e(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    # get node features
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
    for batch_count, indices in enumerate(tqdm(loader)):
        curr_links = links[indices].to(device)
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    if args.wandb:
        wandb.log({f"inference_{split}_epoch_time": time.time() - t0})
    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred, pred, labels


def test_logistic_regression(model, evaluator, train_loader, val_loader, test_loader, args, eval_metric):
    pos_train_pred, neg_train_pred, train_pred, train_true = get_pred_logistic_regression(model, train_loader)
    pos_val_pred, neg_val_pred, val_pred, val_true = get_pred_logistic_regression(model, val_loader)
    pos_test_pred, neg_test_pred, test_pred, test_true = get_pred_logistic_regression(model, test_loader)
    results = {}
    eval_metric = eval_metric.split(',')
    for key in eval_metric:
        if key == 'hits':
            res = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                neg_test_pred, Ks=[args.K])
        elif key == 'mrr':
            res = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
        elif key == 'auc':
            res = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
        elif key == 'acc':
            res = evaluate_acc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
        results.update(res)

    return results

def get_pred_logistic_regression(model, loader):
    data = loader.dataset
    # training
    z = data.z
    #z = torch.cat([z, torch.ones(z.size(0), 1)], dim=-1)
    z = np.array(z)
    # node_features = data.x[train_mask]
    labels = np.array(data.labels)
    pred = model.predict_proba(z)
    n_classes = pred.shape[-1]
    if n_classes == 2:
        pos_pred = pred[labels == 1, 1]
        neg_pred = pred[labels == 0, 1]
        pred = pred[:, 1]
        return torch.tensor(pos_pred), torch.tensor(neg_pred), torch.tensor(pred), torch.tensor(labels)
    else:
        return None, None, torch.tensor(pred), torch.tensor(labels)


