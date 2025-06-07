"""
labelling tricks as described in
https://proceedings.neurips.cc/paper/2021/hash/4be49c79f233b4f4070794825c323733-Abstract.html
"""

import torch
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import diags
from src.walk_profile.magnetic_walk import compute_magnetic_walks
from src.walk_profile.walk_profile import compute_walk_profile


def is_labeling_discrete(method):
    if method in ['zero', 'degree', 'degree+', 'cn', 'cn+', 'drnl', 'de']:
        return True
    elif method in ['rw', 'rw+', 'mw', 'wp', 'katz', 'katz+', 'ppr', 'ppr+']:
        return False
    else:
        raise Exception('Unrecognized node labelling!')


def labeling_dim(args):
    # dimension of node labeling
    method = args.node_label
    factor = 1 if args.entry in ['st', 'ss', 's'] else 2
    if method in ['zero', 'degree', 'cn', 'drnl', 'katz', 'ppr']:
        z_dim = 1
    elif method == 'cn+':
        z_dim = 4
    elif method == 'degree+' or method == 'ppr+' or method == 'katz+':
        z_dim = 2
    elif method == 'rw':
        z_dim = factor * args.max_dist
    elif method == 'rw+':
        z_dim = 2 * factor * args.max_dist
    elif method == 'mw':
        z_dim = args.q_dim * args.max_dist if not args.compact_q else\
            sum([int(min(np.ceil(i/2)+1, args.q_dim)) for i in range(1, args.max_dist+1)])
        z_dim = 2 * factor * z_dim
    elif method == 'wp':
        wp_dim = sum([i + 1 for i in range(1, args.max_dist+1)])
        z_dim = factor * wp_dim
    else:
        raise Exception('Unrecognized node labelling!')
    if args.save_no_subgraph and not args.use_no_subgraph and args.task == 'link':
        # if only keep link features, dimension multiplies 2 as 2*node-level -> link-level
        z_dim = z_dim * 2
    return z_dim


def drnl_hash_function(dist2src, dist2dst):
    """
    mapping from source and destination distances to a single node label e.g. (1,1)->2, (1,2)->3
    @param dist2src: Int Tensor[edges] shortest graph distance to source node
    @param dist2dst: Int Tensor[edges] shortest graph distance to source node
    @return: Int Tensor[edges] of labels
    """
    dist = dist2src + dist2dst

    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    # the src and dst nodes always get a score of 1
    z[dist2src == 0] = 1
    z[dist2dst == 0] = 1
    return z


def get_drnl_lookup(max_dist, num_hops):
    """
    A lookup table from DRNL labels to index into a contiguous tensor. DRNL labels are not contiguous and this
    lookup table is used to index embedded labels
    """
    max_label = get_max_label('drnl', max_dist, num_hops)
    res_arr = [None] * (max_label + 1)
    res_arr[1] = (1, 0)
    for src in range(1, num_hops + 1):
        for dst in range(1, max_dist + 1):
            label = drnl_hash_function(torch.tensor([src]), torch.tensor([dst]))
            res_arr[label] = (src, dst)
    z_to_idx = {}
    idx_to_dst = {}
    counter = 0
    for idx, elem in enumerate(res_arr):
        if elem is not None:
            z_to_idx[idx] = counter
            idx_to_dst[counter] = (elem)
            counter += 1
    return z_to_idx, idx_to_dst


def get_max_label(method, max_dist, num_hops):
    if method in {'de', 'de+'}:
        max_label = max_dist
    elif method in {'drnl-', 'drnl'}:
        max_label = drnl_hash_function(torch.tensor([num_hops]), torch.tensor([max_dist])).item()
    else:
        raise NotImplementedError
    return max_label


def drnl_node_labeling(adj, src, dst, max_dist=100):
    """
    The heuristic proposed in "Link prediction based on graph neural networks". It is an integer value giving the 'distance'
    to the (src,dst) edge such that src = dst = 1, neighours of dst,src = 2 etc. It implements
    z = 1 + min(d_x, d_y) + (d//2)[d//2 + d%2 - 1] where d = d_x + d_y
    z is treated as a node label downstream. Even though the labels measures of distance from the central edge, they are treated as
    categorical objects and embedded in an embedding table of size max_z * hidden_dim
    @param adj:
    @param src:
    @param dst:
    @return:
    """
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)
    dist2src[dist2src > max_dist] = max_dist

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    dist2dst[dist2dst > max_dist] = max_dist

    z = drnl_hash_function(dist2src, dist2dst)
    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 1, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 1, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist

    return dist.to(torch.long)


def cn_node_labeling(adj, src, dst):
    # mark each node with the number of total common neighbors of src and dst
    cn_11 = torch.tensor((adj @ adj)[src, dst])
    cn_10 = torch.tensor((adj @ adj.T)[src, dst])
    cn_01 = torch.tensor((adj.T @ adj)[src, dst])
    cn_00 = torch.tensor((adj.T @ adj.T)[src, dst])
    cn = cn_11 + cn_10 + cn_01 + cn_00
    cn = cn[None].tile(adj.shape[0])
    return cn.to(torch.long)


def cn_plus_node_labeling(adj, src, dst):
    # mark each node with the number of total common neighbors of src and dst
    cn_11 = torch.tensor((adj @ adj)[src, dst])
    cn_10 = torch.tensor((adj @ adj.T)[src, dst])
    cn_01 = torch.tensor((adj.T @ adj)[src, dst])
    cn_00 = torch.tensor((adj.T @ adj.T)[src, dst])
    cn = torch.cat([cn_00[None], cn_01[None], cn_10[None], cn_11[None]])
    cn = cn[None].tile([adj.shape[0], 1])
    return cn.to(torch.long)


def rw_node_labeling(adj, src, dst, walk_length, walk_params):
    _, _, nbt, norm, entry, _ = walk_params
    #adj = torch.tensor((adj+adj.T).toarray()) # the subgraph is typically small to use dense representation
    adj = adj + adj.T
    num_nodes = adj.shape[0]
    adj[adj>1]=1
    if norm:
        degree = adj.sum(dim=0)
        inv_degree = 1. / (degree + 1e-6)
        inv_degree = inv_degree[..., None]
        adj = inv_degree * adj
        walks = [adj]
    else:
        walks = [adj]
    for _ in range(1, walk_length):
        walks.append(walks[-1] @ adj)
    #walks = torch.cat([w[..., None] for w in walks], dim=-1)
    if entry == 'st':
        #node_label = walks[src, dst][None].tile([num_nodes, 1])  # src->dst
        node_label = np.concatenate([w[src, dst][None] for w in walks], axis=-1)
        node_label = torch.tensor(node_label).tile([num_nodes, 1])
    elif entry == 'ss':
        #node_label = walks[src, src][None].tile([num_nodes, 1])  # src->dst
        node_label = np.concatenate([w[src, src][None] for w in walks], axis=-1)
        node_label = torch.tensor(node_label).tile([num_nodes, 1])
    #elif entry == 'st-ts':
        #node_label = torch.cat([walks[src, dst][None], walks[dst, src][None]], dim=-1).tile([num_nodes, 1])
    #elif entry == 'ss-st-ts-tt':
        #node_label = torch.cat([walks[:, src], walks[:, dst]], dim=-1)
    return node_label.float()



def rw_plus_node_labeling(adj, src, dst, walk_length, walk_parms):
    _, _, nbt, norm, entry, _ = walk_parms
    #adj, adjt = torch.tensor(adj.toarray()), torch.tensor(adj.T.toarray())  # the subgraph is typically small to use dense representation
    adj, adjt = adj, adj.T
    num_nodes = adj.shape[0]
    if norm:
        degree_o, degree_i = adj.sum(dim=1), adjt.sum(dim=1)
        # add self-loops
        zero_out_degree_nodes, zero_in_degree_nodes = torch.where(degree_o == 0)[0], torch.where(degree_i == 0)[0]
        #zero_degree_nodes = torch.cat([torch.where(degree_i == 0)[0], torch.where(degree_o == 0)[0]])
        adj[zero_out_degree_nodes, zero_out_degree_nodes] = 1.
        adjt[zero_in_degree_nodes, zero_in_degree_nodes] = 1.

        degree_o, degree_i = adj.sum(dim=1), adjt.sum(dim=1)
        inv_degree_o, inv_degree_i = 1. / (degree_o + 1e-6), 1. / (degree_i + 1e-6)
        inv_degree_o, inv_degree_i = inv_degree_o[..., None], inv_degree_i[..., None]
        adj = inv_degree_o * adj
        adjt = inv_degree_i * adjt

    walks = [[adj], [adjt]]
    for _ in range(1, walk_length):
        walks[0].append(walks[0][-1] @ adj)
        walks[1].append(walks[1][-1] @ adjt)
    #walks[0] = torch.cat([w[..., None] for w in walks[0]], dim=-1)
    #walks[1] = torch.cat([w[..., None] for w in walks[1]], dim=-1)
    #walks = torch.cat([walks[0], walks[1]], dim=-1)
    if entry == 'st':
        #node_label = walks[src, dst][None].tile([num_nodes, 1])# src->dst
        node_label = np.concatenate([w[src, dst][None] for w in walks[0]] + [w[src, dst][None] for w in walks[1]], axis=-1)
        node_label = torch.tensor(node_label).tile([num_nodes, 1])
    elif entry == 'ss':
        #node_label = walks[src, src][None].tile([num_nodes, 1])  # src->dst
        node_label = np.concatenate([w[src, src][None] for w in walks[0]] + [w[src, src][None] for w in walks[1]], axis=-1)
        node_label = torch.tensor(node_label).tile([num_nodes, 1])
    elif entry == 's':
        node_label = (walks[src, :].sum())[None].tile([num_nodes, 1])  # src->dst
    elif entry == 'st-ts':
        node_label = torch.cat([walks[src, dst][None], walks[dst, src][None]], dim=-1).tile([num_nodes, 1])
    elif entry == 'ss-st-ts-tt':
        node_label = torch.cat([walks[:, src], walks[:, dst]], dim=-1)
    return node_label.float()


def mw_node_labeling(adj, src, dst, walk_length, mw_params):
    q, q_dim, nbt, norm, entry, compact_q = mw_params
    q_step = q if q is not None else 1 / (2 * (walk_length + 1))
    #q = torch.tensor([i / 2 / (walk_length + 1) for i in range(q_dim)])
    q = torch.tensor([i * q_step for i in range(q_dim)])
    if q_dim == 1: # if only choose one q, does not include q=0
        q = torch.tensor([q_step])
    edge_index = torch.where(torch.tensor(adj.toarray())==1)
    edge_index = torch.cat([edge_index[0][None], edge_index[1][None]], dim=0)
    num_nodes = adj.shape[0]
    if edge_index.size(-1) > 0:
        #mw = compute_magnetic_walks(edge_index, q, walk_length, source_node=torch.tensor([src, dst]), nbt=nbt, normalize=norm)
        mw = compute_magnetic_walks(edge_index, num_nodes, q, walk_length, nbt=nbt, normalize=norm)
        if compact_q:
            tmp = []
            for i in range(1, mw.size(1)+1):
                compact_num_q = int(min(np.ceil(i/2)+1, q_dim))
                mw_i = mw[0:compact_num_q, i-1] # To Do: for i=1, maybe choose q!=0 is better
                tmp.append(mw_i)
            mw = torch.cat(tmp)
        else:
            mw = mw.flatten(0, 1)
        mw = torch.cat([mw.real, mw.imag], dim=0)
        # node_label = torch.cat([mw[..., src][..., None], mw[..., dst][..., None]], dim=-1)
        if entry == 'st':
            node_label = mw[:, src, dst][None].tile([num_nodes, 1])
        elif entry == 'ss':
            node_label = mw[:, src, src][None].tile([num_nodes, 1])
        elif entry == 's':
            node_label = (mw[:, src, :].sum(-1))[None].tile([num_nodes, 1])
        elif entry == 'st-ts':
            node_label = torch.cat([mw[:, src, dst][None], mw[:, dst, src][None]], dim=-1).tile([num_nodes, 1])
        elif entry == 'ss-st-ts-tt':
            node_label = torch.cat([mw[..., src], mw[..., dst]], dim=0)
            node_label = node_label.transpose(0, 1)
        elif entry == 'ss-tt':
            node_label = torch.cat([mw[:, src, src][None], mw[:, dst, dst][None]], dim=-1).tile([num_nodes, 1])
    else: # in case that two nodes are both isolated (due to test edge removal)
        factor = 2 if entry == 'st' else 4
        z_dim = q_dim * walk_length if not compact_q else \
            sum([int(min(np.ceil(i / 2) + 1, q_dim)) for i in range(1, walk_length + 1)])
        z_dim = factor * z_dim
        node_label = torch.zeros([2, z_dim])
    return node_label


def wp_node_labeling(adj, src, dst, walk_length, mw_params):
    _, _, nbt, norm, entry, _ = mw_params
    edge_index = torch.where(torch.tensor(adj.toarray()) == 1)
    edge_index = torch.cat([edge_index[0][None], edge_index[1][None]], dim=0)
    num_nodes = adj.shape[0]
    if edge_index.size(-1) > 0:
        # mw = compute_magnetic_walks(edge_index, q, walk_length, source_node=torch.tensor([src, dst]), nbt=nbt, normalize=norm)
        wp = compute_walk_profile(edge_index, num_nodes, walk_length, nbt=nbt, normalize=norm)
        # node_label = torch.cat([mw[..., src][..., None], mw[..., dst][..., None]], dim=-1)
        if entry == 'st':
            node_label = wp[:, src, dst][None].tile([num_nodes, 1])
        elif entry == 'ss':
            node_label = wp[:, src, src][None].tile([num_nodes, 1])
        elif entry == 's':
            node_label = (wp[:, src, :].sum(dim=-1))[None].tile([num_nodes, 1])
        elif entry == 'st-ts':
            node_label = torch.cat([wp[:, src, dst][None], wp[:, dst, src][None]], dim=-1).tile([num_nodes, 1])
        elif entry == 'ss-st-ts-tt':
            node_label = torch.cat([wp[..., src], wp[..., dst]], dim=0)
            node_label = node_label.transpose(0, 1)
        elif entry == 'ss-tt':
            node_label = torch.cat([wp[:, src, src][None], wp[:, dst, dst][None]], dim=-1).tile([num_nodes, 1])
    else:  # in case that two nodes are both isolated (due to test edge removal)
        factor = 1 if entry == 'st' else 2
        wp_dim = sum([i+1 for i in range(1, walk_length+1)])
        node_label = torch.zeros([2, factor * wp_dim])
    return node_label


def wp_node_labeling_full_graph(adj, walk_length, mw_params):
    _, _, nbt, norm, entry, _ = mw_params
    edge_index = torch.where(torch.tensor(adj.toarray()) == 1)
    edge_index = torch.cat([edge_index[0][None], edge_index[1][None]], dim=0)
    num_nodes = adj.shape[0]
    # to support batching in case of graphs are large
    nodes = torch.arange(num_nodes)
    batch_size = 256
    batched_nodes = torch.split(nodes, batch_size)
    wp = []
    for batch in batched_nodes:
        tmp = compute_walk_profile(edge_index, num_nodes, walk_length, source_node=batch, nbt=nbt, normalize=norm, history=False)
        wp.append(tmp)
    wp = torch.cat(wp, dim=1)
    if entry == 'ss':
        return wp.diagonal(dim1=1, dim2=2).T
    else:
        raise Exception("Unimplemented walk profile features.")


def mw_node_labeling_full_graph(adj, walk_length, mw_params):
    history=False # hard-coded, should change later
    q, q_dim, nbt, norm, entry, compact_q = mw_params
    q_step = q if q is not None else 1 / (2 * (walk_length + 1))
    #q = torch.tensor([i / 2 / (walk_length + 1) for i in range(q_dim)])
    q = torch.tensor([i * q_step for i in range(q_dim)])
    if q_dim == 1: # if only choose one q, does not include q=0
        q = torch.tensor([q_step])
    edge_index = torch.where(torch.tensor(adj.toarray())==1)
    edge_index = torch.cat([edge_index[0][None], edge_index[1][None]], dim=0)
    num_nodes = adj.shape[0]
    # to support batching in case of graphs are large
    nodes = torch.arange(num_nodes)
    batch_size = 256
    batched_nodes = torch.split(nodes, batch_size)
    mw = []
    for batch in batched_nodes:
        tmp = compute_magnetic_walks(edge_index, num_nodes, q, walk_length, source_node=batch, nbt=nbt, normalize=norm, history=history)
        mw.append(tmp)
    mw = torch.cat(mw, dim=1)
    if compact_q:
        if history:
            tmp = []
            for i in range(1, mw.size(1) + 1):
                compact_num_q = int(min(np.ceil(i / 2) + 1, q_dim))
                mw_i = mw[0:compact_num_q, i - 1]  # To Do: for i=1, maybe choose q!=0 is better
                tmp.append(mw_i)
            mw = torch.cat(tmp)
        else:
            compact_num_q = int(min(np.ceil(walk_length / 2) + 1, q_dim))
            mw = mw[0:compact_num_q]
    #else:
        #mw = mw.flatten(0, 1)
    mw = torch.cat([mw.real, mw.imag], dim=0)
    if entry == 'ss':
        mw = mw[0:mw.size(0)//2] # drop imaginary part as they should be zero
        return mw.diagonal(dim1=1, dim2=2).T
    else:
        raise Exception("Unimplemented magnetic walks features.")


def rw_node_labeling_full_graph(adj, walk_length, walk_params):
    _, _, nbt, norm, entry, _ = walk_params
    adj = ((adj + adj.T) / 2).ceil()
    num_nodes = adj.shape[0]
    if norm:
        degree = adj.sum(0)
        inv_degree = 1. / (degree + 1e-6)
        #inv_degree = inv_degree[..., None]
        inv_degree = np.array(inv_degree)[0]
        adj = diags(inv_degree) * adj
        walks = [adj]
    else:
        walks = [adj]
    for _ in range(1, walk_length):
        walks.append(walks[-1] @ adj)
    #walks = torch.cat([torch.tensor(w.diagonal()[..., None]) for w in walks], dim=-1)
    walks = torch.tensor(walks[-1].diagonal()[..., None])
    return walks


def rw_plus_node_labeling_full_graph(adj,walk_length, walk_parms):
    _, _, nbt, norm, entry, _ = walk_parms
    adjt = adj.T
    num_nodes = adj.shape[0]
    if norm:
        degree_o, degree_i = adj.sum(1), adjt.sum(1)
        degree_o, degree_i = np.array(degree_o)[:, 0], np.array(degree_i)[:, 0]
        # add self-loops
        zero_out_degree_nodes, zero_in_degree_nodes = np.where(degree_o == 0)[0], np.where(degree_i == 0)[0]
        #zero_degree_nodes = torch.cat([torch.where(degree_i == 0)[0], torch.where(degree_o == 0)[0]])
        adj[zero_out_degree_nodes, zero_out_degree_nodes] = 1.
        adjt[zero_in_degree_nodes, zero_in_degree_nodes] = 1.

        degree_o, degree_i = adj.sum(1), adjt.sum(1)
        degree_o, degree_i = np.array(degree_o)[:, 0], np.array(degree_i)[:, 0]
        inv_degree_o, inv_degree_i = 1. / (degree_o + 1e-6), 1. / (degree_i + 1e-6)
        adj = diags(inv_degree_o) * adj
        adjt = diags(inv_degree_i) * adjt

    walks = [[adj], [adjt]]
    for _ in range(1, walk_length):
        walks[0].append(walks[0][-1] @ adj)
        walks[1].append(walks[1][-1] @ adjt)
    #walks[0] = torch.cat([torch.tensor(w.diagonal()[..., None]) for w in walks[0]], dim=-1)
    #walks[1] = torch.cat([torch.tensor(w.diagonal()[..., None]) for w in walks[1]], dim=-1)
    walks[0] = torch.tensor(walks[0][-1].diagonal()[..., None])
    walks[1] = torch.tensor(walks[1][-1].diagonal()[..., None])
    walks = torch.cat([walks[0], walks[1]], dim=-1)
    return walks