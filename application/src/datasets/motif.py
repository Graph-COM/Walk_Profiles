import numpy as np
import torch
import re
import networkx as nx
from networkx.algorithms import isomorphism
from dotmotif import Motif, GrandIsoExecutor
import string

def out_degree(A):
    labels = A.sum(dim=1)
    return labels


def feedforward_loop(A):
    labels = torch.zeros(A.shape[0])
    tmp = A@A@A.T + A@A.T@A + A.T@A@A
    for i in range(len(labels)):
        labels[i] = tmp[i, i]
    return labels


def reciprocal_edges(A):
    labels = torch.zeros(A.shape[0])
    tmp = A@A
    for i in range(len(labels)):
        labels[i] = tmp[i, i]
    return labels

def three_walks(A):
    labels = torch.zeros(A.shape[0])
    tmp = A@A@A
    for i in range(len(labels)):
        labels[i] = tmp[i, i]
    return labels

def common_parents(A):
    tmp = A @ A.T
    labels = torch.tensor(tmp.sum(axis=-1)).view(-1)
    return labels

def common_children(A):
    tmp = A.T @ A
    labels = torch.tensor(tmp.sum(axis=-1)).view(-1)
    return labels


def find_cycle_networkx(A, source=None, orientation=None):
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    if not G.is_directed() or orientation in (None, "original"):

        def tailhead(edge):
            return edge[:2]

    elif orientation == "reverse":

        def tailhead(edge):
            return edge[1], edge[0]

    elif orientation == "ignore":

        def tailhead(edge):
            if edge[-1] == "reverse":
                return edge[1], edge[0]
            return edge[:2]

    explored = set()

    all_cycles = []
    cycle = []
    final_node = None
    for start_node in G.nbunch_iter(source):
        if start_node in explored:
            # No loop is possible.
            continue

        edges = []
        # All nodes seen in this iteration of edge_dfs
        seen = {start_node}
        # Nodes in active path.
        active_nodes = {start_node}
        previous_head = None

        for edge in nx.edge_dfs(G, start_node, orientation):
            # Determine if this edge is a continuation of the active path.
            tail, head = tailhead(edge)
            if head in explored:
                # Then we've already explored it. No loop is possible.
                continue
            if previous_head is not None and tail != previous_head:
                # This edge results from backtracking.
                # Pop until we get a node whose head equals the current tail.
                # So for example, we might have:
                #  (0, 1), (1, 2), (2, 3), (1, 4)
                # which must become:
                #  (0, 1), (1, 4)
                while True:
                    try:
                        popped_edge = edges.pop()
                    except IndexError:
                        edges = []
                        active_nodes = {tail}
                        break
                    else:
                        popped_head = tailhead(popped_edge)[1]
                        active_nodes.remove(popped_head)

                    if edges:
                        last_head = tailhead(edges[-1])[1]
                        if tail == last_head:
                            break
            edges.append(edge)

            if head in active_nodes:
                # We have a loop!
                cycle.extend(edges)
                final_node = head
                break
            else:
                seen.add(head)
                active_nodes.add(head)
                previous_head = head

        if cycle:
            for i, edge in enumerate(cycle):
                tail, head = tailhead(edge)
                if tail == final_node:
                    break
            all_cycles.append(cycle[i:])
            #break
        else:
            explored.update(seen)

    else:
        assert len(cycle) == 0
        raise nx.exception.NetworkXNoCycle("No cycle found.")

    # We now have a list of edges which ends on a cycle.
    # So we need to remove from the beginning edges that are not relevant.

    #for i, edge in enumerate(cycle):
    #    tail, head = tailhead(edge)
    #    if tail == final_node:
    #        break

    #return cycle[i:]
    return all_cycles

def cycles(A, cycle_mode):
    # cycle_mode is string in format like "cycle1010", where 1 represents forward edge and 0 for backward edge
    match = re.search(r'\d', cycle_mode).start()
    cycle_mode = cycle_mode[match:]
    length = len(cycle_mode)
    node_list = [string.ascii_uppercase[i] for i in range(length)] + ['A']
    edge_list = []
    edge_list_str = '\n'
    for i in range(length):
        u, v = node_list[i], node_list[i+1]
        if cycle_mode[i] == '0':
            edge_list.append((v, u))
            edge_list_str += v + ' -> '+ u + '\n'
        elif cycle_mode[i] == '1':
            edge_list.append((u, v))
            edge_list_str += u + ' -> '+ v + '\n'
        else:
            raise Exception('Incorrect cycle format.')

    #G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph())
    motif = Motif(edge_list_str)
    executor = GrandIsoExecutor(graph=G)
    y = torch.zeros(len(G))
    results = executor.find(motif)
    for i, res in enumerate(results):
        #node_idx = list(map.keys())[list(map.values()).index('*')]
        node_idx = list(res.values())
        y[node_idx] += 1

    C = nx.DiGraph()
    C.add_edges_from(edge_list)
    executor = GrandIsoExecutor(graph=C)
    num_automorphism = len(executor.find(motif))
    y = y / num_automorphism # divided by repeated count from cycle automorphism
    return y


def cycles_networkx(A, cycle_mode):
    assert cycle_mode.count('0') <= 1 # currently only support bidirectional cycles with at most one backward edge
    match = re.search(r'\d', cycle_mode).start()
    cycle_mode = cycle_mode[match:]
    cycle_length = len(cycle_mode)
    num_back_edges = cycle_mode.count('0')

    G = nx.from_scipy_sparse_array(A) # undirected version of A
    cycles_iterator = nx.simple_cycles(G, cycle_length)
    y = torch.zeros(len(G))
    count = 0
    for i, c in enumerate(cycles_iterator):
        if len(c) == cycle_length:
            cycle_edge_index = np.array([[c[i], c[i+1]] for i in range(len(c)-1)] + [[c[-1], c[0]]])
            if A[cycle_edge_index[:, 0], cycle_edge_index[:, 1]].sum() == cycle_length - num_back_edges:
                y[c] += 1
                count += 1
        if count % 100 == 0:
            print("%d %d-cycle detected..." % (count, cycle_length))
    return y

