import numpy as np
import networkx as nx
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


def random_tree(nb_nodes):
  G = nx.random_labeled_tree(nb_nodes, seed=args.seed)
  DG = nx.DiGraph()
  DG.add_edges_from(G.edges)
  return nx.adjacency_matrix(DG).todense()

def cycle(nb_nodes):
  n = nb_nodes
  A = np.zeros((n, n))
  for i in range(n):
    A[i, (i + 1) % n] = 1  # Connect i to i+1, with wrap-around
  return A

def clique(nb_nodes):
  n = nb_nodes
  A = np.zeros((n, n))
  for i in range(n):
    for j in range(i+1, n):
      A[i, j] = 1
  return A


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--p_order', type=str, choices=['1-n12', '1-n', 'sqrt_logn-n', 'logn-n'])
parser.add_argument('--p_const', type=float, default=1.)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--type', type=str, default='er')
args = parser.parse_args()


seed = args.seed
np.random.seed(seed)
n = args.n
c = args.p_const
if args.p_order == '1-n12':
  p = c / (n)**(1.2)
elif args.p_order == '1-n':
  p = c / n
elif args.p_order == 'sqrt_logn-n':
  p = c * np.sqrt(np.log(n) / n)
elif args.p_order == 'logn-n':
  p = c * np.log(n) / n
else:
  raise Exception('p_order=%s is not implemented' % args.p_order)


if args.type == 'er':
  A = random_er_graph(n, p, type=args.type)
elif args.type == 'tree':
  A = random_tree(n)
elif args.type == 'cycle':
  A = cycle(n)
elif args.type == 'clique':
  A = clique(n)
edge_index = np.where(A==1)
edge_index = np.concatenate((edge_index[0][None], edge_index[1][None]), axis=0)
if args.type == 'er':
  save_path = '../data/random_graphs/er_n%d_p_%s_c_%.4f_seed_%d.npy'%(n, args.p_order, args.p_const, seed)
elif args.type == 'tree':
  save_path = '../data/random_graphs/tree_n%d.npy' % n
elif args.type == 'cycle':
  save_path = '../data/random_graphs/cycle_n%d.npy' % n
elif args.type == 'clique':
  save_path = '../data/random_graphs/clique_n%d.npy' % n
np.save(save_path, edge_index)