import numpy as np
import torch
import scipy.sparse as ss
import matplotlib.pyplot as plt
from utils.walk_profile import message_passing_nbt_walk_profile_calculation_and_analysis
from utils.walk_profile import get_walk_profile, get_nbt_walk_profile, get_walk_profile_edge
from utils.magnetic_walk import get_magnetic_walks, get_magnetic_walks_edge
from utils.mag_inverse_problem import setup_inverse_problem, q_sampling
import time
import random


def sample_random_q(num_q):
  q = []
  for _ in range(num_q):
    while True:
      q_trial = random.uniform(0, 0.25)
      if np.array(q).size == 0:
        q.append(q_trial)
        break
      else:
        diff = np.abs(np.array(q) - q_trial).min()
        if diff >= 0.1 / num_q:
          q.append(q_trial)
          break
        else:
          continue
  return q



def random_er_graph(nb_nodes, p=0.5, directed=True, acyclic=False,
                     weighted=False, low=0.1, high=1.0):
  """Random Erdos-Renyi graph."""

  mat = np.random.binomial(1, p, size=(nb_nodes, nb_nodes))
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


def relative_error(x, xhat, ignore_all_zeros=True):
  # x, xhat: [L, N, N]
  if ignore_all_zeros: # all-zero vectors can be easily identified
    nonzeros_ind = torch.where(x.sum(0)!=0)
    #x_nonzeros = x[:, nonzeros_ind[0], nonzeros_ind[1]]
    #xhat_nonzeros = xhat[:, nonzeros_ind[0], nonzeros_ind[1]]
    x_nonzeros = x[:, nonzeros_ind[0]]
    xhat_nonzeros = xhat[:, nonzeros_ind[0]]
    return torch.sqrt((x_nonzeros - xhat_nonzeros).square().sum(dim=0)) / torch.sqrt((x_nonzeros).square().sum(dim=0))



def test_nbt(A, walk_length, q, num_sn):
  #nbt_list = nbt_walk(A, walk_length)
  #nbt_wp_list = nbt_walk_profile(A, walk_length)
  # compare to message passing version
  num_nodes = A.shape[0]
  source_node_idx = np.array(random.sample([i for i in range(num_nodes)], num_sn)) if num_sn is not None else None


  #q = torch.tensor([i / 2 / (walk_length+1) for i in range(walk_length+1)])
  #q = torch.tensor([i / 2 / (walk_length+1) for i in range(walk_length)])
  edge_index = torch.cat([(torch.where(torch.tensor(A)!=0)[0])[None], (torch.where(torch.tensor(A)!=0)[1])[None]], dim=0)
  # compute magnetic walks
  #mag_walks = get_magnetic_walA = to_dense_adj(edge_index)[0]ks(edge_index, q, walk_length, nbt=args.nbt, source_node=source_node_idx, normalize=args.norm)
  time1=time.time()
  mag_walks = get_magnetic_walks_edge(edge_index, q, walk_length, source_node=source_node_idx, nbt=args.nbt, normalize=args.norm)
  time2=time.time()
  #inverse_problem = setup_inverse_problem(walk_length, q)
  # inference walk profiles
  mag_walks = mag_walks.flatten(1)
  #if solver == 'pseudo':
  #  walk_profile_hat = inverse_problem.solve_peusdo_inverse(mag_walks)
  #elif solver == 'ridge':
  #  walk_profile_hat = inverse_problem.solve_ridge(mag_walks, sigma)
  # compute ground-truth walk profiles
  #if args.nbt:
    #_ = get_walk_profile_edge(edge_index, walk_length, nbt=True)
    #get_magnetic_walks_edge(edge_index, q, walk_length, nbt=True)
    #walk_profile = get_nbt_walk_profile(edge_index, walk_length, source_node=source_node_idx)
  time3=time.time()
  walk_profile = get_walk_profile_edge(edge_index, walk_length, source_node=source_node_idx, nbt=args.nbt, normalize=args.norm)
  time4=time.time()
  #else:
    #walk_profile = get_walk_profile(edge_index, walk_length, source_node=source_node_idx, normalize=args.norm)
  #walk_profile = get_nbt_walk_profile(edge_index, walk_length, source_node=source_node_idx)
  walk_profile = walk_profile.flatten(1)
  return mag_walks, walk_profile
  #error = relative_error(walk_profile, walk_profile_hat)
  #error = torch.linalg.norm((walk_profile - walk_profile_hat).float(), dim=0) / (torch.linalg.norm(walk_profile.float(), dim=0) + 1e-6)
  #error = error.mean()
  #return error
  #assert error <= 1e-5


  #message_passing_nbt_walk_profile_calculation_and_analysis(edge_index, walk_length)

  #nbt_wp_list_gt, nbt_wp_list_gt_l, nbt_wp_list_r, nbt_wp_list_lr = nbt_walk_profile_gt(A, walk_length)
  #for m in range(2, walk_length):
  #  assert np.sum(np.abs(nbt_list[m] - nbt_wp_list[m][m])) == 0.0
  #for i in range(len(nbt_wp_list_gt)):
    # TO DO: phi_{0,9}(8, 2) mismatch, likely due to precision problem
    #assert torch.linalg.norm(nbt_wp_list_gt[i].float() - torch.tensor(nbt_wp_list[-1][i]).float()) == 0.




import argparse
torch.set_num_threads(10)
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--d', type=float, default=1.)
parser.add_argument('--walk_length', type=int, default=5)
parser.add_argument('--num_sn', type=int, default=None)
parser.add_argument('--norm', action='store_true', default=False)
parser.add_argument('--nbt', action='store_true', default=False)
parser.add_argument('--solver', type=str, default='inverse')
args = parser.parse_args()



np.random.seed(42)
random.seed(42)
m = args.walk_length
n = args.n
d = args.d
p = d / n
n_samples = 2
#sigmas = [0.01, 0.1, 1, 5]
sigmas = []
A_list = []
for i in range(n_samples):
  if i % 10 == 0:
    print('generating random graphs num %d' % i)
  A_list.append(random_er_graph(n, p))
# verify statistical property
error = {'pseudo': [[] for _ in range(m)]}
#error = {'pseudo': [[] for _ in range(m)], 'quadratic_program': [[] for _ in range(m)]}
#error = {'pseudo': [[] for _ in range(m)], 'pseudo+positive': [[] for _ in range(m)]}
#error = {'pseudo (imag)': [[] for _ in range(m)], 'pseudo (real)': [[] for _ in range(m)]}
for sigma in sigmas:
  error['ridge-%.2f'%sigma] = [[] for _ in range(m)]

q_full = torch.tensor([i / 2 / (args.walk_length + 1) for i in range(args.walk_length + 1)])
#q_full = torch.tensor([i / 2 / (args.walk_length + 1) for i in range(1, args.walk_length + 2)])
#q_full = torch.tensor([0.01 * i for i in range(args.walk_length+1)])
#q_full = torch.cat([torch.tensor(sample_random_q(args.walk_length)), torch.tensor([0.])])

for A in A_list:
  A = A - A * A.T  # no multi-edges
  mag_walks, walk_profile = test_nbt(A, m, q_full, args.num_sn)
  for Q in range(2, m + 2):
    print('Q=%d'%Q)
    # subsampling of q
    q, sample_ind = q_sampling(q_full, Q, q_sampling_strategy='top-k')
    mag_walks_sampled = mag_walks[sample_ind]
    inverse_problem = setup_inverse_problem(args.walk_length, q, normalize=args.norm)
    #walk_profile_hat = inverse_problem.solve_pseudo_inverse(mag_walks_sampled)
    if args.solver == 'inverse':
      walk_profile_hat = inverse_problem.solve_pseudo_inverse(mag_walks_sampled)
    elif args.solver == 'lp':
      walk_profile_hat = inverse_problem.solve_linear_programming_with_positive_constraints(mag_walks_sampled)
    #walk_profile_hat2 = inverse_problem.solve_pseudo_inverse(mag_walks_sampled)
    e = relative_error(walk_profile, walk_profile_hat)
    #e2 = relative_error(walk_profile, walk_profile_hat2)
    e = e.mean()
    #e2 = e2.mean()
    print(e)
    # elif solver == 'ridge':
    #  walk_profile_hat = inverse_problem.solve_ridge(mag_walks, sigma)
    #error['quadratic_program'][Q-2].append(e)
    error['pseudo'][Q-2].append(e)
    for sigma in sigmas:
      walk_profile_hat = inverse_problem.solve_ridge(mag_walks_partial, sigma)
      e = relative_error(walk_profile, walk_profile_hat)
      e = e.mean()
      error['ridge-%.2f'%sigma][Q-2].append(e)
    #print(e1, e2)

import matplotlib.pyplot as plt
for key in error.keys():
  temp = np.array(error[key])
  temp = temp.mean(axis=-1)
  plt.plot([Q for Q in range(2, m+2)], temp, label=key)
plt.legend()
plt.xlabel('Number of qs')
plt.ylabel('Relative error')
plt.title('walk length=%d, n=%d, d=%.3f' % (m, n, d))
plt.show()
name = 'nbt' if args.nbt else ''
name += '_norm' if args.norm else ''
name += '_'+args.solver
#plt.savefig('./figs/ip_%s_m%d_n%d_d%d.png' % (name, m, n, d))

