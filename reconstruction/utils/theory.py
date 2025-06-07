# include all the theoretical things we know/conjecture


import numpy as np
from math import comb
from scipy.stats import poisson, binom

def expectation(m, k, n, p, method='ind'):
    # compute Expectation(phi(m, r)) on random graph G(n,p)
    assert method in ['bt', 'nbt', 'ind']
    if method == 'bt':
        raise Exception('expectation of back-tracking walk profile is unknown!')
    else:
        #return comb(m, k) * n ** (m-1) * p ** (m)
        return comb(m, k) * (n * p) ** (m - 1) * p


def moments(m, k, r, n, p, method='ind'):
    # compute r-th moments of phi(m,k) on random graph G(n,p)
    assert method in ['bt', 'nbt', 'ind']
    if r == 1:
        return expectation(m, k, n, p, method)
    else:
        if method == 'bt':
            raise Exception('moments of back-tracking walk profile is unknown!')
        elif method == 'nbt':
            raise Exception('moments of non-back-tracking walk profile is unknown!')
        else:
            raise Exception('moments of independent-graph walk profile is unknown!')


def pmf(m, k, x, n, p, method='ind'):
    # compute Prob(phi(m,k)=x) on random graph G(n,p)
    assert method in ['bt', 'nbt', 'ind']
    if method == 'bt':
        raise Exception('pmf of back-tracking walk profile is unknown!')
    elif method == 'nbt':
        raise Exception('pmf of non-back-tracking walk profile is unknown!')
    elif method == 'ind':
        if m == k: # conjecture: poisson distribution
            d = (n * p) ** (m - 1) * p
            return poisson.pmf(x, d)


