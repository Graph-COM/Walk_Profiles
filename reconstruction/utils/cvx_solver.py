import cvxpy as cp
import numpy as np


def linear_programming_with_positive_constraints(A, b, only_feasibility=False):
    m, n = A.shape[0], A.shape[1]
    N = b.shape[-1]
    x = cp.Variable([n, N])
    # feasibility only
    #prob = cp.Problem(cp.Minimize(1),
                     #[A @ x == b, x >= 0])
    # minimizing L2 norm
    #prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
        #[A @ x == b, x >= 0])
    if only_feasibility:
        prob = cp.Problem(cp.Minimize(1),
                          [A @ x == b, x >= 0])
    else:
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
            [A @ x == b, x >= 0])
    #prob.solve(verbose=True)
    prob.solve()
    return x.value
