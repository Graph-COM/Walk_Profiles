# playground for mag-walk & walk-profile inverse problem
import numpy as np
import torch


def q_sampling(q, q_dim, q_sampling_strategy='top-k'):
    if q_sampling_strategy == 'top-k':
        ind = [i for i in range(q_dim)]
    elif q_sampling_strategy == 'last-k':
        ind = [i for i in range(len(q)-q_dim, len(q))]
    elif q_sampling_strategy == 'random':
        ind = np.random.permutation([i for i in range(len(q))])[0:q_dim]
    return q[ind], ind


class setup_inverse_problem:
    def __init__(self, walk_length, q, normalize=False):
        self.walk_length = walk_length
        self.q = q
        self.Q = len(q)
        self.normalize = normalize
        self.F = self.construct_measurement_matrix() # measurement matrix
        #self.F_inv = torch.linalg.pinv(self.F)


    def construct_measurement_matrix(self):
        # measurement matrix F: [Q, walk length + 1]
        F = torch.tensor([[torch.exp(1j * 4 * torch.pi * qj * k) for k in range(self.walk_length+1)] for qj in self.q])
        return F

    def construct_observations(self, y, real_domain=True):
        # y: [Q, *]
        # return y * exp(1j * 2 * pi * q * m)
        q = self.q.unsqueeze(-1)
        y = y * torch.exp(1j * 2 * torch.pi * q * self.walk_length)
        if real_domain:
            return torch.cat([y.real, y.imag], dim=0)
        else:
            return y


    #def solve_pseudo_inverse(self, y, real_domain=True, positive=True):
    #    y = self.construct_observations(y, real_domain)
    #    F = torch.cat([self.F.real, self.F.imag], dim=0) if real_domain else self.F
        #x = self.F_inv @ y
    #    if positive:
    #        x = linear_programming_with_positive_constraints(np.array(F), np.array(y))
    #        x = torch.tensor(x)
    #    else:
    #        F_inv = torch.linalg.pinv(F)
            #x = torch.einsum('ij, juv->iuv', F_inv, y)
    #        x = F_inv @ y
    #    x = x.real # the unknowns lie in real space
    #    x = x.round() # the unknowns lie in integer space
    #    x[torch.where(x <= 0)] = 0. # the unknowns are positive
    #    return x

    def solve_pseudo_inverse(self, y, real_domain=True):
        y = self.construct_observations(y, real_domain=real_domain)
        if real_domain:
            F = torch.cat([self.F.real, self.F.imag], dim=0)
            F_inv = torch.linalg.pinv(F)
        else:
            F_inv = torch.linalg.pinv(self.F)
        x = F_inv @ y
        x = x.real # the unknowns lie in real space
        if not self.normalize:
            x = x.round() # the unknowns lie in integer space
        #x[torch.where(x <= 0)] = 0. # the unknowns are positive
        return x

    def solve_linear_programming_with_positive_constraints(self, y, only_feasibility=False):
        F = torch.cat([self.F.real, self.F.imag], dim=0)
        if F.shape[0] < F.shape[1]:
            y = self.construct_observations(y)
            x = linear_programming_with_positive_constraints(np.array(F), np.array(y), only_feasibility)
            x = torch.tensor(x)
            x = x.real  # the unknowns lie in real space
            if not self.normalize:
                x = x.round()  # the unknowns lie in integer space
            x[torch.where(x <= 0)] = 0.  # the unknowns are positive
        else:
            x = self.solve_pseudo_inverse(y) # assume this case the solution is unique, no need for positive constraints
        return x


    def solve_ridge(self, y, sigma):
        # solve ridge regression: |Ax-b|_2^2 + sigma * |x|_2^2
        y = self.construct_observations(y)
        F = torch.cat([self.F.real, self.F.imag], dim=0)
        mat = F.T @ F + sigma * torch.eye(F.size(-1))
        mat = torch.linalg.inv(mat) @ F.T
        #x = torch.einsum('ij, juv->iuv', mat, y)
        x = mat @ y
        x = x.real  # the unknowns lie in real space
        x = x.round()  # the unknowns lie in integer space
        x[torch.where(x <= 0)] = 0.  # the unknowns are positive
        return x

    def solve_lasso(self, y, sigma):
        # solve lasso regression: |Ax-b|_2^2 + sigma * |x|_1
        raise Exception('not implemented yet!')


