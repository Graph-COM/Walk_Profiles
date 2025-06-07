import numpy as np
import torch
from test.mag_inverse_problem import setup_inverse_problem
# standard transform

def fullmw2wp(mw, m):
    res = torch.zeros([mw.size(0), m+1]) * (1 + 0*1j)
    q = torch.tensor([l / 2 / (m + 1) for l in range(m + 1)])
    for k in range(0, m+1):
        phases = torch.exp(torch.tensor(1j * 2 * torch.pi * q * (m - 2*k)))
        res[..., k] = (phases * mw).sum(dim=-1) / (m+1)
    return res


def wp2mwfull(wp, m):
    res = torch.zeros([wp.size(0), m+1]) * (1 + 0*1j)
    for l in range(0, m+1):
        q = l / 2 / (m+1)
        phases = torch.tensor([torch.exp(torch.tensor(1j * 4 * torch.pi * q * k)) for k in range(0, m+1)])
        res[..., l] = (phases * wp).sum(-1)
        res[..., l] *= torch.exp(torch.tensor(-1j * 2 * torch.pi * q * m))
    return res


# assume wp is real
def mw2realwp(mw, m):
    # inverse Fourier transform to mw, of length m+1
    # mw: [*, ceil(m/2)+1]
    res = torch.zeros([mw.size(0), m+1])
    for k in range(m+1):
        res[..., k] += mw[..., 0].real
        #if m % 2 != 0: # this term is always zero
            #res[..., k] += mw[..., int(np.ceil(m/2))] * (1j)**m * (-1)**k
        is_m_even = int(m % 2 == 0)
        for l in range(1, int(np.ceil(m/2))+is_m_even):
            #res[..., k] += 2 * (torch.exp(1j * torch.pi * (m-2*k)/(m+1)) * mw[..., l]).real
            res[..., k] += 2 * (torch.exp(torch.tensor(1j * torch.pi * l * (m-2*k)/(m+1))) * mw[..., l]).real
    return res / (m+1)


def wp2mw(wp, m):
    res = torch.zeros([wp.size(0), int(np.ceil(m/2))+1]) * (1 + 0*1j)
    for l in range(0, int(np.ceil(m/2))+1):
        q = l / 2 / (m+1)
        phases = torch.tensor([torch.exp(torch.tensor(1j * 4 * torch.pi * q * k)) for k in range(0, m+1)])
        res[..., l] = (phases * wp).sum(-1)
        res[..., l] *= torch.exp(torch.tensor(-1j * 2 * torch.pi * q * m))
    return res


# assume wp is real and symmetric, real-domain transform
def realwp2realmw(wp, m):
    # assume wp is real and symmetric
    res = torch.zeros([wp.size(0), m + 1], dtype=torch.float64)
    for l in range(0, m + 1):
        q = l / 2 / (m + 1)
        phases = torch.tensor([2 * torch.pi * q * (2*k-m) for k in range(0, m + 1)], dtype=torch.float64)
        res[..., l] = (torch.cos(phases) * wp).sum(-1)
    return res


def realwp2realmw_np(wp, m):
    # assume wp is real and symmetric
    res = np.zeros([wp.shape[0], m + 1], dtype=np.float128)
    for l in range(0, m + 1):
        q = l / 2 / (m + 1)
        phases = torch.tensor([2 * torch.pi * q * (2*k-m) for k in range(0, m + 1)], dtype=torch.float64)
        res[..., l] = (torch.cos(phases) * wp).sum(-1)
    return res

def realmw2realwp(mw, m):
    # inverse Fourier transform to mw, of length m+1
    # mw: [*, ceil(m/2)+1]
    res = torch.zeros([mw.size(0), m + 1], dtype=torch.float64)
    for k in range(m + 1):
        res[..., k] += mw[..., 0]
        #if m % 2 != 0:
            #res[..., k] += mw[..., int(np.ceil(m / 2))] * (1j) ** m * (-1) ** k
        is_m_even = int(m % 2 == 0)
        for l in range(1, int(np.ceil(m / 2)) + is_m_even):
            # res[..., k] += 2 * (torch.exp(1j * torch.pi * (m-2*k)/(m+1)) * mw[..., l]).real
            #res[..., k] += 2 * (torch.exp(torch.tensor(1j * torch.pi * l * (m - 2 * k) / (m + 1))) * mw[..., l]).real
            res[..., k] += 2 * torch.cos(torch.tensor(torch.pi * l * (m - 2 * k) / (m + 1), dtype=torch.float64)) * mw[..., l]
    return res / (m + 1)

def realmw2realwp_np(mw, m):
    # inverse Fourier transform to mw, of length m+1
    # mw: [*, ceil(m/2)+1]
    res = np.zeros([mw.shape[0], m + 1])
    for k in range(m + 1):
        res[..., k] += mw[..., 0]
        #if m % 2 != 0:
        #res[..., k] += mw[..., int(np.ceil(m / 2))] * (1j) ** m * (-1) ** k
        is_m_even = int(m % 2 == 0)
        for l in range(1, int(np.ceil(m / 2)) + is_m_even):
            # res[..., k] += 2 * (torch.exp(1j * torch.pi * (m-2*k)/(m+1)) * mw[..., l]).real
            #res[..., k] += 2 * (torch.exp(torch.tensor(1j * torch.pi * l * (m - 2 * k) / (m + 1))) * mw[..., l]).real
            res[..., k] += 2 * np.cos(np.pi * l * (m - 2 * k) / (m + 1)) * mw[..., l]
    return res / (m + 1)


def realmw2realwp_reconstruct(mw, m, normalize=True):
    Q = mw.size(1)
    if Q >= np.ceil(m/2)+1:
        # directly reverse mw to wp
        wp = realmw2realwp(mw, m)
    else:
        q = torch.tensor([i / 2 / (m + 1) for i in range(Q)])
        inverse_problem = setup_inverse_problem(m, q, normalize=normalize)
        wp = inverse_problem.solve_pseudo_inverse(mw.T).T
    return wp
