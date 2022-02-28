import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from scipy.stats import truncnorm

PROFILE_HEADER = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  'M->M', 'M->I', 'M->D', 'I->M', 'I->I',
                  'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D')  # yapf: disable

AMINO_ACIDS = AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = AA_IDS = {k: i for i, k in enumerate(AA)}

NUC = "ATCG-"
NUC_IDS = {k:i for i,k in enumerate(NUC)}

I = (lambda x: x)

inf = float("Inf")
r2 = math.sqrt(2)
SAFE_BOUND = 1 - 1e-7
EPS = 1e-7


device = "cpu"
DATA = "/home/malbranke/data/"

# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = "GHIBESTC"

abc_codes = {"a": 0, "b": 1, "c": 2}
dssp_codes = {"G": 0,
               "H": 1,
               "I": 2,
               "B": 3,
               "E": 4,
               "S": 5,
               "T": 6,
               "C": 7}


NAd_idx = {"A":[1,0,0,0], "T":[0,1,0,0], "C":[0,0,1,0], "G":[0,0,0,1],
          "W":[1,1,0,0], "S":[0,0,1,1], "M":[1,0,1,0], "K":[0,1,0,1], "R":[1,0,0,1], "Y":[0,1,1,0],
           "B":[0,1,1,1], "D":[1,1,0,1], "H":[1,1,1,0], "V": [1,0,1,1], "N":[1,1,1,1]}

NAd_in = {"A":"A", "T":"T", "C":"C", "G":"G",
          "W":"AT", "S":"CG", "M":"AC", "K":"TG", "R":"AG", "Y":"TC",
           "B":"TCG", "D":"ATG", "H":"ATC", "V": "ACG", "N":"ATCG"}


# Converter for the DSSP secondary pattern elements
# to the classical ones
dssp_to_abc = {"G": "a",
               "H": "a",
               "I": "a",
               "B": "b",
               "E": "b",
               "S": "c",
               "T": "c",
               "C": "c"}

pdb_codes = {0: "I",
                    1: "S",
                    2: "H",
                    3: "E",
                    4: "G",
                    5: "B",
                    6: "T",
                    7: "C"}


def ss8_to_ss3(x):
    if x <= 2:
        return 0
    if x >= 5:
        return 2
    return 1


# Torch utils

def trace(x, offset):
    shape = x.size()
    s1, s2 = shape[-2], shape[-1]
    x = x.reshape(-1, s1, s2)
    idxs = (torch.arange(s1)+offset)%s2
    return x[:, torch.arange(s1), idxs].sum(-1).view(*shape[:-2])

def mode(X, bins = 100):
    modes = []
    for x in X.t():
        idx = plt.hist(x, bins = bins)[0].argmax()
        modes.append(plt.hist(x, bins = bins)[1][idx])
    return torch.tensor(modes)

# Loss

def kld_loss(recon_x, x, mu):
    BCE = F.cross_entropy(recon_x, x.argmax(1), reduction="mean")
    KLD = -0.5 * torch.sum(- mu.pow(2))
    return BCE + KLD

def hinge_loss(model, x, y, m=1):
    e = model(x)
    e_bar = torch.min((e + 1e9 * y), 1, keepdim=True)[0].view(e.size(0), 1,
                                                              e.size(-1))
    loss = F.relu(m + (e - e_bar) * y)
    return loss.sum() / (e.size(0))

def likelihood_loss(model, x, y):
    e = model(x)
    return (-F.log_softmax(F.log_softmax(-e, 1), 1) * y).sum()

# Metrics

def aa_acc(x, recon_x):
    r"""
    Evaluate the ratio of amino acids retrieved in the reconstructed sequences

    Args:
        x (torch.Tensor): true sequence(s)
        recon_x (torch.Tensor): reconstructed sequence(s)
    """
    empty = torch.max(x, 1)[0].view(-1)
    x = torch.argmax(x, 1).view(-1)
    recon_x = torch.argmax(recon_x, 1).view(-1)
    return (((x == recon_x) * (empty != 0)).int().sum().item()) / ((empty != 0).int().sum().item())


# Regularizer

def msa_mean(x, w):
    return (w * x).sum(0) / w.sum(0)

# Probabilistic law


def R(W):
    return W.view(*W.size()[:-1], 21, -1).abs().sum(-2)


def phi(x):
    return (1 + torch.erf(x / r2)) / 2

def phi_inv(x):
    return r2 * torch.erfinv((2 * x - 1).clamp(-SAFE_BOUND, SAFE_BOUND))

def TNP(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = phi(-mu / sigma), torch.tensor(1.)
    a = (phi_a + x * (phi_b - phi_a))
    return phi_inv(a) * sigma + mu

def TNN(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = torch.tensor(0.), phi(-mu / sigma)
    a = (phi_a + x * (phi_b - phi_a))
    return phi_inv(a) * sigma + mu

# Gauges

def ZeroSumGauge(N=31, q=21):
    gauge = torch.eye(q * N)
    for i in range(q * N):
        gauge[i, (i + N * np.arange(q)) % (q * N)] -= 1 / q
    return gauge

# Sparse Tensor

def to_sparse(x):
    """ converts dense tensor x to sparse format """

    x = x.view(-1,x.size(-1))
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def push(file, key, val):
    data = torch.load(file)
    data[key] = val
    torch.save(data, file)

def pull(file, keys):
    data = torch.load(file)
    return {k: v for k, v in data.items() if v in keys}

def to_onehot(a, shape):
    if shape[0] is None:
        shape = len(a), shape[1]
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot
