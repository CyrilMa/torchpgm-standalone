import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch, torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .utils import *
from Bio import SeqIO


class AbstractDataset(Dataset):
    def __init__(self):
        self.x_d = None
        self.x_m = None
        self.weights = None

    def update_pcd(self, idx, samples):
        for i, sample in zip(idx, samples):
            self.x_m[i] = sample

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.weights[i], i

class FastaDataset(AbstractDataset):
    def __init__(self, file):
        super(FastaDataset, self).__init__()
        file = "/home/malbranke/data/cas9/aligned.fasta"
        fasta_sequences = SeqIO.parse(open(file),'fasta')
        seqs = []
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            seqs.append(torch.tensor(to_onehot(torch.tensor([AA_IDS[x]+1 if x in AA else 0 for x in sequence]), (None,21))))
        self.x_d = torch.stack(seqs,0)
        self.L = len(self.x_d)
        self.weights = torch.ones(self.L)
        self.x_m = self.x_d.clone()

class StandardDataset(AbstractDataset):
    def __init__(self, file, subset=None, nnz_idx=None):
        super(StandardDataset, self).__init__()
        data = torch.load(file)
        keys = list(data.keys())
        print("Available : ", *keys)
        if subset is not None:
            idx = data["subset"][subset]
        else:
            idx = torch.arange(data["L"])
        self.L = len(idx)
        self.x_d = list(data["x"][idx]) if "x" in keys else None
        if nnz_idx is not None:
            self.x_d = [x[:, nnz_idx] for x in self.pis]
        self.weights = data["weights"][idx] if "weights" in keys else [1. for _ in self.x_d]
        self.x_m = []
        for i, x_d in enumerate(self.x_d):
            gaps = (x_d.sum(0) == 0).int()
            self.x_m.append(torch.cat([gaps[None], x_d], 0))
            self.x_d[i] = torch.cat([gaps[None], self.x_d[i]], 0)

    def update_pcd(self, idx, samples):
        for i, sample in zip(idx, samples):
            self.x_m[i] = sample

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.weights[i], i


class MNISTData(Dataset):
    def __init__(self, folder, kept_classes = list(range(10))):
        super(MNISTData, self).__init__()
        mnist_data = torchvision.datasets.MNIST(folder, download = True)
        idx = torch.where(torch.stack([y_ == mnist_data.train_labels for y_ in kept_classes]).sum(0).bool())[0]

        self.x_d = (mnist_data.data > 127).int()
        self.x_d = self.x[idx]
        self.L = len(self.x)
        self.x_d = self.x_d.view(self.L,-1)
        self.x_d = torch.stack([torch.tensor(to_onehot(x_, (None,2))) for x_ in self.x_d],0).int().permute(0, 2, 1)
        self.weights = [1 for _ in self.x_d]
        self.x_m = [x for x in self.x_d]

    def update_pcd(self, idx, samples):
        for i, sample in zip(idx, samples):
            self.x_m[i] = sample

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.weights[i], i

class MNISTDataWithClass(Dataset):
    def __init__(self, folder, kept_classes = list(range(10))):
        super(MNISTDataWithClass, self).__init__()
        mnist_data = torchvision.datasets.MNIST(folder, download = True)
        idx = torch.where(torch.stack([y_ == mnist_data.train_labels for y_ in kept_classes]).sum(0).bool())[0]

        self.x_d = (mnist_data.data > 127).int()
        self.x_d = self.x_d[idx].view(len(idx),-1)
        self.x_d = torch.stack([torch.tensor(to_onehot(x_, (None,2))) for x_ in self.x_d],0).int().permute(0, 2, 1)
        self.y = torch.tensor(to_onehot(mnist_data.train_labels, (None, 10))).int()[idx]

        self.L = len(self.x_d)

        self.weights = [1 for _ in self.x_d]
        self.x_m = [x for x in self.x_d]

    def update_pcd(self, idx, samples):
        for i, sample in zip(idx, samples):
            self.x_m[i] = sample

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.y[i], self.weights[i], i
