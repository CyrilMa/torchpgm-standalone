import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
from torch import nn

import numpy as np

from .utils import *
from .subedge import *


class AbstractEdge(nn.Module):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out):
        super(AbstractEdge, self).__init__()

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.freeze = True
        self.gauge = None

        self.in_shape, self.out_shape = lay_in.shape, lay_out.shape
        self.subs = None

    def to(self, device):
        super(AbstractEdge, self).to(device)
        self.subs = nn.ModuleList([sub.to(device) for sub in self.subs])
        return self

    def freeze(self):
        self.freeze = True

    def unfreeze(self):
        self.freeze = False

    def gauge_weights(self):
        for sub in self.subs:
            sub.gauge_weights()

    def get_weights(self):
        return [sub.weight for sub in self.subs]

    def backward(self, h, sample=True):
        h = h.reshape(h.size(0), -1)
        p = torch.stack([sub.backward(h[:,m:M]) for sub, m, M in zip(self.subs, self.cumNks[:-1], self.cumNks[1:])],-1).sum(-1)
        if sample:
            x = self.in_layer.sample([p])
            return x
        return p

    def forward(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        p = torch.cat([sub(x) for sub in self.subs],-1)
        if sample:
            h = self.out_layer.sample([p])
            return h
        return p

    def gibbs_step(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        mu = self.forward(x, sample=False)
        h = mu
        if sample:
            h = self.out_layer.sample([mu])
        else:
            h = self.out_layer.mean([mu])
        mut = self.backward(h, sample=False)
        x_rec = mut
        if sample:
            x_rec = self.in_layer.sample([mut])
        else:
            x_rec = self.in_layer.mean([mut])

        return x_rec, h, mut, mu

    def partial_gibbs_step(self, x, active_visible, inactive_units, sample=True):
        m, M = active_visible
        q = self.in_layer.q
        x = x.reshape(x.size(0), -1).float()
        mu = h = self.forward(x, sample=False)
        if sample:
            h = self.out_layer.sample([mu])
        else:
            h = self.out_layer.mean([mu])
        h[:,inactive_units] = mu[:,inactive_units]
        mut = x_rec = self.backward(h, sample=False)
        if sample:
            x_rec = self.in_layer.sample([mut])
        else:
            x_rec = self.in_layer.mean([mut])
        x_rec, x, mut = x_rec.view(x.size(0),q,-1), x.view(x.size(0),q,-1), mut.view(mut.size(0),q,-1)
        x_rec[:,:,:m] = x[:,:,:m]; x_rec[:,:,M:] = x[:,:,M:]
        mut[:,:,:m] = x[:,:,:m]; mut[:,:,M:] = x[:,:,M:]
        return x_rec.reshape(x.size(0), -1), h, mut.reshape(x.size(0), -1).detach(), mu.detach()

    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.abs(w_).reshape(w_.size(0),-1).mean(-1) for w_ in w],-1)
        return reg.pow(2).mean(0)

    def l2_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.pow(2).mean(-1) for w_ in w],-1)
        return reg.pow(2).mean(0)

    def save(self, filename):
        torch.save(self, filename)


class DenseEdge(AbstractEdge):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out):
        super(DenseEdge, self).__init__(lay_in, lay_out)

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.N, self.q = lay_in.N, lay_in.q

        self.Nks = [self.out_layer.N]
        self.cumNks = [0]+list(np.cumsum(self.Nks))

        # Model
        self.subs = nn.ModuleList([DenseSubEdge(self.N, self.q, self.Nks[0], gauge=ZeroSumGauge(self.N, self.q).to(device))])

    def sim_weights(self, strength = 1):
        W = self.linear.weight
        N = self.out_shape
        return sum(sum(torch.exp(-(W[i]*W[j]).mean(-1) for j in range(i)) for i in range(N)))/(N*(N-1)/2)

    def l1c_reg(self):
        W = self.get_weights()
        rw = R(W)
        rw = (rw[:,None]*rw[:,:,None])
        rw = torch.stack([torch.diagonal(rw,k,1,2).sum(-1) for k in range(1,rw.size(-1))],-1)
        reg = (rw * torch.tensor([1-np.exp(-i) for i in range(1,rw.size(-1)+1)])[None]).mean(-1)
        return reg.sum()


class FilterbankEdge(AbstractEdge):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out, out_channels, window_size, strides, weights=None):
        super(FilterbankEdge, self).__init__(lay_in, lay_out)

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.N, self.q = lay_in.N, lay_in.q
        self.window_size, self.strides, self.out_channels = window_size, strides, out_channels

        self.Nks = FilterbankEdge._Nks(self.N, out_channels, window_size, strides)
        self.cumNks = [0]+list(np.cumsum(self.Nks))
        gauges = [ZeroSumGauge(w, self.q) for w in window_size]

        # Model
        self.subs = nn.ModuleList([FilterbankSubEdge(self.N, self.q, d, w, s, gauge) for d, w, s, gauge in zip(out_channels, window_size, strides, gauges)])


    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([math.sqrt(self.N/size) * torch.abs(w_).sum(-1) for size, w_ in zip(self.window_size, w)],-1)
        return reg.pow(2).sum(0)

    def linfb_reg(self):
        w = self.get_weights()
        reg = torch.cat([torch.max(w_)[0].sum(-1) for size, w_ in zip(self.window_size, w)],-1)
        return reg.pow(2).sum(0)


    def l2_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.pow(2).sum(-1) for w_ in w],-1)
        return reg.pow(2).sum(0)

    def _Nks(N, out_channels, window_size, strides):
        return [(d*(int((N-w)/s - EPS)+1)) for w, s, d in zip(window_size, strides, out_channels)]

class ConvolutionalEdge(AbstractEdge):
    r"""
    Class to handle a Convolutional Edge where the Dense connexion between visible and hidden units are replace by
    convolutional connexion of different scales

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out, out_channels, window_size, strides, weights=None):
        super(ConvolutionalEdge, self).__init__(lay_in, lay_out)

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.N, self.q = lay_in.N, lay_in.q

        self.Nks = ConvolutionalEdge._Nks(self.N, out_channels, window_size, strides)
        self.opads = [self.N - ((Nk//d-1)*s+w) for Nk,s,w,d in zip(self.Nks, strides, window_size, out_channels)]
        self.cumNks = [0]+list(np.cumsum(self.Nks))
        self.gauges = [ZeroSumGauge(w, self.q).to(device) for w in window_size]

        # Model
        self.subs = nn.ModuleList([ConvolutionalSubEdge(self.N, self.q, d, w, s, opad, gauge) for w, s, d, opad, gauge in zip(window_size, strides, out_channels, self.opads, self.gauges)])

    def _Nks(N, out_channels, window_size, strides):
        return [d*int((N-(w-1)-1)/s+1) for w,s,d in zip(window_size,strides,out_channels)]
