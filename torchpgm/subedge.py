import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
from torch import nn

import numpy as np

from .utils import *


class AbstractSubEdge(nn.Module):
    def __init__(self, in_features, in_channels, out_channels, gauge = None):
        super(AbstractSubEdge, self).__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = None
        self.gauge = gauge
        if gauge is not None:
            self.gauge = gauge.detach()

    def to(self, device):
        super(AbstractSubEdge, self).to(device)
        self.gauge = self.gauge.to(device)
        return self

    def gauge_weights(self):
        if (self.gauge is not None) and (self.weight is not None):
            size = self.weight.size()
            self.weight.data = nn.Parameter(self.weight.reshape(-1,self.gauge.size(0)).data.mm(self.gauge)).reshape(*size)

    def get_weights(self):
        self.gauge_weights()
        return self.weight

    def forward(self, x):
        pass

    def backward(self, h):
        pass

class DenseSubEdge(AbstractSubEdge):
    def __init__(self, in_features, in_channels, out_channels, gauge=None, weights=None):
        super(DenseSubEdge, self).__init__(in_features, in_channels, out_channels, gauge)
        self.weight = nn.Parameter(2*(torch.rand(out_channels, in_features*in_channels)-0.5)/math.sqrt(self.in_features*self.in_channels))
        nn.init.xavier_uniform(self.weight)
        if weights is not None:
            self.weight = weights

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h = F.linear(x, self.weight)
        return h

    def backward(self, h):
        h = h.reshape(h.size(0), -1)
        x = F.linear(h, self.weight.t())
        return x

class FilterbankSubEdge(AbstractSubEdge):
    def __init__(self, in_features, in_channels, out_channels, window_size, strides, gauge):
        super(FilterbankSubEdge, self).__init__(in_features, in_channels, out_channels, gauge)
        self.window_size = window_size
        self.strides = strides

        self.Nk, self.fbank = self.build_fbank()
        self.weight = nn.Parameter(2*(torch.rand(self.Nk, self.window_size*self.in_channels)-0.5)/math.sqrt(self.window_size*self.in_channels))

    def to(self, device):
        super(FilterbankSubEdge, self).to(device)
        self.fbank = self.fbank.to(device)
        return self

    def build_fbank(self):
        Nk = self.Nk = self.out_channels*(int((self.in_features-self.window_size)/self.strides - EPS)+1)
        filters = torch.zeros(self.Nk, self.in_channels, self.window_size, self.in_channels, self.in_features)
        cursor = 0
        for j in range(int((self.in_features-self.window_size)/self.strides - EPS)):
            for k in range(self.in_channels):
                filters[self.out_channels*j:self.out_channels*(j+1),k, torch.arange(self.window_size), k, cursor+torch.arange(self.window_size)]=1
            cursor += self.strides
        for k in range(self.in_channels):
            filters[self.out_channels*int((self.in_features-self.window_size)/self.strides - EPS):, k, torch.arange(self.window_size),k,self.in_features-self.window_size+torch.arange(self.window_size)] = 1

        filters = filters.view(self.Nk, self.in_channels * self.window_size, self.in_channels * self.in_features)
        return Nk, to_sparse(filters)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).t()
        fx = torch.sparse.mm(self.fbank,x).t().view(batch_size, self.Nk, -1)
        h = fx * self.weight[None]
        return h.sum(-1)

    def backward(self, h):
        batch_size = h.size(0)
        wh = (h.view(batch_size, -1, 1) * self.weight[None]).view(batch_size,-1).t()
        x = torch.sparse.mm(self.fbank.t(),wh).t()
        return x

class ConvolutionalSubEdge(AbstractSubEdge):
    def __init__(self, in_features, in_channels, out_channels, window_size, strides, opad = 0, gauge = None):
        super(ConvolutionalSubEdge, self).__init__(in_features, in_channels, out_channels, gauge)
        self.window_size = window_size
        self.strides = strides
        self.opad = opad
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.window_size))
        self.conv = (lambda x : F.conv1d(x, self.weight, stride = self.strides))
        self.reverse = (lambda h : F.conv_transpose1d(h, self.weight, stride = strides, output_padding = self.opad))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_channels, -1)
        h = self.conv(x)
        return h.reshape(batch_size,-1)

    def backward(self, h):
        batch_size = h.size(0)
        h = h.reshape(batch_size,self.out_channels, -1)
        x = self.reverse(h)
        return x.view(batch_size,-1)
