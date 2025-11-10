"""# Layers Module
contain:
- LinearLayer
- Conv2dLayer
- LinearLayer
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
"""
import torch
import torch.nn.functional as F
from torch import nn
from cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp
from .ops_collections import *
from .conv import *
from .embedding import *
from .linear import *
from .norms import *
from .pooling import *
from .upsample import *
class EmbaeddableModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.list=nn.ModuleList()
    def forward(self,x):
        for layer in self.list:
            x = layer(x)
        return x
    def add_layer(self,layer):
        self.list.append(layer)

class flattenLayer(torch.nn.Module):
    def __init__(self,inputRank,outputRank=2,dtype=torch.float32,start_dim=1,end_dim=-1):
        super(flattenLayer, self).__init__()
        self.inputRank=inputRank
        self.outputRank=outputRank
        self.startdim = start_dim
        self.end_dim = end_dim
        self.dtype=dtype

    def forward(self, x):
        return torch.flatten(x, start_dim=self.startdim, end_dim=self.end_dim)
    def to_cpp(self,layer_num):
        return "", "", f"flatten<float, {self.inputRank}, {self.outputRank}> (input{'_'+str(layer_num-1) if layer_num>0 else ''}, {self.startdim}, {self.end_dim});"
class reshapeLayer(torch.nn.Module):
    def __init__(self,shape,dtype=torch.float32):
        super(reshapeLayer, self).__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x):
        return torch.reshape(x, self.shape)
    def to_cpp(self):
        return f"reshape(x, {self.shape},{self.dtype})"
class dropoutLayer(torch.nn.Module):
    def __init__(self, p=0.5,dtype=torch.float32):
        super(dropoutLayer, self, dtype=torch.float32).__init__()
        self.p = p
        self.dtype = dtype

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)
    def to_cpp(self):
        return f"dropout(x, {self.p}), {self.dtype})"
