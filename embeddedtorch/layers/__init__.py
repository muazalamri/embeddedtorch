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
from cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp # type: ignore
from .ops_collections import models_col,models_row # type: ignore
from .conv import Conv1dLayer,Conv2dLayer,Conv3dLayer # type: ignore
from .embedding import EmbeddingLayer # type: ignore
from .linear import LinearLayer # type: ignore
from .norms import batchNorm1dLayer,batchNorm2dLayer,batchNorm3dLayer,LayerNorm1dLayer,LayerNorm2dLayer,LayerNorm3dLayer # type: ignore
from .pooling import MaxPool1dLayer,MaxPool2dLayer,MaxPool3dLayer # type: ignore
from .upsample import UpSample1dLayer,UpSample2dLayer,UpSample3dLayer # type: ignore
class EmbaeddableModel(nn.Module):
    def __init__(self, dtype:torch.dtype):
        super().__init__() # type: ignore
        self.list=nn.ModuleList()
    def forward(self,x:torch.Tensor):
        for layer in self.list:
            x = layer(x)
        return x
    def add_layer(self,layer: nn.Module):
        self.list.append(layer)

class flattenLayer(torch.nn.Module):
    def __init__(self,inputRank:int=4,outputRank:int=2,dtype:torch.dtype=torch.float32,start_dim:int=1,end_dim:int=-1):
        super().__init__() # type: ignore
        self.inputRank=inputRank
        self.outputRank=outputRank
        self.startdim = start_dim
        self.end_dim = end_dim
        self.dtype=dtype

    def forward(self, x:torch.Tensor):
        return torch.flatten(x, start_dim=self.startdim, end_dim=self.end_dim)
    def to_cpp(self,layer_num:int):
        return "", "", f"flatten<float, {self.inputRank}, {self.outputRank}> (input{'_'+str(layer_num-1) if layer_num>0 else ''}, {self.startdim}, {self.end_dim});"
class reshapeLayer(torch.nn.Module):
    def __init__(self,shape:torch.Size,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.shape = shape
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return torch.reshape(x, self.shape)
    def to_cpp(self):
        return f"reshape(x, {self.shape},{self.dtype})"
class dropoutLayer(torch.nn.Module):
    def __init__(self, p:float=0.5,dtype:torch.dtype=torch.float32):
        super(dropoutLayer, self, dtype=torch.float32).__init__() # type: ignore
        self.p = p
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return F.dropout(x, p=self.p, training=self.training)
    def to_cpp(self):
        return f"dropout(x, {self.p}), {self.dtype})"
