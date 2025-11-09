import torch
import torch.nn.functional as F
from torch import nn
from ..cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp

class batchNorm1dLayer(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32):
        super(batchNorm1dLayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.bn(x)
    def to_cpp(self):
        return f"batch_norm_1d(x,{self.bn.num_features}, {self.bn.eps}, {self.bn.momentum}, {self.bn.affine}), {self.dtype})"
class batchNorm2dLayer(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32):
        super(batchNorm2dLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.bn(x)
    def to_cpp(self):
        return f"batch_norm_2d(x,{self.bn.num_features}, {self.bn.eps}, {self.bn.momentum}, {self.bn.affine}), {self.dtype})"
class batchNorm3dLayer(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32):
        super(batchNorm3dLayer, self).__init__()
        self.bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.bn(x)
    def to_cpp(self):
        return f"batch_norm_3d(x,{self.bn.num_features}, {self.bn.eps}, {self.bn.momentum}, {self.bn.affine}), {self.dtype})"
class LayerNorm1dLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,dtype=torch.float32):
        super(LayerNorm1dLayer, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.ln(x)
    def to_cpp(self):
        return f"layer_norm_1d(x,{self.ln.normalized_shape}, {self.ln.eps}, {self.ln.elementwise_affine}), {self.dtype})"
class LayerNorm2dLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,dtype=torch.float32):
        super(LayerNorm2dLayer, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.ln(x)
    def to_cpp(self):
        return f"layer_norm_2d(x,{self.ln.normalized_shape}, {self.ln.eps}, {self.ln.elementwise_affine}), {self.dtype})"
class LayerNorm3dLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,dtype=torch.float32):
        super(LayerNorm3dLayer, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.ln(x)
    def to_cpp(self):
        return f"layer_norm_3d(x,{self.ln.normalized_shape}, {self.ln.eps}, {self.ln.elementwise_affine}), {self.dtype})"
