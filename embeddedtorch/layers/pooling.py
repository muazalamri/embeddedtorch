import torch
import torch.nn.functional as F
from torch import nn
from ..cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp
class MaxPool2dLayer(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=(2,2),dtype=torch.float32):
        super(MaxPool2dLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride).to(dtype)
        self.dtype = dtype
        self.ker_size=kernel_size
        self.stride=stride

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self,layer_num):
        return "","",f"maxPool2D<float>(input{'_'+str(layer_num-1) if layer_num>0 else ''}, {self.ker_size[0]}, {self.ker_size[1]}, {self.stride}, {self.stride})"
class MaxPool1dLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,dtype=torch.float32):
        super(MaxPool1dLayer, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self,layer_num):
        self.pool.padding
        return "", "", f"max_pool1d<float>(input{'_'+str(layer_num-1) if layer_num<0 else ''}, {self.pool.kernel_size[0]}, {self.pool.kernel_size[1]}, {self.pool.stride[0]}, {self.pool.stride[1]}, {self.pool.padding}, 0);"
class MaxPool3dLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,dtype=torch.float32):
        super(MaxPool3dLayer, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self):
        return f"max_pool3d(x, {self.pool.kernel_size}, {self.pool.stride}, {self.pool.padding}, {self.pool.dilation}, {self.pool.ceil_mode}), {self.dtype})"
