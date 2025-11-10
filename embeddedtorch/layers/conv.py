import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.append("../")
from cpp import tensor2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp

class Conv2dLayer(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
            super(Conv2dLayer, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
            self.dtype = dtype

        def forward(self, x):
            return self.conv(x)
        def to_cpp(self,layer_num):
            shape=self.conv.weight.shape
            return conv2D2cpp(layer_num,tensor2cpp(self.conv.weight.reshape(shape[1],shape[0],1,shape[2],shape[3]),float),self.conv.in_channels,self.conv.out_channels,list(self.conv.kernel_size),list(self.conv.stride),self.conv.padding)
class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
        super(Conv1dLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.conv(x)
    def to_cpp(self,layer_num):
        return conv1D2cpp(layer_num,tensor2cpp(self.conv.weight,float),self.conv.in_channels,self.conv.out_channels,self.conv.kernel_size,self.conv.padding[0],self.conv.padding[1],tensor2cpp(torch.tensor(self.conv.stride),int))
class Conv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
        super(Conv3dLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.conv(x)
    def to_cpp(self):
        return conv3D2cpp(self.conv.out_channels,self.conv.in_channels,tensor2cpp(self.conv.weight,float),tensor2cpp(torch.tensor(self.conv.stride),int),list(self.conv.kernel_size),list(self.conv.stride),self.conv.padding[0],self.conv.padding[1])
