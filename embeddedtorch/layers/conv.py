import torch
import torch.nn.functional as F # type: ignore
from torch import nn
import sys
sys.path.append("../")
from cpp import tensor2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp

class Conv2dLayer(nn.Module):
        def __init__(self, in_channels:int, out_channels:int , kernel_size:int|tuple[int,int],stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=True,dtype:torch.dtype=torch.float32):
            super(Conv2dLayer, self).__init__() # type: ignore
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
            self.dtype = dtype

        def forward(self, x:torch.Tensor):
            return self.conv(x)
        def to_cpp(self,layer_num:int):
            shape=self.conv.weight.shape
            return conv2D2cpp(layer_num,tensor2cpp(self.conv.weight.reshape(shape[1],shape[0],1,shape[2],shape[3]),torch.float),self.conv.in_channels,self.conv.out_channels,list(self.conv.kernel_size),list(self.conv.stride),list([int(i) for i in self.conv.padding]))
class Conv1dLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=True,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.conv(x)
    def to_cpp(self,layer_num:int):
        return conv1D2cpp(layer_num=layer_num,kerVal=tensor2cpp(self.conv.weight,torch.float),chanel_in=self.conv.in_channels,chanel_out=self.conv.out_channels,kernal_size=list(self.conv.kernel_size),padding_left=int(self.conv.padding[0]),padding_right=0,stridesVal=tensor2cpp(torch.tensor(self.conv.stride),dtype=torch.float))
class Conv3dLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, dtype:torch.dtype=torch.float32):
        super(Conv3dLayer, self).__init__() # type: ignore
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.conv(x)
    def to_cpp(self, layer_num:int):
        return conv3D2cpp(layer_num=layer_num,chanel_out=self.conv.out_channels,chanel_in=self.conv.in_channels,kerVal=tensor2cpp(self.conv.weight,torch.float),strides=list(self.conv.stride),kernal_size=list(self.conv.kernel_size),stridesVal=str(list(self.conv.stride)),padding=list([int(i) for i in self.conv.padding]))
