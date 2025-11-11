import torch
import torch.nn.functional as F # type: ignore
from torch import nn
import sys
from cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp # type: ignore
sys.path.append("../")

class LinearLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool=True,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.linear = nn.Linear(in_features, out_features, bias=bias).to(dtype)
        self.dtype = dtype
        self.cpp_name = "linear"
    def cpp_call_pram(self):
        return "<{dtype},{in_features},{out_features}>".format(dtype=self.dtype,in_features=self.linear.in_features,out_features=self.linear.out_features)
    def forward(self, x:torch.Tensor):
        return self.linear(x)
    def to_cpp(self,layer_num:int):
        return linear2cpp(self.linear,layer_num)