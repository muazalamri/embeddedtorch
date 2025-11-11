import torch
import torch.nn.functional as F # type: ignore
from torch import nn
import sys
sys.path.append("../")

class UpSample2dLayer(nn.Module):
    def __init__(self, size:None|tuple[int,int]=None,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.upsample = nn.Upsample(size=size).to(dtype=dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample2d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"
class UpSample3dLayer(nn.Module):
    def __init__(self, size:None|tuple[int,int]=None,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.upsample = nn.Upsample(size=size).to(dtype=dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample3d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"
class UpSample1dLayer(nn.Module):
    def __init__(self, size:None|tuple[int,int]=None,dtype:torch.dtype=torch.float32):
        super().__init__() # type: ignore
        self.upsample = nn.Upsample(size=size).to(dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample1d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"