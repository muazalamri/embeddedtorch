import torch
import torch.nn.functional as F
from torch import nn
from ..cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp

class UpSample2dLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None,dtype=torch.float32):
        super(UpSample2dLayer, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample2d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"
class UpSample3dLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None,dtype=torch.float32):
        super(UpSample3dLayer, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample3d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"
class UpSample1dLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None,dtype=torch.float32):
        super(UpSample1dLayer, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.upsample(x)
    def to_cpp(self):
        return f"upsample1d(x, {self.upsample.size}, {self.upsample.scale_factor}, {self.upsample.mode}, {self.upsample.align_corners}), {self.dtype})"