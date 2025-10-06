import torch
import torch.nn.functional as F
from torch import nn
class EmbaeddableModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.list=nn.ModuleList()
    def forward(self,x):
        for layer in self.list:
            x = layer(x)
        return x
    def to_cpp(self):
        
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,dtype=torch.float32):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias).to(dtype)
    def forward(self, x):
        return self.linear(x)
    def to_cpp(self):
        return f"linear(x, weight, bias)" if self.linear.bias is not None else "linear(x, weight)"
class flattenLayer(torch.nn.Module):
    def __init__(self,dim=1,dtype=torch.float32):
        super(flattenLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flatten(x, start_dim=self.dim)
    def to_cpp(self):
        return f"flatten(x, {self.dim}, {self.dim})"
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

class Conv2dLayer(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
            super(Conv2dLayer, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
            self.dtype = dtype

        def forward(self, x):
            return self.conv(x)
        def to_cpp(self):
            return f"conv2d(x, weight, bias, {self.conv.stride}, {self.conv.padding}, {self.conv.dilation}, {self.conv.groups}), {self.dtype})"
class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
        super(Conv1dLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.conv(x)
    def to_cpp(self):
        return f"conv1d(x, weight, bias, {self.conv.stride}, {self.conv.padding}, {self.conv.dilation}, {self.conv.groups}), {self.dtype})"
class Conv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,dtype=torch.float32):
        super(Conv3dLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.conv(x)
    def to_cpp(self):
        return f"conv3d(x, weight, bias, {self.conv.stride}, {self.conv.padding}, {self.conv.dilation}, {self.conv.groups}), {self.dtype})"
class MaxPool2dLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,dtype=torch.float32):
        super(MaxPool2dLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self):
        return f"max_pool2d(x, {self.pool.kernel_size}, {self.pool.stride}, {self.pool.padding}, {self.pool.dilation}, {self.pool.ceil_mode}), {self.dtype})"
class MaxPool1dLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,dtype=torch.float32):
        super(MaxPool1dLayer, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self):
        return f"max_pool1d(x, {self.pool.kernel_size}, {self.pool.stride}, {self.pool.padding}, {self.pool.dilation}, {self.pool.ceil_mode}), {self.dtype})"
class MaxPool3dLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,dtype=torch.float32):
        super(MaxPool3dLayer, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.pool(x)
    def to_cpp(self):
        return f"max_pool3d(x, {self.pool.kernel_size}, {self.pool.stride}, {self.pool.padding}, {self.pool.dilation}, {self.pool.ceil_mode}), {self.dtype})"
class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,dtype=torch.float32):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.embedding(x)
    def to_cpp(self):
        return f"embedding(x, weight, {self.embedding.num_embeddings}, {self.embedding.embedding_dim}, {self.embedding.padding_idx}), {self.dtype})"