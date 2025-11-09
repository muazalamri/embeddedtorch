import torch
import torch.nn.functional as F
from torch import nn
from ..cpp import tensor2cpp,linear2cpp,conv2D2cpp,conv1D2cpp,conv3D2cpp

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,dtype=torch.float32):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse).to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.embedding(x)
    def to_cpp(self):
        return f"embedding(x, weight, {self.embedding.num_embeddings}, {self.embedding.embedding_dim}, {self.embedding.padding_idx}), {self.dtype})"