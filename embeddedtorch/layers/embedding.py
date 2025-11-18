import torch
import torch.nn.functional as F # type: ignore
from torch import nn
import sys
sys.path.append("../")

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int,dtype:torch.dtype=torch.float32):
        super(EmbeddingLayer, self).__init__() # type: ignore
        self.embedding = nn.Embedding(num_embeddings, embedding_dim).to(dtype)
        self.dtype = dtype

    def forward(self, x:torch.Tensor):
        return self.embedding(x)
    def to_cpp(self, layer_num:int):
        return f"embedding(x, weight, {self.embedding.num_embeddings}, {self.embedding.embedding_dim}), {self.dtype})"