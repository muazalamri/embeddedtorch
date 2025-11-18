import torch.nn as nn
import torch
class embeddableClass(nn.Module):
    def toCpp(self,layerNum:int)->None:
        ...
    def forward(self,x:torch.Tensor)->None:
        ...