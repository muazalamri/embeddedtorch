"""# Operitions for model
contain:
- addLayer
- subLayer
- mulLayer
- divLayer
- powLayer
- matmulLayer
- catLayer
- stackLayer
- meanLayer
- sumLayer
- maxLayer
- minLayer
- reluLayer
- sigmoidLayer
- tanhLayer
- softmaxLayer
- logLayer
- expLayer
- sqrtLayer
- absLayer
"""
import torch
import torch.nn.functional as F
from cpp import activiton2cpp
class addLayer(torch.nn.Module):
    def forward(self, x1, x2):
        return x1 + x2
    def to_cpp(self, layer_num:int):
        return "x1 + x2"
class subLayer(torch.nn.Module):

    def forward(self, x1, x2):
        return x1 - x2
    def to_cpp(self, layer_num:int):
        return "sub(x1, x2)"
class mulLayer(torch.nn.Module):

    def forward(self, x1, x2):
        return x1 * x2
    def to_cpp(self, layer_num:int):
        return "mul(x1, x2)"
class divLayer(torch.nn.Module):

    def forward(self, x1, x2):
        return x1 / x2
    def to_cpp(self, layer_num:int):
        return "div(x1, x2)"
class powLayer(torch.nn.Module):
    def forward(self, x1 : torch.tensor, x2 : torch.scalar_tensor):
        return x1 ** x2
    def to_cpp(self, layer_num:int):
        return "pow(x1, x2)"
class matmulLayer(torch.nn.Module):
    def __init__(self):
        super(matmulLayer, self).__init__() # type: ignore

    def forward(self, x1, x2):
        return torch.matmul(x1, x2)
    def to_cpp(self, layer_num:int):
        return "matmul(x1, x2)"
class catLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(catLayer, self).__init__() # type: ignore
        self.dim = dim

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=self.dim)
    def to_cpp(self, layer_num:int):
        return f"cat(x1, x2, {self.dim})"
class stackLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(stackLayer, self).__init__() # type: ignore
        self.dim = dim

    def forward(self, x1, x2):
        return torch.stack((x1, x2), dim=self.dim)
    def to_cpp(self, layer_num:int):
        return f"stack(x1, x2, {self.dim})"
class meanLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(meanLayer, self).__init__() # type: ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x : torch.tensor):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
    def to_cpp(self, layer_num:int):
        return f"mean(x, {self.dim}, {str(self.keepdim).lower()})"
class sumLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(sumLayer, self).__init__() # type: ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x:torch.Tensor):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)
    def to_cpp(self, layer_num:int):
        return f"sum(x, {self.dim}, {str(self.keepdim).lower()})"
class maxLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(maxLayer, self).__init__() # type: ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x:torch.Tensor):
        return torch.max(x, dim=self.dim, keepdim=self.keepdim).values
    def to_cpp(self, layer_num:int):
        return f"max(x, {self.dim}, {str(self.keepdim).lower()})"
class minLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(minLayer, self).__init__() # type: ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x:torch.Tensor):
        return torch.min(x, dim=self.dim, keepdim=self.keepdim).values
    def to_cpp(self, layer_num:int):
        return f"min(x, {self.dim}, {str(self.keepdim).lower()})"
class reluLayer(torch.nn.Module):
    def __init__(self,dtype=torch.float32):
        super(reluLayer, self).__init__() # type: ignore
        self.dtype=dtype

    def forward(self, x:torch.Tensor):
        return F.relu(x)
    def to_cpp(self,layer_num,inputRank=2):
        return activiton2cpp("relu",inputRank,layer_num)
class sigmoidLayer(torch.nn.Module):
    def __init__(self):
        super(sigmoidLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.sigmoid(x)
    def to_cpp(self,layer_num,inputRank=2):
        return activiton2cpp("sigmoid",inputRank,layer_num)
class tanhLayer(torch.nn.Module):
    def __init__(self):
        super(tanhLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.tanh(x)
    def to_cpp(self,layer_num,inputRank=2):
        return activiton2cpp("tanh",inputRank,layer_num)
class softmaxLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(softmaxLayer, self).__init__() # type: ignore
        self.dim = dim

    def forward(self, x:torch.Tensor):
        return F.softmax(x, dim=self.dim)
    def to_cpp(self,layer_num,inputRank=2):
        return activiton2cpp("softmax",inputRank,layer_num)
class logLayer(torch.nn.Module):
    def __init__(self):
        super(logLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.log(x)
    def to_cpp(self, layer_num:int):
        return "log(x)"
class expLayer(torch.nn.Module):
    def __init__(self):
        super(expLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.exp(x)
    def to_cpp(self, layer_num:int):
        return "exp(x)"
class sqrtLayer(torch.nn.Module):
    def __init__(self):
        super(sqrtLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.sqrt(x)
    def to_cpp(self, layer_num:int):
        return "sqrt(x)"
class absLayer(torch.nn.Module):
    def __init__(self):
        super(absLayer, self).__init__() # type: ignore

    def forward(self, x:torch.Tensor):
        return torch.abs(x)
    def to_cpp(self, layer_num:int):
        return "abs(x)"