import torch
import torch.nn.functional as F
class addLayer(torch.nn.Module):
    def __init__(self):
        super(addLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2
    def to_cpp(self):
        return "add(x1, x2)"
class subLayer(torch.nn.Module):
    def __init__(self):
        super(subLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 - x2
    def to_cpp(self):
        return "sub(x1, x2)"
class mulLayer(torch.nn.Module):
    def __init__(self):
        super(mulLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 * x2
    def to_cpp(self):
        return "mul(x1, x2)"
class divLayer(torch.nn.Module):
    def __init__(self):
        super(divLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 / x2
    def to_cpp(self):
        return "div(x1, x2)"
class powLayer(torch.nn.Module):
    def __init__(self):
        super(powLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 ** x2
    def to_cpp(self):
        return "pow(x1, x2)"
class matmulLayer(torch.nn.Module):
    def __init__(self):
        super(matmulLayer, self).__init__()

    def forward(self, x1, x2):
        return torch.matmul(x1, x2)
    def to_cpp(self):
        return "matmul(x1, x2)"
class catLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(catLayer, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=self.dim)
    def to_cpp(self):
        return f"cat(x1, x2, {self.dim})"
class stackLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(stackLayer, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return torch.stack((x1, x2), dim=self.dim)
    def to_cpp(self):
        return f"stack(x1, x2, {self.dim})"
class meanLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(meanLayer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
    def to_cpp(self):
        return f"mean(x, {self.dim}, {str(self.keepdim).lower()})"
class sumLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(sumLayer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)
    def to_cpp(self):
        return f"sum(x, {self.dim}, {str(self.keepdim).lower()})"
class maxLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(maxLayer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.max(x, dim=self.dim, keepdim=self.keepdim).values
    def to_cpp(self):
        return f"max(x, {self.dim}, {str(self.keepdim).lower()})"
class minLayer(torch.nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(minLayer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.min(x, dim=self.dim, keepdim=self.keepdim).values
    def to_cpp(self):
        return f"min(x, {self.dim}, {str(self.keepdim).lower()})"
class reluLayer(torch.nn.Module):
    def __init__(self):
        super(reluLayer, self).__init__()

    def forward(self, x):
        return F.relu(x)
    def to_cpp(self):
        return "relu(x)"
class sigmoidLayer(torch.nn.Module):
    def __init__(self):
        super(sigmoidLayer, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)
    def to_cpp(self):
        return "sigmoid(x)"
class tanhLayer(torch.nn.Module):
    def __init__(self):
        super(tanhLayer, self).__init__()

    def forward(self, x):
        return torch.tanh(x)
    def to_cpp(self):
        return "tanh(x)"
class softmaxLayer(torch.nn.Module):
    def __init__(self, dim=1):
        super(softmaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)
    def to_cpp(self):
        return f"softmax(x, {self.dim})"
class logLayer(torch.nn.Module):
    def __init__(self):
        super(logLayer, self).__init__()

    def forward(self, x):
        return torch.log(x)
    def to_cpp(self):
        return "log(x)"
class expLayer(torch.nn.Module):
    def __init__(self):
        super(expLayer, self).__init__()

    def forward(self, x):
        return torch.exp(x)
    def to_cpp(self):
        return "exp(x)"
class sqrtLayer(torch.nn.Module):
    def __init__(self):
        super(sqrtLayer, self).__init__()

    def forward(self, x):
        return torch.sqrt(x)
    def to_cpp(self):
        return "sqrt(x)"
class absLayer(torch.nn.Module):
    def __init__(self):
        super(absLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)
    def to_cpp(self):
        return "abs(x)"