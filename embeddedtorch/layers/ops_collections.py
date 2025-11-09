class models_col(nn.Module):
    def __init__(self, models:list):
        super(models_col, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        for model in self.models:
            x = model(x)
        return x
    def to_cpp(self):
        code=""
        for model in self.models:
            code+=model.to_cpp()+"\n"
        return code
class models_row(nn.Module):
    def __init__(self, models:list,split_points:list[int],split_dim:int=1):
        super(models_row, self).__init__()
        self.models = nn.ModuleList(models)
        self.split_points = split_points
        self.split_dim = split_dim

    def forward(self, x):
        split_x = torch.split(x, self.split_points, dim=1)
        outputs = [model(part) for model, part in zip(self.models, split_x)]
        return torch.cat(outputs, dim=1)
    def to_cpp(self):
        code=""
        for model in self.models:
            code+=model.to_cpp()+"\n"
        return code