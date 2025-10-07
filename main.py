from layers import EmbaeddableModel,LinearLayer
from cpp import cpp_code
import torch

if __name__=="__main__":
    model=EmbaeddableModel(torch.float32)
    model.add_layer(LinearLayer(4,3,dtype=torch.float32))
    model.add_layer(LinearLayer(3,2,dtype=torch.float32))
    model.add_layer(LinearLayer(2,1,dtype=torch.float32))
    with open("out/code.cpp","w",encoding="utf-8") as f:
        print(cpp_code(model.list),file=f)