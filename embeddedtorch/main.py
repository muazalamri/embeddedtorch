from layers import EmbaeddableModel,LinearLayer,flattenLayer,Conv1dLayer,MaxPool1dLayer
from cpp import cpp_code,write_dep
import torch
from operitions import reluLayer


if __name__=="__main__":
    model=EmbaeddableModel(torch.float32)
    model.add_layer(Conv1dLayer(2,2,1,1,(0,1)))
    model.add_layer(MaxPool1dLayer((2,2)))
    model.add_layer(flattenLayer(4))
    model.add_layer(LinearLayer(16,8,dtype=torch.float32))#2
    model.add_layer(reluLayer(dtype=torch.float32))
    model.add_layer(LinearLayer(8,4,dtype=torch.float32))#3
    model.add_layer(reluLayer(dtype=torch.float32))
    model.add_layer(LinearLayer(4,2,dtype=torch.float32))#4
    model.add_layer(reluLayer(dtype=torch.float32))
    model.add_layer(LinearLayer(2,1,dtype=torch.float32))#5
    model.add_layer(reluLayer(dtype=torch.float32))
    write_dep('out')
    with open("out/code.cpp","w",encoding="utf-8") as f:
        print(cpp_code(model.list),file=f)