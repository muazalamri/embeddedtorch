import torch
def tensor2cpp(tensor,dtype)->str:
    tensor=list(tensor.cpu().flatten().numpy().tolist())
    return str(tensor).replace('[','{').replace(']','}')

def linear2cpp(layer:torch.nn.Module,layer_num:int)->str:
    weight=layer.weight.detach().cpu()
    bias=layer.bias.detach().cpu()
    return f"""Eigen::Matrix<float,{weight.shape[0]},{weight.shape[1]}> weight_{layer_num}={tensor2cpp(weight,float)};
Eigen::Matrix<float,{bias.shape[0]},1> bias_{layer_num}={tensor2cpp(bias,float)};
"""
def cpp_code(layers:list)->str:
    code='#include "func.cpp"\n'
    for i,layer in enumerate(layers):
        if hasattr(layer,'to_cpp'):
            code+=layer.to_cpp(i)+'\n'
    
    return code
if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))