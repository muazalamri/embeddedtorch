import torch
def tensor2cpp(tensor,dtype)->str:
    tensor=list(tensor.cpu().numpy().tolist())
    return str(tensor).replace('[','{').replace(']','}')

def linear2cpp(layer:torch.nn.Module,layer_num:int)->str:
    weight=layer.weight.detach().cpu()
    bias=layer.bias.detach().cpu()
    return f"""Eigen::Matrix<float,{weight.shape[0]},{len(weight.shape)}> weight_{layer_num}({weight.shape[1]},{weight.shape[0]});
    float d_w_{layer_num}[] = {tensor2cpp(weight,float)};
    weight_{layer_num}.setValues(d_w_{layer_num});
    Eigen::Matrix<float,{bias.shape[0]},1> bias_{layer_num}({bias.shape[0]});
    float d_b_{layer_num}[] = {tensor2cpp(bias,float)};
    bias_{layer_num}.setValues(d_b_{layer_num});
"""
def write_dep():
    with open("classes.cpp","r",encoding="utf-8") as f:
        open("out/classes.cpp","w",encoding="utf-8").write(f.read())
    with open("func.cpp","r",encoding="utf-8") as f:
        open("out/func.cpp","w",encoding="utf-8").write(f.read())
def cpp_code(layers:list)->str:
    init_layers=""
    body="\n"
    code=open("template.cpp","r").read()
    for i,layer in enumerate(layers):
        if hasattr(layer,'to_cpp'):
            init_layers+=layer.to_cpp(i)
            body+=f"    auto input= {layer.cpp_name+'_'+str(i)}(input);\n"
    return code.format(init_layers=init_layers,body=body)
if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))