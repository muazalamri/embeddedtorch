import torch
def tensor2cpp(tensor,dtype)->str:
    tensor=list(tensor.cpu().numpy().tolist())
    return str(tensor).replace('[','{').replace(']','}')

def linear2cpp(layer:torch.nn.Module,layer_num:int)->str:
    weight=layer.weight.T.detach().cpu()
    bias=layer.bias.detach().cpu()
    return f"""Tensor<float,{len(weight.shape)}> weight_{layer_num}({weight.shape[0]},{weight.shape[1]});
Tensor<float,1> bias_{layer_num}({bias.shape[-1]});
""",f"""
    weight_{layer_num}.setValues({tensor2cpp(weight,float)});
    bias_{layer_num}.setValues({tensor2cpp(bias,float)});""",f"linearLayer<float, 2, 2>(input{'_'+str(layer_num-1) if layer_num > 0 else ''}, weight_{layer_num}, bias_{layer_num});"
def relu2cpp(layer:torch.nn.Module,inputRank:int,layer_num:int)->str:
    return "", "", f"relu<float, {inputRank}>(input" + ('_'+str(layer_num-1) if layer_num > 0 else '') + ");"
def write_dep():
    with open("classes.cpp","r",encoding="utf-8") as f:
        open("out/classes.cpp","w",encoding="utf-8").write(f.read())
    with open("func.cpp","r",encoding="utf-8") as f:
        open("out/func.cpp","w",encoding="utf-8").write(f.read())
def cpp_code(layers:list)->str:
    init_layers=""
    set_values=""
    sets=""
    layer_count=len(layers)
    body="\n"
    code=open("template.cpp","r").read()
    for i,layer in enumerate(layers):
        if layer.cpp_name=="linear":
            init,sets,body_line=layer.to_cpp(i)
            init_layers+=init
            set_values+=sets
            if i<layer_count-1:body+=f"    auto input_{i}= {body_line}\n"
            else:body+=f"    auto output= {body_line}\n"
    return code.format(init_layers=init_layers,body=body,set_values=set_values)
if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))