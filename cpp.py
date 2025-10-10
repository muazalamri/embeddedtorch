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
def activiton2cpp(name,inputRank:int,layer_num:int)->str:
    return "", "", f"{name}<float, {inputRank}>(input" + ('_'+str(layer_num-1) if layer_num > 0 else '') + ");"
def conv2D2cpp(layer_num:int,kerVal:str,stridesVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],strides:list[int],padding:list[int]):
    init=f"""Tensor<float, 4> ker_{layer_num}({chanel_out}, {chanel_in}, {kernal_size[0]}, {kernal_size[1]});
Eigen::array<int, 2> strides_{layer_num};
Eigen::array<std::pair<int, int>, 3> padding_{layer_num};"""
    sets=f"""   ker_{layer_num}.setValues({kerVal});
    strides_{layer_num}={{{stridesVal}}};
    padding_{layer_num}= {{std::make_pair(0, 0), std::make_pair({padding}, {padding}), std::make_pair({padding}, {padding})}};"""
    call=f'conv2DLayer<float>(input{'_'+str(layer_num) if layer_num>0 else ''}, ker_{layer_num}, strides_{layer_num}, padding_{layer_num});'
    return init,sets,call
def conv1D2cpp(layer_num:int,kerVal:str,stridesVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],padding:int):
    sets=f"""Tensor<float, 3>({chanel_out}, {chanel_in}, {kernal_size}) ker_{layer_num};
Eigen::array<int, 1> strides_{layer_num};
Eigen::array<std::pair<int, int>, 2> padding_{layer_num};
"""
    init=f"""ker_{layer_num}.setValues({kerVal});
strides_{layer_num}={{{stridesVal}}};
padding_{layer_num}={{std::make_pair(0, 0), std::make_pair({padding}, {padding})}}"""
    call=f"""conv1DLayer<float>(input{'_'+str(layer_num) if layer_num>0 else ''}, ker_{layer_num}, strides_{layer_num}, padding_{layer_num});"""
    return sets,init,call
def conv3D2cpp(layer_num:int,kerVal:str,stridesVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],strides:list[int],padding:list[int]):
    init=f"""Tensor<float, 5> ker_{layer_num}({chanel_out}, {chanel_in}, {kernal_size[0]}, {kernal_size[1]}, {kernal_size[2]});
Eigen::array<int, 3> strides_{layer_num};
Eigen::array<std::pair<int, int>, 4> padding_{layer_num};"""
    sets=f"""   ker_{layer_num}.setValues({kerVal});
strides_{layer_num}={{{stridesVal}}};
padding_{layer_num}= {{std::make_pair(0, 0), std::make_pair({padding}, {padding}), std::make_pair({padding}, {padding}), std::make_pair({padding}, {padding})}};"""
    call=f'conv3DLayer<float>(input{'_'+str(layer_num) if layer_num>0 else ''}, ker_{layer_num}, strides_{layer_num}, padding_{layer_num});'
    return init,sets,call
def flatten2cpp(InputRank:int,OutputRank:int,start_dim:int,end_dim:int,layer_num:int)->str:
    return "", "", f"flatten<float, {InputRank}, {OutputRank}> (input{'_'+str(layer_num) if layer_num > 0 else ''}, {start_dim}, {end_dim});"
def write_dep():
    with open("func.hpp","r",encoding="utf-8") as f:
        open("out/func.hpp","w",encoding="utf-8").write(f.read())
def cpp_code(layers:list)->str:
    init_layers=""
    set_values=""
    sets=""
    layer_count=len(layers)
    body="\n"
    code=open("template.cpp","r").read()
    for i,layer in enumerate(layers):
        init,sets,body_line=layer.to_cpp(i)
        init_layers+=init
        set_values+=sets
        if i<layer_count-1:body+=f"    auto input_{i}= {body_line}\n"
        else:body+=f"    auto output= {body_line}\n"
    return code.format(init_layers=init_layers,body=body,set_values=set_values)
if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))