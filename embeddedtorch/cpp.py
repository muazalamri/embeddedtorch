"""# C++
Tools to transform model to cpp
"""
import torch
from template import cpp_tempalte,funcs_code
def tensor2cpp(tensor:torch.Tensor,dtype:torch.dtype)->str:
    out_tensor:str=str(list(tensor.to(dtype=dtype).detach().numpy().tolist()))
    return out_tensor.replace('[','{').replace(']','}')

def linear2cpp(layer:torch.nn.Linear,layer_num:int)->tuple[str,str,str]:
    weight=layer.weight.T.detach().cpu()
    bias=layer.bias.detach().cpu()
    return f"""Tensor<float,{len(weight.shape)}> weight_{layer_num}({weight.shape[0]},{weight.shape[1]});
Tensor<float,1> bias_{layer_num}({bias.shape[-1]});
""",f"""
    weight_{layer_num}.setValues({tensor2cpp(weight,torch.float)});
    bias_{layer_num}.setValues({tensor2cpp(bias,torch.float)});""",f"linearLayer<float, 2, 2>(input{'_'+str(layer_num-1) if layer_num > 0 else ''}, weight_{layer_num}, bias_{layer_num});"""
def activiton2cpp(name:str,inputRank:int,layer_num:int)->tuple[str,str,str]:
    return "", "", f"{name}<float, {inputRank}>(input" + ("_"+str(layer_num-1) if layer_num > 0 else "") + ");"
def conv2D2cpp(layer_num:int,kerVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],strides:list[int],padding:list[int])->tuple[str,str,str]:
    init=f"""Tensor<float, 5> ker_{layer_num}({chanel_in}, {chanel_out}, 1, {kernal_size[0]}, {kernal_size[1]});
Eigen::array<int, 3> strides_{layer_num};
Eigen::array<std::pair<int, int>, 4> padding_{layer_num};\n"""
    sets=f"""ker_{layer_num}.setValues({kerVal});
    strides_{layer_num}={{1, 1, 1}};
    padding_{layer_num}= {{std::make_pair(0, 0), std::make_pair(0, 0), std::make_pair({padding[0]}, {padding[1]}), std::make_pair({padding[0]}, {padding[1]})}};"""
    call=f'Conv<float, 4, 4, 5, 3, 4>(input{"_"+str(layer_num-1) if layer_num > 0 else ""}.shuffle(F_L).eval(), ker_{layer_num}, strides_{layer_num}, padding_{layer_num}).shuffle(F_L).eval();'
    return init,sets,call
def conv1D2cpp(layer_num:int,kerVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],padding_left:int,padding_right:int,stridesVal:str)->tuple[str,str,str]:
    
    init=f"""Tensor<float, 4> ker_{layer_num}({chanel_out}, {chanel_in}, 1, {kernal_size[0]});
Eigen::array<int, 2> strides_{layer_num};
Eigen::array<std::pair<int, int>, 3> padding_{layer_num};\n"""
    sets=f"""ker_{layer_num}.setValues({kerVal});
    strides_{layer_num}={stridesVal};
    padding_{layer_num}= {{std::make_pair(0, 0), std::make_pair({padding_left}, {padding_right} , std::make_pair(0, 0)}};"""
    call=f'Conv<float, 3, 3, 4, 2, 3>(input{"_"+str(layer_num-1) if layer_num > 0 else ""}.shuffle(S_1D).eval(), ker_{layer_num}, strides_{layer_num}, padding_{layer_num}).shuffle(S_1D).eval();'
    return init,sets,call
def conv3D2cpp(layer_num:int,kerVal:str,stridesVal:str,chanel_in:int,chanel_out:int,kernal_size:list[int],strides:list[int],padding:list[int])->tuple[str,str,str]:
    init=f"""Tensor<float, 5> ker_{layer_num}({chanel_out}, {chanel_in}, {kernal_size[0]}, {kernal_size[1]}, {kernal_size[2]});
Eigen::array<int, 3> strides_{layer_num};
Eigen::array<std::pair<int, int>, 4> padding_{layer_num};"""
    sets=f"""   ker_{layer_num}.setValues({kerVal});
strides_{layer_num}={{{stridesVal}}};
padding_{layer_num}= {{std::make_pair(0, 0), std::make_pair({padding}, {padding}), std::make_pair({padding}, {padding}), std::make_pair({padding}, {padding})}};"""
    call=f"conv3DLayer<float>(input{'_'+str(layer_num) if layer_num>0 else ''}, ker_{layer_num}, strides_{layer_num}, padding_{layer_num});"
    return init,sets,call
def flatten2cpp(InputRank:int,OutputRank:int,start_dim:int,end_dim:int,layer_num:int)->tuple[str,str,str]:
    return "", "", f"flatten<float, {InputRank}, {OutputRank}> (input{'_'+str(layer_num-1) if layer_num > 0 else ''}, {start_dim}, {end_dim});"
def write_dep(folder:str='out'):
    open("out/func.hpp","w",encoding="utf-8").write(funcs_code)
def cpp_code(layers:torch.nn.ModuleList,folder:str)->str:
    init_layers:str=""
    set_values:str=""
    sets=""
    layer_count=len(layers)
    body="\n"
    for i,layer in enumerate(layers):
        init,sets,body_line=layer.to_cpp(i) # type: ignore
        init_layers+=init # type: ignore
        set_values+=sets # type: ignore
        if i<layer_count-1:body+=f"    auto input_{i}= {body_line}\n"
        else:body+=f"    auto output= {body_line}\n"
    return cpp_tempalte.format(init_layers=init_layers,body=body,set_values=set_values) # type: ignore