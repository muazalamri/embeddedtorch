import torch
def tensor2cpp(tensor,dtype)->str:
    tensor=torch.flatten(tensor)
    ndim=len(tensor.shape)
    if ndim==0:
        return str(tensor)
    tensor=str([dtype(i) for i in list(tensor)])
    return tensor.replace('[','{').replace(']','}')
if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))