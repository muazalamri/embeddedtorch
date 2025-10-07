import torch
def tensor2cpp(tensor,dtype)->str:
    tensor=list(tensor.cpu().flatten().numpy().tolist())
    return str(tensor).replace('[','{').replace(']','}')

if __name__=="__main__":
    a=torch.randn(3,4,5)
    print(tensor2cpp(a,float))