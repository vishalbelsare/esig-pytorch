import iisignature
import torch
from torch.autograd import Function

class SigFn(Function):
    def __init__(self, m):
        super(SigFn, self).__init__()
        self.m = m
    def forward(self, X):
        result=iisignature.sig(X.detach().numpy(), self.m)
        self.save_for_backward(X)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        (X,) = self.saved_tensors
        result = iisignature.sigbackprop(grad_output.numpy(), X.detach().numpy(),self.m)
        return torch.FloatTensor(result)

class LogSigFn(Function):
    def __init__(self, s, method):
        super(LogSigFn, self).__init__()
        self.s = s
        self.method = method
    def forward(self,X):
        result=iisignature.logsig(X.detach().numpy(), self.s, self.method)
        self.save_for_backward(X)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        (X,) = self.saved_tensors
        g = grad_output.numpy()
        result = iisignature.logsigbackprop(g, X.detach().numpy(),self.s,self.method)
        return torch.FloatTensor(result)

def Sig(X,m):
    return SigFn(m)(X)

def LogSig(X,s,method=""):
    return LogSigFn(s,method)(X)

if __name__ == "__main__":

    # importing esig in pytorch...
    from tosig_pytorch import EsigPyTorch
    sig_PT = EsigPyTorch()

    # input streams
    inp = torch.randn((8, 2), dtype=torch.double, requires_grad=True)
    inp2 = torch.tensor(inp.data, dtype=torch.float, requires_grad=True)

    # Our sigs
    my_sigs = sig_PT.stream2sig(inp, 2)

    # Jeremy's sigs
    j_sigs = Sig(inp2, 2)

    print('This is our signature: \n {} \n \n'.format(my_sigs.data[1:]))
    print('This is Jeremy s signature: \n {} \n \n'.format(j_sigs.data))

    # Our gradients with backprop
    my_sigs.backward(torch.randn(my_sigs.size(), dtype=torch.double))

    # Jeremy's gradients with backprop
    j_sigs.backward(torch.randn(j_sigs.size()))

    print('\n This is our gradient with respect to the input: \n {} \n \n'.format(inp.grad))
    print('\n This is Jeremy s gradient with respect to the input: \n {} \n \n'.format(inp2.grad))






    a = SigFn(2)