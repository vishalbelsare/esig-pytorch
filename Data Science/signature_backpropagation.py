import iisignature
import numpy as np
import math
import time
import seaborn as sns
import scipy.sparse as sp
import pandas as pd
import torch
from torch.autograd import Function
import tensorflow as tf

##text, index -> (either number or [res, res]), newIndex
#def parseBracketedExpression(text,index):
#    if(text[index] == '['):
#        left, m = parseBracketedExpression(text,index + 1)
#        right, n = parseBracketedExpression(text,m + 1)
#        return [left,right],n + 1
#    else:
#        n = 0
#        while(n < len(text) and text[index + n] in ['1','2','3','4','5','6','7','8','9']): #this should always just happen once if input is a bracketed expression of
#                                                                                           #letters
#            n = n + 1
#        return int(text[index:index + n]), index + n

##print (parseBracketedExpression("[23,[2,[[22,1],2]]]",0))

##bracketed expression, dim -> numpy array of its value, depth
#def multiplyOut(expn, dim):
#    if isinstance(expn,list):
#        left, leftDepth = multiplyOut(expn[0],dim)
#        right,rightDepth = multiplyOut(expn[1],dim)
#        a = np.outer(left,right).flatten()
#        b = np.outer(right,left).flatten()
#        return a - b, leftDepth + rightDepth
#    else:
#        a = np.zeros(dim)
#        a[expn - 1] = 1
#        return a,1

##string of bracketed expression, dim -> numpy array of its value, depth
##for example:
##valueOfBracket("[1,2]",2) is ([0,1,-1,0],2)
#def valueOfBracket(text,dim):
#    return multiplyOut(parseBracketedExpression(text,0)[0],dim)


##inputs are values in the tensor algebra given as lists of levels (from 1 to
##level), assumed 0 in level 0.
##returns their concatenation product
#def multiplyTensor(a,b):
#    level = len(a)
#    dim = len(a[0])
#    sum = [np.zeros(dim ** m) for m in range(1,level + 1)]
#    for leftLevel in range(1,level):
#        for rightLevel in range(1, 1 + level - leftLevel):
#            sum[leftLevel + rightLevel - 1]+=np.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
#    return sum

##input is a value in the tensor algebra given as lists of levels (from 1 to
##level), assumed 0 in level 0.
##returns its exp - assumed 1 in level 0
##exp(x)-1 = x+x^2/2 +x^3/6 +x^4/24 + ...
#def exponentiateTensor(a):
#    out = [i.copy() for i in a]
#    level = len(a)
#    products = [out]
#    for m in range(2,level + 1):
#        t = multiplyTensor(a,products[-1])
#        for j in t:
#            j *= (1.0 / m)
#        products.append(t)
#    return [np.sum([p[i] for p in products],0) for i in range(level)]

##input is a value in the tensor algebra given as lists of levels (from 1 to
##level), assumed 1 in level 0.
##returns its log - assumed 0 in level 0
##log(1+x) = x -x^2/2 +x^3/3 -x^4/4 + ...
#def logTensor(a):
#    out = [i.copy() for i in a]
#    level = len(a)
#    products = [out]
#    for m in range(2,level + 1):
#        t = multiplyTensor(a,products[-1])
#        products.append(t)
#    neg = True
#    for m in range(2,level + 1):
#        for j in products[m - 1]:
#            if neg:
#                j *= (-1.0 / m)
#            else:
#                j *= (1.0 / m)
#        neg = not neg
#    return [np.sum([p[i] for p in products],0) for i in range(level)]

##given a tensor as a concatenated 1D array, return it as a list of levels
#def splitConcatenatedTensor(a, dim, level):
#    start = 0
#    out = []
#    for m in range(1,level + 1):
#        levelLength = dim ** m
#        out.append(a[start:(start + levelLength)])
#        start = start + levelLength
#    assert(start == a.shape[0])
#    return out

##returns the signature of a straight line as a list of levels, and the number
##of multiplications used
#def sigOfSegment(displacement, level):
#    d = displacement.shape[0]
#    sig = [displacement]
#    mults = 0
#    denominator = 1
#    for m in range(2,level + 1):
#        other = sig[-1]
#        mults += 2 * other.shape[0] * d
#        sig.append(np.outer(other,displacement).flatten() * (1.0 / m))
#    return sig, mults
    
##inputs are values in the tensor algebra given as lists of levels (from 1 to
##level), assumed 1 in level 0.
##returns their concatenation product, and also the number of multiplications
##used
##c.f.  multiplyTensor
#def chen(a,b):
#    level = len(a)
#    dim = len(a[0])
#    mults = 0
#    sum = [a[m] + b[m] for m in range(level)]
#    for leftLevel in range(1,level):
#        for rightLevel in range(1, 1 + level - leftLevel):
#            sum[leftLevel + rightLevel - 1]+=np.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
#            mults += a[leftLevel - 1].shape[0] * b[rightLevel - 1].shape[0]
#    return sum, mults

#def diff(a,b):
#    return numpy.max(numpy.abs(a - b))

##numpy's lstsq has a different default in different versions
##I think this function agrees with newer versions (1.14+) 
##and with scipy.
##This doesn't really affect the tests anyway.
#def lstsq(a,b):
#    rcond_to_use=max(a.shape)*np.finfo(float).eps
#    return np.linalg.lstsq(a,b,rcond=rcond_to_use)






#Finite difference derivative
#estimate bump dot (the derivative of np.sum(f(X)) wrt X at X=x)
# i.e. the change in np.sum(f(X)) caused by bump 
# with a finite difference approximation
def fdDeriv(f,x,bump,order,nosum=False):
    if order==0:#SIMPLE
        o=f(x+bump)-f(x)
    elif order ==2:#2nd order central
        o=0.5 * (f(x+bump)-f(x-bump))
    elif order ==4:#4th order central
        o=(8*f(x+bump)-8*f(x-bump)+f(x-2*bump)-f(x+2*bump))/12
    elif order ==6:#6th order central
        o=(45*f(x+bump)-45*f(x-bump)+9*f(x-2*bump)-9*f(x+2*bump)+f(x+3*bump)-f(x-3*bump))/60
    if nosum:
        return o
    return np.sum(o)

#def signature_derivative1(path, increment, depth):
#    return np.tensordot(increment, iisignature.sigjacobian(path, depth))

#def signature_derivative2(path, increment, depth, order=2):
#    sig_transform = lambda x:iisignature.sig(x,depth)
#    return fdDeriv(sig_transform, path, increment, order, nosum=True)

def signature_backpropagation(g, path, depth):
    jacobian_matrix = iisignature.sigjacobian(path, depth)
    return np.dot(jacobian_matrix, g).astype(np.float32)

# ================================================================
# PyTorch
# ================================================================
class SigActivationFnPyTorch(Function):
    def __init__(self, depth):
        super(SigActivationFnPyTorch, self).__init__()
        self.depth = depth
    def forward(self, X):
        result = iisignature.sig(X.numpy(), self.depth)
        self.save_for_backward(X)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        (X,) = self.saved_tensors
        result = signature_backpropagation(grad_output.numpy(),X.numpy(),self.depth)
        return torch.FloatTensor(result)

def SigActivationPyTorch(X, depth):
    return SigActivationFnPyTorch(depth)(X)


# ================================================================
# TensorFlow
# ================================================================

# gradient
_zero=np.array(0., dtype="float32")

# my way
def _sigGradImp(g,path,depth):
    o = []
    for p,g_individual in zip(path,g):
        o.append(signature_backpropagation(g_individual, p, depth))
    o = np.array(o, dtype=np.float32)
    return o, _zero

# Jeremy's way
#def _sigGradImp(g,path,m):
#    print(path.shape)
#    o=iisignature.sigbackprop(g,path,m)
#    return o, _zero

# op for gradient
def _sigGrad(op, grad):
    return tf.py_func(_sigGradImp,
                      [grad]+list(op.inputs),
                      [tf.float32]*2, 
                      name="SigGrad",
                      stateful=False)

## forward operations
def _sigImp(path, depth):
    return iisignature.sig(path, depth).astype(np.float32)

## op for forward operation
def Sig(path, depth):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigImp, 
                          [path,depth], 
                          tf.float32, 
                          name="Sig",
                          stateful=True) 

