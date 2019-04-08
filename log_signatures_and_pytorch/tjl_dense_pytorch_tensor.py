import doctest
import math
import torch
import bisect
from tjl_timer import timeit
try:
    import functools
except:
    import functools32 as functools

class EsigPyTorchTensor:
    
    def __init__(self, device='cpu'):
        # to make code machine-agnostic: it runs automatically on cpu or gpu (if latter is available)
        self.device = device

    # blob_size(x,N) gives the full size of the footprint of a tensor algebra element with x letters and developed to N layers
    # level -1 is used for the empty or minimal zero tensor and points to just after the information that defines the shape of the data
    # ie the beginning of the actual data for the tensor if any
    @functools.lru_cache(maxsize=None)
    def blob_size(self, width, max_degree_included_in_blob=-1):
        """ 
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.blob_size(x,y) for (x,y) in [(3,1),(3,0),(3,3),(2,6)]]
        [5, 2, 41, 128]
        >>> [pyt.blob_size(x) for x in [3,0]]
        [1, 1]

        """
        if max_degree_included_in_blob >= 0:
            if width == 0:
                return 2
            if width == 1:
                return max_degree_included_in_blob + 2
            return 1 + (-1 + width ** (1 + max_degree_included_in_blob)) // (-1 + width)
        else:
            return 1

    # gives the tuple that defines the shape of the tensor component at given degree
    @functools.lru_cache(maxsize=None)
    def tensor_shape(self, degree, width):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.tensor_shape(x,y) for (x,y) in [(3,5),(1,5),(0,5)]]
        [(5, 5, 5), (5,), ()]
        """
        return tuple([width for i in range(degree)])

    ### the degree of the smallest tensor whose blob_size is at least blobsz
    ### layers(blobsz, width) := inf{k : blob_size(width, k) >= blobsz}
    @functools.lru_cache(maxsize=None)
    def layers(self, blobsz, width):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.layers(pyt.blob_size(x, y) + z, x) for x in [2] for y in [0,1,17] for z in [-1,0,1]]
        [-1, 0, 1, 1, 1, 2, 17, 17, 18]
        >>> [(z, pyt.layers(z,2), pyt.blob_size(2,z), pyt.layers(pyt.blob_size(2,z),2)) for z in range(9)]
        [(0, -1, 2, 0), (1, -1, 4, 1), (2, 0, 8, 2), (3, 1, 16, 3), (4, 1, 32, 4), (5, 2, 64, 5), (6, 2, 128, 6), (7, 2, 256, 7), (8, 2, 512, 8)]

        """
        return next((k for k in range(-1, blobsz) if self.blob_size(width, k) >= blobsz), None)
    
    @functools.lru_cache(maxsize=None)
    def layers_new(self, blobsz, width): 
        return bisect.bisect([self.blob_size(width, k) for k in range(-1, blobsz)], blobsz) - 1

    def blob_overflow(self, x, N):
        return self.layers(self.blob_size(x, N), N) != x

    def blob_misssize(self, bs, N):
        return self.blob_size(self.layers(bs, N), N) != bs

    def zero(self, width, depth=-1):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.zero(x,y) for x in range(1) for y in range(2)]
        [tensor([0., 0.], dtype=torch.float64), tensor([0., 0.], dtype=torch.float64)]
        """
        ans = torch.zeros(self.blob_size(width, depth), dtype=torch.double)
        ans[0:1] = torch.tensor([width], dtype=torch.double)
        return ans

    def one(self, width, depth=0):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.one(x,y) for x in range(3) for y in range(2)]
        [tensor([0., 1.], dtype=torch.float64), tensor([0., 1.], dtype=torch.float64), tensor([1., 1.], dtype=torch.float64), tensor([1., 1., 0.], dtype=torch.float64), tensor([2., 1.], dtype=torch.float64), tensor([2., 1., 0., 0.], dtype=torch.float64)]
        """
        ans = torch.zeros(self.blob_size(width, depth), dtype=torch.double)
        ans[0:2] = torch.tensor([width, 1.], dtype=torch.double)
        return ans

    # all ones not useful except for testing
    def ones(self, width, depth=0):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.ones(x,y) for x in range(3) for y in range(2)]
        [tensor([0., 1.], dtype=torch.float64), tensor([0., 1.], dtype=torch.float64), tensor([1., 1.], dtype=torch.float64), tensor([1., 1., 1.], dtype=torch.float64), tensor([2., 1.], dtype=torch.float64), tensor([2., 1., 1., 1.], dtype=torch.float64)]
        """
        ans = torch.ones(self.blob_size(width, depth), dtype=torch.double)
        ans[0:2] = torch.tensor([width, 1.0], dtype=torch.double)
        return ans

    # count entries not useful except for testing
    def arange(self, width, depth=0):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> [pyt.arange(x,y) for x in range(3) for y in range(2)]
        [tensor([0., 1.], dtype=torch.float64), tensor([0., 1.], dtype=torch.float64), tensor([1., 1.], dtype=torch.float64), tensor([1., 1., 2.], dtype=torch.float64), tensor([2., 1.], dtype=torch.float64), tensor([2., 1., 2., 3.], dtype=torch.float64)]
        """
        ans = torch.arange(self.blob_size(width, depth), dtype=torch.double)
        ans[0:2] = torch.tensor([width, 1.0], dtype=torch.double)
        return ans

    def tensor_add(self, lhs, rhs):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> pyt.tensor_add(pyt.arange(3,2), pyt.arange(3,2))
        tensor([ 3.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20., 22., 24., 26.],
               dtype=torch.float64)
        >>> pyt.tensor_add(pyt.arange(3,2), pyt.arange(4,2))
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "<stdin>", line 120, in tensor_add
        ValueError: ('different width tensors cannot be added:', tensor(3., dtype=torch.float64), '!=', tensor(4., dtype=torch.float64))
        >>> 
        """
        if int(rhs[0:1]) != int(lhs[0:1]):
            raise ValueError(
                "different width tensors cannot be added:", lhs[0], "!=", rhs[0]
            )
        if lhs.nelement() >= rhs.nelement():
            ans = lhs
            ans[1 : rhs.nelement()] += rhs[1:]
        else:
            ans = rhs
            ans[1 : lhs.nelement()] += lhs[1:]
        return ans

    def pytorch_outer_tensordot(self, lhs, rhs):
        '''
        !!! It does not handle tensors whose sum of dimensions are larger than 26 (letters in alphabet)!!!
        '''
        dim_lhs = lhs.dim()
        dim_rhs = rhs.dim()
        l1 = ''.join([chr(97+i) for i in list(range(dim_lhs))])
        l2 = ''.join([chr(97+i) for i in list(range(dim_lhs, dim_lhs+dim_rhs))])
        return torch.einsum(l1 + ','+ l2 + '->' + (l1 + l2), [lhs, rhs])

    def rescale(self, arg, factor, top=None):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> pyt.rescale(pyt.arange(3,2), torch.tensor(.5, dtype=torch.double))
        tensor([3.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000,
                4.5000, 5.0000, 5.5000, 6.0000, 6.5000], dtype=torch.float64)

        """
        if top is None:
            top = arg[: self.blob_size(int(arg[0]))]     
        xx = self.pytorch_outer_tensordot(factor, arg)
        xx[0 : top.nelement()] = top[:]
        return xx

    def tensor_sub(self, lhs, rhs):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> pyt.tensor_sub(pyt.arange(3,2), pyt.ones(3,2))
        tensor([ 3.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.],
               dtype=torch.float64)

        """
        return self.tensor_add(lhs, self.rescale(rhs, torch.tensor(-1., dtype=torch.double)))

    ##@timeit
    def tensor_multiply(self, lhs, rhs, depth):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> print(pyt.tensor_multiply(pyt.arange(3,2), pyt.arange(3,2), 2))
        tensor([ 3.,  1.,  4.,  6.,  8., 14., 18., 22., 22., 27., 32., 30., 36., 42.],
               dtype=torch.float64)
        >>> print(pyt.tensor_multiply(pyt.arange(3,2), pyt.ones(3,2), 2))
        tensor([ 3.,  1.,  3.,  4.,  5.,  8.,  9., 10., 12., 13., 14., 16., 17., 18.],
               dtype=torch.float64)
        >>> print(pyt.tensor_multiply(pyt.arange(3,2), pyt.one(3,2), 2))
        tensor([ 3.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],
               dtype=torch.float64)
        >>> 
        """
        # lhs and rhs same width
        if int(rhs[0:1]) != int(lhs[0:1]):
            raise ValueError(
                "different width tensors cannot be combined:", lhs[0], "!=", rhs[0]
            )
        # extract width
        width = int(lhs[0])
        lhs_layers = self.layers(lhs.nelement(), width)
        rhs_layers = self.layers(rhs.nelement(), width)
        out_depth = min(depth, lhs_layers + rhs_layers)
        ans = self.zero(int(lhs[0]), depth).to(self.device)
        # i is the total degree 
        for i in range(min(out_depth, lhs_layers + rhs_layers) + 1):  
            # j is the degree of the rhs term
            for j in range(max(i - lhs_layers, 0), min(i, rhs_layers) + 1):
                ## nxt row the tensors must be shaped before multiplicaton and flattened before assignment
                ansb = self.blob_size(width, i - 1)
                anse = self.blob_size(width, i)
                lhsb = self.blob_size(width, (i - j) - 1)
                lhse = self.blob_size(width, (i - j))
                rhsb = self.blob_size(width, j - 1)
                rhse = self.blob_size(width, j)
                ans[ansb:anse] += self.pytorch_outer_tensordot(
                                     lhs[lhsb:lhse].view(self.tensor_shape(i - j, width)),
                                     rhs[rhsb:rhse].view(self.tensor_shape(j, width))
                                     ).view(-1)
        return ans

    def white(self, steps=int(100), width=int(1), time=1.0):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> torch.sum((torch.sum(pyt.white(10000, 3, 2.)**2, dim=0) - 2)**2) < torch.tensor(0.05, dtype=torch.double)
        tensor(1, dtype=torch.uint8)
        """
        mu, sigma = 0, torch.sqrt(torch.tensor(time / steps, dtype=torch.double))  # mean and standard deviation
        return torch.distributions.Normal(mu, sigma).sample((steps, width))
      
    def brownian(self, steps=int(100), width=int(1), time=1.0):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> pyt.brownian()[0]
        tensor([0.], dtype=torch.float64)
        >>> pyt.brownian(50,4).shape
        torch.Size([51, 4])
        >>> 
        """
        path = torch.zeros((steps + 1, width), dtype=torch.double)
        torch.cumsum(self.white(steps, width, time), dim=0, out=path[1:, :])
        return path

    def tensor_exp(self, arg, depth):
        """"
        >>> pyt = EsigPyTorchTensor()
        >>> d = 7
        >>> s = pyt.stream2sigtensor(pyt.brownian(100,2), d)
        >>> t = pyt.tensor_log(s,d)
        >>> torch.sum(pyt.tensor_sub(s, pyt.tensor_exp(t, d))[pyt.blob_size(2):]**2) < 1e-25
        tensor(1, dtype=torch.uint8)
        """
        # Computes the truncated exponential of arg
        #     1 + arg + arg^2/2! + ... + arg^n/n! where n = depth
        width = int(arg[0])
        result = self.one(width).to(self.device)
        if arg.nelement() > self.blob_size(width):
            top = arg[0 : self.blob_size(width) + 1]
            scalar = top[-1]
            top[-1] = 0.0
            x = arg
            x[self.blob_size(width)] = 0.0
            for i in range(depth, 0, -1):
                xx = self.rescale(arg, 1.0 / torch.tensor(i, dtype=torch.double, device=self.device), top)  
                # top resets the shape and here is extended to set the scalar coefficient to 
                result = self.tensor_multiply(result, xx, depth)
                result[self.blob_size(width)] += 1.0
            result = self.pytorch_outer_tensordot(torch.exp(scalar), result)
            result[: self.blob_size(width)] = top[: self.blob_size(width)]
        return result

    def tensor_log(self, arg, depth, normalised=True):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> d = 7
        >>> s = pyt.stream2sigtensor(pyt.brownian(100,2), d)
        >>> t = pyt.tensor_log(s,d)
        >>> torch.sum(pyt.tensor_sub(s, pyt.tensor_exp(t, d))[pyt.blob_size(2):]**2) < 1e-25
        tensor(1, dtype=torch.uint8)
        >>> 
        """

        """" 
        Computes the truncated log of arg up to degree depth.
        The coef. of the constant term (empty word in the monoid) of arg 
        is forced to 1.
        log(arg) = log(1+x) = x - x^2/2 + ... + (-1)^(n+1) x^n/n.
        arg must have a nonzero scalar term and depth must be > 0
        """

        width = int(arg[0])
        top = torch.tensor(arg[0 : self.blob_size(width) + 1], device=self.device)  
        # throw an error if there is no body to tensor as log zero not allowed
        if normalised:
            x = torch.tensor(arg, device=self.device)
            x[self.blob_size(width)] = 0.0
            # x = (arg - 1)
            result = self.zero(width, -1).to(self.device)
            # result will grow as the computation grows
            for i in range(depth, 0, -1):
                top[self.blob_size(width)] = torch.tensor((2 * (i % 2) - 1), dtype=torch.double, device=self.device) / torch.tensor(i, dtype=torch.double, device=self.device)
                result = self.tensor_add(result, top)
                result = self.tensor_multiply(result, x, depth)
        else:
            scalar = top[self.blob_size(width)]
            x = self.rescale(arg, torch.tensor(1.0 / scalar, dtype=torch.double, device=self.device))
            result = self.tensor_log(x, depth, True)
            #ans[self.blob_size(width)] += torch.log(scalar)
            result[self.blob_size(width)] += torch.log(scalar)

        return result

    def _stream2sigtensor(self, increments, depth):
        length, width = increments.shape
        if length > 1:
            lh = int(length / 2)
            return self.tensor_multiply(self._stream2sigtensor(increments[:lh, :], depth),
                                        self._stream2sigtensor(increments[lh:, :], depth),
                                        depth)
        else:
            lie = self.zero(width, 1).to(self.device)
            lie[self.blob_size(width, 0) : self.blob_size(width, 1)] = increments[0, :]
            return self.tensor_exp(lie, depth)


    def stream2sigtensor(self, stream, depth):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> s = torch.tensor([[0.],[1.]])
        >>> (torch.sum(pyt.stream2sigtensor(s, 7)[1:]) - torch.exp(torch.tensor(1., dtype=torch.double)))**2 < 1e-6
        tensor(1, dtype=torch.uint8)
        >>> s = pyt.brownian(100,2)
        >>> t  = torch.flip(s,[0])
        >>> d = 7
        >>> torch.sum(pyt.tensor_sub(pyt.tensor_multiply(pyt.stream2sigtensor(s,d),pyt.stream2sigtensor(t,d), d), pyt.one(2,d))[pyt.blob_size(2):]**2) < 1e-25
        tensor(1, dtype=torch.uint8)
        """
        increments = stream[1:, :] - stream[:-1, :]
        return self._stream2sigtensor(increments, depth)


    def stream2sig(self, stream, depth):
        """
        >>> pyt = EsigPyTorchTensor()
        >>> d = 7
        >>> s = pyt.brownian(100,2)
        >>> pyt.stream2sig(s,d).shape[0] < pyt.stream2sigtensor(s,d).shape[0] 
        True
        >>> from esig import tosig as ts
        >>> import numpy as np
        >>> s = pyt.brownian(200,2)
        >>> np.sum((pyt.stream2sig(s,5).numpy() - ts.stream2sig(s.numpy(),5))**2) < 1e-25
        True
        >>> 
        """
        return self.stream2sigtensor(stream, depth)[1:]

if __name__ == "__main__":
    doctest.testmod()
    #import time
    #pyt = EsigPyTorchTensor()
    #print([pyt.layers(pyt.blob_size(x, y) + z, x) for x in [2] for y in [0,1,17] for z in [-1,0,1]])
    #print([pyt.layers_new(pyt.blob_size(x, y) + z, x) for x in [2] for y in [0,1,17] for z in [-1,0,1]])
    #t1 = time.time()
    #print([(z, pyt.layers(z,2), pyt.blob_size(2,z), pyt.layers(pyt.blob_size(2,z),2)) for z in range(10)])
    #t2 = time.time()
    #print([(z, pyt.layers_new(z,2), pyt.blob_size(2,z), pyt.layers_new(pyt.blob_size(2,z),2)) for z in range(17)])
    #t3 = time.time()
    #print(t2-t1)
    #print(t3-t2)