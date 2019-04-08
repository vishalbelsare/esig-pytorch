import doctest
from esig import tosig as esig
from esig import tests as tests
import torch
from tjl_hall_pytorch_lie import *

try:
    import functools
except:
    import functools32

# ask the user whether to use the lyndon basis for the log-signature or the Coropa Hall basis
var = input('do you want to use the lyndon basis? (yes or no)')

class EsigPyTorch((EsigPyTorchLie_Lyndon if var=='yes' else EsigPyTorchLie)):
    
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(device)

    def logsigdim(self, signal_dimension, signature_degree):
        # logsigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the dimension of the log signature vector returned by array2logsig
        if var == 'yes':
            hall_set, _, _, _ = self.hall_basis(signal_dimension, signature_degree)
            return len(hall_set) - 1
        else:
            hall_set, _, _, _, _ = self.hall_basis(signal_dimension, signature_degree)
            return hall_set.shape[0] - 1

    def sigdim(self, signal_dimension, signature_degree):
        # sigdim(signal_dimension, signature_degree) returns a Py_ssize_t integer giving the length of the signature vector returned by array2logsig
        return self.blob_size(signal_dimension, signature_degree) - self.blob_size(signal_dimension)

    if var == 'yes':
        def stream2logsig(self, stream, signature_degree):
            # stream2logsig(torch.tensor(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, "the data in stream space" and returns a numpy vector containing the logsignature of the vector series up to given signature_degree
        
            """
            >>> sig = EsigPyTorch()
            >>> width = 4
            >>> depth = 4
            >>> stream = sig.brownian(100, width)
            >>> import iisignature
            >>> import numpy as np
            >>> s = iisignature.prepare(width, depth)
            >>> print(np.abs(sig.stream2logsig(stream, depth).sum().tolist() - np.sum(iisignature.logsig(stream, s))) < 1e-12)
            True
            """
            return self.sparse_to_dense(
                                        self.t2l(
                                                 self.tensor_log(
                                                                 self.stream2sigtensor(stream, signature_degree), signature_degree
                                                                 ),
                                                 ), stream.shape[1], signature_degree,
                                        )
    else:
        def stream2logsig(self, stream, signature_degree):
            # stream2logsig(torch.tensor(no_of_ticks x signal_dimension), signature_degree) reads a 2 dimensional numpy array of floats, "the data in stream space" and returns a numpy vector containing the logsignature of the vector series up to given signature_degree
        
            """
            >>> sig = EsigPyTorch()
            >>> width = 4
            >>> depth = 4
            >>> stream = sig.brownian(100, width)
            >>> print(torch.max(torch.abs(esig.stream2logsig(stream.numpy(), depth) - sig.stream2logsig(stream, depth))).tolist() < 1e-12)
            True
            """
            return self.sparse_to_dense(
                                        self.t2l(
                                                 self.tensor_log(
                                                                 self.stream2sigtensor(stream, signature_degree), signature_degree
                                                                 ),
                                                 ), stream.shape[1], signature_degree,
                                    )

    def stream2sig(self, stream, signature_degree):

        """
        >>> sig = EsigPyTorch()
        >>> width = 4
        >>> depth = 4
        >>> stream = sig.brownian(100, width)
        >>> print(torch.max(torch.abs(esig.stream2sig(stream.numpy(), depth) - sig.stream2sig(stream, depth))).tolist() < 1e-12)
        True
        """
        width = stream.shape[1]
        return self.stream2sigtensor(stream, signature_degree)[self.blob_size(width) :]

if __name__ == "__main__":
    doctest.testmod()

    #pyt = EsigPyTorch()
    #from time import time
    #from esig import tosig

    #width = 4
    #depth = 4

    #stream = pyt.brownian(100, width)

    #print(torch.max(torch.abs(esig.stream2logsig(stream.numpy(), depth) - pyt.stream2logsig(stream, depth))).tolist())

    #t = time()
    #log_sigs = pyt.stream2logsig(stream, depth)
    #print("t: {}".format(time() - t))

    #t = time()
    #log_sigs = tosig.stream2logsig(stream.numpy(), depth)
    #print("t: {}".format(time() - t))