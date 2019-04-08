import doctest
import torch
import ujson
from free_lie_algebra import *
from collections import defaultdict
import operator
from tjl_dense_pytorch_tensor import EsigPyTorchTensor
import copy
try:
    import functools
except:
    import functools32 as functools


# log-signatures in Coropa Hall basis

class EsigPyTorchLie(EsigPyTorchTensor):

    def __init__(self, device='cpu'):
        super().__init__(device)
        self.rbraketing_cache = {}
        self.expand_cache = {}
        self.prod_cache = {}

    @functools.lru_cache(maxsize=None)
    def hall_basis(self, width, desired_degree=0):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.hall_basis(1, 0)
        (tensor([[0, 0]], dtype=torch.int32), tensor([0], dtype=torch.int32), tensor([1], dtype=torch.int32), defaultdict(<class 'int'>, {}), 1)
        >>> lie.hall_basis(1, 3)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "<stdin>", line 61, in hall_basis
        IndexError: list index out of range
        >>> lie.hall_basis(1, 1)
        (tensor([[0, 0],
                [0, 1]], dtype=torch.int32), tensor([0, 1], dtype=torch.int32), tensor([1, 1], dtype=torch.int32), defaultdict(<class 'int'>, {(0, 1): 1}), 1)
        >>> lie.hall_basis(2,3)
        (tensor([[0, 0],
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 3]], dtype=torch.int32), tensor([0, 1, 1, 2, 3, 3], dtype=torch.int32), tensor([1, 1, 3, 4], dtype=torch.int32), defaultdict(<class 'int'>, {(0, 1): 1, (0, 2): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}), 2)

        """

        degrees = []
        hall_set = []
        degree_boundaries = []
        reverse_map = defaultdict(int)

        # the first entry in hall_set is not part of the basis but instead is
        # the nominal parent of all self parented elements (letters)
        # its values can be used for wider information about the lie element
        curr_degree = 0
        degrees.append(0)
        p = (0, 0)
        hall_set.append(p)
        degree_boundaries.append(1)
        if desired_degree > 0:
            # level 1 the first basis terms
            degree_boundaries.append(1)
            for i in range(1, width + 1):
                hall_set.append((0, i))
                degrees.append(1)
                reverse_map[(0, i)] = i
            curr_degree += 1
            for d in range(curr_degree + 1, desired_degree + 1):
                bound = len(hall_set)
                degree_boundaries.append(bound)
                for i in range(1, bound + 1):
                    for j in range(i + 1, bound + 1):
                        if (degrees[i] + degrees[j] == d) & (hall_set[j][0] <= i):
                            hall_set.append((i, j))
                            degrees.append(d)
                            reverse_map[(i, j)] = len(hall_set) - 1
                curr_degree += 1
        return (torch.tensor(hall_set, dtype=torch.int), 
                torch.tensor(degrees, dtype=torch.int), 
                torch.tensor(degree_boundaries, dtype=torch.int), 
                reverse_map, 
                width,)

    @functools.lru_cache(maxsize=None)
    def hb_to_string(self, z, width, desired_degree):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.hb_to_string(7, 3, 6)
        '[1,[1,2]]'

        """
        torch_hall_set = self.hall_basis(width, desired_degree)[0]
        (n, m) = torch_hall_set[z]
        n = n.tolist()
        m = m.tolist()
        if n:
            return (
                "["
                + self.hb_to_string(n, width, desired_degree)
                + ","
                + self.hb_to_string(m, width, desired_degree)
                + "]"
            )
        else:
            return str(m)
    
    @functools.lru_cache(maxsize=None)
    def logsigkeys(self, width, desired_degree):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.logsigkeys(3,6)
        ' 1 2 3 [1,2] [1,3] [2,3] [1,[1,2]] [1,[1,3]] [2,[1,2]] [2,[1,3]] [2,[2,3]] [3,[1,2]] [3,[1,3]] [3,[2,3]] [1,[1,[1,2]]] [1,[1,[1,3]]] [2,[1,[1,2]]] [2,[1,[1,3]]] [2,[2,[1,2]]] [2,[2,[1,3]]] [2,[2,[2,3]]] [3,[1,[1,2]]] [3,[1,[1,3]]] [3,[2,[1,2]]] [3,[2,[1,3]]] [3,[2,[2,3]]] [3,[3,[1,2]]] [3,[3,[1,3]]] [3,[3,[2,3]]] [[1,2],[1,3]] [[1,2],[2,3]] [[1,3],[2,3]] [1,[1,[1,[1,2]]]] [1,[1,[1,[1,3]]]] [2,[1,[1,[1,2]]]] [2,[1,[1,[1,3]]]] [2,[2,[1,[1,2]]]] [2,[2,[1,[1,3]]]] [2,[2,[2,[1,2]]]] [2,[2,[2,[1,3]]]] [2,[2,[2,[2,3]]]] [3,[1,[1,[1,2]]]] [3,[1,[1,[1,3]]]] [3,[2,[1,[1,2]]]] [3,[2,[1,[1,3]]]] [3,[2,[2,[1,2]]]] [3,[2,[2,[1,3]]]] [3,[2,[2,[2,3]]]] [3,[3,[1,[1,2]]]] [3,[3,[1,[1,3]]]] [3,[3,[2,[1,2]]]] [3,[3,[2,[1,3]]]] [3,[3,[2,[2,3]]]] [3,[3,[3,[1,2]]]] [3,[3,[3,[1,3]]]] [3,[3,[3,[2,3]]]] [[1,2],[1,[1,2]]] [[1,2],[1,[1,3]]] [[1,2],[2,[1,2]]] [[1,2],[2,[1,3]]] [[1,2],[2,[2,3]]] [[1,2],[3,[1,2]]] [[1,2],[3,[1,3]]] [[1,2],[3,[2,3]]] [[1,3],[1,[1,2]]] [[1,3],[1,[1,3]]] [[1,3],[2,[1,2]]] [[1,3],[2,[1,3]]] [[1,3],[2,[2,3]]] [[1,3],[3,[1,2]]] [[1,3],[3,[1,3]]] [[1,3],[3,[2,3]]] [[2,3],[1,[1,2]]] [[2,3],[1,[1,3]]] [[2,3],[2,[1,2]]] [[2,3],[2,[1,3]]] [[2,3],[2,[2,3]]] [[2,3],[3,[1,2]]] [[2,3],[3,[1,3]]] [[2,3],[3,[2,3]]] [1,[1,[1,[1,[1,2]]]]] [1,[1,[1,[1,[1,3]]]]] [2,[1,[1,[1,[1,2]]]]] [2,[1,[1,[1,[1,3]]]]] [2,[2,[1,[1,[1,2]]]]] [2,[2,[1,[1,[1,3]]]]] [2,[2,[2,[1,[1,2]]]]] [2,[2,[2,[1,[1,3]]]]] [2,[2,[2,[2,[1,2]]]]] [2,[2,[2,[2,[1,3]]]]] [2,[2,[2,[2,[2,3]]]]] [3,[1,[1,[1,[1,2]]]]] [3,[1,[1,[1,[1,3]]]]] [3,[2,[1,[1,[1,2]]]]] [3,[2,[1,[1,[1,3]]]]] [3,[2,[2,[1,[1,2]]]]] [3,[2,[2,[1,[1,3]]]]] [3,[2,[2,[2,[1,2]]]]] [3,[2,[2,[2,[1,3]]]]] [3,[2,[2,[2,[2,3]]]]] [3,[3,[1,[1,[1,2]]]]] [3,[3,[1,[1,[1,3]]]]] [3,[3,[2,[1,[1,2]]]]] [3,[3,[2,[1,[1,3]]]]] [3,[3,[2,[2,[1,2]]]]] [3,[3,[2,[2,[1,3]]]]] [3,[3,[2,[2,[2,3]]]]] [3,[3,[3,[1,[1,2]]]]] [3,[3,[3,[1,[1,3]]]]] [3,[3,[3,[2,[1,2]]]]] [3,[3,[3,[2,[1,3]]]]] [3,[3,[3,[2,[2,3]]]]] [3,[3,[3,[3,[1,2]]]]] [3,[3,[3,[3,[1,3]]]]] [3,[3,[3,[3,[2,3]]]]] [[1,2],[1,[1,[1,2]]]] [[1,2],[1,[1,[1,3]]]] [[1,2],[2,[1,[1,2]]]] [[1,2],[2,[1,[1,3]]]] [[1,2],[2,[2,[1,2]]]] [[1,2],[2,[2,[1,3]]]] [[1,2],[2,[2,[2,3]]]] [[1,2],[3,[1,[1,2]]]] [[1,2],[3,[1,[1,3]]]] [[1,2],[3,[2,[1,2]]]] [[1,2],[3,[2,[1,3]]]] [[1,2],[3,[2,[2,3]]]] [[1,2],[3,[3,[1,2]]]] [[1,2],[3,[3,[1,3]]]] [[1,2],[3,[3,[2,3]]]] [[1,2],[[1,2],[1,3]]] [[1,2],[[1,2],[2,3]]] [[1,3],[1,[1,[1,2]]]] [[1,3],[1,[1,[1,3]]]] [[1,3],[2,[1,[1,2]]]] [[1,3],[2,[1,[1,3]]]] [[1,3],[2,[2,[1,2]]]] [[1,3],[2,[2,[1,3]]]] [[1,3],[2,[2,[2,3]]]] [[1,3],[3,[1,[1,2]]]] [[1,3],[3,[1,[1,3]]]] [[1,3],[3,[2,[1,2]]]] [[1,3],[3,[2,[1,3]]]] [[1,3],[3,[2,[2,3]]]] [[1,3],[3,[3,[1,2]]]] [[1,3],[3,[3,[1,3]]]] [[1,3],[3,[3,[2,3]]]] [[1,3],[[1,2],[1,3]]] [[1,3],[[1,2],[2,3]]] [[1,3],[[1,3],[2,3]]] [[2,3],[1,[1,[1,2]]]] [[2,3],[1,[1,[1,3]]]] [[2,3],[2,[1,[1,2]]]] [[2,3],[2,[1,[1,3]]]] [[2,3],[2,[2,[1,2]]]] [[2,3],[2,[2,[1,3]]]] [[2,3],[2,[2,[2,3]]]] [[2,3],[3,[1,[1,2]]]] [[2,3],[3,[1,[1,3]]]] [[2,3],[3,[2,[1,2]]]] [[2,3],[3,[2,[1,3]]]] [[2,3],[3,[2,[2,3]]]] [[2,3],[3,[3,[1,2]]]] [[2,3],[3,[3,[1,3]]]] [[2,3],[3,[3,[2,3]]]] [[2,3],[[1,2],[1,3]]] [[2,3],[[1,2],[2,3]]] [[2,3],[[1,3],[2,3]]] [[1,[1,2]],[1,[1,3]]] [[1,[1,2]],[2,[1,2]]] [[1,[1,2]],[2,[1,3]]] [[1,[1,2]],[2,[2,3]]] [[1,[1,2]],[3,[1,2]]] [[1,[1,2]],[3,[1,3]]] [[1,[1,2]],[3,[2,3]]] [[1,[1,3]],[2,[1,2]]] [[1,[1,3]],[2,[1,3]]] [[1,[1,3]],[2,[2,3]]] [[1,[1,3]],[3,[1,2]]] [[1,[1,3]],[3,[1,3]]] [[1,[1,3]],[3,[2,3]]] [[2,[1,2]],[2,[1,3]]] [[2,[1,2]],[2,[2,3]]] [[2,[1,2]],[3,[1,2]]] [[2,[1,2]],[3,[1,3]]] [[2,[1,2]],[3,[2,3]]] [[2,[1,3]],[2,[2,3]]] [[2,[1,3]],[3,[1,2]]] [[2,[1,3]],[3,[1,3]]] [[2,[1,3]],[3,[2,3]]] [[2,[2,3]],[3,[1,2]]] [[2,[2,3]],[3,[1,3]]] [[2,[2,3]],[3,[2,3]]] [[3,[1,2]],[3,[1,3]]] [[3,[1,2]],[3,[2,3]]] [[3,[1,3]],[3,[2,3]]]'
        >>> width = 4
        >>> depth = 4
        >>> lie.logsigkeys(width, depth)
        ' 1 2 3 4 [1,2] [1,3] [1,4] [2,3] [2,4] [3,4] [1,[1,2]] [1,[1,3]] [1,[1,4]] [2,[1,2]] [2,[1,3]] [2,[1,4]] [2,[2,3]] [2,[2,4]] [3,[1,2]] [3,[1,3]] [3,[1,4]] [3,[2,3]] [3,[2,4]] [3,[3,4]] [4,[1,2]] [4,[1,3]] [4,[1,4]] [4,[2,3]] [4,[2,4]] [4,[3,4]] [1,[1,[1,2]]] [1,[1,[1,3]]] [1,[1,[1,4]]] [2,[1,[1,2]]] [2,[1,[1,3]]] [2,[1,[1,4]]] [2,[2,[1,2]]] [2,[2,[1,3]]] [2,[2,[1,4]]] [2,[2,[2,3]]] [2,[2,[2,4]]] [3,[1,[1,2]]] [3,[1,[1,3]]] [3,[1,[1,4]]] [3,[2,[1,2]]] [3,[2,[1,3]]] [3,[2,[1,4]]] [3,[2,[2,3]]] [3,[2,[2,4]]] [3,[3,[1,2]]] [3,[3,[1,3]]] [3,[3,[1,4]]] [3,[3,[2,3]]] [3,[3,[2,4]]] [3,[3,[3,4]]] [4,[1,[1,2]]] [4,[1,[1,3]]] [4,[1,[1,4]]] [4,[2,[1,2]]] [4,[2,[1,3]]] [4,[2,[1,4]]] [4,[2,[2,3]]] [4,[2,[2,4]]] [4,[3,[1,2]]] [4,[3,[1,3]]] [4,[3,[1,4]]] [4,[3,[2,3]]] [4,[3,[2,4]]] [4,[3,[3,4]]] [4,[4,[1,2]]] [4,[4,[1,3]]] [4,[4,[1,4]]] [4,[4,[2,3]]] [4,[4,[2,4]]] [4,[4,[3,4]]] [[1,2],[1,3]] [[1,2],[1,4]] [[1,2],[2,3]] [[1,2],[2,4]] [[1,2],[3,4]] [[1,3],[1,4]] [[1,3],[2,3]] [[1,3],[2,4]] [[1,3],[3,4]] [[1,4],[2,3]] [[1,4],[2,4]] [[1,4],[3,4]] [[2,3],[2,4]] [[2,3],[3,4]] [[2,4],[3,4]]'
        
        """
        torch_hall_set, _, _, _, width = self.hall_basis(width, desired_degree)
        return " " + " ".join([self.hb_to_string(z, width, desired_degree) for z in range(1, torch_hall_set.shape[0])])

    def multiply(self, lhs, rhs, width, depth):
        ## WARNING assumes all multiplications are in range -
        ## if not then the product should use the coproduct and the max degree
        ans = defaultdict(float)
        for k1 in sorted(lhs.keys()):
            for k2 in sorted(rhs.keys()):
                a1 = self.prod(k1, k2, width, depth)
                factor = lhs[k1] * rhs[k2]
                a1 = self.scale_into(a1, factor)
                self.add_into(ans, a1)
        return self.sparsify(ans)

    def prod(self, k1, k2, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lhs = lie.scale_into(lhs, 3)
        >>> print(lhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
        >>> lie.subtract_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
        >>> lie.multiply(lhs,rhs, 3, 6)
        defaultdict(<class 'float'>, {13: 3.0})
        >>> lie.multiply(rhs,lhs,3,6)
        defaultdict(<class 'float'>, {13: -3.0})
        >>> lie.multiply(lhs,lhs,3,6)
        defaultdict(<class 'float'>, {})
        >>> 
        """
        if (k1, k2, width, depth) in self.prod_cache:
            return self.prod_cache[(k1, k2, width, depth)]

        ans = defaultdict(float)
        hall_set, degrees, foliages, reverse_map, width = self.hall_basis(width, depth)

        if k1 > k2:
            ans = self.prod(k2, k1, width, depth)
            ans = self.scale_into(ans, -1.)
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        if k1 == k2:
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        if degrees[k1] + degrees[k2] > depth:
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        t = reverse_map.get((k1, k2), 0)
        if t:
            ans[t] = 1.

        else:
            (k3, k4) = hall_set[k2]  
            k3 = int(k3)
            k4 = int(k4)

            ### We use Jacobi: [k1,k2] = [k1,[k3,k4]]] = [[k1,k3],k4]-[[k1,k4],k3]

            t1 = self.multiply(
                                self.prod(k1, k3, width, depth), self.key_to_sparse(k4), width, depth
                                )
            t2 = self.multiply(
                                self.prod(k1, k4, width, depth), self.key_to_sparse(k3), width, depth
                                )
            ans = self.subtract_into(t1, t2)


                                                  
        self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
        return ans

    def lie_to_string(self, li, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.lie_to_string(lie.prod(7, 6, 3, 6), 3, 6)
        '-1.0 [[2,3],[1,[1,2]]]'

        """
        return " + ".join([str(li[x]) + " " + self.hb_to_string(x, width, depth) for x in sorted(li.keys())])

    def sparse_to_dense(self, sparse, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lhs = lie.scale_into(lhs, 3)
        >>> print(lhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
        >>> lie.subtract_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
        >>> lie.add_into(lhs, lie.multiply(rhs, lhs, 3, 6))
        defaultdict(<class 'float'>, {3: 3.0, 5: 2.0, 13: -3.0})
        >>> lie.sparse_to_dense(lhs, 3, 2)
        tensor([0., 0., 3., 0., 2., 0.], dtype=torch.float64)
        """
        hall_set, degrees, _, reverse_map, width = self.hall_basis(width, depth)
        dense = torch.zeros(len(hall_set), dtype=torch.double, device=self.device)
        for k in sparse.keys():
            if k < len(hall_set):
                dense[k] = sparse[k]
        return dense[1:]

    def dense_to_sparse(self, dense, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> hall_set, degrees, degree_boundaries, reverse_map, width = lie.hall_basis(2, 3)
        >>> l = torch.tensor([i for i in range(1, len(hall_set))], dtype=torch.double)
        >>> print(l, " ", lie.dense_to_sparse(l,2,3))
        tensor([1., 2., 3., 4., 5.], dtype=torch.float64)   defaultdict(<class 'float'>, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
        >>> lie.sparse_to_dense(lie.dense_to_sparse(l,2,3), 2, 3) == l
        tensor([1, 1, 1, 1, 1], dtype=torch.uint8)
        >>> 
        """
        sparse = defaultdict(float)
        for k in range(len(dense)):
            if dense[k]:
                sparse[k+1] = dense[k].tolist()
        return sparse


    ## expand is a map from hall basis keys to tensors
    def expand(self, k, width, depth):

        if (k, width, depth) in self.expand_cache:
            return self.expand_cache[(k, width, depth)]

        _expand = functools.partial(self.expand, width=width, depth=depth)
        _tensor_multiply = functools.partial(self.tensor_multiply, depth=depth)

        if k:
            hall_set, degrees, degree_boundaries, reverse_map, width = self.hall_basis(width, depth)
            (k1, k2) = hall_set[k].tolist()
            if k1:
                ans = self.tensor_sub(_tensor_multiply(_expand(k1), _expand(k2)),
                                        _tensor_multiply(_expand(k2), _expand(k1))
                                        )
                self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
                return ans
            else:
                ans = self.zero(width, 1)
                ans[self.blob_size(width, 0) - 1 + k2] = 1.  ## recall k2 will never be zero
                self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
                return ans
        ans = self.zero(width)
        self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
        return ans


    ## tuple a1,a2,...,an is converted into [a1,[a2,[...,an]]] as a LIE element recursively.
    #@functools.lru_cache(maxsize=0)
    def rbraketing(self, tk, width, depth):

        if (tk, width, depth) in self.rbraketing_cache:
            return self.rbraketing_cache[(tk, width, depth)]

        _rbracketing = functools.partial(self.rbraketing, width=width, depth=depth)
        _multiply = functools.partial(self.multiply, width=width, depth=depth)


        if len(tk) > 1:
            ans1 = _rbracketing(tk[:1])
            ans2 = _rbracketing(tk[1:])
            ans = _multiply(ans1, ans2)
            self.rbraketing_cache[(tk, width, depth)] = copy.deepcopy(ans)
            return ans
        else:
            ans = defaultdict(float)
            if tk:
                ans[tk[0]] = float(1)

            self.rbraketing_cache[(tk, width, depth)] = copy.deepcopy(ans)
            return ans

    def key_to_sparse(self, k):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.key_to_sparse(7)
        defaultdict(<class 'float'>, {7: 1.0})

        """
        ans = defaultdict(float)
        ans[k] = float(1)
        return ans

    def add_into(self, lhs, rhs):
        """
        >>> lie = EsigPyTorchLie()
        >>> lhs = lie.key_to_sparse(3)
        >>> rhs = lie.key_to_sparse(5)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        """
        for k in rhs.keys():
            lhs[k] += rhs.get(k, float())
        return lhs

    def subtract_into(self, lhs, rhs):
        """
        >>> lie = EsigPyTorchLie()
        >>> lhs = lie.key_to_sparse(3)
        >>> rhs = lie.key_to_sparse(5)
        >>> lie.add_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lie.subtract_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
        """
        for k in rhs.keys():
            lhs[k] -= rhs.get(k, float())
        return lhs

    def scale_into(self, lhs, s):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lhs = lie.scale_into(lhs, 3)
        >>> print(lhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
        >>> lhs = lie.scale_into(lhs, 0)
        >>> print(lhs)
        defaultdict(<class 'float'>, {})
        >>> 
        """
        ans = defaultdict(float)
        if s:
            for k in lhs.keys():
                ans[k] = lhs[k]*s
        else:
            ans = defaultdict(float)
        return ans

    def sparsify(self, arg):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lie.subtract_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
        >>> lie.sparsify(lhs)
        defaultdict(<class 'float'>, {3: 1.0})
        >>> 
        """
        empty_key_vals = list(k for k in arg.keys() if not arg[k])
        for k in empty_key_vals:
            del arg[k]
        return arg

    def l2t(self, arg, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.tensor_log(lie.stream2sigtensor(lie.brownian(100, width), depth), depth)
        >>> print(torch.sum(lie.tensor_sub(lie.l2t(lie.t2l(t), width, depth), t)[2:]**2).tolist() < 1e-30)
        True
        >>> 
        """
        _expand = functools.partial(self.expand, width=width, depth=depth)

        ans = self.zero(width)
        for k in arg.keys():
            if k:
                ans = self.tensor_add(ans, (self.rescale(_expand(k), arg[k])))
        
        # this is just to measure the  sparsity coefficient of the projection from lie to tensor.
        nonempty = len([val for val in self.expand_cache.values() if len(val) > 0])
        sig_dim = self.blob_size(width, depth) - self.blob_size(width)
        hall_set, _, _, _, _ = self.hall_basis(width, depth)
        logsig_dim = hall_set.shape[0] - 1
        total = sig_dim * logsig_dim
        #self.sparsity_coeff_l2t = float(nonempty/total)  
        self.sparsity_coeff_l2t = float(nonempty/sig_dim)
                
        return ans


    # the shape of the tensor that contains
    # \sum t[k] index_to_tuple(k,width) is the tensor
    # None () (0) (1) ...(w-1) (0,0) ...(n1,...,nd) ...
    @functools.lru_cache(maxsize=None)
    def index_to_tuple(self, i, width):
        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.arange(width, depth)
        >>> print(t)
        tensor([ 2.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
                14., 15.], dtype=torch.float64)
        >>> for t1 in [t]:
        ...     for k, coeff in enumerate(t1):
        ...         print (coeff.tolist(), lie.index_to_tuple(k, width))
        ... 
        2.0 None
        1.0 ()
        2.0 (1,)
        3.0 (2,)
        4.0 (1, 1)
        5.0 (1, 2)
        6.0 (2, 1)
        7.0 (2, 2)
        8.0 (1, 1, 1)
        9.0 (1, 1, 2)
        10.0 (1, 2, 1)
        11.0 (1, 2, 2)
        12.0 (2, 1, 1)
        13.0 (2, 1, 2)
        14.0 (2, 2, 1)
        15.0 (2, 2, 2)
        >>> 
        """

        bz = i + 1
        d = self.layers(bz, width)  ## this index is in the d-tensors
        if d < 0:
            return
        j = bz - 1 - self.blob_size(width, d-1)
        ans = ()
        ## remove initial offset to compute the index
        if j >= 0:
            for jj in range(d):
                ans = (1 + (j % width),) + ans
                j = j // width
        return ans
    
    def t_to_string(self, i, width):
        j = self.index_to_tuple(i, width)
        if self.index_to_tuple(i, width) == None:
            return " "
        return "(" + ",".join([str(k) for k in self.index_to_tuple(i, width)]) + ")"

    @functools.lru_cache(maxsize=None)
    def sigkeys(self, width, desired_degree):
        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> from esig import tosig as ts
        >>> ts.sigkeys(width , depth) == lie.sigkeys(width , depth)
        True
        >>> lie.sigkeys(width , depth)
        ' () (1) (2) (1,1) (1,2) (2,1) (2,2) (1,1,1) (1,1,2) (1,2,1) (1,2,2) (2,1,1) (2,1,2) (2,2,1) (2,2,2)'
        >>> sig = EsigPyTorchLie()
        >>> width = 4
        >>> depth = 4
        >>> lie.sigkeys(width, depth)
        ' () (1) (2) (3) (4) (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (2,3) (2,4) (3,1) (3,2) (3,3) (3,4) (4,1) (4,2) (4,3) (4,4) (1,1,1) (1,1,2) (1,1,3) (1,1,4) (1,2,1) (1,2,2) (1,2,3) (1,2,4) (1,3,1) (1,3,2) (1,3,3) (1,3,4) (1,4,1) (1,4,2) (1,4,3) (1,4,4) (2,1,1) (2,1,2) (2,1,3) (2,1,4) (2,2,1) (2,2,2) (2,2,3) (2,2,4) (2,3,1) (2,3,2) (2,3,3) (2,3,4) (2,4,1) (2,4,2) (2,4,3) (2,4,4) (3,1,1) (3,1,2) (3,1,3) (3,1,4) (3,2,1) (3,2,2) (3,2,3) (3,2,4) (3,3,1) (3,3,2) (3,3,3) (3,3,4) (3,4,1) (3,4,2) (3,4,3) (3,4,4) (4,1,1) (4,1,2) (4,1,3) (4,1,4) (4,2,1) (4,2,2) (4,2,3) (4,2,4) (4,3,1) (4,3,2) (4,3,3) (4,3,4) (4,4,1) (4,4,2) (4,4,3) (4,4,4) (1,1,1,1) (1,1,1,2) (1,1,1,3) (1,1,1,4) (1,1,2,1) (1,1,2,2) (1,1,2,3) (1,1,2,4) (1,1,3,1) (1,1,3,2) (1,1,3,3) (1,1,3,4) (1,1,4,1) (1,1,4,2) (1,1,4,3) (1,1,4,4) (1,2,1,1) (1,2,1,2) (1,2,1,3) (1,2,1,4) (1,2,2,1) (1,2,2,2) (1,2,2,3) (1,2,2,4) (1,2,3,1) (1,2,3,2) (1,2,3,3) (1,2,3,4) (1,2,4,1) (1,2,4,2) (1,2,4,3) (1,2,4,4) (1,3,1,1) (1,3,1,2) (1,3,1,3) (1,3,1,4) (1,3,2,1) (1,3,2,2) (1,3,2,3) (1,3,2,4) (1,3,3,1) (1,3,3,2) (1,3,3,3) (1,3,3,4) (1,3,4,1) (1,3,4,2) (1,3,4,3) (1,3,4,4) (1,4,1,1) (1,4,1,2) (1,4,1,3) (1,4,1,4) (1,4,2,1) (1,4,2,2) (1,4,2,3) (1,4,2,4) (1,4,3,1) (1,4,3,2) (1,4,3,3) (1,4,3,4) (1,4,4,1) (1,4,4,2) (1,4,4,3) (1,4,4,4) (2,1,1,1) (2,1,1,2) (2,1,1,3) (2,1,1,4) (2,1,2,1) (2,1,2,2) (2,1,2,3) (2,1,2,4) (2,1,3,1) (2,1,3,2) (2,1,3,3) (2,1,3,4) (2,1,4,1) (2,1,4,2) (2,1,4,3) (2,1,4,4) (2,2,1,1) (2,2,1,2) (2,2,1,3) (2,2,1,4) (2,2,2,1) (2,2,2,2) (2,2,2,3) (2,2,2,4) (2,2,3,1) (2,2,3,2) (2,2,3,3) (2,2,3,4) (2,2,4,1) (2,2,4,2) (2,2,4,3) (2,2,4,4) (2,3,1,1) (2,3,1,2) (2,3,1,3) (2,3,1,4) (2,3,2,1) (2,3,2,2) (2,3,2,3) (2,3,2,4) (2,3,3,1) (2,3,3,2) (2,3,3,3) (2,3,3,4) (2,3,4,1) (2,3,4,2) (2,3,4,3) (2,3,4,4) (2,4,1,1) (2,4,1,2) (2,4,1,3) (2,4,1,4) (2,4,2,1) (2,4,2,2) (2,4,2,3) (2,4,2,4) (2,4,3,1) (2,4,3,2) (2,4,3,3) (2,4,3,4) (2,4,4,1) (2,4,4,2) (2,4,4,3) (2,4,4,4) (3,1,1,1) (3,1,1,2) (3,1,1,3) (3,1,1,4) (3,1,2,1) (3,1,2,2) (3,1,2,3) (3,1,2,4) (3,1,3,1) (3,1,3,2) (3,1,3,3) (3,1,3,4) (3,1,4,1) (3,1,4,2) (3,1,4,3) (3,1,4,4) (3,2,1,1) (3,2,1,2) (3,2,1,3) (3,2,1,4) (3,2,2,1) (3,2,2,2) (3,2,2,3) (3,2,2,4) (3,2,3,1) (3,2,3,2) (3,2,3,3) (3,2,3,4) (3,2,4,1) (3,2,4,2) (3,2,4,3) (3,2,4,4) (3,3,1,1) (3,3,1,2) (3,3,1,3) (3,3,1,4) (3,3,2,1) (3,3,2,2) (3,3,2,3) (3,3,2,4) (3,3,3,1) (3,3,3,2) (3,3,3,3) (3,3,3,4) (3,3,4,1) (3,3,4,2) (3,3,4,3) (3,3,4,4) (3,4,1,1) (3,4,1,2) (3,4,1,3) (3,4,1,4) (3,4,2,1) (3,4,2,2) (3,4,2,3) (3,4,2,4) (3,4,3,1) (3,4,3,2) (3,4,3,3) (3,4,3,4) (3,4,4,1) (3,4,4,2) (3,4,4,3) (3,4,4,4) (4,1,1,1) (4,1,1,2) (4,1,1,3) (4,1,1,4) (4,1,2,1) (4,1,2,2) (4,1,2,3) (4,1,2,4) (4,1,3,1) (4,1,3,2) (4,1,3,3) (4,1,3,4) (4,1,4,1) (4,1,4,2) (4,1,4,3) (4,1,4,4) (4,2,1,1) (4,2,1,2) (4,2,1,3) (4,2,1,4) (4,2,2,1) (4,2,2,2) (4,2,2,3) (4,2,2,4) (4,2,3,1) (4,2,3,2) (4,2,3,3) (4,2,3,4) (4,2,4,1) (4,2,4,2) (4,2,4,3) (4,2,4,4) (4,3,1,1) (4,3,1,2) (4,3,1,3) (4,3,1,4) (4,3,2,1) (4,3,2,2) (4,3,2,3) (4,3,2,4) (4,3,3,1) (4,3,3,2) (4,3,3,3) (4,3,3,4) (4,3,4,1) (4,3,4,2) (4,3,4,3) (4,3,4,4) (4,4,1,1) (4,4,1,2) (4,4,1,3) (4,4,1,4) (4,4,2,1) (4,4,2,2) (4,4,2,3) (4,4,2,4) (4,4,3,1) (4,4,3,2) (4,4,3,3) (4,4,3,4) (4,4,4,1) (4,4,4,2) (4,4,4,3) (4,4,4,4)'

        """
        self.t_to_string(0, width)
        return " " + " ".join([self.t_to_string(z, width) for z in range(1, self.blob_size(width, desired_degree))])

    def t2l(self, arg):
        # projects a lie element in tensor form to a lie in lie basis form

        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.tensor_log(lie.stream2sigtensor(lie.brownian(100, width), depth), depth)
        >>> print(torch.sum(lie.tensor_sub(lie.l2t(lie.t2l(t), width, depth), t)[2:]**2).tolist()  < 1e-30)
        True
        >>> 
        """
        width = int(arg[0])
        depth = self.layers(len(arg), width)
        ans = defaultdict(float)
        ibe = self.blob_size(width, 0)  # just beyond the zero tensors
        ien = self.blob_size(width, depth)

        for i in range(ibe, ien):
            t = self.index_to_tuple(i, width)
            if t:
                ## must normalise to get the dynkin projection to make it a projection
                res = self.rbraketing(t, width, depth)
                res = self.scale_into(res, arg[i] / float(len(t)))
                self.add_into(ans, res)
                
        # this is just to measure the  sparsity coefficient of the projection from tensor to lie.
        nonempty = len([val for val in self.rbraketing_cache.values() if len(val) > 0])
        sig_dim = self.blob_size(width, depth) - self.blob_size(width)
        hall_set, _, _, _, _ = self.hall_basis(width, depth)
        logsig_dim = hall_set.shape[0] - 1
        total = sig_dim * logsig_dim
        #self.sparsity_coeff_t2l = float(nonempty/total)
        self.sparsity_coeff_t2l = float(nonempty/sig_dim)

        return ans


# log-signature FLA set-up with Lyndon basis
class EsigPyTorchLie_Lyndon(EsigPyTorchTensor):

    def __init__(self, device='cpu'):
        super().__init__(device)
        self.rbraketing_cache = {}
        self.expand_cache = {}
        self.prod_cache = {}

    def hall_basis(self, width, desired_degree):

        # define total order relation
        lessExpression = lessExpressionLyndon

        # generate all letters
        out=[[(i,) for i in range(1, width+1)]]
        all_letters = [('', '')] + [('', '{}'.format(i)) for i in range(1, width+1)]
        letters_foliages = [''] + ['{}'.format(i) for i in range(1, width+1)]
    
        # initialize hall set
        hall_set = dict(zip(all_letters, letters_foliages))
    
        # keep track of degrees
        degrees = [0] + [1]*len(out[0])

        # keep track of number of lie elements in basis
        ordered_lie_elts = dict(zip(out[0], range(1, width+1)))
        reverse_map = dict(zip(range(width+1), [(0,i) for i in range(width+1)]))
        count_elts = width

        # ordered foliages
        foliages = dict(zip(range(width+1), letters_foliages))
    
        # loop through degrees
        for mm in range(2, desired_degree+1):

            out.append([])

            # for each degree mm, span from 1 to mm
            for firstLev in range(1,mm):

                # loop through all lie elements stored so far, from the start
                for x in out[firstLev-1]:

                    # fixing the first parent, loop through all lie elements stored so far, from the end
                    for y in out[mm-firstLev-1]:

                        # check if the pair (x, y) is Hall...
                        if lessExpression(x,y) and (firstLev==1 or not lessExpression(x[1],y)):

                            # ...if yes, add it to the list of lie elements
                            out[-1].append((x,y))
                            hall_set[(foliageFromTree(x), foliageFromTree(y))] = foliageFromTree((x,y))

                            # update degrees
                            degrees.append(mm)
                        
                            # update ordered lie basis elements
                            count_elts += 1
                            reverse_map[count_elts] = (ordered_lie_elts[x], ordered_lie_elts[y])
                            ordered_lie_elts[(x, y)] = count_elts
                        
                            # update foliages
                            foliages[count_elts] = foliageFromTree((x,y))
    
        return hall_set, reverse_map, np.array(degrees, dtype=int), foliages


    def hb_to_string(self, z, width, desired_degree):
        """hall basis element to string"""
        _, reverse_map, _, _ = self.hall_basis(width, desired_degree)
        (n, m) = reverse_map[z]
    
        if n != 0:
            return (
                    "["
                    + self.hb_to_string(n, width, desired_degree)
                    + ","
                    + self.hb_to_string(m, width, desired_degree)
                    + "]"
                    )
    
        else:
            return str(m)


    def logsigkeys(self, width, desired_degree):

        _, reverse_map, _, _ = self.hall_basis(width, desired_degree)
    
        return " " + " ".join(
            [self.hb_to_string(z, width, desired_degree) for z in range(1, len(reverse_map))]
        )

    def prod(self, k1, k2, width, depth):
        """product of 2 lyndon words"""

        # if seen before, use it!
        if (k1, k2, width, depth) in self.prod_cache:
            return self.prod_cache[(k1, k2, width, depth)]

        ans = defaultdict(float)
        hall_set, reverse_map, degrees, foliages = self.hall_basis(width, depth)
        hall_set_reversed = {f:v for v,f in hall_set.items()}
        foliages_reversed = {f:v for v,f in foliages.items()}

        # if the product exits max depth then don't do it
        if len(foliages[k1]) + len(foliages[k2]) > depth:
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        # prod of a lyndon letter by itself gives the empty word
        if k1 == k2:
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        # [a,b] = -[b,a]
        if foliages[k1] > foliages[k2]:
            ans = self.prod(k2, k1, width, depth)
            ans = self.scale_into(ans, -float(1))
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        conc = foliages[k1] + foliages[k2]
        if conc < foliages[k2] and (len(foliages[k1])==1 or hall_set_reversed[foliages[k1]][1]>=foliages[k2]):
            ans = self.key_to_sparse(foliages_reversed[conc])
            self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
            return ans

        t = hall_set.get((foliages[k1], foliages[k2]), 0)
        if t:
            ans[foliages_reversed[t]] = float(1)

        else:
            ## We use Jacobi:
            # k1 = (k3, k4)
            # [k1, k2] = [k4, [k2, k3]] + [k3, [k4, k2]]
        
            (k3_fol, k4_fol) = hall_set_reversed[foliages[k1]]  

            t1 = self.multiply(self.key_to_sparse(foliages_reversed[k4_fol]), 
                                self.prod(k2, foliages_reversed[k3_fol], width, depth),  
                                width, depth)

            t2 = self.multiply(self.key_to_sparse(foliages_reversed[k3_fol]),
                                self.prod(foliages_reversed[k4_fol], k2, width, depth),
                                width, depth)

            ans = self.add_into(t1, t2)

        self.prod_cache[(k1, k2, width, depth)] = copy.deepcopy(ans)
        return ans

    def multiply(self, lhs, rhs, width, depth):
        """calculates the lie product of polynomials lhs and rhs in the lyndon basis
            ignoring any words created longer than depth
        """
        ans = defaultdict(float)
        for k1 in sorted(lhs.keys()):
            for k2 in sorted(rhs.keys()):
                a1 = self.prod(k1, k2, width, depth)
                factor = lhs[k1] * rhs[k2]
                a1 = self.scale_into(a1, factor)
                self.add_into(ans, a1)
        return self.sparsify(ans)

    def lie_to_string(self, li, width, depth):
        return " + ".join([str(li[x]) + " " + self.hb_to_string(x, width, depth) for x in sorted(li.keys())])

    def sparse_to_dense(self, sparse, width, depth):
        hall_set, _, _, _ = self.hall_basis(width, depth)
        dense = torch.zeros(len(hall_set), dtype=torch.double, device=self.device)
        for k in sparse.keys():
            if k < len(hall_set):
                dense[k] = sparse[k]
        return dense[1:]

    def dense_to_sparse(self, dense, width, depth):
        sparse = defaultdict(float)
        for k in range(len(dense)):
            if dense[k]:
                sparse[k+1] = dense[k].tolist()
        return sparse

    def expand(self, k, width, depth):
        """ expand is a map from hall basis keys to tensors
        """

        if (k, width, depth) in self.expand_cache:
            return self.expand_cache[(k, width, depth)]

        _expand = functools.partial(self.expand, width=width, depth=depth)
        _tensor_multiply = functools.partial(self.tensor_multiply, depth=depth)

        if k:

            hall_set, reverse_map, degrees, foliages = self.hall_basis(width, depth)
            hall_set_reversed = {f:v for v,f in hall_set.items()}
            foliages_reversed = {f:v for v,f in foliages.items()}

            (k1_fol, k2_fol) = hall_set_reversed[foliages[k]]

            if k1_fol:

                k1 = foliages_reversed[k1_fol]
                k2 = foliages_reversed[k2_fol]

                ans = self.tensor_sub(_tensor_multiply(_expand(k1), _expand(k2)),
                                        _tensor_multiply(_expand(k2), _expand(k1))
                                        )

                self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
                return ans

            else:
                k2 = foliages_reversed[k2_fol]
                ans = self.zero(width, 1)
                ans[self.blob_size(width, 0) - 1 + k2] = float(1.)  
                self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
                return ans

        ans = self.zero(width)
        self.expand_cache[(k, width, depth)] = copy.deepcopy(ans)
        return ans

    ## tuple a1,a2,...,an is converted into [a1,[a2,[...,an]]] as a LIE element recursively
    def rbraketing(self, tk, width, depth):

        if (tk, width, depth) in self.rbraketing_cache:
            return self.rbraketing_cache[(tk, width, depth)]

        _rbracketing = functools.partial(self.rbraketing, width=width, depth=depth)
        _multiply = functools.partial(self.multiply, width=width, depth=depth)

        #if len(tk) > 1:
        if tk[1:]:
            ans = _multiply(_rbracketing(tk[:1]), _rbracketing(tk[1:]))
            
            self.rbraketing_cache[(tk, width, depth)] = copy.deepcopy(ans)
            return ans
       
        else:
            ans = defaultdict(float)
            if tk:
                ans[tk[0]] = float(1)

            self.rbraketing_cache[(tk, width, depth)] = copy.deepcopy(ans)
            return ans
  
    def key_to_sparse(self, k):
        """
        >>> lie = EsigPyTorchLie()
        >>> lie.key_to_sparse(7)
        defaultdict(<class 'float'>, {7: 1.0})

        """
        ans = defaultdict(float)
        ans[k] = float(1)
        return ans

    def add_into(self, lhs, rhs):
        """
        >>> lie = EsigPyTorchLie()
        >>> lhs = lie.key_to_sparse(3)
        >>> rhs = lie.key_to_sparse(5)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        """
        for k in rhs.keys():
            lhs[k] += rhs.get(k, float())
        return lhs

    def subtract_into(self, lhs, rhs):
        """
        >>> lie = EsigPyTorchLie()
        >>> lhs = lie.key_to_sparse(3)
        >>> rhs = lie.key_to_sparse(5)
        >>> lie.add_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lie.subtract_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
        """
        for k in rhs.keys():
            lhs[k] -= rhs.get(k, float())
        return lhs

    def scale_into(self, lhs, s):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lhs = lie.scale_into(lhs, 3)
        >>> print(lhs)
        defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
        >>> lhs = lie.scale_into(lhs, 0)
        >>> print(lhs)
        defaultdict(<class 'float'>, {})
        >>> 
        """
        ans = defaultdict(float)
        if s:
            for k in lhs.keys():
                ans[k] = lhs[k]*s
        else:
            ans = defaultdict(float)
        return ans

    def sparsify(self, arg):
        """
        >>> lie = EsigPyTorchLie()
        >>> rhs = lie.key_to_sparse(5)
        >>> lhs = lie.key_to_sparse(3)
        >>> lie.add_into(lhs,rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
        >>> lie.subtract_into(lhs, rhs)
        defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
        >>> lie.sparsify(lhs)
        defaultdict(<class 'float'>, {3: 1.0})
        >>> 
        """
        empty_key_vals = list(k for k in arg.keys() if not arg[k])
        for k in empty_key_vals:
            del arg[k]
        return arg

    def l2t(self, arg, width, depth):
        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.tensor_log(lie.stream2sigtensor(lie.brownian(100, width), depth), depth)
        >>> print(torch.sum(lie.tensor_sub(lie.l2t(lie.t2l(t), width, depth), t)[2:]**2).tolist() < 1e-25)
        True
        >>> 
        """
        _expand = functools.partial(self.expand, width=width, depth=depth)

        ans = self.zero(width)
        for k in arg.keys():
            if k:
                ans = self.tensor_add(ans, (self.rescale(_expand(k), arg[k])))
        
        # this is just to measure the  sparsity coefficient of the projection from lie to tensor.
        nonempty = len([val for val in self.expand_cache.values() if len(val) > 0])
        sig_dim = self.blob_size(width, depth) - self.blob_size(width)
        hall_set, _, _, _ = self.hall_basis(width, depth)
        logsig_dim = len(hall_set) - 1
        total = sig_dim * logsig_dim
        #self.sparsity_coeff_l2t = float(nonempty/total)  
        self.sparsity_coeff_l2t = float(nonempty/sig_dim)
                
        return ans


    # the shape of the tensor that contains
    # \sum t[k] index_to_tuple(k,width) is the tensor
    # None () (0) (1) ...(w-1) (0,0) ...(n1,...,nd) ...
    @functools.lru_cache(maxsize=None)
    def index_to_tuple(self, i, width):
        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.arange(width, depth)
        >>> print(t)
        tensor([ 2.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
                14., 15.], dtype=torch.float64)
        >>> for t1 in [t]:
        ...     for k, coeff in enumerate(t1):
        ...         print (coeff.tolist(), lie.index_to_tuple(k, width))
        ... 
        2.0 None
        1.0 ()
        2.0 (1,)
        3.0 (2,)
        4.0 (1, 1)
        5.0 (1, 2)
        6.0 (2, 1)
        7.0 (2, 2)
        8.0 (1, 1, 1)
        9.0 (1, 1, 2)
        10.0 (1, 2, 1)
        11.0 (1, 2, 2)
        12.0 (2, 1, 1)
        13.0 (2, 1, 2)
        14.0 (2, 2, 1)
        15.0 (2, 2, 2)
        >>> 
        """

        bz = i + 1
        d = self.layers(bz, width)  ## this index is in the d-tensors
        if d < 0:
            return
        j = bz - 1 - self.blob_size(width, d-1)
        ans = ()
        ## remove initial offset to compute the index
        if j >= 0:
            for jj in range(d):
                ans = (1 + (j % width),) + ans
                j = j // width
        return ans
    
    def t_to_string(self, i, width):
        j = self.index_to_tuple(i, width)
        if self.index_to_tuple(i, width) == None:
            return " "
        return "(" + ",".join([str(k) for k in self.index_to_tuple(i, width)]) + ")"

    @functools.lru_cache(maxsize=None)
    def sigkeys(self, width, desired_degree):
        self.t_to_string(0, width)
        return " " + " ".join([self.t_to_string(z, width) for z in range(1, self.blob_size(width, desired_degree))])

    def t2l(self, arg):
        # projects a lie element in tensor form to a lie in lie basis form

        """
        >>> lie = EsigPyTorchLie()
        >>> width = 2
        >>> depth = 3
        >>> t = lie.tensor_log(lie.stream2sigtensor(lie.brownian(100, width), depth), depth)
        >>> print(torch.sum(lie.tensor_sub(lie.l2t(lie.t2l(t), width, depth), t)[2:]**2).tolist()  < 1e-25)
        True
        >>> 
        """
        width = int(arg[0])
        depth = self.layers(len(arg), width)
        ans = defaultdict(float)
        ibe = self.blob_size(width, 0)  # just beyond the zero tensors
        ien = self.blob_size(width, depth)

        for i in range(ibe, ien):
            t = self.index_to_tuple(i, width)
            if t:
                ## must normalise to get the dynkin projection to make it a projection
                res = self.rbraketing(t, width, depth)
                res = self.scale_into(res, arg[i] / float(len(t)))
                self.add_into(ans, res)
                
        # this is just to measure the  sparsity coefficient of the projection from tensor to lie.
        nonempty = len([val for val in self.rbraketing_cache.values() if len(val) > 0])
        sig_dim = self.blob_size(width, depth) - self.blob_size(width)
        hall_set, _, _, _ = self.hall_basis(width, depth)
        logsig_dim = len(hall_set) - 1
        total = sig_dim * logsig_dim
        #self.sparsity_coeff_t2l = float(nonempty/total)
        self.sparsity_coeff_t2l = float(nonempty/sig_dim)

        return ans

if __name__ == "__main__":
    doctest.testmod()

