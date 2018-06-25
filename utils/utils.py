#coding=utf-8
import os
import torch
import codecs
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def wrap_with_variable(tensor, volatile, gpu):
    if gpu > -1:
        return Variable(tensor.cuda(gpu), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var


def unsort(x, ind):
    '''
    ind: batch x 1, 倒排索引，使得按长度排序的表示重新排列回去
    '''
    outs = []
    for i in x:
        if i is not None:
            if isinstance(i, list) or isinstance(i, tuple):
                outs.append([list(i)[_x] for _x in ind.squeeze(1).data])
            else:
                # outs.append(torch.gather(i, 0, ind.expand(*i.size())))
                outs.append(i[ind.squeeze(1)])
        else:
            outs.append(None)
    return outs


def get_I(batch_size, hops, cuda):
    I = Variable(torch.zeros(batch_size, hops, hops))
    for i in range(batch_size):
        for j in range(hops):
            I.data[i][j][j] = 1
    if cuda != -1:
        I = I.cuda()
    return I


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')