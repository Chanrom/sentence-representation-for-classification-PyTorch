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
