import math
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

class Optim(object):
    def __init__(self, config):
        self.lr = config['lr']
        self.max_grad_norm = config['max_grad_norm']
        self.method = config['optim']
        self.lr_decay = config['learning_rate_decay']
        self.weight_decay = config['weight_decay']
        self.momentum = config['momentum']
        self.reduce_lr_at = config['reduce_lr_at']

        self.scheduler = None

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(self.params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, mode='max', factor=0.5,
                patience=self.reduce_lr_at, verbose=True)

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
