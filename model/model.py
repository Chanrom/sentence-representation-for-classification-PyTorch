#coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.mem_size = config['mem_size']
        self.layers = config['layer']
        self.num_directions = 2 if config['brnn'] else 1
        assert self.mem_size % self.num_directions == 0
        self.hidden_size = self.mem_size // self.num_directions
        self.merge = config['merge'] # how to use all LSTM hidden states

        self.rnn = nn.LSTM(config['word_vec_size'], self.hidden_size,
                        num_layers=self.layers,
                        dropout=config['dropout'],
                        bidirectional=config['brnn'],
                        batch_first=True)

        self.dropout = nn.Dropout(config['dropout'])

        self.gpu = config['gpu']


    def avg_reps(self, out, lens):
        lens = Variable(torch.FloatTensor(list(lens)))
        if self.gpu != -1:
            lens = lens.cuda()

        # 可以使用broadcast改写
        avg_m = out.sum(1).squeeze(1) # batch x hidden
        lens = lens.unsqueeze(1).expand_as(avg_m)
        avg_m = torch.div(avg_m, lens) # batch x hidden
        return avg_m


    def forward(self, tokens_emb, length):
        batch_size = tokens_emb.size(0)

        # transporse because of LSTM accepting seq_len*batch_size
        tokens_emb = pack(tokens_emb, length, batch_first=True)

        outputs, states_t = self.rnn(tokens_emb)
        reps, _ = unpack(outputs, batch_first=True)

        # fetch the top layer's hidden state
        h_t = states_t[0][(self.layers - 1)*self.num_directions]
        h_t = h_t.transpose(0, 1).contiguous().view(batch_size, -1)

        rep = None
        if self.merge == 'last':
            rep = last_hidden
        elif self.merge == 'mean':
            rep = self.avg_reps(reps, length)
        elif self.merge == 'last':
            rep = h_t

        return self.dropout(rep), None



class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.kernel_sizes = [int(x) for x in config['kernel_sizes'].split(',')]
        self.max_filter_size = max(self.kernel_sizes)
        self.kernel_num = config['mem_size']
        assert self.kernel_num % len(self.kernel_sizes) == 0

        self.convs = nn.ModuleList([nn.Conv2d(1,
                    self.kernel_num/len(self.kernel_sizes),
                    (k, config['word_vec_size'])) for k in self.kernel_sizes])

        self.dropout = nn.Dropout(config['dropout'])


    def forward(self, tokens_emb, length):
        max_len = tokens_emb.size(1)

        # if longest sentence in the batch is too short
        if self.max_filter_size > max_len:
            tokens_zeros = Variable(tokens_emb.data.new(tokens_emb.size(0),
                                self.max_filter_size - max_len,
                                tokens_emb.size(2)))
            tokens_emb = torch.cat([tokens_emb, tokens_zeros], 1)
        tokens_emb = tokens_emb.unsqueeze(1)  # (batch_size, 1, max_len, word_dim)

        # [(batch_size, kernel_num/len(kernel_size), max_len), ...]*len(kernel_size)
        sentence_embs = [F.relu(conv(tokens_emb)).squeeze(3)\
                         for conv in self.convs]

        # [(batch_size, kernel_num/len(kernel_size)), ...]*len(kernel_size)
        sentence_embs = [F.max_pool1d(i, i.size(2)).squeeze(2)\
                         for i in sentence_embs]

        rep = torch.cat(sentence_embs, 1) # (batch_size, kernel_num)

        return self.dropout(rep), None



class Classifier(nn.Module):
    def __init__(self, config, classes):
        super(Classifier, self).__init__()

        self.classes = classes
        self.input_size = config['mem_size']
        self.hidden_size = config['hid_size_cls']
        self.layers = config['layer_cls'] + 1 # including the output layer
        self.param_init = config['param_init']

        self.dropout = nn.Dropout(config['dropout'])

        self.mlps = nn.ModuleList()
        for i in range(self.layers):
            if i == 0:
                self.mlps.append(nn.Linear(self.input_size, self.hidden_size))
            elif i < self.layers - 1:
                self.mlps.append(nn.Linear(self.hidden_size, self.hidden_size))
            else:
                self.mlps.append(nn.Linear(self.hidden_size, self.classes))

        self.tanh = nn.Tanh()
        self.log_sm = nn.LogSoftmax()

        self.reset_parameters()


    def reset_parameters(self):
        for i in range(self.layers-1):
            init.kaiming_normal(self.mlps[i].weight.data)
            init.constant(self.mlps[i].bias.data, val=0)
        init.uniform(self.mlps[self.layers-1].weight.data,
                     -self.param_init, self.param_init)
        init.constant(self.mlps[self.layers-1].bias, val=0)


    def forward(self, rep):
        '''
        rep: batch x mem_size
        '''
        for i in range(self.layers-1):
            rep = self.dropout(self.tanh(self.mlps[i](rep)))
        logit = self.mlps[self.layers-1](rep)

        output_sm = self.log_sm(logit)
        return output_sm



class RepModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self, config, dataset_info):
        super(RepModel, self).__init__()

        num_tokens = dataset_info['num_tokens']
        num_classes = dataset_info['num_classes']
        PAD_ID = dataset_info['PAD_ID']

        self.word_lut = nn.Embedding(num_tokens, config['word_vec_size'],
                                    padding_idx=PAD_ID)

        if config['encoder'] == 'LSTM':
            self.encoder = LSTM(config)
        elif config['encoder'] == 'CNN':
            self.encoder = CNN(config)
        else:
            raise RuntimeError("Invalid model name: " + config['encoder'])

        self.classifier = Classifier(config, num_classes)

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tokens, length):
        tokens_emb = self.dropout(self.word_lut(tokens))

        outputs = self.encoder(tokens_emb, length)
        sentence_emb = outputs[0]

        logits = self.classifier(sentence_emb)

        return logits, None # None for placeholder

