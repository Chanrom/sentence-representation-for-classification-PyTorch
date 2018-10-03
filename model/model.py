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
        elif self.merge == 'all':
            rep = reps

        return self.dropout(rep), None



class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.kernel_sizes = [int(x) for x in 
                             config['kernel_sizes'].split(',')]
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
        sentence_embs = [F.relu(conv(tokens_emb)).squeeze(3)
                         for conv in self.convs]

        # [(batch_size, kernel_num/len(kernel_size)), ...]*len(kernel_size)
        sentence_embs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                         for i in sentence_embs]

        rep = torch.cat(sentence_embs, 1) # (batch_size, kernel_num)

        return self.dropout(rep), None


class SelfAttn(nn.Module):
    def __init__(self, config):
        super(SelfAttn, self).__init__()

        self.mem_size = config['mem_size']
        self.attn_size = config['attn_size']
        self.hops = config['hops']
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

        self.ws1 = nn.Linear(self.mem_size, self.attn_size, bias=False)
        self.ws2 = nn.Linear(self.attn_size, self.hops, bias=False)

        self.dropout = nn.Dropout(config['dropout'])

        self.tanh = nn.Tanh()
        self.sm = nn.Softmax()
        self.gpu = config['gpu']


    def get_mask(self, lens):
        mask = []
        max_len = max(lens)
        for l in lens:
            mask.append([1]*l + [0]*(max_len - l))
        mask = Variable(torch.FloatTensor(mask))
        if self.gpu != -1:
            return mask.cuda()
        else:
            return mask


    def forward(self, tokens_emb, length):
        batch_size = tokens_emb.size(0)

        tokens_emb = pack(tokens_emb, length, batch_first=True)

        outputs, states_t = self.rnn(tokens_emb)
        reps, _ = unpack(outputs, batch_first=True)
        # print 'reps', reps

        size = reps.size()
        compressed_reps = reps.contiguous().view(-1, size[2])  # (batch_size x seq_len) * mem_size

        hbar = self.tanh(self.ws1(compressed_reps))  # (batch_size x seq_len) * attn_size
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # batch_size * seq_len * hops
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # batch_size * hops * seq_len

        mask = self.get_mask(length)
        # print 'mask', mask
        multi_mask = [mask.unsqueeze(1) for i in range(self.hops)]
        multi_mask = torch.cat(multi_mask, 1)
        # print 'multi_mask', multi_mask

        penalized_alphas = alphas + -1e7 * (1 - multi_mask)
        alphas = self.sm(penalized_alphas.view(-1, size[1]))  # (batch_size x hops) * seq_len
        alphas = alphas.view(size[0], self.hops, size[1])  # batch_size * hops * seq_len
        # print 'alphas', alphas

        reps = torch.bmm(alphas, reps) # batch_size * hops * hidden_size
        # here we use mean pooling of all hops
        rep = reps.mean(1)
        assert len(rep.size()) == 2

        # batch_size * classes, batch_size * hops * seq_len
        return self.dropout(rep), alphas



class BCN(nn.Module):
    """implementation of Biattentive Classification Network in 
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""
    def __init__(self, config, pad_id):
        super(BCN, self).__init__()
        self.word_vec_size = config['word_vec_size']
        self.mtlstm_hidden_size = config['mtlstm_hidden_size']
        self.cove_size = self.mtlstm_hidden_size + self.word_vec_size
        self.fc_hidden_size = config['fc_hidden_size']
        self.bilstm_encoder_size = config['bilstm_encoder_size']
        self.bilstm_integrator_size = config['bilstm_integrator_size']
        self.fc_hidden_size1 = config['fc_hidden_size1']
        self.mem_size = config['mem_size']

        # model parameters is downloaded
        # model_urls = {
        #     'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth' 
        # }
        model_cache = './.vector_cache/'

        self.rnn = nn.LSTM(self.word_vec_size, self.mtlstm_hidden_size/2,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True,
                           dropout=config['dropout'])
        # self.rnn.load_state_dict(model_zoo.load_url(model_urls['wmt-lstm'],
        #                          model_dir=model_cache))
        # load parameters to CPU (then put on GPU use .cuda())
        self.rnn.load_state_dict(torch.load(model_cache + 'wmtlstm-chanrom.pth',
                            map_location=lambda storage, loc: storage))

        self.fc = nn.Linear(self.cove_size, self.fc_hidden_size)

        self.bilstm_encoder =  nn.LSTM(self.fc_hidden_size,
                               self.bilstm_encoder_size/2,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True,
                               dropout=config['dropout'])

        self.bilstm_integrator = nn.LSTM(self.bilstm_encoder_size * 3,
                               self.bilstm_integrator_size/2,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True,
                               dropout=config['dropout'])

        self.attentive_pooling_proj = nn.Linear(self.bilstm_integrator_size,
                                                1)

        self.fc1 = nn.Linear(self.bilstm_integrator_size * 4,
                                 self.fc_hidden_size1)
        self.fc2 = nn.Linear(self.fc_hidden_size1, self.mem_size)

        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        self.log_sm = nn.LogSoftmax()
        self.dropout = nn.Dropout(config['dropout'])

        self.pad_id = pad_id

        self.gpu = config['gpu']


    def makeMask(self, lens, hidden_size):
        mask = []
        max_len = max(lens)
        for l in lens:
            mask.append([1]*l + [0]*(max_len - l))
        mask = Variable(torch.FloatTensor(mask))
        if hidden_size == 1:
            trans_mask = mask
        else:
            trans_mask = mask.unsqueeze(2).expand(mask.size(0),
                                                  mask.size(1), 
                                                  hidden_size)
        if self.gpu != -1:
            return trans_mask.cuda()
        else:
            return trans_mask


    def forward(self, tokens_emb, length):
        batch_size = tokens_emb.size(0)

        tokens_emb_pack = pack(tokens_emb, length, batch_first=True)
        outputs_pack, states_t = self.rnn(tokens_emb_pack)

        mt_outputs, _ = unpack(outputs_pack, batch_first=True)
        reps = torch.cat([mt_outputs, tokens_emb], 2)
        reps = self.dropout(reps)

        max_len = max(length)

        compressed_reps = reps.view(-1, self.cove_size)
        task_specific_reps = (self.relu(self.fc(compressed_reps))).view(
                                batch_size,
                                max_len,
                                self.fc_hidden_size)
        task_specific_reps = pack(task_specific_reps,
                                  length, 
                                  batch_first=True)

        outputs, _ = self.bilstm_encoder(task_specific_reps)
        X, _ = unpack(outputs, batch_first=True)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = X.bmm(X.permute(0, 2, 1).contiguous())

        attention_mask1 = Variable((-1e7 * (attention_logits <= 1e-7).float()).data)
        masked_attention_logits = attention_logits + attention_mask1
        compressed_Ay = self.sm(masked_attention_logits.view(-1, max_len))
        attention_mask2 = Variable((attention_logits >= 1e-7).float().data) # mask those all zeros
        Ay = compressed_Ay.view(batch_size, max_len, max_len) * attention_mask2

        Cy = torch.bmm(Ay, X) # batch_size * max_len * bilstm_encoder_size

        # Build the input to the integrator
        integrator_input = torch.cat([Cy,
                                      X - Cy,
                                      X * Cy], 2)
        integrator_input = pack(integrator_input, length, batch_first=True)

        outputs, _ = self.bilstm_integrator(integrator_input) # batch_size * max_len * bilstm_integrator_size
        Xy, _ = unpack(outputs, batch_first=True)

        # Simple Pooling layers
        max_masked_Xy = Xy + -1e7 * (1 - self.makeMask(length,
                                     self.bilstm_integrator_size))
        max_pool = torch.max(max_masked_Xy, 1)[0]
        min_masked_Xy = Xy + 1e7 * (1 - self.makeMask(length, 
                                     self.bilstm_integrator_size))
        min_pool = torch.min(min_masked_Xy, 1)[0]
        mean_pool = torch.sum(Xy, 1) / torch.sum(self.makeMask(length, 1),
                                                 1, 
                                                 keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attentive_pooling_proj(Xy.contiguous().view(-1,
                                                             self.bilstm_integrator_size))
        self_attentive_logits = self_attentive_logits.view(batch_size, max_len) \
                                        + -1e7 * (1 - self.makeMask(length, 1))
        self_weights = self.sm(self_attentive_logits)
        self_attentive_pool = torch.bmm(self_weights.view(batch_size,
                                                          1, 
                                                          max_len),
                                                          Xy).squeeze(1)

        pooled_representations = torch.cat([max_pool, 
                                            min_pool, 
                                            mean_pool, 
                                            self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout(pooled_representations)

        rep = self.dropout(self.relu(self.fc1(pooled_representations_dropped)))
        rep = self.dropout(self.relu(self.fc2(rep)))

        return rep, None



class Classifier(nn.Module):
    def __init__(self, config, classes):
        super(Classifier, self).__init__()

        self.classes = classes
        self.input_size = config['mem_size']
        self.hidden_sizes = [int(x) for x in 
                        config['hid_sizes_cls'].split(',') if x]
        assert len(self.hidden_sizes) >= 1
        self.layers = len(self.hidden_sizes) + 1 # including the output layer
        self.param_init = config['param_init']

        self.dropout = nn.Dropout(config['dropout'])

        self.mlps = nn.ModuleList()
        for i in range(self.layers):
            if i == 0:
                self.mlps.append(nn.Linear(self.input_size, 
                                           self.hidden_sizes[i]))
            elif i < self.layers - 1:
                self.mlps.append(nn.Linear(self.hidden_sizes[i], 
                                           self.hidden_sizes[i + 1]))
        self.mlps.append(nn.Linear(self.hidden_sizes[-1],
                                           self.classes))

        self.tanh = nn.Tanh()
        self.log_sm = nn.LogSoftmax()


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
        elif config['encoder'] == 'SelfAttn':
            self.encoder = SelfAttn(config)
        elif config['encoder'] == 'BCN':
            self.encoder = BCN(config, PAD_ID)
        else:
            raise RuntimeError("Invalid model name: " + config['encoder'])

        self.classifier = Classifier(config, num_classes)

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tokens, length):
        tokens_emb = self.dropout(self.word_lut(tokens))

        outputs = self.encoder(tokens_emb, length)
        sentence_emb = outputs[0]
        extra_out = outputs[1]

        logits = self.classifier(sentence_emb)

        return logits, extra_out # extra_out can be 'attention weight' or other intermediate information

