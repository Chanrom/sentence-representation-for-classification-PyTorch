#coding=utf-8

import math
import time
import sys
import os
import codecs
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets

from model.model import *
from utils.optim import *
from utils.utils import *
from utils.config import *


parser = argparse.ArgumentParser(description='train.py')

# dataset configuration
parser.add_argument('-config_file', type=str, default='./trec/trec-6.conf',
                    help='model configuration for a dataset, i.e. \'./trec/trec-6.conf\'')
parser.add_argument('-flag', type=str, default='',
                    help='unique flag ')
# GPU
parser.add_argument('-gpu', type=int, default=0,
                    help="Use CUDA on the listed devices, -1 for CPU")
parser.add_argument('-cudnn', action='store_false', default=True,
                    help="use cudnn by default")
parser.add_argument('-seed', type=int, default=1234,
                    help="seed for reproductive.")

opt = parser.parse_args()
if os.path.exists(opt.config_file):
    # 解析
    config = Config()
    config.load_config(opt.config_file)
else:
    print '\n* NO CONFIG FILE *'
    sys.exit()

config.set('gpu', opt.gpu)
config.set('cudnn', opt.cudnn)
config.set('seed', opt.seed)

print config.lists()
del opt

if config['gpu'] != -1:
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.set_device(config['gpu'])
    torch.backends.cudnn.enabled=config['cudnn']
torch.manual_seed(config['seed'])


def eval(model, criterion, data_loader):

    model.eval()

    loss_sum = right_sum = 0.0
    num = 0.0
    for batch in data_loader:
        tokens, length = batch.text
        label = batch.label
        batch_size = len(label)
        length = tuple(length.cpu().numpy())

        outputs = model(tokens, length)
        logits = outputs[0]

        loss = criterion(input=logits, target=label)

        num += batch_size
        batch_right = unwrap_scalar_variable(torch.eq(label,
                                 logits.max(1)[1]).float().sum())
        batch_total_loss = unwrap_scalar_variable(loss)*batch_size
        loss_sum += batch_total_loss
        right_sum += batch_right

    loss = loss_sum / num
    acc = right_sum / num

    model.train()

    return loss, acc


def train_models(model, train_loader, valid_loader, test_loader, optim):

    print '\n* Begin training...'

    nll_crit = nn.NLLLoss()
    best_valid_acc, best_valid_test_acc = 0, 0
    num_train_batches = len(train_loader)

    # config['eval_epoch']: how many times of evaluation in one epoch
    validate_every = num_train_batches // config['eval_epoch']
    start_time = time.time()
    try:
        # training
        model.train()

        train_loss_sum = train_right_sum = 0.0
        train_loss_sum_report = train_right_sum_report = 0.0
        num_train = num_train_report = 0.0

        iterations = 0
        for batch_iter, train_batch in enumerate(train_loader):
            iterations += 1
            tokens, length = train_batch.text
            length = tuple(length.cpu().numpy())
            label = train_batch.label
            batch_size = len(label)
            outputs = model(tokens, length)
            logits = outputs[0]
            loss = nll_crit(input=logits, target=label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # stats
            num_train += batch_size
            num_train_report += batch_size
            batch_right = unwrap_scalar_variable(torch.eq(label,
                                     logits.max(1)[1]).float().sum())
            batch_total_loss = unwrap_scalar_variable(loss)*batch_size
            train_loss_sum += batch_total_loss
            train_right_sum += batch_right
            train_loss_sum_report += batch_total_loss
            train_right_sum_report += batch_right

            if (batch_iter + 1) % validate_every == 0:
                # evaluation
                valid_loss, valid_acc = eval(model, nll_crit, valid_loader)

                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc

                    # if we have test dataset
                    if test_loader:
                        test_loss, best_valid_test_acc = eval(model, nll_crit, test_loader)

            # config['interval']: how many times of printing information in one epoch
            interval_every = num_train_batches // config['interval']
            if (batch_iter + 1) % interval_every == 0:
                _s = 'loss %6.4f, acc: %4.2f;' % (
                    train_loss_sum_report/num_train_report,
                    train_right_sum_report/num_train_report
                    )
                train_loss_sum_report = train_right_sum_report = 0.0
                num_train_report = 0.0
                print "Iteration %6d; %s%5.0fs elapsed" %\
                                (iterations, _s, time.time()-start_time)

            # print some information at every epoch
            if (batch_iter + 1) % num_train_batches == 0:

                train_loss = train_loss_sum / num_train
                train_acc = train_right_sum / num_train

                print 'Train loss: %6.4f, acc: %5.4f' % (train_loss, train_acc)
                print 'Valid loss: %6.4f, cur_acc: %5.4f, best_acc: %5.4f, test_acc: %5.4f' % (valid_loss,
                                                                              valid_acc,
                                                                              best_valid_acc,
                                                                              best_valid_test_acc)
                train_loss_sum = train_right_sum = 0.0
                num_train = 0.0

                # schedule the learning rate
                optim.scheduler.step(valid_acc)

            if train_loader.epoch > config['epochs']:
                break

        print '\n* Best valid acc.: %5.4f'%best_valid_acc
        print '* Test accuracy with best valid acc.: %5.4f'%best_valid_test_acc

    except KeyboardInterrupt:
        print '\n* Best valid acc.: %5.4f'%best_valid_acc
        print '* Test accuracy with best valid acc.: %5.4f'%best_valid_test_acc


def main():

    print '\n* Preparing dataset...'
    text_field = data.Field(lower=config['lower'], include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False, unk_token=None)

    train_loader = None
    valid_loader = None
    test_loader = None
    if config['dataset'] == 'TREC-6':
        dataset_splits = datasets.TREC.splits(
            root='data', text_field=text_field, label_field=label_field,
            fine_grained=config['fine_grained'])

        text_field.build_vocab(*dataset_splits, vectors=config['pretrained'])
        label_field.build_vocab(*dataset_splits)
        num_tokens = len(text_field.vocab)
        num_classes = len(label_field.vocab)
        PAD_ID = text_field.vocab.stoi['<pad>']
        dataset_info = {'num_tokens':num_tokens,
                        'num_classes':num_classes,
                        'PAD_ID':PAD_ID}

        print '  initialize with pretrained vectors: %s' % config['pretrained']
        print '  number of classes: %d' % num_classes
        print '  number of tokens: %d' % num_tokens
        print '  max batch size: %d' % config['batch_size']

        # sort_within_batch is set True because we may use nn.LSTM
        train_loader, valid_loader = data.BucketIterator.splits(
            datasets=dataset_splits, batch_size=config['batch_size'],
            sort_within_batch=True, device=config['gpu'])
        print '* OK.'

    elif config['dataset'] == 'SST-2':
        pass

    print '* Building model...'
    cls_model = RepModel(config, dataset_info)

    if config['pretrained']:
        cls_model.word_lut.weight.data.set_(text_field.vocab.vectors)
    if config['fix_word_emb']:
        print '  will not update word embeddings'
        cls_model.word_lut.weight.requires_grad = False

    cls_model = cls_model.cuda() if config['gpu'] != -1 else cls_model
    params = [p for p in cls_model.parameters() if p.requires_grad]
    n_params = sum([p.nelement() for p in params])
    print '  number of total parameters: %d' % n_params
    print '* OK.'

    optim = Optim(config)
    optim.set_parameters(params)

    train_models(cls_model, train_loader, valid_loader, test_loader, optim)


if __name__ == "__main__":
    main()
