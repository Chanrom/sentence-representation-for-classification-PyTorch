# Deep learning models for sentence representation on classification in PyTorch

This repository contains some popular deep learning models for sentence representation (also apply for document-level text) that built in [PyTorch](http://pytorch.org/). Intended for learning PyTorch, this repo is made understandable for someone with basic python and deep learning knowledge. Links to some papers are also given.

## Requirement
* python 2.7
* pytorch 0.2
* torchtext 0.2

## Folder Structure
* model file: model/model.py, contains the deep models for sentence representation.
* training framework: train.py - preprocesses the data and trains the network.
* configuration files: i.e. trec/trec.conf, the config file used to set the dataset and networks.
* help function: utils/utils.py. some helper functions.

## Models [IN PROGRESS]

For now, the models listed bellow are add into this repo. Some benchmarks for these models are also given.


|   Model     | TREC6-valid<sup>[1](#foottime)</sup> | TREC6-test  |   SST2-valid<sup>[2](#foottime)</sup>   |    SST2-test   |
| ------------| :----: | :---------: | :-------: | :----------: |
|   LSTM      |  -          |   93.6      |   84.9         |    87.2        |

<a name="foottime">1</a>: The best accuracy on test set is reported since it has no development set.

<a name="foottime">2</a>: Only the sentence-level training samples are used.

### LSTMs
* [Long short-term memory network](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)

### CNNs
* Image Classification: [Microsoft-ResNet 2015](https://arxiv.org/pdf/1512.03385.pdf)
