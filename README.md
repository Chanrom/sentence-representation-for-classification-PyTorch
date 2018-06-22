# Deep learning models for sentence representation on classification in PyTorch

This repository contains some popular deep learning models for sentence representation (also apply for document-level text) that built in [PyTorch](http://pytorch.org/). Intended for learning PyTorch, this repo is made understandable for someone with basic python and deep learning knowledge. Links to some papers are also given.

## Requirement
* python 2.7
* pytorch 0.2
* torchtext 0.2

## Folder Structure
* model file: 
   * model/model.py, contains the deep models for sentence representation.
* training framework: train.py - preprocesses the data and trains the network.
* configuration files
   * i.e. trec/trec.conf, the config file used to set the dataset and networks.
* help function: utils/utils.py. some helper functions.

## Models [IN PROGRESS]

For now, the models listed bellow are add into this repo. Some benchmarks for these models are also given (the hyper-parameters are far from being optimal, the performances of these models can be improved with carefully tuning).


|   Model     |  TREC6-valid<sup>[1](#foottime)</sup> | TREC6-test  |   SST2-valid<sup>[2](#foottime)</sup>   |    SST2-test   |
| ------------|   :----:   | :----------: | :--------: | :----------: |
|   LSTM      |      -     |     94.6     |   84.98    |    85.45     |
|   Bi-LSTM   |      -     |     94.4     |   85.21    |    86.44     |
|   CNN       |      -     |     95.2     |   84.63    |    84.73     |


<a name="foottime">1</a>: The best accuracy on test set is reported since it has no development set.

<a name="foottime">2</a>: Only the sentence-level training samples are used.

### LSTMs
* [Long short-term memory network](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)

### CNNs
* heavily based on [cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
