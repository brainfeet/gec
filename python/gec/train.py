import argparse
from functools import reduce
import glob
import json
import math
import os
import os.path as path
import shutil

from funcy import *
import funcy
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter
import torch
import torch.autograd as autograd
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchtext.vocab as vocab
import sys

from gec.clojure import *

sys.argv = ["", "--timestamp", "20180218012812"]

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp")
timestamp = first(parser.parse_known_args()).timestamp
resources_path = "../resources"
runs_path = path.join(resources_path, "runs")


def get_hyperparameter_path():
    return path.join(runs_path, timestamp, "hyperparameter.json")


get_hyperparameter = compose(json.loads,
                             slurp,
                             partial(get_hyperparameter_path))

bag_size = 128


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.gru = nn.GRU(bag_size, m["hidden_size"], batch_first=True,
                          bidirectional=True)

    def forward(self, m):
        embedding, hidden = self.gru(m["encoder_input"], m["hidden"])
        return {"embedding": embedding,
                "hidden": hidden}


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()

        self.embedding = nn.Embedding(m["vocabulary_size"], m["hidden_size"])
        self.attention = nn.Linear(m["hidden_size"] * 2, m["max_length"])
        self.attention_combine = nn.Linear(m["hidden_size"] * 2,
                                           m["hidden_size"])
        self.dropout = nn.Dropout(m["dropout_probability"])
        self.gru = nn.GRU(m["hidden_size"], m["hidden_size"])
        self.out = nn.Linear(m["hidden_size"], m["vocabulary_size"])


def get_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_bidirectional_size(n):
    return n * 2


num_layers = 1


def get_hidden(m):
    return autograd.Variable(init.kaiming_normal(
        get_cuda(torch.zeros(get_bidirectional_size(1),
                             m["batch_size"],
                             m["hidden_size"]))))


def get_glob(m):
    return path.join(resources_path,
                     "dataset",
                     m["dataset"],
                     "split",
                     "training",
                     "*")


def get_raw_data(filename):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)


def apply(f, *more):
    return f(*butlast(more), *last(more))


def vector(*more):
    return tuple(more)


def flip(f):
    def g(x, *more):
        if empty(more):
            def h(y, *more_):
                apply(f, y, x, more_)
            return h
        return apply(f, first(more), x, rest(more))
    return g


def get(m, k):
    return m[k]


def if_(test, then, else_):
    if test:
        return then
    return else_


def get_training_variables_(m):
    # TODO transform word and bpe
    return map(compose(if_(m["k"] == "bag",
                           compose(autograd.Variable, torch.FloatTensor, tuple),
                           identity),
                       partial(map, partial(flip(get), m["k"]))),
               m["raw_batches"])


def make_get_training_variables(m):
    def get_training_variables(k):
        return get_training_variables_(
            merge(m, {"k": k,
                      "raw_batches":
                          mapcat(compose(partial(partition, m["batch_size"]),
                                         get_raw_data),
                                 cycle(glob.glob(get_glob(m))))}))
    return get_training_variables


def get_batches(m):
    return apply(partial(map, vector),
                 map(make_get_training_variables(m),
                     ["bag", "word", "bpe"]))


def get_index_path(m):
    return path.join(resources_path,
                     "dataset",
                     m["dataset"],
                     "index.json")


get_index_dictionary = compose(json.loads,
                               slurp,
                               get_index_path)

get_vocabulary_size = compose(len,
                              get_index_dictionary)
