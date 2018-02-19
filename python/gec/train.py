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
        self.gru = nn.GRU(bag_size, m["hidden_size"], bidirectional=True)

    def forward(self, m):
        embedding, states = self.gru(m["input"], m["states"])
        return {"embedding": embedding,
                "states": states}


def get_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_bidirectional_size(n):
    return n * 2


num_layers = 1


def get_state(m):
    return autograd.Variable(init.kaiming_normal(
        get_cuda(torch.zeros(get_bidirectional_size(1),
                             m["batch_size"],
                             m["hidden_size"]))))


def get_states(m):
    return tuple(repeatedly(partial(get_state, m), 2))


def get_glob(m):
    return path.join(resources_path,
                     "dataset",
                     m["dataset"],
                     "split",
                     "training",
                     "*")


def get_data(filename):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)


def get_raw_batches(m):
    return map(tuple, (
        mapcat(compose(partial(partition, m["batch_size"]), get_data),
               cycle(glob.glob(get_glob(m))))))


if __name__ == "__main__":
    pass
