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

if __name__ == "__main__":
    pass
