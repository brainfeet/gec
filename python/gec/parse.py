import argparse
import glob
import json
import os
import re

from funcy import *
import spacy

from gec.clojure import *

nlp = spacy.load("en")


def get_isolated_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    return glob.glob(parser.parse_args().path + "/*")


# TODO rename this function as get_token_map
def get_dictionary(token):
    return {"lemma_": token.lemma_, "text": token.text, "tag_": token.tag_}


parse_stringify = compose(json.dumps,
                          tuple,
                          partial(map, get_dictionary),
                          nlp)

get_parsed_path = partial(re.sub, r"isolated/([^\/]+).txt$", r"parsed/\1.json")


def spit(path, s):
    with open(path, "w") as file:
        file.write(s)


def mkdirs(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def spit_parents(path, s):
    mkdirs(path)
    spit(path, s)


def parse():
    tuple(map(spit_parents,
              map(get_parsed_path, get_isolated_paths()),
              map(compose(parse_stringify, slurp),
                  get_isolated_paths())))


if __name__ == "__main__":
    print(parse())
