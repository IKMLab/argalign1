"""Data utilities for alignment work."""
import json

import numpy as np


def load_json(file):
    with open('data/%s.json' % file) as f:
        return json.loads(f.read())


def load_npy(file):
    with open('data/%s.npy' % file, 'rb') as f:
        return np.load(f)


def logit(p):
    if p == 0.:
        return -np.inf
    return -np.log(1/p - 1)


def rev_dict(d):
    return {v: k for k, v in d.items()}
