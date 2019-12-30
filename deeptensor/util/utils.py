from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

sys_rnd = random.SystemRandom()

def token_in_list(line_list, token):
    in_list = False

    if line_list and len(token) > 0:
        for line in line_list:
            if token in line:
                in_list = True
                break

    return in_list

def list_in_token(line_list, token):
    in_token = False

    if line_list and len(token) > 0:
        for line in line_list:
            if line in token:
                in_token = True
                break

    return in_token

def split_list(line_list, parts, rand=False):
    splits = None

    if line_list and parts > 0:
        splits = []
        for i in range(parts):
            splits.append([])

        if rand:
            random.shuffle(line_list)

        for i, line in enumerate(line_list):
            splits[i%parts].append(line)

    return splits

def random_int(a, b):
    return sys_rnd.randint(a, b)
