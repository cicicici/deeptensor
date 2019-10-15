from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil


def load_file(filename, mode='r'):
    content = None
    if os.path.exists(filename):
        with open(filename, mode) as f:
            content = f.read()
            f.close
    return content

def load_line_list(line_file):
    line_list = None
    if os.path.exists(line_file):
        line_list = []
        with open(line_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip('\n').lstrip(' ')
                if len(line) > 0 and not line.startswith('#'):
                    line_list.append(line)
            f.close()
    return line_list

def save_line_list(line_list, line_file):
    if line_list is not None:
        with open(line_file, 'w') as f:
            for line in line_list:
                f.write(line + '\n')
            f.close()

def file_exist(filename):
    return os.path.exists(filename)

def delete_file(filename):
    if file_exist(filename):
        os.remove(filename)
        return True
    return False

def duplicate_file(src, dst):
    if src != dst:
        shutil.copy2(src, dst)
        return True
    return False

def move_file(src, dst):
    if not file_exist(src):
        return False
    delete_file(dst)
    shutil.move(src, dst)
    return True

def is_link(filename):
    return os.path.islink(filename)

