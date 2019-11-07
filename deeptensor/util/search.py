from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import re

from ..debug import log
from ..util.opt import Opt


def search_text(text, pattern, find_all=True, ignore_case=False):
    if text is None:
        return None

    findings = []

    lines = text.splitlines()
    #log.debug(log.DC.STD, " [{}]".format(len(lines)))

    flags = 0
    if ignore_case:
        flags |= re.I

    found_count = 0
    for i, line in enumerate(lines):
        match = re.search(pattern, line, flags)
        if match:
            found_count += 1
            marked_line = re.sub('('+pattern+')', r'\033[1;91m\1\033[0m', line)
            found = Opt(idx=found_count, row=i+1, col=match.start()+1, line=line, marked=marked_line)
            findings.append(found)
            if not find_all:
                break

    return findings
