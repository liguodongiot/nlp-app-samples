# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains project version information.
.. currentmodule:: nlp_app_samples.version
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
