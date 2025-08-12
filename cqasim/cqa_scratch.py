#!/usr/bin/env python3
"""Scratch file for the CQA model."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from itertools import combinations
from typing import Optional

from cqasim.cqa_utils import pad_with_nans
from cqasim.cqa_vectutils import broken_gaussian_1d, gaussian_1d
