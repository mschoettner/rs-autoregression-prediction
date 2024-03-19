"""Convert HCP data to HDF5 format"""

import argparse

import h5py
import pandas as pd
import numpy as np

timeseries_dir = "/home/mschoett/projects/rrg-pbellec/mschoett/HCP_fMRI_timeseries"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=str,
    default="inputs/connectomes/hcp.h5",
    help="path to save the output",
)
args = parser.parse_args()
print(args)

path_concat = str(args.output)
