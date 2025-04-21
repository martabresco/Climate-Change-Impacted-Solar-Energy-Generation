from __future__ import annotations
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)
import atlite


import datetime as dt
import logging
from collections import namedtuple
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import absolute, arccos, cos, maximum, mod, radians, sin, sqrt, arcsin, arctan2, radians, arctan
from dask.diagnostics import ProgressBar
from numpy import pi
from scipy.sparse import csr_matrix
from atlite.aggregate import aggregate_matrix
from atlite.gis import spdiag
from numpy import pi
import sys
from numpy import logical_and

from atlite import csp as cspm

from regridding_functions import read_and_average_era5_4y
from regridding_functions import regrid


def retieve_filepath(model, variant, period):
    
def albedo_for_model(models):
    for model in models:
        filepath=[]

def power_calculation(models, variants, period, output_dir, albedo):
    
    return 

     

def main():
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    period = ["historical","ssp585"]
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power/"

if __name__ == "__main__":
    main()