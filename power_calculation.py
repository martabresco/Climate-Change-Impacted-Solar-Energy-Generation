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

from regridding_functions import read_and_average_era5_marta
from regridding_functions import regrid


import os
import glob
import os
import re

def collect_files(base_path, models, variants, periods):
    files_model = {}

    # Define the year ranges for each period
    year_ranges = {
        "historical": range(1980, 2015),  # 2014 included
        "ssp585": range(2065, 2100)       # 2099 included
    }
    for model, variant in zip(models, variants):
        model_files = {}
        for period in periods:
            # Construct the path
            search_path = os.path.join(base_path, model, period, variant)
            # Match files with the desired pattern
            file_pattern = os.path.join(search_path, "rsds_rsdsdiff_tas_*.nc")
            matched_files = glob.glob(file_pattern)

            # Filter files by year
            filtered_files = []
            for file_path in matched_files:
                filename = os.path.basename(file_path)
                # Extract year from filename
                match = re.search(r"(\d{4})", filename)
                if match:
                    year = int(match.group(1))
                    if year in year_ranges[period]:
                        filtered_files.append(file_path)

            model_files[period] = filtered_files
        files_model[model] = model_files

    return files_model

    
def albedo_for_model(models, variants):
    mean_albedo_era5 = read_and_average_era5_marta("albedo")
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/albedo/"
    os.makedirs(output_dir, exist_ok=True)
    for model, variant in zip(models, variants):
        try:
            filepath = f"/groups/FutureWind/SFCRAD/{model}/historical/{variant}/rsds_rsdsdiff_tas_2010.nc"
            ds_model = xr.open_dataset(filepath, engine="netcdf4")  # Explicitly specify the engine
            regridder_model = regrid(mean_albedo_era5, ds_model)
            albedo_model = regridder_model(mean_albedo_era5)
            output_path = os.path.join(output_dir, f"mean_albedo_grid_{model}.nc")
            albedo_model.to_netcdf(output_path)
            print(f"Saved regridded mean albedo for {model} to {output_path}")
        except Exception as e:
            print(f"Failed for model {model}: {e}")
    


def power_calculation(models, variants, period, output_dir, albedo):
    
    return 

     

def main():
    base_path="/groups/FutureWind/SFCRAD/"
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    period = ["historical","ssp585"]
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power/"
    files=collect_files(base_path, models, variants, period)
    print(files)
    albedo_for_model(models, variants)

if __name__ == "__main__":
    main()