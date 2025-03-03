import sys
import xarray as xr
import numpy as np
from datetime import timedelta
from glob import glob
import cftime
import os
from future_wind_copy import combine_hemispheres
from datetime import datetime,timedelta
import re
from collections import defaultdict


def open_nc_files_by_range(folder_path, target_year):
    """
    Opens NetCDF files for rsds, rsdsdiff, and tas that contain the given target year.

    Parameters:
    - folder_path (str): Path to the directory containing the NetCDF files.
    - target_year (int or str): The target year to filter the files.

    Returns:
    - ds_rsds (xarray.Dataset): Dataset for rsds. 
    - ds_rsdsdiff (xarray.Dataset): Dataset for rsdsdiff.
    - ds_tas (xarray.Dataset): Dataset for tas.
    """
    # Ensure target_year is a string
    target_year = str(target_year)

    # Regex to extract variable name, start year, and end year
    pattern = re.compile(r"(rsds|rsdsdiff|tas)_.*_(\d{4})\d{6}-(\d{4})\d{6}.nc")

    # Dictionary to store file paths, grouped by (start_year, end_year)
    files_dict = defaultdict(dict)

    # Loop through files in the folder
    for file in os.listdir(folder_path):
        match = pattern.search(file)
        if match:
            var_type, start_year, end_year = match.groups()
            file_path = os.path.join(folder_path, file)
            
            # Store file paths in a dictionary grouped by year range
            files_dict[(start_year, end_year)][var_type] = file_path

    # Find the correct files that contain the target year
    for (start_year, end_year), file_group in files_dict.items():
        if start_year <= target_year < end_year:  # Check if the target year falls within this range
            # Open the datasets if all three files are found
            ds_rsds = xr.open_dataset(file_group["rsds"]) if "rsds" in file_group else None
            ds_rsdsdiff = xr.open_dataset(file_group["rsdsdiff"]) if "rsdsdiff" in file_group else None
            ds_tas = xr.open_dataset(file_group["tas"]) if "tas" in file_group else None

            if ds_rsds and ds_rsdsdiff and ds_tas:
                return ds_rsds, ds_rsdsdiff, ds_tas

    print(f"Warning: No matching datasets found for year {target_year}.")
    return None, None, None

# Example usage:
folder = "/groups/FutureWind/SFCRAD/CanESM5/historical/r1i1p2f1"
year = 1981

ds_rsds, ds_rsdsdiff, ds_tas = open_nc_files_by_range(folder, year)

# Print dataset info
print(ds_rsds)
print(ds_rsdsdiff)
print(ds_tas)











