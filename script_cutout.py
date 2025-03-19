import xarray as xr
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import atlite
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)

import atlite

start_date = "2014-01-01"
end_date = "2014-01-02"

cutout_1x1 = atlite.Cutout(
    path="europe_1x1",
    module=["era5"],
    #sarah_dir="Sarah_data_2014_v4",  # Change directory
    x=slice(-12, 35),  # Updated longitude bounds
    y=slice(33, 72),   # Updated latitude bounds
    dx=1.0,            # Set grid resolution to 1º (longitude)
    dy=1.0,            # Set grid resolution to 1º (latitude)
    time=slice(start_date, end_date),
    chunks={"time": 100, "lat": -1, "lon": -1},
)

cutout_1x1.prepare()
