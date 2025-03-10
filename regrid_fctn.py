import numpy as np
import xesmf as xe
import numpy as np

import numpy as np
import xarray as xr
import xesmf as xe

import numpy as np
import xarray as xr
import xesmf as xe
from datetime import datetime
import pandas as pd

def read_and_average_era5_marta(field="influx_direct"):
    """Read a single NetCDF file containing multiple years and compute the long-term mean"""
    
    # Path to the single file containing data for multiple years
    diri = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/"
    file = "europe-2013.nc"  # Update this to your actual file name

    # Open the single NetCDF file
    df = xr.open_dataset(diri + file, decode_times=False)

    # If the file contains monthly data over multiple years, 
    # manually assign the correct time range (1979–2014).
    # Adjust the 'time' dimension as needed.
    first_time = datetime(2013, 1, 15)
    end_time = datetime(2013, 12, 15)
    
    # Ensure the time range matches the length of the time dimension in the dataset
    time_length = len(df['time'])  # or whichever dimension holds time
    df['time'] = pd.date_range(first_time, end_time, freq="1M", periods=time_length)
    
    # Rename the month dimension to 'time' if necessary (depending on how it's stored in your file)
    df = df.rename({'month': 'time'})

    # Compute the long-term mean of the selected field over the time dimension
    return df[field].mean(dim="time")



def read_and_average_sarah(diri,field="influx_direct"): #diri: "SARAH-3"
    path = "/groups/EXTREMES/"+diri
    
       # List of years for which we have files (1996, 2010, 2012, 2013)
    years = [2013]
    
    # Create the list of files based on the years
    files = [f'{path}europe-{year}-sarah3-era5.nc' for year in years]
    #print(files)
    df = xr.open_mfdataset(files,combine="by_coords")

    return df[field].mean(dim="time")



def regrid(ds_in, ds_out, method='conservative'):
    """Setup coordinates for esmf regridding"""

    lon = ds_in.lon   # centers
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0]-dlon/2.,lon[-1]+dlon/2.,len(lon)+1)
    print(lon.size,lon_b.size)

    lat = ds_in.lat
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0]-dlat/2.,lat[-1]+dlat/2.,len(lat)+1)
    print(lat.size,lat_b.size)
    
    grid_in = {'lon': lon, 'lat': lat,
               'lon_b': lon_b, 'lat_b': lat_b}

    lon = ds_out.lon   # centers
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0]-dlon/2.,lon[-1]+dlon/2.,len(lon)+1)
    print(lon.size,lon_b.size)

    lat = ds_out.lat
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0]-dlat/2.,lat[-1]+dlat/2.,len(lat)+1)
    print(lat.size,lat_b.size)

    grid_out = {'lon': lon, 'lat': lat,
                'lon_b': lon_b, 'lat_b': lat_b}

    regridder = xe.Regridder(grid_in, grid_out, method, periodic=False)
    regridder.clean_weight_file()
    return regridder



