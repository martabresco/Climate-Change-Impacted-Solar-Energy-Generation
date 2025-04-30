from __future__ import annotations
import xarray as xr
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
import pandas as pd
import xarray as xr
import pv_functions
import os
import glob
import os
import re
import cftime

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

from datetime import datetime
def transform_360_to_gregorian(ds: xr.Dataset, year: int) -> xr.Dataset:
    """
    Transform a dataset with a 360-day calendar to a proleptic Gregorian calendar.

    - Duplicates day 30 for all months that should have 31 days.
    - Removes Feb 30 in leap years, and Feb 29 + 30 in non-leap years.
    - Preserves rsds, rsdsdiff, and tas values exactly — no interpolation.
    - Assumes 3-hourly time steps starting at 01:30 on January 1.

    Parameters:
        ds (xarray.Dataset): The dataset with a 360-day calendar.
        year (int): The target Gregorian year to convert into.

    Returns:
        xarray.Dataset: The transformed dataset with Gregorian-style time axis.
    """

    def is_leap_year(y):
        return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

    months_with_31 = [1, 3, 5, 7, 8, 10, 12]
    time_dim = ds.dims["time"]

    # Step 1: Build synthetic time axis assuming 3-hourly steps starting at 01:30
    df = ds[["rsds", "rsdsdiff", "tas"]].to_dataframe().reset_index()

    # Replace CFTimeIndex with synthetic datetime based on time index order
    time_axis = pd.date_range(start=datetime(year, 1, 1, 1, 30), periods=time_dim, freq="3H")
    df["time"] = df["time"].map(dict(zip(df["time"].unique(), time_axis)))


    # Step 3: Duplicate day 30 if month has 31 days
    extended_rows = []
    for month in range(1, 13):
        month_data = df[df["time"].dt.month == month]
        for day in sorted(month_data["time"].dt.day.unique()):
            day_data = month_data[month_data["time"].dt.day == day]
            extended_rows.append(day_data)
            if day == 30 and month in months_with_31:
                day_31 = day_data.copy()
                day_31["time"] = day_31["time"] + pd.Timedelta(days=1)
                extended_rows.append(day_31)

    full_df = pd.concat(extended_rows).reset_index(drop=True)

    # Step 4: Remove invalid February days
    feb_mask = full_df["time"].dt.month == 2
    if is_leap_year(year):
        full_df = full_df[~((feb_mask) & (full_df["time"].dt.day == 30))]
    else:
        full_df = full_df[~((feb_mask) & (full_df["time"].dt.day.isin([29, 30])))]

    # Step 5: Drop duplicates to ensure unique (time, lat, lon) combinations
    full_df = full_df.drop_duplicates(subset=["time", "lat", "lon"])

    # Step 6: Convert back to xarray Dataset
    full_df = full_df.set_index(["time", "lat", "lon"])
    ds_out = full_df.to_xarray()

    # Step 7: Annotate time axis
    ds_out["time"].attrs["calendar"] = "proleptic_gregorian"
    ds_out["time"].attrs["units"] = f"hours since {year}-01-01 00:00:00"

    return ds_out

def is_leap_year(year):
    """
    Check if a year is a leap year in the Gregorian calendar.
    
    Parameters:
        year (int): The year to check.
    
    Returns:
        bool: True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

import xarray as xr
import numpy as np
import pandas as pd

def add_february_29(ds, year):
    """
    Add February 29 to a dataset with a noleap calendar by duplicating all Feb 28 timestamps.
    Handles 3-hourly (or any sub-daily) datasets correctly.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    if is_leap_year(year):
        # Check if the time coordinate is in cftime format
        if isinstance(ds['time'].values[0], cftime.DatetimeNoLeap):
            # Convert cftime to pandas datetime64 using to_datetimeindex()
            ds['time'] = ds.indexes['time'].to_datetimeindex()
        elif not np.issubdtype(ds['time'].dtype, np.datetime64):
            try:
                # Convert time to datetime64[ns] if not already in that format
                ds['time'] = pd.to_datetime(ds['time'].values)
            except Exception as e:
                raise ValueError(f"Failed to convert time to datetime64[ns]: {e}")

        # Select Feb 28 times
        feb_28 = ds.sel(time=(ds['time'].dt.month == 2) & (ds['time'].dt.day == 28))

        if feb_28.time.size > 0:
            # Duplicate Feb 28 and shift by +1 day
            feb_29 = feb_28.copy()
            feb_29 = feb_29.assign_coords(time=feb_28['time'] + pd.Timedelta(days=1))

            # Concatenate the new Feb 29 data with the original dataset
            ds = xr.concat([ds, feb_29], dim="time")
            ds = ds.sortby('time')
        else:
            print(f"Warning: February 28 not found for year {year}")
    else:
        # Handle non-leap years gracefully
        print(f"Year {year} is not a leap year. No February 29 added.")

    return ds
   

def power_calculation(files, orientation1, trigon_model, clearsky_model, tracking, panel, output_dir):
    for model, periods in files.items():
        print(f"Processing model: {model}")
        for period, file_list in periods.items():
            print(f"  Processing Period: {period}")
            for file_path in file_list:
                try:
                    print(f"    Processing file: {file_path}")
                    
                    # Prepare output directory and filenames
                    output_dir_period = os.path.join(output_dir, model, period)
                    os.makedirs(output_dir_period, exist_ok=True)  # Ensure the output directory exists
                    
                    file_name = os.path.basename(file_path)
                    # Replace "rsds_rsdsdiff_tas" with "solar_power"
                    file_name = file_name.replace("rsds_rsdsdiff_tas", "solar_power")
                    output_file = os.path.join(output_dir_period, file_name)

                    file_name_1h="1h"+file_name
                    output_file_1h = os.path.join(output_dir_period, file_name_1h)
                    
                    # Check if the solar power file already exists
                    if os.path.exists(output_file):
                        print(f"    Skipping {file_path} as {output_file} already exists.")
                        continue
                    
                    # Prepare aggregated generation file name
                    file_name_agg = "aggregated_"+file_name 
                    output_file_agg = os.path.join(output_dir_period, file_name_agg)
                    
                    # Check if the aggregated generation file already exists
                    if os.path.exists(output_file_agg):
                        print(f"    Skipping {file_path} as {output_file_agg} already exists.")
                        continue

                    # Extract the year as a string for later
                    year = int(re.search(r"(\d{4})", file_name).group(1))
                    
                    # Open the file
                    #for H models and gregorian: 
                    #ds = xr.open_dataset(file_path, engine="netcdf4", decode_times=False)
                    ds = xr.open_dataset(file_path, engine="netcdf4", decode_times=True)
                    print('File opened')

                    # Transform the time coordinate for HadGEM models
                    if model in ["HadGEM3-GC31-LL", "HadGEM3-GC31-MM"]:
                        ds = transform_360_to_gregorian(ds, year)
                        if isinstance(ds['time'].values[0], cftime.datetime):
                            print(f"Calendar type after transform_to_gregorian for {model}: {ds['time'].values[0].calendar}")
                        else:
                            print(f"Time coordinate is not in cftime format after transform_to_gregorian for {model}.")
                    elif model in ["CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2"]:
                        ds = add_february_29(ds, year)
                        if isinstance(ds['time'].values[0], cftime.datetime):
                            print(f"Calendar type after add_february_29 for {model}: {ds['time'].values[0].calendar}")
                        else:
                            print(f"Time coordinate is not in cftime format after add_february_29 for {model}.")
                    else:
                        # Decode time normally for other models
                        ds = xr.decode_cf(ds)
                        if isinstance(ds['time'].values[0], cftime.datetime):
                            print(f"Calendar type after decode_cf for {model}: {ds['time'].values[0].calendar}")
                        else:
                            print(f"Time coordinate is not in cftime format after decode_cf for {model}.")
                    print('Calendar transformation done')

                    # Select the region again after resampling
                    ds = ds.sel(lon=slice(-12, 35), lat=slice(33, 64.8))
                    print('Coordinates selection')

                    # Calculate the solar position
                    solar_position_model = pv_functions.SolarPosition(ds, time_shift="0h")
                    print('Solar position calculated')

                    # Calculate panel orientation
                    orientation = pv_functions.get_orientation(orientation1)
                    surface_orientation_model = pv_functions.SurfaceOrientation(ds, solar_position_model, orientation, tracking)
                    print('Surface orientation calculated')

                    # Open mean albedo for each model
                    ds_albedo = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/albedo/mean_albedo_grid_{model}.nc")
                    albedo = ds_albedo['__xarray_dataarray_variable__'].sel(lon=slice(-12, 35), lat=slice(33, 64.8))
                    # Opening each bias factor 
                    #for var in variables1:
                    ds_bias_factor_direct = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/direct_bias_factor_{model}.nc")
                    ds_bias_factor_diffuse = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/diffuse_bias_factor_{model}.nc")
                    ds_bias_factor_temp = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/temp_bias_factor_{model}.nc")  
                    ds_bias_factor_total = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/total_bias_factor_{model}.nc")
                    bf_direct= ds_bias_factor_direct['bias_factor']
                    bf_diffuse= ds_bias_factor_diffuse['bias_factor']
                    bf_temp= ds_bias_factor_temp['bias_factor']
                    bf_total= ds_bias_factor_total['bias_factor']
                    print('opened bias factors')  

                    # Calculate tilted irradiation. bias factor is included in the function
                    irradiation_model = pv_functions.TiltedIrradiation(
                        ds,
                        albedo,
                        solar_position_model,
                        surface_orientation_model,
                        trigon_model,
                        clearsky_model,
                        bf_direct,
                        bf_diffuse,
                        bf_total,
                        tracking=0,
                        altitude_threshold=1.0,
                        irradiation="total", 
                    )
                    print('Tilted irradiation calculated')

                    # Calculate power
                    solar_panel = pv_functions.SolarPanelModel(ds, irradiation_model, panel, bf_temp)
                    print('Solar power calculated with hourly means')

                    total_solar_power=3 * solar_panel #I multiply times 3 to get the 


                    aggregated_generation = total_solar_power.sum(dim="time")
                    print('Aggregated generation calculated')

                    # For total_solar_power
                    if "time" in total_solar_power.coords and "units" in total_solar_power.coords["time"].attrs:
                        del total_solar_power.coords["time"].attrs["units"]

                    # For aggregated_generation (only if it has 'time')
                    if "time" in aggregated_generation.coords and "units" in aggregated_generation.coords["time"].attrs:
                        del aggregated_generation.coords["time"].attrs["units"]
                    for key in ["units", "calendar"]:
                        if "time" in total_solar_power.coords and key in total_solar_power.coords["time"].attrs:
                            del total_solar_power.coords["time"].attrs[key]

                    # Save the solar panel data to a NetCDF file
                    try:
                        total_solar_power.to_netcdf(output_file)
                        print(f"Saved total solar power data to {output_file}")
                    except Exception as e:
                        print(f"Failed to save solar power data: {e}")

                    # Save the aggregated generation data to a NetCDF file
                    try:
                        aggregated_generation.to_netcdf(output_file_agg)
                        print(f"Saved aggregated solar power data to {output_file_agg}")
                    except Exception as e:
                        print(f"Failed to save aggregated solar power data: {e}")

                except Exception as e:
                    print(f"    Failed to process file {file_path}: {e}")

     

def main():
    base_path="/groups/FutureWind/SFCRAD/" 
    # ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", ,"HadGEM3-GC31-LL", "HadGEM3-GC31-MM" "MRI-ESM2-0 , 
    # "r1i1p1f1" "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", ,"r1i1p1f3", "r1i1p1f3", "r1i1p1f1"
    #models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    #variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    #period = ["historical","ssp585"]
    models = ["HadGEM3-GC31-LL", "HadGEM3-GC31-MM"]  # Test with only one model
    variants = ["r1i1p1f3", "r1i1p1f3"]  # Corresponding variant for the model
    period = ["historical","ssp585"]


    orientation1='latitude_optimal'
    trigon_model='simple'
    clearsky_model='simple'
    tracking=None
    panel = {
        "model": "huld",  # Model type
        "name": "CSi",  # Panel name
        "source": "Huld 2010",  # Source of the model

        # Used for calculating capacity per m2
        "efficiency": 0.2,  # Efficiency of the panel

        # Panel temperature coefficients
        "c_temp_amb": 1,  # Panel temperature coefficient of ambient temperature
        "c_temp_irrad": 0.035,  # Panel temperature coefficient of irradiance (K / (W/m2))

        # Reference conditions
        "r_tamb": 293,  # Reference ambient temperature (20 degC in Kelvin)
        "r_tmod": 298,  # Reference module temperature (25 degC in Kelvin)
        "r_irradiance": 1000,  # Reference irradiance (W/m^2)

        # Fitting parameters
        "k_1": -0.017162,
        "k_2": -0.040289,
        "k_3": -0.004681,
        "k_4": 0.000148,
        "k_5": 0.000169,
        "k_6": 0.000005,

        # Inverter efficiency
        "inverter_efficiency": 0.9
}
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power/"

    files=collect_files(base_path, models, variants, period)
    # Filter files for the year 1988
     
    """ for model in files:
        for period_key in files[model]:
            files[model][period_key] = [
                file for file in files[model][period_key] if "1988" in file
            ] 
 """
    power_calculation(files, orientation1, trigon_model, clearsky_model, tracking, panel, output_dir)

if __name__ == "__main__":
    main()