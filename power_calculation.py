from __future__ import annotations
import xarray as xr
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
import pandas as pd
import xarray as xr
from regridding_functions import read_and_average_era5_marta
from regridding_functions import regrid
import pv_functions
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
    


def power_calculation(files, orientation, trigon_model, clearsky_model, tracking, panel, output_dir):
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
                    
                    # Check if the solar power file already exists
                    if os.path.exists(output_file):
                        print(f"    Skipping {file_path} as {output_file} already exists.")
                        continue
                    
                    # Prepare aggregated generation file name
                    file_name_agg = file_name + "_aggregated"
                    output_file_agg = os.path.join(output_dir_period, file_name_agg)
                    
                    # Check if the aggregated generation file already exists
                    if os.path.exists(output_file_agg):
                        print(f"    Skipping {file_path} as {output_file_agg} already exists.")
                        continue
                    
                    # Open the file
                    ds = xr.open_dataset(file_path, engine="netcdf4", decode_times=True)
                    
                    # Bias correct rsds, rsdsdiff, tas
                    variables1 = ["direct", "diffuse", "temp"]
                    for var in variables1:
                        bias_factor = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/{var}_bias_factor_{model}.nc")
                        if var == "direct":
                            ds["rsds"] = ds["rsds"] * bias_factor
                        elif var == "diffuse":
                            ds["rsdsdiff"] = ds["rsdsdiff"] * bias_factor
                        elif var == "temp":
                            ds["tas"] = ds["tas"] * bias_factor
                    
                    # Resample to 1h
                    ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
                    ds = ds.assign_coords(time=ds.time - np.timedelta64(90, "m"))
                    ds = ds.resample(time="1H").ffill()
                    new_time_index = pd.date_range(start=ds.time.min().values, end="2014-12-31T23:00", freq="1H")
                    ds_h = ds.reindex(time=new_time_index, method="ffill")
                    
                    # Calculate the solar position
                    solar_position_model = pv_functions.SolarPosition(ds_h, time_shift="+30min")
                    
                    # Calculate panel orientation
                    orientation = pv_functions.get_orientation(orientation)
                    surface_orientation_model = pv_functions.SurfaceOrientation(ds_h, solar_position_model, orientation, tracking)
                    
                    # Open mean albedo for each model
                    albedo = xr.open_dataset(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/albedo/mean_albedo_grid_{model}.nc")
                    
                    # Calculate tilted irradiation
                    irradiation_model = pv_functions.TiltedIrradiation(
                        ds_h,
                        solar_position_model,
                        surface_orientation_model,
                        trigon_model,
                        clearsky_model,
                        tracking=0,
                        altitude_threshold=1.0,
                        irradiation="total",
                    )
                    
                    # Calculate power
                    solar_panel = pv_functions.SolarPanelModel(ds_h, irradiation_model, panel)
                    aggregated_generation = solar_panel.sum(dim="time")
                    
                    # Save the solar panel data to a NetCDF file
                    try:
                        solar_panel.to_netcdf(output_file)
                        print(f"Saved solar power data to {output_file}")
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
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    period = ["historical","ssp585"]
    orientation='latitude_optimal'
    trigon_model='simple'
    clearsky_model='simple'
    tracking=None
    panel = {
        "model": "huld",  # Model type
        "name": "CSi",  # Panel name
        "source": "Huld 2010",  # Source of the model

        # Used for calculating capacity per m2
        "efficiency": 0.1,  # Efficiency of the panel

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
    power_calculation(files, orientation, trigon_model, clearsky_model, tracking, panel, output_dir)

    print(files)

if __name__ == "__main__":
    main()