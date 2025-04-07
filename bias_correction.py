import xarray as xr
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)
#import xesmf as xe
from regridding_functions import read_and_average_era5_3h
from regridding_functions import read_and_average_era5_4y
from regridding_functions import read_and_average_sarah
from regridding_functions import regrid
from regridding_functions import read_and_average_era5_marta
from regridding_functions import read_and_average_cmip
import os

def fill_nans_with_last_valid(bias_factor):
    #for Sarah, fills the Nan values of the bias factor along the latitude axis with the last valid value
    for lon in range(bias_factor.shape[1]):  # Iterate over longitudes (second axis)
        valid_values = ~np.isnan(bias_factor[:, lon])  # Find valid (non-NaN) indices along latitude
        if np.any(valid_values):  # If there's at least one valid value
            last_valid_idx = np.max(np.where(valid_values))  # Get the last valid latitude index
            bias_factor[last_valid_idx+1:, lon] = bias_factor[last_valid_idx, lon]  # Fill NaNs downward
    
    return bias_factor

def bias_factor_era5_sarah(var):
    #calculates the bias factor between era5 and sarah for the variable var, for now only 4 years data
    rsds_era5_mean_4y= read_and_average_era5_4y(var) #read and av the 4 years of era5 for bias correction with sarah
    rsds_sarah_mean= read_and_average_sarah(var) #same for sarah
    rsds_era5_mean_cut=rsds_era5_mean_4y.sel(x=slice(-12, 35), y=slice(33, 72)) #cut to the max latitude covered by sarah
    rsds_sarah_mean_cut=rsds_sarah_mean.sel(x=slice(-12, 35), y=slice(33, 72))
    regridder=regrid(rsds_era5_mean_cut, rsds_sarah_mean_cut, method='conservative')  #regrid era5 (0.25x0.25) to the sarah grid (0.3x0.3)
    rsds_era5_mean_interp_cut_4y=regridder(rsds_era5_mean_cut)
    denominator_era5_sarah= rsds_era5_mean_interp_cut_4y.values  # ERA5 dataset
    numerator_era5_sarah= rsds_sarah_mean_cut.values  # SARAH dataset
    # Ensure valid bias factor calculation
    mask_valid = (denominator_era5_sarah != 0) & (numerator_era5_sarah != 0) # Avoid division by zero and all values in sarah that have mean 0
    bias_factor_era5_sarah = np.where(mask_valid, numerator_era5_sarah / denominator_era5_sarah, np.nan)  # Replace invalid cases with NaN
    bias_factor_era5_sarah = fill_nans_with_last_valid(bias_factor_era5_sarah)  # Fill NaNs downward
    return bias_factor_era5_sarah

def bias_factor_era5_model(var, var2, model, period, variant, bias_factor_era5_sarah, output_dir):
    # Define the output file path
    filename = f"bias_factor_era5_{model}_{var}.nc"
    filepath = os.path.join(output_dir, filename)

    # Check if the file already exists
    if os.path.exists(filepath):
        logging.info(f"File already exists: {filepath}. Skipping computation.")
        return

    # Compute bias factor if the file does not exist
    #use the 3h function
    rsds_era5_mean_BOC = read_and_average_era5_3h(var)  # mean of era5 historical period for each grid cell
    rsds_model_mean_BOC = read_and_average_cmip(f'SFCRAD/{model}/{period}/{variant}/', var2)  # mean of model of historical period for each grid cell
    rsds_era5_mean_BOC = rsds_era5_mean_BOC.sel(x=slice(-12, 35), y=slice(33, 72))
    rsds_model_mean_BOC = rsds_model_mean_BOC.sel(lon=slice(-12, 35), lat=slice(33, 72))
    ds_03 = xr.open_dataset('europe_03.nc')  # grid 0.3x0.3
    regridder_era5 = regrid(rsds_era5_mean_BOC, ds_03, method='conservative')  # regrid era5 to the 0.3x0.3º grid
    rsds_era5_03 = regridder_era5(rsds_era5_mean_BOC)  # regridded historical mean from era5 to 0.3x0.3º grid
    rsds_era5_correct = rsds_era5_03.sel(lon=slice(-12, 35), lat=slice(33, 72)) * bias_factor_era5_sarah  # apply bias factor to era5 rsds
    regridder_era503_model = regrid(rsds_era5_correct, rsds_model_mean_BOC, method='conservative')  # regrid corrected era5 to the model grid
    rsds_era5_correct_model = regridder_era503_model(rsds_era5_correct)  # regrid corrected era5 to the model grid
    rsds_era5_correct_model = rsds_era5_correct_model.sel(lon=slice(-12, 35), lat=slice(33, 72))
    numerator_era5_model = rsds_era5_correct_model.values
    denominator_era5_model = rsds_model_mean_BOC.values

    # Ensure valid bias factor calculation
    mask_valid_2 = (denominator_era5_model != 0) & (numerator_era5_model != 0)  # Avoid values 0
    bias_factor_era5_model = np.where(mask_valid_2, numerator_era5_model / denominator_era5_model, np.nan)  # Replace invalid cases with NaN

    # Print shapes for debugging
    print(f"Shape of bias_factor_era5_model: {bias_factor_era5_model.shape}")
    print(f"Shape of rsds_model_mean_BOC.lat: {len(rsds_model_mean_BOC.lat)}")
    print(f"Shape of rsds_model_mean_BOC.lon: {len(rsds_model_mean_BOC.lon)}")

    # Ensure the shape of bias_factor_era5_model matches lat and lon dimensions
    if bias_factor_era5_model.shape != (len(rsds_model_mean_BOC.lat), len(rsds_model_mean_BOC.lon)):
        raise ValueError("Shape of bias_factor_era5_model does not match lat/lon dimensions of rsds_model_mean_BOC")

    # Convert to xarray dataset
    ds = xr.Dataset(
        {"bias_factor": (["lat", "lon"], bias_factor_era5_model)},
        coords={
            "lat": rsds_model_mean_BOC.lat,
            "lon": rsds_model_mean_BOC.lon,
        },
    )

    # Save to .nc file
    ds.to_netcdf(filepath)
    logging.info(f"Saved bias factor to {filepath}")


def main():
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    variables = ["influx_direct", "influx_diffuse", "temperature"]
    variables_cmip = ["rsds", "rsdsdiff", "tas"]
    period = "historical"
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/"
    os.makedirs(output_dir, exist_ok=True)
    
    for var, var2 in zip(variables, variables_cmip):
        if var == "temperature" and var2 == "tas":
            bias_factor_era5_sarah_result = 1  # Set bias factor to 1 for temperature
            logging.info(f"Set bias_factor_era5_sarah_result to 1 for variable {var} and CMIP variable {var2}")
        else:
            bias_factor_era5_sarah_result = bias_factor_era5_sarah(var)
        
        for model, variant in zip(models, variants):
            bias_factor_era5_model(var, var2, model, period, variant, bias_factor_era5_sarah_result, output_dir)
            logging.info(f"Computed and saved bias factor for model {model}, variable {var}")


if __name__ == "__main__":
    main()