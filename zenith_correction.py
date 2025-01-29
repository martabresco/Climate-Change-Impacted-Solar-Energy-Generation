# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:13:54 2025

@author: marta
"""


import atlite
import logging
import xarray as xr
import numpy as np

# Open the file
data = xr.open_dataset('C:/Users/marta/Desktop/Thesis/Climate-Change-Impacted-Solar-Energy-Generation/rsds_3hr_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001010130-201412312230.nc')

# Explore the file structure
print(data.variables)
sliced_data = data['rsds'].isel(time=0, lat=slice(0, 10), lon=slice(0, 10))
print(sliced_data.values)

# Get the latitude in radians

lat_radians = np.deg2rad(data['lat'])
# Get the solar declination angle from the calendar day
#calendar_day = 

#solar_Declination = 23.45*np.sin(2*np.pi/365*(248*calendar_day))*np.pi/180