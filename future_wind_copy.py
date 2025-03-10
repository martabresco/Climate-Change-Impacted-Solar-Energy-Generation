#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Tue May 17 09:16:05 CEST 2022

@author: Andrea Hahmann, DTU Wind
"""

from calendar import calendar
import cftime
import xarray as xr 
import numpy as np
from datetime import datetime,timedelta
# import wget
import os
import time

def find_file(field,model,root_dir,
              root_url,experiment,variant,table,grid,version,
              start_date,end_date):

    start_date_string = start_date.strftime("%Y%m%d%H%M")
    end_date_string = end_date.strftime("%Y%m%d%H%M")
    date_string = "-".join((start_date_string,end_date_string))

    File = "_".join((field,table,model,experiment,variant,grid,date_string))+".nc"
    print("Find",File)

    if not os.path.exists(File):
        print("File",File,"not found")
        url = "/".join((root_url,experiment,variant,table,field,grid,version,File))
        print("URL",url)
        
        got_it = False; tries = 1

        while not got_it and tries < 10:
            try:
                filename = wget.download(url)
                got_it = True
            except:
                print("An error occured, retry after 60s")
                time.sleep(60)
                tries = tries + 1
    else:
        print("File already exists")

    File = "/".join((root_dir,File))
    return File

def datetime_to_cftime(date,calendar='proleptic_gregorian'):
    '''Returns a cftime from a datetime

    Args:
        date (string): date in the form yyyymmddhh

    Returns:
        cftime: date in cftime 
    '''

    y,m,d,h = (
        date.dt.year, 
        date.dt.month, 
        date.dt.day, 
        date.dt.hour)
    
    return cftime.datetime(y,m,d,h,
        calendar=calendar)

def date_to_cftime(date,calendar='noleap'):
    '''Returns a cftime from a yyyymmddhh string

    Args:
        date (string): date in the form yyyymmddhh

    Returns:
        cftime: date in cftime 
    '''

    # print(date,len(date))
    # print("Using calendar:",calendar)
    if len(date) >= 10:
        y,m,d,h = (
            int(date[0:4]),int(date[4:6]),\
            int(date[6:8]),int(date[8:10])
        )
        return cftime.datetime(y,m,d,h,calendar=calendar)
    else:
        y,m,d = (
            int(date[0:4]),int(date[4:6]),\
            int(date[6:8])
        )
        return cftime.datetime(y,m,d,0,calendar=calendar)

    
def search_esgf(conn,
    field,model,experiment,variant,
    date=None,table='6hrLev',
    replica=True,latest=True,
    data_node=None,
    verbose=False):

    '''Search dataset in the ESGF database for the field and 
    desired year'''

    ctx = conn.new_context(
        project='CMIP6',
        source_id=model,
        experiment_id=experiment,
        variable=field,
        table_id=table,
        variant_label=variant,
        data_node = data_node,
        latest=latest,
        replica=replica)  # sometimes, "latests=True" works, other times "replica=False"

    ff = []  # empty list

    if verbose:
        print('Hits: {}, Realms: {}, Ensembles: {}'.format(
        ctx.hit_count,
        ctx.facet_counts['realm'],
        ctx.facet_counts['ensemble']))
            
    if (ctx.hit_count > 0):
            
        result = ctx.search()[0]
        
        files = result.file_context().search()
        print(files)

        if (date is None):
            ff.append(files[0].opendap_url)
            print(ff)
        else:
            calendar = date.calendar
            print("Calendar is",calendar)

            for File in files:
                print(File.opendap_url)
                filename = File.opendap_url.split("/")[-1]
                if (table in ['6hrLev','6hrPlevPt','3hr','E3hr']):

                    filedate = filename.split("_")[-1].split("-")[0][0:10]
                    start_time = date_to_cftime(filedate,calendar=calendar)
                    filedate = filename.split("_")[-1].split("-")[1][0:10]
                    end_time = date_to_cftime(filedate,calendar=calendar)

                elif (table in ['day','Eday']):

                    filedate = filename.split("_")[-1].split("-")[0][0:8]
                    start_time = date_to_cftime(filedate,calendar=calendar)
                    filedate = filename.split("_")[-1].split("-")[1][0:8]
                    end_time = date_to_cftime(filedate,calendar=calendar)

                elif (table in ['LImon','Emon','Lmon']):

                    filedate = filename.split("_")[-1].split("-")[0][0:6] + "1612"
                    start_time = date_to_cftime(filedate,calendar=calendar)
                    filedate = filename.split("_")[-1].split("-")[1][0:6] + "1612"
                    end_time = date_to_cftime(filedate,calendar=calendar)
                    print(start_time,end_time)

                if (start_time <= date) and (date <= end_time):
                    print("Date found:",date, 'in',start_time,end_time)  
                    ff.append(File.opendap_url)

                elif (table in ['LImon','Emon','Lmon'] and date >= end_time):
                    print("Date found:",date, 'in',start_time,end_time)  
                    ff.append(File.opendap_url)

                elif (table == 'day' and date >= end_time):
                    print("Date found:",date, 'in',start_time,end_time)  
                    ff.append(File.opendap_url)
                
        return ff ##File.opendap_url
    else:
        print("No matching files found")

    return

def combine_hemispheres(var,time=None,lev=None,for_interp=False, 
    reverse_lev=False, 
    minlat=20.,maxlat=75.,minlon=330.,maxlon=50.):  
    '''Combine array (var) from both hemispheres with continuous
    coordinates.
    In some datasets the vertical coordinate is reversed.'''

    min_lat = minlat ; max_lat = maxlat
    min_lon = minlon ; max_lon = maxlon

    # print(lev)

    if (for_interp):
        min_lat = minlat - 5. ; max_lat = maxlat + 5.
        min_lon = minlon - 5. ; max_lon = maxlon + 5.

    if time is None and lev is None:
        west = var.sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,360.))
        west['lon'] = west['lon'] - 360.
        east = var.sel(lat=slice(min_lat,max_lat),lon=slice(0.,max_lon))

    elif lev is None:
        west = var.sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,360.),
                       time=time)
        west['lon'] = west['lon'] - 360.
        east = var.sel(lat=slice(min_lat,max_lat),lon=slice(0.,max_lon),
                       time=time)
        
    elif isinstance(lev,int) and lev==0:
        west = var.sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,360.),
                       time=time).isel(lev=0)
        west['lon'] = west['lon'] - 360.
        east = var.sel(lat=slice(min_lat,max_lat),lon=slice(0.,max_lon),
                       time=time).isel(lev=0)

    else:   ##isinstance(lev,slice) or isinstance(lev,np.ndarray):
        west = var.sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,360.),
                       lev=lev,time=time)
        west['lon'] = west['lon'] - 360.
        east = var.sel(lat=slice(min_lat,max_lat),lon=slice(0.,max_lon),
                       lev=lev,time=time)
        
    west_east = xr.concat([west,east],'lon')

    if (reverse_lev):
        west_east = west_east.reindex(lev=west_east.lev[::-1])

    return west_east

def virtual_temperature(t,q):
    '''
    Calculates virtual temperature (K) from temperature (K) and specific humidity (kg/kg)
    '''
    eps = 0.62198
    return t * ((q + eps) / (eps * (1. + q)))    

def air_density(ps,t,q):
    '''
    Calculates air density (kg/m3) from 
    pressure (hPa), temperature (K) and specific humidity (kg/kg)
    '''
    Rd = 287.058
    tv = virtual_temperature(t,q)
    return ps / (Rd * tv)

def make_data_set(du,ws,wd,rho):
    """Creates xarray DataArray for netCDF write

    Args:
        du (dataset): sample dataset with attributes
        ws (DataArray): wind speed 
        wd (DataArray): wind direction
        rho (DataArray): surface air density

    Returns:
        xarray DataArray: DataArray for write
    """
    lat = xr.DataArray(
        data=ws.lat.values.astype('float32'),
        dims=["lat"],
        coords=dict(
            lat=(["lat"], ws.lat.values)
        ),
        attrs=dict(
        long_name="latitude",
        units="degrees_north",
        axis="Y"
        ),
    )
    lon = xr.DataArray(
        data=ws.lon.values.astype('float32'),
        dims="lon",
        coords=dict(
            lon=(["lon"], ws.lon.values)
        ),
        attrs=dict(
        long_name="longitude",
        units="degrees_east",
        axis="X"
        ),
    )
    level = xr.DataArray(
        data=ws.level.values.astype('float32'),
        dims="level",
        coords=dict(
            level=(["level"], ws.level.values)
        ),
        attrs=dict(
        long_name="level",
        units="m",
        axis="Z"
        ),
    )
    ds = xr.Dataset(
        data_vars=dict(
            wind_speed = (
                ["time","level","lat","lon"],ws.values.astype('float32'),
                dict(long_name = "wind speed",
                units = "m s-1",
                vert_units = "m")),
            wind_direction = (
                ["time","level","lat","lon"],wd.values.astype('float32'),
                dict(long_name = "wind direction",
                units = "degrees",
                vert_units = "m")),
            air_density = (
                ["time","lat","lon"],rho.values.astype('float32'),
                dict(long_name = "surface air density",
                units = "kg m-3",
                height = "surface")),
            ),
        coords=dict(
            lon=lon,
            lat=lat,
            level=level,
            time=ws.time
            ),
        attrs=dict(
            data_source = "Processed data from CMIP6 runs",
            experiment = du.experiment_id,
            source = du.source_id,
            variant_label = du.variant_label,
            data_written = datetime.now().strftime("%d/%m/%Y %H:%M")
            )
    )   
    return ds