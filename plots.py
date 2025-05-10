from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

def map_plots(variable, cmap='viridis', setnan=True, vmin=None, vmax=None, title='', label=''):
    """
    Function to plot a map with the given variable and customization options.

    Parameters:
    - variable: xarray.DataArray, the variable to plot (must include lon and lat coordinates).
    - cmap: str, colormap for the plot (default: 'viridis').
    - vmin: float, minimum value for the color scale (default: None).
    - vmax: float, maximum value for the color scale (default: None).
    - title: str, title of the plot.
    - label: str, label for the colorbar.

    Returns:
    - None (displays the plot).
    """
    # Replace zeros with NaNs in the variable data and slice to the desired range
    if setnan==True:
        variable = xr.where(variable != 0, variable, float('nan')).sel(x=slice(-12, 35), y=slice(33, 64))
    else:
        variable = variable.sel(x=slice(-12, 35), y=slice(33, 64))

    # Extract longitude and latitude
    lon = variable.x
    lat = variable.y

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-12, 35, 33, 64], crs=ccrs.PlateCarree())  # Restrict the extent to the specified range

    # Plot the data
    c = ax.pcolormesh(
        lon, lat, variable,  # Ensure correct shapes
        transform=ccrs.PlateCarree(),
        cmap=cmap,  # Colormap for the plot
        shading='auto',
        vmin=vmin,
        vmax=vmax,  # Smooth shading
    )

    # Add map features
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.set_title(title, fontsize=16)  # Set the plot title

    # Add gridlines aligned with x and y coordinates
    gridlines = ax.gridlines(
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        linestyle='--', 
        x_inline=False, 
        y_inline=False
    )

    # Reduce the number of gridlines for better readability
    gridlines.xlocator = plt.FixedLocator(lon.values)
    gridlines.ylocator = plt.FixedLocator(lat.values)  # Use latitude values for y gridlines

    # Set custom formatters for longitude and latitude
    gridlines.xformatter = mticker.FuncFormatter(lambda x, _: f"{x:.2f}") 
    gridlines.yformatter = mticker.FuncFormatter(lambda y, _: f"{y:.2f}") 

    # Configure gridline labels
    gridlines.top_labels = False  # Disable labels on the top
    gridlines.right_labels = False  # Disable labels on the right
    gridlines.xlabel_style = {'fontsize': 12, 'rotation': 45, 'ha': 'right'}  # Rotate and align x-axis labels
    gridlines.ylabel_style = {'fontsize': 12}  # Increased font size for y-axis labels

    # Add colorbar
    cbar = fig.colorbar(c, ax=ax, orientation='vertical')
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=14)  # Set colorbar label
    cbar.ax.tick_params(labelsize=12)  # Increase font size for colorbar ticks

    # Show the plot
    plt.show()


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point, box

def country_plots(
    variable: xr.DataArray,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    title: str = '',
    label: str = '',
    missing_frac_thresh: float = 0.7  # fraction of missing cells to flag a country as unavailable
):
    """
    Take a DataArray on dims (y,x), compute per-country means, and plot.
    
    - variable: xarray.DataArray with coords .x/.y in degrees
    - cmap: matplotlib colormap name
    - vmin/vmax: if None→use full data min/max; else→clip colorbar
    - title: plot title
    - label: colorbar label
    - missing_frac_thresh: float in [0,1]; if a country has this fraction or more of its
      cells missing, it will be treated as unavailable and colored gray.
    """
    # 1) Subset region
    da = variable.sel(x=slice(-12, 35), y=slice(33, 64))
    
    # 2) Flatten to a Pandas table of (lon, lat, value), KEEPING NaNs
    xs, ys = da.x.values, da.y.values
    lon2d, lat2d = np.meshgrid(xs, ys)
    df = pd.DataFrame({
        'lon': lon2d.ravel(),
        'lat': lat2d.ravel(),
        'value': da.values.ravel()
    })
    
    # 3) Make a GeoDataFrame of points (including those with NaN value)
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df.lon, df.lat)],
        crs="EPSG:4326"
    )
    
    # 4) Load high-res countries and clip to the box
    ne_50m = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(ne_50m).to_crs("EPSG:4326")
    bbox = box(-12, 33, 35, 64)
    world = world.clip(bbox)
    
    # 5) Spatial join
    joined = gpd.sjoin(
        gdf_pts, 
        world[['NAME_LONG','geometry']],
        how='inner',
        predicate='within'
    )
    
    # 6) Compute availability stats per country
    stats = joined.groupby('NAME_LONG').agg(
        total_cells = ('value','size'),
        avail_cells = ('value', lambda x: x.notna().sum())
    )
    stats['missing_frac'] = 1 - stats['avail_cells'] / stats['total_cells']
    
    # 7) Identify countries with too many missing cells
    bad_countries = stats.index[stats['missing_frac'] >= missing_frac_thresh]
    
    # 8) Compute mean_diff only for sufficiently available countries
    valid = joined[~joined['NAME_LONG'].isin(bad_countries)]
    country_means = (
        valid
        .groupby('NAME_LONG')['value']
        .mean()
        .reset_index(name='mean_diff')
    )
    
    # 9) Merge means into world; others remain NaN
    world = world.merge(country_means, on='NAME_LONG', how='left')
    
    # 10) Decide on vmin/vmax
    if vmin is None and vmax is None:
        vmin_plot = world['mean_diff'].min()
        vmax_plot = world['mean_diff'].max()
    else:
        vmin_plot, vmax_plot = vmin, vmax
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
    
    # 11) Plot
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    world.plot(
        column='mean_diff',
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        linewidth=0.5,
        edgecolor='black',
        ax=ax,
        missing_kwds={
            'color':'lightgrey',
            #'edgecolor':'red',
            #'hatch':'///',
            'label':'no data'
        }
    )
    
    # map styling
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines('10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-12, 35, 33, 64], ccrs.PlateCarree())
    ax.set_title(title, fontsize=16)
    
    # 1° gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        xlocs=np.arange(-12, 36, 1),
        ylocs=np.arange(33, 65, 1)
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize':12,'rotation':45,'ha':'right'}
    gl.ylabel_style = {'fontsize':12}
    
    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.show()

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box

def country_plots_weighted(
    variable: xr.DataArray,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    title: str = '',
    label: str = '',
    missing_frac_thresh: float = 0.7,
    proj_crs: str = "EPSG:3035"
):
    """
    Plot per‐country means of `variable`, weighting each grid cell by the
    fraction of its area within the country.  
    Countries with ≥ missing_frac_thresh fraction of their total cell‐area
    missing — or which lie outside Europe — will be gray.
    """
    # 1) Subset region & extract coords + values
    da = variable.sel(x=slice(-12, 35), y=slice(33, 64))
    xs, ys = da.x.values, da.y.values
    vals = da.values  # shape (ny, nx)

    # 2) Build GeoDataFrame of grid‐cell polygons with a unique cell_id
    dx = np.diff(xs).mean()
    dy = np.diff(ys).mean()
    records = []
    cell_id = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            records.append({
                'cell_id': cell_id,
                'value': vals[j, i],
                'geometry': box(x - dx/2, y - dy/2, x + dx/2, y + dy/2)
            })
            cell_id += 1
    gdf_cells = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # 3) Load & filter to Europe only
    ne50 = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(ne50).to_crs("EPSG:4326")
    # keep only European continent
    world = world[world['CONTINENT'] == 'Europe']
    # clip to our bounding box
    region_box = box(-12, 33, 35, 64)
    world = world.clip(region_box)

    # 4) Compute intersections: each cell ∩ country
    inter = gpd.overlay(gdf_cells, world[['NAME_LONG','geometry']],
                        how='intersection')

    # 5) Project intersections & cells to an equal‐area CRS
    inter_proj = inter.to_crs(proj_crs)
    cells_proj = gdf_cells.to_crs(proj_crs)[['cell_id','geometry']]
    cells_proj['cell_area'] = cells_proj.geometry.area

    # 6) Compute overlap area & weight for each piece
    inter_proj['overlap_area'] = inter_proj.geometry.area
    inter_proj = inter_proj.merge(cells_proj[['cell_id','cell_area']], on='cell_id')
    inter_proj['weight'] = inter_proj['overlap_area'] / inter_proj['cell_area']

    # 7) Compute per-country missing‐area fraction
    grp          = inter_proj.groupby('NAME_LONG')
    total_weight = grp['weight'].sum()
    avail_weight = grp.apply(lambda g: (g['weight'] * g['value'].notna()).sum())
    avail_frac   = avail_weight / total_weight

    # 8) Identify “good” countries
   # keep only those where at least `missing_frac_thresh` of the AREA is available
    good  = avail_frac[avail_frac >= missing_frac_thresh].index
    valid = inter_proj[inter_proj['NAME_LONG'].isin(good)]


    # 9) Compute weighted mean per country
    weighted = valid.groupby('NAME_LONG').apply(
        lambda g: (g['weight'] * g['value']).sum() / g['weight'].sum()
    ).rename('mean_diff').reset_index()

    # 10) Merge into our Europe‐only world for plotting
    world = world.merge(weighted, on='NAME_LONG', how='left')

    # 11) Color‐limits
    if vmin is None and vmax is None:
        vmin_plot, vmax_plot = world['mean_diff'].min(), world['mean_diff'].max()
    else:
        vmin_plot, vmax_plot = vmin, vmax
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax_plot)

    # 12) Plot
    fig, ax = plt.subplots(figsize=(12, 8),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    world.plot(
        column='mean_diff',
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        linewidth=0.5,
        edgecolor='black',
        ax=ax,
        missing_kwds={'color':'lightgrey'}
    )
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines('10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-12, 35, 33, 64], ccrs.PlateCarree())
    ax.set_title(title, fontsize=16)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--',
                      xlocs=np.arange(-12, 36, 1), ylocs=np.arange(33, 65, 1))
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'fontsize':12,'rotation':45,'ha':'right'}
    gl.ylabel_style = {'fontsize':12}

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.show()


