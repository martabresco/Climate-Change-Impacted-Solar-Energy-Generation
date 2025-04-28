from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

def map_plots(variable, cmap='viridis', vmin=None, vmax=None, title='', label=''):
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
    # Replace zeros with NaNs in the variable data
    variable = xr.where(variable != 0, variable, float('nan'))

    # Extract longitude and latitude
    lon = variable.lon
    lat = variable.lat

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

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