import numpy as np
import xesmf as xe

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