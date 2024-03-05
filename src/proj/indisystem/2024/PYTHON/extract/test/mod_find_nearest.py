# mod_find_nearest.py
import numpy as np
import netCDF4 as nc

def find_nearest_grid_point(file_name, target_lat, target_lon):
    """
    Finds the nearest grid point in a WRF output file to a specified lat/lon.

    Parameters:
    - file_name: Path to the WRF output file.
    - target_lat: Target latitude.
    - target_lon: Target longitude.

    Returns:
    - min_dist_idx: Tuple of indices (i, j) for the nearest grid point.
    """
    ds = nc.Dataset(file_name)
    lat2d = ds.variables['XLAT'][0]  # Assuming time dimension is the first one
    lon2d = ds.variables['XLONG'][0]  # Assuming time dimension is the first one

    # Calculate squared distances
    dist_squared = (lat2d - target_lat)**2 + (lon2d - target_lon)**2

    # Find the index of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(dist_squared), dist_squared.shape)
    
    # Close the dataset
    ds.close()
    
    return min_dist_idx

