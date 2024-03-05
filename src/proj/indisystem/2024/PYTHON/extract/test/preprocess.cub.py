import netCDF4 as nc
import glob
import numpy as np
import sys
from mod_find_nearest import find_nearest_grid_point

# Case configuration
fcode = '2024-02-05_12_vg_cub'

# HMS01
ref_file   = 'def/wrfout_d01_2024-02-05_12:00:00'
target_lat = 35.465325
target_lon = 126.12869444444443

hms_idx = find_nearest_grid_point(ref_file, target_lat, target_lon)
j_hms, i_hms = hms_idx # j, i index of WRF file at target point

# Use glob to find all files matching the pattern 'wrfwind*.nc' and sort them
file_list = sorted(glob.glob('cub/wrfwind*'))

# Initialize empty lists to store the data from all files
U_all, V_all, PH_all, PHB_all = [], [], [], []

k_max = 9

# Loop through each file in the list
for file_name in file_list:
    print('=== Reading file',file_name,'...')

    # Open the NetCDF file
    ds = nc.Dataset(file_name)
    
    # Extract the lower 5 levels for each variable and append to the respective lists
    # Assuming the dimensions are in the order (time, level, y, x)
    U_all.append(ds.variables['U'][:, :k_max, j_hms, i_hms])
    V_all.append(ds.variables['V'][:, :k_max, j_hms, i_hms])
    PH_all.append(ds.variables['PH'][:, :k_max+1, j_hms, i_hms])
    PHB_all.append(ds.variables['PHB'][:, :k_max+1, j_hms, i_hms])
    
    # Close the dataset after processing
    ds.close()

# Concatenate the lists along the time dimension to create single arrays for each variable
U = np.concatenate(U_all, axis=0)
V = np.concatenate(V_all, axis=0)
PH = np.concatenate(PH_all, axis=0)
PHB = np.concatenate(PHB_all, axis=0)
H_s = ( PH + PHB ) / 9.80665
H = 0.5 * ( H_s[:,:-1] + H_s[:,1:]) 

nt, nz = U.shape
WSP86 = np.zeros(nt)
WSP99 = np.zeros(nt)
WDR86 = np.zeros(nt)
WDR99 = np.zeros(nt)

for i in range(nt):
    U_int = U[i, :]
    V_int = V[i, :]
    H_int = H[i, :]

    # Wind speed using power-law
    WSP0 = np.sqrt( U_int[0]**2 + V_int[0]**2 )
    WSP = np.log( np.sqrt( U_int**2 + V_int**2 ) ) - np.log( WSP0 )
    LHS = np.log( H_int / H_int[0] )
    alp, res, rank, s = np.linalg.lstsq(LHS[:, np.newaxis], WSP, rcond=None )
    WSP86[i] = WSP0 * ( 86.0 / H_int[0] )**alp
    WSP99[i] = WSP0 * ( 99.0 / H_int[0] )**alp
 
    # Wind direction using linear interpolation
    k86 = np.argmax( H_int > 86.0 )
    aa = ( H_int[k86+1] - 86.0 ) 
    bb = ( 86.0 - H_int[k86] )
    U86 = ( U_int[k86] * aa + U_int[k86+1] * bb ) / ( aa + bb )
    V86 = ( V_int[k86] * aa + V_int[k86+1] * bb ) / ( aa + bb )

    k99 = np.argmax( H_int > 99.0 )
    aa = ( H_int[k99+1] - 99.0 ) 
    bb = ( 99.0 - H_int[k99] )
    U99 = ( U_int[k99] * aa + U_int[k99+1] * bb ) / ( aa + bb )
    V99 = ( V_int[k99] * aa + V_int[k99+1] * bb ) / ( aa + bb )

    WDR86[i] = ( np.arctan2( -U86, -V86 )*180.0 / np.pi )%360.0
    WDR99[i] = ( np.arctan2( -U99, -V99 )*180.0 / np.pi )%360.0

np.save('tmp_np_arr/wsp86_'+fcode+'.npy', WSP86)
np.save('tmp_np_arr/wsp99_'+fcode+'.npy', WSP99)
np.save('tmp_np_arr/wdr86_'+fcode+'.npy', WDR86)
np.save('tmp_np_arr/wdr99_'+fcode+'.npy', WDR99)

