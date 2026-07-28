import pandas as pd
import xarray as xr
import glob
import os
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    meta_file = 'doc/META_SST_관측지점정보.csv'
    
    # 1. Read metadata
    df_meta = pd.read_csv(meta_file, encoding='cp949')
    
    # 2. Read observation data
    obs_files = glob.glob('doc/OBS_BUOY_TIM_*.csv')
    if not obs_files:
        print("No observation files found.")
        return
        
    print(f"Found {len(obs_files)} observation files. Reading data...")
    df_obs_list = []
    for f in sorted(obs_files):
        try:
            df_temp = pd.read_csv(f, encoding='cp949')
            df_obs_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    df_obs = pd.concat(df_obs_list, ignore_index=True)
    
    # 3. Preprocess
    print("Preprocessing data...")
    df_obs['time'] = pd.to_datetime(df_obs['일시'], errors='coerce')
    df_obs = df_obs.dropna(subset=['time', '지점'])
    
    # Ensure '지점' is integer
    df_obs['지점'] = df_obs['지점'].astype(int)
    
    # 4. Pivot to 2D format (time x station)
    print("Pivoting data to 2D (time x station)...")
    pivot_sst = df_obs.pivot_table(index='time', columns='지점', values='수온(°C)', aggfunc='mean')
    
    # 5. Build xarray Dataset
    times = pivot_sst.index.values
    stations = pivot_sst.columns.values
    sst_values = pivot_sst.values
    
    df_meta['지점'] = df_meta['지점'].astype(int)
    df_meta = df_meta.drop_duplicates(subset=['지점'])
    df_meta_filtered = df_meta.set_index('지점').reindex(stations)
    
    lats = df_meta_filtered['위도'].values
    lons = df_meta_filtered['경도'].values
    names = df_meta_filtered['지점명'].astype(str).tolist()
    
    ds = xr.Dataset(
        {
            "sst": (["time", "station_code"], sst_values),
        },
        coords={
            "time": times,
            "station_code": stations,
            "lat": (["station_code"], lats),
            "lon": (["station_code"], lons),
            "station_name": (["station_code"], names)
        }
    )
    
    # Add attributes
    ds.sst.attrs = {
        'long_name': 'Sea Surface Temperature',
        'units': 'degC'
    }
    ds.lat.attrs = {'units': 'degrees_north', 'long_name': 'latitude'}
    ds.lon.attrs = {'units': 'degrees_east', 'long_name': 'longitude'}
    ds.time.attrs = {'long_name': 'time'}
    ds.station_code.attrs = {'long_name': 'Station Code'}
    ds.attrs = {
        'description': 'Validation BUOY SST Data (Time x Station Code)',
        'source': 'META_SST_관측지점정보.csv, OBS_BUOY_TIM_*.csv'
    }
    
    # 6. Save to NetCDF
    out_nc = 'doc/Validation_BUOY_SST_2D.nc'
    ds.to_netcdf(out_nc)
    print(f"Successfully saved NetCDF to {out_nc}")

if __name__ == "__main__":
    main()
