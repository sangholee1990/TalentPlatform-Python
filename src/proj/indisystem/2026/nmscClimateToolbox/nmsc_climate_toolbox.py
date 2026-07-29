import os
import glob
import xarray as xr
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
try:
    import xcdat as xc
except ImportError:
    pass
HAS_XCDAT = True

class NMSCClimateToolbox:
    @staticmethod
    def search(pattern):
        """Search for files matching the pattern."""
        return glob.glob(pattern)

    @staticmethod
    def _make_xcdat_compliant(ds):
        """Ensure dataset is CF compliant and ready for xCDAT spatial/temporal operations."""
        # Add required CF attributes for axis detection
        if 'lat' in ds.coords: ds['lat'].attrs.update({'standard_name': 'latitude', 'axis': 'Y', 'units': 'degrees_north'})
        elif 'latitude' in ds.coords: ds['latitude'].attrs.update({'standard_name': 'latitude', 'axis': 'Y', 'units': 'degrees_north'})
        
        if 'lon' in ds.coords: ds['lon'].attrs.update({'standard_name': 'longitude', 'axis': 'X', 'units': 'degrees_east'})
        elif 'longitude' in ds.coords: ds['longitude'].attrs.update({'standard_name': 'longitude', 'axis': 'X', 'units': 'degrees_east'})
        
        if 'time' in ds.coords: ds['time'].attrs.update({'standard_name': 'time', 'axis': 'T'})
        
        # Let xcdat add bounds for conservative regridding and averaging
        try:
            import xcdat as xc
            # For 1D coordinates, xcdat can generate bounds automatically
            ds = ds.bounds.add_missing_bounds(axes=['X', 'Y'])
            if 'time' in ds.coords:
                ds = ds.bounds.add_missing_bounds(axes=['T'])
        except Exception as e:
            print(f"Warning: xCDAT bounds generation failed: {e}")
            
        return ds

    @staticmethod
    def open(filepaths):
        ds = NMSCClimateToolbox._open_raw(filepaths)
        return NMSCClimateToolbox._make_xcdat_compliant(ds)

    @staticmethod
    def _open_raw(filepaths):
        """Open one or multiple NetCDF/GeoTIFF files with graceful fallback."""
        if isinstance(filepaths, str):
            if filepaths.lower().endswith(('.tif', '.tiff')):
                import rioxarray
                try:
                    da = rioxarray.open_rasterio(filepaths, chunks='auto')
                except Exception:
                    # Fallback to no chunks
                    da = rioxarray.open_rasterio(filepaths)
                
                if da.name is None: da.name = 'band_data'
                ds = da.to_dataset()
                if 'x' in ds.coords and 'y' in ds.coords:
                    ds = ds.rename({'x': 'lon', 'y': 'lat'})
                return ds
            
            try:
                return xr.open_dataset(filepaths, chunks='auto')
            except Exception:
                return xr.open_dataset(filepaths)
        else:
            try:
                return xr.open_mfdataset(filepaths, combine='by_coords', parallel=False, chunks='auto')
            except Exception:
                return xr.open_mfdataset(filepaths, combine='by_coords', parallel=False)

    @staticmethod
    def filter(dataset, time_slice=None, lon_slice=None, lat_slice=None):
        """Filter dataset by time and spatial domain."""
        selection = {}
        if time_slice:
            selection['time'] = time_slice
        if lon_slice:
            lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
            selection[lon_name] = lon_slice
        if lat_slice:
            lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
            selection[lat_name] = lat_slice
            
        if selection:
            return dataset.sel(**selection)
        return dataset

    @staticmethod
    def clean(dataset):
        """Handle missing values."""
        return dataset.dropna(dim='time', how='all')

    @staticmethod
    def weg(dataset):
        """Generate latitude/longitude bounds automatically."""
        if HAS_XCDAT:
            try:
                return dataset.bounds.add_missing_bounds()
            except AttributeError:
                pass
        return dataset

    @staticmethod
    def spaMean(dataset, variable):
        """Calculate spatial mean using xCDAT with a fallback."""
        if HAS_XCDAT:
            try:
                return dataset.spatial.average(variable, axis=['X', 'Y'])[variable]
            except Exception:
                pass
        
        # Fallback to simple mean
        lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
        lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
        return dataset[variable].mean(dim=[lon_name, lat_name], skipna=True)

    @staticmethod
    def timeMean(dataset, variable):
        """Calculate temporal mean using xCDAT with a fallback."""
        if HAS_XCDAT:
            return dataset.temporal.average(variable)[variable]
        return dataset[variable].mean(dim='time', skipna=True)

    @staticmethod
    def cli(dataset, variable=None, target_month=None):
        """Calculate standard climatology using xCDAT with a fallback."""
        if target_month is not None:
            dataset = dataset.sel(time=dataset['time'].dt.month.isin([target_month]))
            
        if HAS_XCDAT and variable:
            try:
                return dataset.temporal.climatology(variable, freq="month")
            except:
                pass
        if 'time' in dataset.dims:
            # 일별 데이터 등을 먼저 각 년/월별 대표 평균값으로 변환
            ds_monthly = dataset.resample(time='1MS').mean('time')
            return ds_monthly.groupby('time.month').mean(dim='time')
        return dataset

    @staticmethod
    def ano(dataset, variable=None, target_month=None):
        """Calculate anomalies from climatology using xCDAT with a fallback."""
        if target_month is not None:
            dataset = dataset.sel(time=dataset['time'].dt.month.isin([target_month]))
            
        if HAS_XCDAT and variable:
            try:
                return dataset.temporal.departures(variable, freq="month")
            except:
                pass
        if 'time' in dataset.dims:
            # 일별 데이터 등을 먼저 각 년/월별 대표 평균값으로 변환
            ds_monthly = dataset.resample(time='1MS').mean('time')
            clim = ds_monthly.groupby('time.month').mean(dim='time')
            return ds_monthly.groupby('time.month') - clim
        return dataset

    @staticmethod
    def trend(timeseries, time_dim='time'):
        """Calculate linear trend."""
        # timeseries is expected to be a 1D xarray DataArray
        y = timeseries.values
        # Handle datetime conversion for regression
        x = np.arange(len(y))
        
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            return None, None, None
            
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[valid], y[valid])
        trend_line = intercept + slope * x
        
        # Return as DataArray
        trend_da = xr.DataArray(trend_line, coords={time_dim: timeseries[time_dim]}, dims=[time_dim])
        return trend_da, slope, p_value

    @staticmethod
    def generate_static_map(dataset, variable, time_idx=0, data_layer='original', bounds=None, cmap_name='jet'):
        import io
        import base64
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from scipy.ndimage import median_filter
        import matplotlib.patheffects as pe

        lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
        lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
        
        if 'time' in dataset.dims and dataset.sizes['time'] > 1:
            data = dataset[variable].isel(time=time_idx)
        elif 'time' in dataset.dims:
            data = dataset[variable].isel(time=0)
        else:
            data = dataset[variable]

        data_2d = data.values
        lons = dataset[lon_name].values
        lats = dataset[lat_name].values

        if bounds:
            vmin, vmax = bounds
        else:
            vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))

        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        pcm = ax.pcolormesh(lons, lats, data_2d, cmap=cmap_name, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')

        size_val = 15
        smoothed_data = median_filter(data_2d, size=size_val)
        smoothed_ma = np.ma.masked_where(np.isnan(data_2d), smoothed_data)

        DLEV = (vmax - vmin) / 9.0 if bounds else 4.0
        LEVELS = np.arange(vmin, vmax + 1e-6, DLEV)
        
        try:
            c = ax.contour(lons, lats, smoothed_ma, levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8, transform=ccrs.PlateCarree())
            labels = ax.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black')
            for label in labels:
                label.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])
        except Exception:
            pass

        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, extend='both', spacing='proportional')
        cbar.set_label('Value', fontsize=10, fontweight='bold')
        title_str = f'{data_layer.capitalize()} Map ({variable})'
        if 'time' in dataset.dims:
            try:
                title_str += f" - {str(dataset['time'].values[time_idx])[:10]}"
            except:
                pass
        ax.set_title(title_str, fontsize=12, fontweight='bold')

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, transparent=False)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @staticmethod
    def resMap(dataset, variable, time_idx=0, cmap='RdYlBu_r'):
        """Plot 2D resource map."""
        lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
        lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
        
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Get data for a specific time index
        if 'time' in dataset.dims and dataset.sizes['time'] > 1:
            data = dataset[variable].isel(time=time_idx)
            title_time = str(dataset['time'].values[time_idx])[:10]
        elif 'time' in dataset.dims:
            data = dataset[variable].isel(time=0)
            title_time = str(dataset['time'].values[0])[:10]
        else:
            data = dataset[variable]
            title_time = "Average"
            
        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, cbar_kwargs={'shrink': 0.8})
        ax.set_title(f"{variable} Distribution ({title_time})")
        # ax.set_global()  # Commented out to allow dynamic extent
        return fig

    @staticmethod
    def get_map_overlay_data(dataset, variable, time_idx=0, cmap='RdYlBu_r', vmin=None, vmax=None):
        """Generate base64 image and extent for OpenLayers overlay."""
        import io
        import base64
        import matplotlib.pyplot as plt
        
        lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
        lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
        
        if 'time' in dataset.dims and dataset.sizes['time'] > 1:
            data = dataset[variable].isel(time=time_idx)
            title_time = str(dataset['time'].values[time_idx])[:10]
        elif 'time' in dataset.dims:
            data = dataset[variable].isel(time=0)
            title_time = str(dataset['time'].values[0])[:10]
        else:
            data = dataset[variable]
            title_time = "Average"
            
        lon = dataset[lon_name].values
        lat = dataset[lat_name].values
        
        min_lon, max_lon = float(lon.min()), float(lon.max())
        min_lat, max_lat = float(lat.min()), float(lat.max())
        extent = [min_lon, min_lat, max_lon, max_lat]
        
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        
        if lat[0] > lat[-1]:
            origin = 'upper'
        else:
            origin = 'lower'
            
        # Encode raw data to base64 Float32Array for tooltip hover
        data_filled = data.fillna(np.nan).values.astype(np.float32)
        data_bytes = data_filled.tobytes()
        data_b64 = base64.b64encode(data_bytes).decode('utf-8')
        height, width = data_filled.shape
            
        im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto', interpolation='nearest')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'image_base64': f"data:image/png;base64,{img_base64}",
            'data_base64': data_b64,
            'extent': extent,
            'width': width,
            'height': height,
            'origin': origin,
            'min_val': float(vmin) if vmin is not None else float(data.min(skipna=True)),
            'max_val': float(vmax) if vmax is not None else float(data.max(skipna=True)),
            'title_time': title_time,
            'variable': variable
        }

    @staticmethod
    def timeGrp(timeseries, trend_line=None, variable_name="Variable"):
        """Plot time series with optional trend line."""
        fig, ax = plt.subplots(figsize=(10, 5))
        timeseries.plot.line(ax=ax, label='Original Data', color='b', marker='o', markersize=3, linewidth=1)
        
        if trend_line is not None:
            trend_line.plot.line(ax=ax, label='Trend', color='r', linestyle='--')
            
        ax.set_title(f"Time Series Analysis: {variable_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel(variable_name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig

    @staticmethod
    def scaGrp(dataset_prod, dataset_valid, var_prod, var_valid):
        """Plot scatter plot for comparative analysis between product and validation data."""
        # Align datasets spatially and temporally using inner join
        ds_prod, ds_valid = xr.align(dataset_prod, dataset_valid, join='inner')
        
        # Extract and flatten the values
        val1 = ds_prod[var_prod].values.flatten()
        val2 = ds_valid[var_valid].values.flatten()
        
        # Remove NaNs
        mask = ~np.isnan(val1) & ~np.isnan(val2)
        v1, v2 = val1[mask], val2[mask]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(v1) == 0:
            ax.text(0.5, 0.5, 'No overlapping data', ha='center', va='center')
            return fig
            
        # Hexbin scatter plot
        hb = ax.hexbin(v1, v2, gridsize=50, cmap='Blues', mincnt=1)
        
        # 1:1 Reference Line
        min_val = min(v1.min(), v2.min())
        max_val = max(v1.max(), v2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
        
        # Calculate metrics
        r = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 0.0
        bias = np.mean(v1 - v2)
        
        ax.set_title(f"검증자료 비교 산점도 (Scatter Plot)/nR: {r:.3f}, Bias: {bias:.3f}")
        ax.set_xlabel(f"Product: {var_prod}")
        ax.set_ylabel(f"Validation: {var_valid}")
        fig.colorbar(hb, ax=ax, label='Density Count')
        ax.legend()
        
        return fig

nct = NMSCClimateToolbox()

if __name__ == '__main__':
    nct = NMSCClimateToolbox()

    data = nct.open('C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/indisystem/2026/nmscClimateToolbox/doc/L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled22.nc')
    rdata = nct.open('C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/indisystem/2026/nmscClimateToolbox/doc/Validation_BUOY_SST_2D.nc')

    stnCodeList = rdata['station_code'].values
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    # 결과를 저장할 빈 리스트 생성
    matched_results = []

    for stn in stnCodeList:
        # ---------------------------------------------------------
        # 1. 부이(Buoy) 데이터 추출 및 전처리 (시간 일치 준비)
        # ---------------------------------------------------------
        buoy_da = rdata['sst'].sel(station_code=stn)
        buoy_df = buoy_da.to_dataframe().reset_index()

        # 관측소의 위경도 추출 (NetCDF 구조에 따라 lat/lon 변수명이 다를 수 있음)
        # rdata에 위경도가 변수 또는 좌표로 존재한다고 가정합니다.
        stn_lat = rdata['lat'].sel(station_code=stn).values.item()
        stn_lon = rdata['lon'].sel(station_code=stn).values.item()

        # 시간 컬럼을 datetime 형식으로 확실히 변환
        buoy_df['time'] = pd.to_datetime(buoy_df['time'])

        # 부이 데이터가 월간(Monthly) 단위가 아니라면 위성 데이터와 맞추기 위해 월평균 리샘플링
        # 위성 자료가 월초(MS, Month Start)를 기준으로 저장되었다고 가정
        buoy_monthly = buoy_df.set_index('time').resample('MS')['sst'].mean().reset_index()
        buoy_monthly = buoy_monthly.rename(columns={'sst': 'SST_Buoy'})

        # ---------------------------------------------------------
        # 2. 위성(Satellite/Grid) 데이터 공간 추출 (공간 일치)
        # ---------------------------------------------------------
        # method='nearest'를 사용하여 부이 위치와 가장 가까운 격자 데이터 추출
        sat_da = data['SST_L3_monthly'].sel(lat=stn_lat, lon=stn_lon, method='nearest')
        sat_df = sat_da.to_dataframe().reset_index()

        sat_df['time'] = pd.to_datetime(sat_df['time'])
        sat_df = sat_df[['time', 'SST_L3_monthly']].rename(columns={'SST_L3_monthly': 'SST_Sat'})

        # ---------------------------------------------------------
        # 3. 시간축 기준으로 두 데이터 병합 (시공간 매칭 완료)
        # ---------------------------------------------------------
        # 두 데이터셋에 모두 존재하는 시간(inner join)만 남김
        merged_df = pd.merge(sat_df, buoy_monthly, on='time', how='inner')

        # 메타데이터 추가 (어떤 관측소인지, 매칭된 위경도 정보 등)
        merged_df['station_code'] = stn
        merged_df['station_lat'] = stn_lat
        merged_df['station_lon'] = stn_lon

        # 추출된 격자의 실제 위경도 (부이 위치와 얼마나 차이나는지 확인용)
        merged_df['grid_lat'] = sat_da.lat.values.item()
        merged_df['grid_lon'] = sat_da.lon.values.item()

        matched_results.append(merged_df)

        # target_station = stn
        # plot_df = merged_df
        #
        # # 그래프 크기 설정
        # plt.figure(figsize=(14, 6))
        #
        # # 위성 데이터 (파란색 선)
        # plt.plot(plot_df['time'], plot_df['SST_Sat'],
        #          label='Satellite SST', color='blue', marker='o', markersize=4, linestyle='-')
        #
        # # 부이 데이터 (빨간색 선) - NaN 값은 자동으로 끊겨서 또는 점으로 그려짐
        # plt.plot(plot_df['time'], plot_df['SST_Buoy'],
        #          label='Buoy SST', color='red', marker='x', markersize=6, linestyle='-')
        #
        # # 그래프 꾸미기
        # plt.title(f'SST Time Series Comparison (Station: {target_station})', fontsize=14)
        # plt.xlabel('Time', fontsize=12)
        # plt.ylabel('Sea Surface Temperature (°C)', fontsize=12)
        # plt.legend(fontsize=12)
        # plt.grid(True, linestyle='--', alpha=0.7)
        #
        # # x축 날짜 포맷 겹치지 않게 회전
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        #
        # # 그래프 출력
        # plt.show()

    # 리스트에 담긴 개별 관측소 매칭 결과를 하나의 데이터프레임으로 합침
    final_matched_df = pd.concat(matched_results, ignore_index=True)

    # 결과 확인 (오차 계산 등 추가 분석에 활용)
    print(final_matched_df.head())