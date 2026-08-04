import warnings
warnings.filterwarnings('ignore')
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
import os
import glob
import xarray as xr
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    def process_climate_data(dataset, freq='1MS', op='평균', start_yr=None, end_yr=None):
        """Process climate data: resample, compute climatology, and compute anomalies.

        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset with a 'time' dimension.
        freq : str
            Resampling frequency, e.g. '1MS' (monthly) or '1YS' (yearly).
        op : str
            Aggregation operation: '평균' (mean), '합계' (sum), '최대' (max), '최소' (min).
        start_yr : str or None
            Start year for climatology baseline (e.g. '1991').
        end_yr : str or None
            End year for climatology baseline (e.g. '2020').

        Returns
        -------
        tuple of (ds_resampled, ds_cli, ds_ano)
        """
        # --- 1. Resample along time axis ---
        if 'time' in dataset.dims:
            resampler = dataset.resample(time=freq)
            if op == '합계':
                ds_resampled = resampler.sum('time')
            elif op == '최대':
                ds_resampled = resampler.max('time')
            elif op == '최소':
                ds_resampled = resampler.min('time')
            else:  # 평균
                ds_resampled = resampler.mean('time')
        else:
            ds_resampled = dataset

        # --- 2. Subset for climatology baseline period ---
        if start_yr is not None and end_yr is not None and 'time' in ds_resampled.dims:
            import pandas as pd
            # Use full date string if provided, otherwise default to start of year/end of year
            if len(str(start_yr)) == 4:
                t0 = pd.Timestamp(f'{start_yr}-01-01')
            else:
                t0 = pd.Timestamp(str(start_yr))
                
            if len(str(end_yr)) == 4:
                t1 = pd.Timestamp(f'{end_yr}-12-31')
            else:
                t1 = pd.Timestamp(str(end_yr))
                
            ds_base = ds_resampled.sel(time=slice(t0, t1))
        else:
            ds_base = ds_resampled

        # --- 3. Climatology (long-term monthly/yearly mean) ---
        if 'time' in ds_base.dims:
            if freq == '1YS':
                ds_cli = ds_base.mean(dim='time')
            else:
                ds_cli = ds_base.groupby('time.month').mean(dim='time')
        else:
            ds_cli = ds_base

        # --- 4. Anomaly (departure from climatology) ---
        if 'time' in ds_resampled.dims:
            if freq == '1YS':
                ds_ano = ds_resampled - ds_cli
            else:
                ds_ano = ds_resampled.groupby('time.month') - ds_cli
        else:
            ds_ano = ds_resampled

        return ds_resampled, ds_cli, ds_ano

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
    def resMap(dataset, variable, time_idx=0, data_layer='original', bounds=None, cmap_name='jet', dataset_cli=None, custom_title=None, custom_legend=None, return_type='b64'):
        import io
        import base64
        import numpy as np
        import matplotlib
        if return_type == 'b64':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False
        from mpl_toolkits.basemap import Basemap
        from scipy.ndimage import median_filter
        import matplotlib.patheffects as pe

        lon_name = 'lon' if 'lon' in dataset.dims else 'longitude'
        lat_name = 'lat' if 'lat' in dataset.dims else 'latitude'
        
        if 'time' in dataset.dims and dataset.sizes['time'] > 0:
            if data_layer == 'climatology':
                target_month = time_idx + 1
                if dataset_cli is not None and 'month' in dataset_cli.dims:
                    data = dataset_cli.sel(month=target_month)[variable]
                else:
                    ds_month = dataset.isel(time=(dataset.time.dt.month == target_month))
                    data = ds_month[variable].mean(dim='time')
            elif data_layer == 'anomaly':
                # Anomaly uses the current time_idx to find the month, computes climatology, and subtracts
                t_slice = dataset.isel(time=time_idx)
                target_month = int(dataset.time[time_idx].dt.month)
                if dataset_cli is not None and 'month' in dataset_cli.dims:
                    clim = dataset_cli.sel(month=target_month)[variable]
                else:
                    clim = dataset.isel(time=(dataset.time.dt.month == target_month))[variable].mean(dim='time')
                data = t_slice[variable] - clim
            else:
                data = dataset[variable].isel(time=time_idx)
        else:
            data = dataset[variable]

        data_2d = data.values
        lons = dataset[lon_name].values
        lats = dataset[lat_name].values
        
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            data_2d = data_2d[::-1, :]

        fig = plt.figure(figsize=(2400 / 300, 2000 / 300))
        ax = fig.add_subplot(111)

        m = Basemap(
            projection='lcc', resolution='i',
            lat_0=38, lon_0=126,
            llcrnrlat=11.308528, urcrnrlat=53.303712,
            llcrnrlon=101.395259, urcrnrlon=175.188166,
            lat_1=30, lat_2=60, ax=ax
        )
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        x, y = m(lon_grid, lat_grid)
        
        if data_layer == 'anomaly':
            import matplotlib.colors as mcolors
            SST_ANOM_COLORS = [
                '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                '#A00000', '#800000', '#600000'
            ]
            VMIN = bounds[0] if bounds else -6.0
            VMAX = bounds[1] if bounds else 6.0
            DLEV = (VMAX - VMIN) / 24.0 if bounds else 0.5
            LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)
            
            if cmap_name == 'SST_ANOM (custom)':
                cmap = mcolors.ListedColormap(SST_ANOM_COLORS)
                norm = mcolors.BoundaryNorm(LEVELS, cmap.N)
                pcm = m.pcolormesh(x, y, data_2d, cmap=cmap, norm=norm, shading='auto')
            else:
                pcm = m.pcolormesh(x, y, data_2d, cmap=cmap_name, shading='auto', vmin=VMIN, vmax=VMAX)
                
            line_levels = np.arange(VMIN, VMAX + 1e-6, 2.0)
            line_levels = line_levels[line_levels != 0.0]
            line_levels = line_levels.astype(int)
        else:
            VMIN = bounds[0] if bounds else 0.0
            VMAX = bounds[1] if bounds else 36.0
            DLEV = (VMAX - VMIN) / 9.0 if bounds else 4.0
            LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)
            
            if cmap_name == 'SST_ANOM (custom)':
                import matplotlib.colors as mcolors
                SST_ANOM_COLORS = [
                    '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                    '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                    '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                    '#A00000', '#800000', '#600000'
                ]
                cmap = mcolors.ListedColormap(SST_ANOM_COLORS)
                pcm = m.pcolormesh(x, y, data_2d, cmap=cmap, shading='auto', vmin=VMIN, vmax=VMAX)
            else:
                pcm = m.pcolormesh(x, y, data_2d, cmap=cmap_name, shading='auto', vmin=VMIN, vmax=VMAX)
            line_levels = LEVELS

        m.drawcoastlines(color='k', linewidth=0.5)
        m.drawcountries(color='gray', linewidth=0.5)
        m.fillcontinents(color='lightgray', lake_color='white')
        
        m.drawparallels(np.arange(-10, 61, 10), labels=[1,0,0,0], fontsize=10, fontname='DejaVu Sans', fmt='%d', fontweight='bold')
        m.drawmeridians(np.arange(50, 181, 10), labels=[0,0,0,1], fontsize=10, fontname='DejaVu Sans', fmt='%d', fontweight='bold')

        size_val = 15
        smoothed_data = median_filter(data_2d, size=size_val)
        smoothed_ma = np.ma.masked_where(np.isnan(data_2d), smoothed_data)

        cs = m.contour(x, y, smoothed_ma, levels=line_levels, colors='k', linewidths=1.0, alpha=0.8)
        if data_layer == 'anomaly':
            cs0 = m.contour(x, y, smoothed_ma, levels=[0.0], colors='k', linewidths=0.5, alpha=0.5)

        candidates = []
        paths = []
        if hasattr(cs, 'collections'):
            for collection in cs.collections:
                paths.extend(collection.get_paths())
        else:
            paths = cs.get_paths()
        for path in paths:
            for poly in path.to_polygons():
                line_length = len(poly)
                if line_length > 30:
                    mid_idx = line_length // 2
                    candidates.append({
                        'length': line_length,
                        'coord': (poly[mid_idx][0], poly[mid_idx][1])
                    })
        candidates.sort(key=lambda x: x['length'], reverse=True)
        MAX_LABELS = 15
        manual_locs = [cand['coord'] for cand in candidates[:MAX_LABELS]]

        if manual_locs:
            try:
                labels = plt.clabel(cs, inline=True, fontsize=8, fmt='%d', colors='black', manual=manual_locs)
                for label in labels:
                    label.set_rotation(0)
                    label.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])
            except Exception:
                pass

        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, extend='both', spacing='proportional')
        if data_layer == 'anomaly':
            cbar.set_ticks(LEVELS)
        
        cbar.set_label(custom_legend if custom_legend else 'Value', fontsize=10, fontname='Malgun Gothic', fontweight='bold')
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontname('Malgun Gothic')
            lbl.set_fontsize(10)
            lbl.set_fontweight('bold')

        title_str = f'{data_layer.capitalize()} Map ({variable})'
        if 'time' in dataset.dims:
            if data_layer == 'climatology':
                title_str += f" - Month {time_idx + 1}"
            else:
                try:
                    title_str += f" - {str(dataset['time'].values[time_idx])[:7]}"
                except:
                    pass
        
        final_title = custom_title if custom_title else title_str

        # Replace strftime placeholders in final_title if any exist
        if final_title and ('%' in final_title):
            if 'time' in dataset.dims:
                if data_layer == 'climatology':
                    month_str = f"{time_idx + 1:02d}"
                    final_title = final_title.replace('%m', month_str).replace('%Y.%m', month_str)
                else:
                    try:
                        import pandas as pd
                        time_val = dataset['time'].values[time_idx]
                        ts = pd.to_datetime(str(time_val))
                        final_title = ts.strftime(final_title)
                    except Exception as e:
                        print("Error formatting title:", e)

        plt.title(final_title, fontsize=12, fontweight='bold', fontname='Malgun Gothic')
        plt.tight_layout()
        
        if return_type == 'fig':
            return fig
            
        buf = io.BytesIO()
        plt.savefig(buf, dpi=300, format='png', transparent=True, pad_inches=0.1, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_b64

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

    @staticmethod
    def histGrp(dataarray, bins=30, variable_name="Variable"):
        """Plot histogram of data."""
        fig, ax = plt.subplots(figsize=(8, 6))
        # Flatten and drop nan
        data = dataarray.values.flatten()
        data = data[~np.isnan(data)]
        
        ax.hist(data, bins=bins, density=True, alpha=0.6, color='b', edgecolor='black')
        
        # Overlay KDE or normal distribution if seaborn is available
        try:
            import seaborn as sns
            sns.kdeplot(data, ax=ax, color='r', linewidth=2)
        except ImportError:
            pass
            
        ax.set_title(f"Histogram and PDF: {variable_name}")
        ax.set_xlabel(variable_name)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig

    @staticmethod
    def barGrp(x_data, y_data, title="Bar Graph", xlabel="X", ylabel="Y"):
        """Plot bar graph for discrete comparisons."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x_data, y_data, color='skyblue', edgecolor='black')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels if they are dates or strings
        fig.autofmt_xdate(rotation=45)
        return fig

    @staticmethod
    def ext(timeseries, percentile=0.95):
        """
        Calculate extreme climate indices including Sen's Slope and Percentile thresholds.
        
        Parameters
        ----------
        timeseries : xr.DataArray
            1D time series data
        percentile : float
            Percentile threshold for extreme events (default 0.95 for 95th percentile)
            
        Returns
        -------
        dict
            Dictionary containing threshold, extreme events count, and Sen's slope
        """
        # Ensure timeseries is computed to avoid dask array issues
        if hasattr(timeseries, 'compute'):
            timeseries = timeseries.compute()
            
        # Calculate threshold
        threshold = timeseries.quantile(percentile, dim='time').item()
        
        # Calculate extreme events count
        extreme_events = timeseries.where(timeseries > threshold, drop=True)
        count = len(extreme_events)
        
        # Calculate Sen's Slope (using scipy.stats.mstats.theilslopes)
        y = timeseries.values
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            sens_slope = np.nan
        else:
            from scipy.stats import mstats
            x = np.arange(len(y))
            # theilslopes returns: slope, intercept, lower_bound, upper_bound
            res = mstats.theilslopes(y[valid], x[valid])
            sens_slope = res[0]
            
        return {
            'threshold': threshold,
            'extreme_count': count,
            'sens_slope': sens_slope
        }

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
