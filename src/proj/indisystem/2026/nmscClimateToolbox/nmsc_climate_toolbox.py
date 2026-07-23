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
    def open(filepaths):
        """Open one or multiple NetCDF/GeoTIFF files."""
        if isinstance(filepaths, str):
            if filepaths.lower().endswith(('.tif', '.tiff')):
                import rioxarray
                # Convert rioxarray to standard dataset
                da = rioxarray.open_rasterio(filepaths)
                if da.name is None: da.name = 'band_data'
                ds = da.to_dataset()
                # Rename x/y to lon/lat if applicable
                if 'x' in ds.coords and 'y' in ds.coords:
                    ds = ds.rename({'x': 'lon', 'y': 'lat'})
                return ds
            return xr.open_dataset(filepaths)
        else:
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
    def cli(dataset, variable):
        """Calculate standard climatology using xCDAT with a fallback."""
        if HAS_XCDAT:
            return dataset.temporal.climatology(variable, freq="month")[variable]
        if 'time' in dataset.dims:
            return dataset[variable].groupby('time.month').mean(dim='time')
        return dataset[variable]

    @staticmethod
    def ano(dataset, variable):
        """Calculate anomalies from climatology using xCDAT with a fallback."""
        if HAS_XCDAT:
            return dataset.temporal.departures(variable, freq="month")[variable]
        if 'time' in dataset.dims:
            clim = dataset[variable].groupby('time.month').mean(dim='time')
            return dataset[variable].groupby('time.month') - clim
        return dataset[variable]

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
        
        ax.set_title(f"검증자료 비교 산점도 (Scatter Plot)\nR: {r:.3f}, Bias: {bias:.3f}")
        ax.set_xlabel(f"Product: {var_prod}")
        ax.set_ylabel(f"Validation: {var_valid}")
        fig.colorbar(hb, ax=ax, label='Density Count')
        ax.legend()
        
        return fig

nct = NMSCClimateToolbox()

if __name__ == '__main__':
    nct = NMSCClimateToolbox()

    data = nct.open('C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/indisystem/2026/nmscClimateToolbox\doc\L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc')

    aa = xr.open_dataset('C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/indisystem/2026/nmscClimateToolbox\doc\L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc')
