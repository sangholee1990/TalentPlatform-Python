import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import netCDF4 as nc
import platform
import argparse
from scipy.ndimage import median_filter
from scipy.ndimage import generic_filter
import warnings

# ============================================================
#  사용자 입력 설정 (여기만 수정하세요)
# ============================================================
# 2026.05.29 수정
# LATLON_NC = "/spd/yejihwang/latlon/gk2a_ami_eazzzlc_latlon/gk2a_ami_ea020lc_latlon.nc"
# #NPY_DIR   = "/spd/yejihwang/sst/2025/longterm/EA/npy/monthly"
# #PNG_DIR   = "/spd/yejihwang/sst/2025/longterm/EA/png/anomaly"
# NPY_DIR   = "/sdd/icshin/WORK_2026/GK2A_SST/EA/npy/monthly"
# PNG_DIR   = "/sdd/icshin/WORK_2026/GK2A_SST/EA/png/anomaly"
# os.makedirs(PNG_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--CTX_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst')
parser.add_argument('--NC_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m_ea020lc_202606010000.nc')
parser.add_argument('--PNG_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m_ke020lc_202606010000.png')
parser.add_argument('--VAR_INFO', type=str, required=False, default='SST01M')
parser.add_argument('--BASE_MONTH', type=str, required=False, default='')
parser.add_argument('--COMP_MONTH', type=str, required=False, default='2026.05')
args = parser.parse_args()

CTX_DIR = args.CTX_DIR
os.chdir(CTX_DIR)
print(f"CTX_DIR : {os.getcwd()}")

LATLON_NC = f"{CTX_DIR}/cfg/gk2a_ami_ea020lc_latlon.nc"
NC_DIR = args.NC_DIR
PNG_DIR =  args.PNG_DIR
COMP_MONTH = args.COMP_MONTH
BASE_MONTH = args.BASE_MONTH
VAR_INFO = args.VAR_INFO

# ── lat/lon 로드 ─────────────────────────────────────────────
with nc.Dataset(LATLON_NC) as ds:
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]

# 2026.05.29 수정
# ── npy 로드 ─────────────────────────────────────────────────
# base_path = os.path.join(NPY_DIR, f'gk2a_sst_monthly_{BASE_MONTH}_ko_ext.npy')
# comp_path = os.path.join(NPY_DIR, f'gk2a_sst_monthly_{COMP_MONTH}_ko_ext.npy')
#
# base = np.load(base_path)
# comp = np.load(comp_path)
# anomaly = comp - base

ds = nc.Dataset(NC_DIR)
data_raw = np.squeeze(ds[VAR_INFO][:])
data = np.ma.masked_invalid(data_raw) - 273.15

print(f"min : {np.nanmin(data):.2f}°C")
print(f"mean : {np.nanmean(data):.2f}°C")
print(f"max : {np.nanmax(data):.2f}°C")

# ── plot ───────────────────────────────────────────────
# year_base,  month_base  = BASE_MONTH[:4],  BASE_MONTH[4:6]
year_comp,  month_comp  = COMP_MONTH[:4],  COMP_MONTH[4:6]

# 2026.05.29 수정
# fig = plt.figure(figsize=(9,9))
fig = plt.figure(figsize=(2200/300, 2000/300))
ax  = fig.add_subplot(111)

# 2026.05.29 수정
#   m = Basemap(projection='lcc', resolution='i',
#                          lat_0=38, lon_0=126,
#                          llcrnrlat=11.308528, urcrnrlat=53.303712,
#                          llcrnrlon=101.395259, urcrnrlon=175.188166,
#                          lat_1=30, lat_2=60)

m = Basemap(projection='cyl',
        llcrnrlat=15, urcrnrlat=55,
        llcrnrlon=110, urcrnrlon=150,
        resolution='h')

# 등온선 설정
VMIN, VMAX, DLEV = 0, 36, 4
LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)

# 2026.05.29 수정
VMIN2, VMAX2, DLEV2 = 0, 35, 5
LEVELS2 = np.arange(VMIN2, VMAX2 + 1e-6, DLEV2)

x,y = m(lon,lat)
pcm = m.pcolormesh(x, y, data, cmap='jet', shading='auto', vmin=VMIN, vmax=VMAX)
m.drawcoastlines(color='k', linewidth=0.5)
m.drawcountries(color='gray', linewidth=0.5)
m.fillcontinents(color='lightgray', lake_color='white')

# 2026.05.29 수정
#   m.drawparallels(np.arange(float(np.nanmin(lat)),float(np.nanmax(lat)) + 1, 5), labels=[1,0,0,0], fontsize=15, fontname='arial', fmt='%d')
#   m.drawmeridians(np.arange(float(np.nanmin(lon)),float(np.nanmax(lon)) + 1, 10), labels=[0,0,0,1], fontsize=15, fontname='arial', fmt='%d')
m.drawparallels(np.arange(15,56,10), labels=[1,0,0,0], fontsize=10, fontname='DejaVu Sans', fmt='%d',fontweight='bold')
m.drawmeridians(np.arange(110,151,10), labels=[0,0,0,1], fontsize=10, fontname='DejaVu Sans', fmt='%d',fontweight='bold')

# 2026.05.29 수정
# smoothed_data = gaussian_filter(data, sigma=3)

# 가우시안 필터
weight = np.ones_like(data.data)
weight[data.mask] = 0.0
data_zeroed = data.filled(0.0)
sigma_val = 15
smoothed_data = gaussian_filter(data_zeroed, sigma=sigma_val)
smoothed_weight = gaussian_filter(weight, sigma=sigma_val)
with np.errstate(invalid='ignore', divide='ignore'):
    smoothed_corrected = smoothed_data / smoothed_weight
smoothed_ma = np.ma.masked_array(smoothed_corrected, mask=data.mask)

# 2026.05.29 수정
# c = m.contour(x,y,smoothed_data,levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8)
c = m.contour(x, y, smoothed_ma, levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8)


# 2026.05.29 수정
# plt.clabel(c, inline=True, fontsize=10, fmt='%d', colors='black')
# labels = plt.clabel(c, levels=LEVELS, inline=False, fontsize=8, fmt='%d', colors='black')
labels = plt.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black')
# labels = plt.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black', manual=manual_locs)
for label in labels:
    label.set_rotation(0)

# 2026.05.29 수정
# cbar = plt.colorbar(pcm,shrink=0.8)
cbar = plt.colorbar(pcm, shrink=0.8, extend='both', spacing='proportional')
cbar.set_ticks(LEVELS2)

cbar.set_label('Sea Surface Temperature (°C)', fontsize=10, fontname='DejaVu Sans',fontweight='bold')
for lbl in cbar.ax.get_yticklabels():
    lbl.set_fontname('DejaVu Sans')
    lbl.set_fontsize(10)
    lbl.set_fontweight('bold')

# 2026.05.29 수정
# plt.title(f'Monthly GK2A Sea Surface Temperature ({year_comp}.{month_comp})',fontsize=12, fontweight='bold', fontname='DejaVu Sans')
plt.title(f'Monthly Mean GK2A Sea Surface Temperature ({COMP_MONTH})',fontsize=12, fontweight='bold', fontname='DejaVu Sans')
plt.tight_layout()

# 2026.05.29 수정
save_path = PNG_DIR

# 2026.05.29 수정
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.savefig(save_path, dpi=300, transparent=True)
plt.close()
print(f"save_path : {save_path}")
