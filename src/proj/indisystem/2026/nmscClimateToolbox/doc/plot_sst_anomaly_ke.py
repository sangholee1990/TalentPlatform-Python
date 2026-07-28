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
import matplotlib.patheffects as pe

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
# parser.add_argument('--CTX_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst')
# parser.add_argument('--NC_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m-anom_ea020lc_202606010000.nc')
# parser.add_argument('--PNG_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m-anom_ke020lc_202606010000.png')
# parser.add_argument('--VAR_INFO', type=str, required=False, default='SST01M-ANOM')
# parser.add_argument('--BASE_MONTH', type=str, required=False, default='Climatology')
# parser.add_argument('--COMP_MONTH', type=str, required=False, default='2026.05')

parser.add_argument('--CTX_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst')
parser.add_argument('--NC_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m-diff_ea020lc_202606010000.nc')
parser.add_argument('--PNG_DIR', type=str, required=False, default='C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\output\gk2a_ami_le3_sst01m-diff_ke020lc_202606010000.png')
parser.add_argument('--VAR_INFO', type=str, required=False, default='SST01M-DIFF')
parser.add_argument('--BASE_MONTH', type=str, required=False, default='2025.05')
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

#VMAX = 3.0   # 편차 색상범위 (-VMAX ~ +VMAX)

# ------------------------------------------------------------------
# 1) 컬러바 정의 — 업로드된 이미지의 24개 색을 hex로 재현
#    (낮은 값 → 높은 값 순서: -6.0 ~ +6.0 / 0.5°C 간격)
# ------------------------------------------------------------------
SST_ANOM_COLORS = [
    '#FF66FF',  # -6.0 ~ -5.5  bright magenta
    '#FF33CC',  # -5.5 ~ -5.0  magenta
    '#CC33CC',  # -5.0 ~ -4.5  pink-purple
    '#9933CC',  # -4.5 ~ -4.0  purple
    '#6633CC',  # -4.0 ~ -3.5  dark purple
    '#3333CC',  # -3.5 ~ -3.0  blue-purple
    '#0033CC',  # -3.0 ~ -2.5  dark blue
    '#0066CC',  # -2.5 ~ -2.0  blue
    '#3399FF',  # -2.0 ~ -1.5  medium blue
    '#66CCFF',  # -1.5 ~ -1.0  light blue
    '#99FFFF',  # -1.0 ~ -0.5  cyan
    '#CCFFFF',  # -0.5 ~  0.0  pale cyan
    '#FFFFCC',  #  0.0 ~  0.5  cream
    '#FFFF99',  #  0.5 ~  1.0  pale yellow
    '#FFFF33',  #  1.0 ~  1.5  yellow
    '#FFCC33',  #  1.5 ~  2.0  yellow-orange
    '#FF9933',  #  2.0 ~  2.5  light orange
    '#FF6633',  #  2.5 ~  3.0  orange
    '#FF3333',  #  3.0 ~  3.5  red-orange
    '#FF0000',  #  3.5 ~  4.0  red
    '#CC0000',  #  4.0 ~  4.5  dark red
    '#A00000',  #  4.5 ~  5.0
    '#800000',  #  5.0 ~  5.5
    '#600000',  #  5.5 ~  6.0  deep red
]

VMIN, VMAX, DLEV = -6.0, 6.0, 0.5
LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)
CUSTORM_CMAP   = mcolors.ListedColormap(SST_ANOM_COLORS)
NORM   = mcolors.BoundaryNorm(LEVELS, ncolors=CUSTORM_CMAP.N,clip=True)

# ── lat/lon 로드 ─────────────────────────────────────────────
with nc.Dataset(LATLON_NC) as ds:
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]

# ── npy 로드 ─────────────────────────────────────────────────
# base_path = os.path.join(NPY_DIR, f'gk2a_sst_monthly_{BASE_MONTH}_ko_ext.npy')
# comp_path = os.path.join(NPY_DIR, f'gk2a_sst_monthly_{COMP_MONTH}_ko_ext.npy')
#
# base = np.load(base_path)
# comp = np.load(comp_path)
# anomaly = comp - base

ds = nc.Dataset(NC_DIR)
anomaly_raw = np.squeeze(ds[VAR_INFO][:])
anomaly = np.ma.masked_invalid(anomaly_raw)

print(f"min : {np.nanmin(anomaly):.2f}°C")
print(f"mean : {np.nanmean(anomaly):.2f}°C")
print(f"max : {np.nanmax(anomaly):.2f}°C")

# ── 편차장 plot ───────────────────────────────────────────────
year_base,  month_base  = BASE_MONTH[:4],  BASE_MONTH[4:6]
year_comp,  month_comp  = COMP_MONTH[:4],  COMP_MONTH[4:6]

#fig = plt.figure(figsize=(12, 10))
# fig = plt.figure(figsize=(8, 9))
# fig = plt.figure(figsize=(10, 2200/300))
# fig = plt.figure(figsize=(2000/300, 1800/300))
fig = plt.figure(figsize=(2200/300, 2000/300))

ax  = fig.add_subplot(111)
#m = Basemap(
#    projection='lcc', resolution='i',
#    lat_0=38, lon_0=126,
#    llcrnrlat=11.308528, urcrnrlat=53.303712,
#    llcrnrlon=101.395259, urcrnrlon=175.188166,
#    lat_1=30, lat_2=60
#)

m = Basemap(projection='cyl',
          llcrnrlat=15, urcrnrlat=55,
          llcrnrlon=110, urcrnrlon=150,
          resolution='h',ax=ax)

x, y = m(lon, lat)
#pcm = m.pcolormesh(x, y, anomaly, cmap='bwr', shading='auto',
#                   vmin=VMIN, vmax=VMAX)
pcm = m.pcolormesh(x, y, anomaly, cmap=CUSTORM_CMAP, norm=NORM, shading='auto')
m.drawcoastlines(color='k', linewidth=0.5)
m.drawcountries(color='gray', linewidth=0.5)
m.fillcontinents(color='lightgray', lake_color='white')
m.drawparallels(np.arange(15,56,10), labels=[1,0,0,0], fontsize=10, fontname='DejaVu Sans', fmt='%d',fontweight='bold')
m.drawmeridians(np.arange(110,151,10), labels=[0,0,0,1], fontsize=10, fontname='DejaVu Sans', fmt='%d',fontweight='bold')

#  컨투어 선 (1°C 간격, 0 라인 강조)
# line_levels = np.arange(VMIN, VMAX + 1e-6, 1.0)     # -6,-5,...,5,6
line_levels = np.arange(VMIN, VMAX + 1e-6, 2.0)
line_levels = line_levels[line_levels != 0.0]
line_levels = line_levels.astype(int)

# 2026.05.29 수정
# smoothed_data = gaussian_filter(anomaly, sigma=3)

# 중간값 필터
size_val = 15
smoothed_data = median_filter(anomaly_raw, size=size_val)
smoothed_ma = np.ma.masked_where(np.isnan(anomaly_raw), smoothed_data)

# 2026.05.29 수정
# cs = m.contour(x, y, smoothed_data, levels=line_levels, colors='k', linewidths=1.0, alpha=0.8)
cs = m.contour(x, y, smoothed_ma, levels=line_levels, colors='k', linewidths=1.0, alpha=0.8)

# 2026.05.29 수정
# 0°C 등치선 굵게
# cs0 = m.contour(x, y, smoothed_data, levels=[0.0], colors='k', linewidths=1.2)
cs0 = m.contour(x, y, smoothed_ma, levels=[0.0], colors='k', linewidths=0.5, alpha=0.5)

# 컨투어 선 라벨
# plt.clabel(cs, inline=True, fontsize=12, fmt='%d', colors='black')
# plt.clabel(cs, inline=True, fontsize=10, fmt='%d', colors='black')
# plt.clabel(cs, levels=line_levels, inline=True, fontsize=8, fmt='%d', colors='black')

# 2026.06.10 수정
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
                # 'coord': (poly[mid_idx, 0], poly[mid_idx, 1])
                'coord': (poly[mid_idx][0], poly[mid_idx][1])
            })
candidates.sort(key=lambda x: x['length'], reverse=True)
MAX_LABELS = 15
manual_locs = [c['coord'] for c in candidates[:MAX_LABELS]]

# labels = plt.clabel(cs, inline=True, fontsize=8, fmt='%d', colors='black')
labels = plt.clabel(cs, inline=True, fontsize=8, fmt='%d', colors='black', manual=manual_locs)
for label in labels:
    label.set_rotation(0)
    label.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])

# cbar = plt.colorbar(pcm, shrink=0.8,extend='both',spacing='proportinal')
cbar = plt.colorbar(pcm, shrink=0.8, extend='both', spacing='proportional')
cbar.set_ticks(LEVELS)
cbar.set_label('SST Anomaly (°C)', fontsize=10, fontname='DejaVu Sans',fontweight='bold')
for lbl in cbar.ax.get_yticklabels():
    lbl.set_fontname('DejaVu Sans')
    lbl.set_fontsize(10)
    lbl.set_fontweight('bold')

# plt.title(f'GK2A SST Anomaly ({year_comp}.{month_comp} - {year_base}.{month_base})', fontsize=12, fontweight='bold', fontname='DejaVu Sans')
plt.title(f'GK2A SST Anomaly ({COMP_MONTH} - {BASE_MONTH})', fontsize=12, fontweight='bold', fontname='DejaVu Sans')
plt.tight_layout()

# save_path = os.path.join(PNG_DIR, f'gk2a_sst_anomaly_{COMP_MONTH}_{BASE_MONTH}_ko_ext.png')
save_path = PNG_DIR
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.savefig(save_path, dpi=300, transparent=True)
plt.close()
print(f"save_path : {save_path}")