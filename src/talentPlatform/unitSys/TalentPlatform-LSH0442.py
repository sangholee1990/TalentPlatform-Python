# -*- coding: utf-8 -*-
import glob
import os
import platform
import warnings
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from pandas.tseries.offsets import Day
from scipy import stats

import re
import argparse
import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import os
import cartopy.crs as ccrs
import glob
# import geopandas as gpd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib as mpl

# sysOpt = {
#     # 관측소 목록
#     'stnList': [
#         {'name': 'fossil0001', 'abbr': None, 'lon': 126.97, 'lat': 37.58}
#         , {'name': 'fossil0049', 'abbr': None, 'lon': 129.15, 'lat': 35.45}
#         , {'name': 'fossil0112', 'abbr': None, 'lon': 128.60, 'lat': 35.85}
#         , {'name': 'fossil0122', 'abbr': None, 'lon': 127.20, 'lat': 36.15}
#         , {'name': 'fossil0194', 'abbr': None, 'lon': 126.50, 'lat': 37.05}
#         , {'name': 'fossil0225', 'abbr': None, 'lon': 129.15, 'lat': 35.65}
#         , {'name': 'fossil0227', 'abbr': None, 'lon': 127.70, 'lat': 34.9}
#         , {'name': 'tccon100', 'abbr': None, 'lon': 126.35, 'lat': 36.64}
#     ]
# }

# ****************************************************************************
# 시작/종료일 설정
# ****************************************************************************
dtSrtDate = pd.to_datetime('2020-12-01', format='%Y-%m-%d')
dtEndDate = pd.to_datetime('2020-12-31', format='%Y-%m-%d')
dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))


#I have created this function in python to transforms the coordinates
# applying the inverse mapping method..
import numpy as np


# https://www.researchgate.net/post/How_can_I_plot_MODIS_HDF-EOS_grid_images_into_geographic_coordinates_lat-lon
def inverse_maps(H, V, size):

    R = 6371007.181
    T = 1111950
    xmin = -20015109.
    ymax = 10007555.

    Lat = np.zeros((size, size))
    Lon = np.zeros((size, size))

    w = T / size
    y = np.array([(ymax - (i + .5) * w - V * T) for i in range(size)])
    x = np.array([((j + .5) * w + (H) * T + xmin) for j in range(size)])

    for i, yy in enumerate(y):
        for j, xx in enumerate(x):
            ll = yy / R
            Lat[i, j] = ll * 180 / np.pi
            Lon[i, j] = 180 / np.pi * (xx / (R * np.cos(ll)))

    return Lat, Lon

H = 35  # horizontal tile number
V = 8   # vertical tile number
size = 2400  # the number of cells along each axis within a tile
Lat, Lon = inverse_maps(H, V, size)

print(Lat, Lon)

# ****************************************************************************
# 자료 처리
# ****************************************************************************
# 날짜 기준으로 반복문
for dtDayIdx, dtDayInfo in enumerate(dtDayList):
    # print(f'[CHECK] dtDayInfo : {dtDayInfo}')

    # inpFile = '{}/{}'.format('/home/data/satellite/OCO3/OCO3_L2_Lite_SIF.10r', 'oco3_LtSIF_%y%m%d_B*_*.nc4')
    inpFile = '{}/{}'.format( '/DATA/INPUT/LSH0442', 'MOD10A1F.A%Y%j.h*v*.*.*.hdf')
    inpFileDate = dtDayInfo.strftime(inpFile)
    fileList = sorted(glob.glob(inpFileDate))

    if fileList is None or len(fileList) < 1:
        continue

    # NetCDF 파일 읽기
    fileInfo = fileList[0]

    data = xr.open_dataset(fileInfo, engine='pynio')
    # data = xr.open_mfdataset(fileList, combine='nested', engine='pynio')

    # 0-100=NDSI snow, 200=missing data, 201=no decision, 211=night, 237=inland water, 239=ocean, 250=cloud, 254=detector saturated, 255=fill
    data['CGF_NDSI_Snow_Cover'].values

    # 0=best, 1=good, 2=ok, 3=poor-not used, 4=other- not used, 211=night, 239=ocean, 255=unusable L1B or no data
    data['Basic_QA'].values

    # data['MOD10A1_NDSI_Snow_Cover'].plot()
    # data['CGF_NDSI_Snow_Cover'].plot()
    # data['Cloud_Persistence'].plot()
    # data['Basic_QA'].plot()
    # data['Algorithm_Flags_QA'].plot()
    # plt.show()

    dataL1 = data.to_dataframe().reset_index(drop=False)

    # # 수행 목록 기준으로 반복문
    # for satIdx, satType in enumerate(['OCO3-SIF']):
    #
    #     inpFile = '{}/{}'.format('/home/data/satellite/OCO3/OCO3_L2_Lite_SIF.10r', 'oco3_LtSIF_%y%m%d_B*_*.nc4')
    #     # inpFile = '{}/{}'.format( '/DATA/INPUT/LSH0455', 'oco3_LtSIF_%y%m%d_B*_*.nc4')
    #     inpFileDate = dtDayInfo.strftime(inpFile)
    #     fileList = sorted(glob.glob(inpFileDate))
    #
    #     if fileList is None or len(fileList) < 1:
    #         continue
    #
    #     # NetCDF 파일 읽기
    #     fileInfo = fileList[0]
    #
    #     data = xr.open_dataset(fileInfo, group=None)
    #     print(f'[CHECK] dtDayInfo : {dtDayInfo}')
    #     print(f'[CHECK] fileInfo : {fileInfo}')
    #
    #     # 관측소 목록에 따른 반복문
    #     for j, stnInfo in enumerate(sysOpt['stnList']):
    #         print(f'[CHECK] stnInfo : {stnInfo}')
    #
    #         # 영역 설정
    #         minLon = stnInfo['lon'] - 0.4
    #         maxLon = stnInfo['lon'] + 0.4
    #         minLat = stnInfo['lat'] - 0.4
    #         maxLat = stnInfo['lat'] + 0.4
    #         cenLon = stnInfo['lon']
    #         cenLat = stnInfo['lat']
    #
    #         # 관측소 위경도 및 flag 문턱값 설정
    #         dataL1 = data.where(
    #             (minLon <= data['Longitude']) & (data['Longitude'] <= maxLon)
    #             & (minLat <= data['Latitude']) & (data['Latitude'] <= maxLat)
    #             & (data['Quality_Flag'] == 0)
    #         )
    #
    #         dataL3 = dataL1
    #
    #         # 관측시간 계산
    #         obsDateTime = dtDayInfo.strftime('%Y%m%d')
    #
    #         # 주요 변수 가져오기
    #         val1D = dataL3['Daily_SIF_740nm']
    #
    #         # 자료 개수
    #         cnt = np.count_nonzero(~np.isnan(val1D))
    #
    #         if cnt < 1: continue
    #
    #         # 자료 전처리
    #         dataL4 = dataL3[['Longitude', 'Latitude', 'Longitude_Corners', 'Latitude_Corners', 'Daily_SIF_740nm']].to_dataframe().reset_index(drop=False).dropna()
    #         dataL5 = dataL4.pivot(index=['sounding_dim', 'Longitude', 'Latitude', 'Daily_SIF_740nm'], columns='vertex_dim')
    #         dataL5.columns = ['vertex_longitude_1', 'vertex_longitude_2', 'vertex_longitude_3', 'vertex_longitude_4', 'vertex_latitude_1', 'vertex_latitude_2', 'vertex_latitude_3', 'vertex_latitude_4']
    #         dataL6 = dataL5.reset_index()
    #
    #         # 자료 저장
    #         saveFile = '{}/{}_{}.csv'.format('/home/sbpark/analysis/python_resources/4satellites/20230723/figs', satType, obsDateTime)
    #         # saveFile = '{}/{}_{}.csv'.format('/DATA/OUTPUT/LSH0455', satType, obsDateTime)
    #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
    #         dataL6.to_csv(saveFile, index=False)
    #
    #         minVal = np.nanmin(val1D)
    #         maxVal = np.nanmax(val1D)
    #
    #         mainTitle = f'{satType} ({stnInfo["name"]}) {obsDateTime}'
    #         subTitle = f'N = {cnt} / range = {minVal:.1f} ~ {maxVal:.1f}'
    #
    #         # 시각화
    #         saveImg = '{}/{}-{}.png'.format('/home/sbpark/analysis/python_resources/4satellites/20230723/figs', satType, obsDateTime)
    #         # saveImg = '{}/{}_{}.png'.format('/DATA/FIG/LSH0455', satType, obsDateTime)
    #         os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    #
    #         fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()), dpi=600)
    #         extent_area = [minLon, maxLon, minLat, maxLat]
    #         ax.set_extent(extent_area)
    #
    #         # Map Gridline and formatting xticks and yticks
    #         gl = ax.gridlines(draw_labels=False, linewidth=0.1)
    #         g1 = plt.gca().gridlines(color='dimgrey', linestyle='--', linewidth=0.4)
    #         line_scale = 0.2  # 간격
    #         xticks = np.arange(125, 130, line_scale)
    #         yticks = np.arange(30, 40, line_scale)
    #
    #         g1.xformatter = LONGITUDE_FORMATTER
    #         g1.yformatter = LATITUDE_FORMATTER
    #         g1.xlocator = mticker.FixedLocator(xticks)
    #         g1.ylocator = mticker.FixedLocator(yticks)
    #         g1.top_labels = True
    #         g1.left_labels = True
    #
    #         for index, row in dataL6.iterrows():
    #             # vertex 그리드
    #             gridLon = np.array([row['vertex_longitude_1'], row['vertex_longitude_2'], row['vertex_longitude_4'], row['vertex_longitude_3']]).reshape(2, 2)
    #             gridLat = np.array([row['vertex_latitude_1'], row['vertex_latitude_2'], row['vertex_latitude_4'], row['vertex_latitude_3']]).reshape(2, 2)
    #             gridVal = np.array([[row['Daily_SIF_740nm']]])
    #             cbar = ax.pcolormesh(gridLon, gridLat, gridVal, transform=ccrs.PlateCarree(), vmin=0, vmax=1)
    #
    #             # 산점도
    #             # cbar = ax.scatter(row['Longitude'], row['Latitude'], c=row['Daily_SIF_740nm'], s=10, vmin=410, vmax=430)
    #
    #         plt.plot(cenLon, cenLat, marker='*', markersize=10, color='red', linestyle='none')
    #
    #         # smaplegend = plt.colorbar(al, ticks=np.arange(minVal, maxVal), extend='both')
    #         cbar.set_clim([0, 1])
    #         plt.colorbar(cbar, ax=ax, extend='both')
    #
    #         # SHP 파일 (시도, 시군구)
    #         shapefile = '/home/sbpark/data/GIS/shapefiles/KOR_adm/gadm36_KOR_0.shp'
    #         shapefile2 = '/home/sbpark/data/GIS/shapefiles/KOR_adm/gadm36_KOR_1.shp'
    #         # shapefile = '/DATA/INPUT/LSH0455/gadm36_KOR_shp/gadm36_KOR_0.shp'
    #         # shapefile2 = '/DATA/INPUT/LSH0455/gadm36_KOR_shp/gadm36_KOR_1.shp'
    #
    #         shp1 = gpd.read_file(shapefile)
    #         shp2 = gpd.read_file(shapefile2)
    #         shp1.plot(ax=ax, color='None', edgecolor='black', linewidth=3)
    #         shp2.plot(ax=ax, color='None', edgecolor='black')
    #
    #         plt.ylabel(None)
    #         plt.xlabel(None)
    #         cbar.set_label(None)
    #
    #         plt.suptitle(mainTitle)
    #         plt.title(subTitle)
    #         plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()