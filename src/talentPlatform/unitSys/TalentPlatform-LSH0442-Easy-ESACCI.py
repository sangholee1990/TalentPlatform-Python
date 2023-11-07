# -*- coding: utf-8 -*-
import glob
import os
import platform
import warnings
import re
import argparse
import sys
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from pandas.tseries.offsets import Day
from scipy import stats
import matplotlib.colors as mcolors

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
import numpy as np

# ****************************************************************************
# 요청 사항
# ****************************************************************************
# 안녕하세요 의뢰내용은 대략적으로 아래와 같습니다.
# 의뢰내용: 파이썬으로 두종류 위성자료 처리 및 시각화
#
# 위성자료 샘플자료 받는경로: 한달 테스트 자료를 이용
# 1. cloud-gap-filled (CGF) daily snow cover tiled product L3 500m SIN grid
# https://search.earthdata.nasa.gov/search/granules?p=C1646609734-NSIDC_ECS&pg[0][v]=f&pg[0][gsk]=-start_date&q=MOD10A1F&tl=1697290105.586!3!!
# https://nsidc.org/sites/default/files/c61_modis_snow_user_guide.pdf
#
# 2. ESA CCI daily snow cover fraction on the ground product
# https://data.ceda.ac.uk/neodc/esacci/snow/data/scfg/MODIS/v2.0/
#
# 자료처리 내용
# 1. qc flag 이용한 데이터 확인작업
# 두개 위성산출물에서 임의의 몇개 날짜 자료를 읽고 best자료만 선택하여 mapping하기
# 2. 0.25x0.25도 해상도로 재격자화 한 후 북반구 영역만 nc 파일로 생성
# 3. end of snowmelt (day of year) 구하기
# end of snowmelt는 각 격자별로 daily snow cover 자료가 0이 되는 날짜 (end of snowmelt)를 구하는데
# 10일 이동평균을 계산하여 snow cover fracction이 0이 연속적으로 며칠간 0이 되는 날짜의 마지막날 (idl 코드 참고)을 계산하여
# 아웃풋이 각 년도별로 day of year 로 산출되도록 코딩.
#
# 사용할 라이브러리: xarray, pandas, numpy, pyhdf (netCDF4가 더 유용하다면 사용하셔도 됩니다)
# 예시: https://stackoverflow.com/questions/57990038/genrate-grid-information-file-from-modis-hdfeos-data


# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2020-12-01'
    , 'endDate': '2020-12-31'

    # 신규 격자
    , 'grid': {
        # 경도 최소/최대/간격
        'lonMin': -180
        , 'lonMax': 180
        , 'lonInv': 0.25

        # 위도 최소/최대/간격
        , 'latMin': 0
        , 'latMax': 90
        , 'latInv': 0.25
    }
}

# ****************************************************************************
# 시작/종료일 설정
# ****************************************************************************
dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

# ****************************************************************************
# 자료 처리
# ****************************************************************************
# 날짜 기준으로 반복문
# dataL3 = xr.Dataset()
# for dtDayIdx, dtDayInfo in enumerate(dtDayList):
#     # print(f'[CHECK] dtDayInfo : {dtDayInfo}')
#
#     # inpFile = '{}/{}'.format('/home/data/satellite/OCO3/OCO3_L2_Lite_SIF.10r', 'oco3_LtSIF_%y%m%d_B*_*.nc4')
#     inpFile = '{}/{}'.format( '/DATA/INPUT/LSH0442', '%Y%m%d-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv2.0.nc')
#     inpFileDate = dtDayInfo.strftime(inpFile)
#     fileList = sorted(glob.glob(inpFileDate))
#
#     if fileList is None or len(fileList) < 1: continue
#
#     for fileInfo in fileList:
#
#         print(f'[CHECK] fileInfo : {fileInfo}')
#
#         fileNameNoExt = os.path.basename(fileInfo).split('.nc')[0]
#
#         # NetCDF 파일 읽기
#         data = xr.open_dataset(fileInfo)
#
#         # data['scfg'].values
#         # 0=snow free, 1-100=SCF[%], 205=Cloud, 206=Night, 210=Water, 215=Glaciers, icecaps, ice sheets, 252|253|254=ERROR, 255=Not valid data
#         # data['scfg_unc'].values
#         # data['spatial_ref'].values
#
#         lonList = np.arange(sysOpt['grid']['lonMin'], sysOpt['grid']['lonMax'], sysOpt['grid']['lonInv'])
#         latList = np.arange(sysOpt['grid']['latMin'], sysOpt['grid']['latMax'], sysOpt['grid']['latInv'])
#         dataL1 = data.isel(nv=0, time=0).sel(lon=lonList, lat=latList, method='nearest')
#
#         # dataL1['scfg'].plot()
#         # plt.show()
#
#         # flag 정보 없음
#         # scfg 기준으로 252 이하 설정
#         dataL2 = dataL1['scfg'].where((dataL1['scfg'] < 252))
#
#         lat1D = dataL2['lat'].values
#         lon1D = dataL2['lon'].values
#
#         selData = xr.Dataset(
#             {
#                 'scfg': (('time', 'lat', 'lon'), (dataL2.values).reshape(1, len(lat1D), len(lon1D)))
#             }
#             , coords={
#                 'time': pd.date_range(dtDayInfo, periods=1)
#                 , 'lat': lat1D
#                 , 'lon': lon1D
#             }
#         )
#
#         # ***********************************************************
#         # 2. 0.25x0.25도 해상도로 재격자화 한 후 북반구 영역만 nc 파일로 생성
#         # ***********************************************************
#         # saveNcFile = '{}/{}-{}.png'.format('/home/sbpark/analysis/python_resources/4satellites/20230723/figs', satType, obsDateTime)
#         saveNcFile = '{}/{}-{}.nc'.format('/DATA/OUTPUT/LSH0442', fileNameNoExt, 'prop')
#         os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
#         selData.to_netcdf(saveNcFile)
#         print(f'[CHECK] saveNcFile : {saveNcFile}')
#
#         # ***********************************************************
#         # 시각화
#         # ***********************************************************
#         dataL3 = dataL2.to_dataframe().reset_index(drop=False).dropna()
#         lon1D = dataL3['lon'].values
#         lat1D = dataL3['lat'].values
#         val1D = dataL3['scfg'].values
#
#         #   # 0=snow free, 1-100=SCF[%], 205=Cloud, 206=Night, 210=Water, 215=Glaciers, icecaps, ice sheets, 252|253|254=ERROR, 255=Not valid data
#         cateList = {0: 'snow free', 1: 'SCF %', 205: 'Cloud', 206: 'Night', 210: 'Water', 215: 'Glaciers', 251: 'Null'}
#         colors = ['blue', 'gray', 'orange', 'yellow', 'green', 'cyan', 'black']
#         bounds = list(cateList.keys())
#
#         # Create a custom colormap and norm based on your categories
#         cmap = mcolors.ListedColormap(colors)
#         norm = mcolors.BoundaryNorm(bounds, cmap.N)
#
#         # Create a scatter plot
#         fig, ax = plt.subplots()
#         sc = ax.scatter(lon1D, lat1D, c=val1D, cmap=cmap, norm=norm, s=1.0)
#
#         # Create a colorbar with the custom colormap
#         cb = plt.colorbar(sc, ticks=bounds)
#         cb.set_ticklabels(list(cateList.values()))
#
#         # saveImg = '{}/{}-{}.png'.format('/home/sbpark/analysis/python_resources/4satellites/20230723/figs', satType, obsDateTime)
#         saveImg = '{}/{}-{}.png'.format('/DATA/FIG/LSH0442', fileNameNoExt, 'prop')
#         os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#
#         cb.set_label(None)
#         plt.xlabel(None)
#         plt.ylabel(None)
#         plt.grid(True)
#         plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
#         plt.tight_layout()
#         plt.show()
#         plt.close()
#
#         print(f'[CHECK] saveImg : {saveImg}')

inpFile = '{}/{}'.format('/DATA/OUTPUT/LSH0442', '*-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv2.0-prop.nc')
fileList = sorted(glob.glob(inpFile))

data = xr.open_mfdataset(fileList)

scfg = data['scfg']
# 0=snow free, 1-100=SCF[%], 205=Cloud, 206=Night, 210=Water, 215=Glaciers, icecaps, ice sheets, 252|253|254=ERROR, 255=Not valid data

scfgL1 = scfg.where((scfg <= 100))

# 10일 이동평균 계산
movMean = scfgL1.rolling(time=10, center=True).sum()

# movMean.isel(time = 11).plot()
# plt.show()

# 각 격자점에서 end of snowmelt를 찾기 위한 함수
def find_end_of_snowmelt(da):
    # 연속적인 0 값을 찾기 위한 Boolean 마스크 생성
    zeros_mask = (da == 0)

    # 연속적인 0 값을 갖는 시간의 길이를 계산
    lengths = zeros_mask.groupby('time.year').apply(lambda y: y.cumsum(dim='time') - y.cumsum(dim='time').where(~y).ffill(dim='time').fillna(0))

    # 각 해마다 마지막 연속 0값 찾기
    end_of_snowmelt = lengths.groupby('time.year').apply(lambda y: y.where(y.groupby('time.year').max() == y).dropna(dim='time', how='all').time)

    # day of year로 변환
    return end_of_snowmelt.dt.dayofyear

# (movMean == 0)

movMean['time'] = pd.to_datetime(movMean['time']).dt.strftime('%j')

aa = movMean.to_dataframe().reset_index(drop=False)

# aa['time']

# jd = pd.to_datetime(aa['time']).dt.strftime('%j')

bb = aa.pivot(index=['lon', 'lat'], columns='time', values='scfg').reset_index(drop=False)

grouped = movMean.stack(points=('lat', 'lon'))
end_of_snowmelt = grouped.groupby('points').apply(find_end_of_snowmelt)
# end_of_snowmelt2 = end_of_snowmelt.unstack('points')

# 각 격자점에서 end of snowmelt 계산
# test = movMean.groupby(('time')).apply(find_end_of_snowmelt)
# end_of_snowmelt = grouped.groupby('lat').apply(find_end_of_snowmelt)
# end_of_snowmelt = grouped.apply(lambda x: x.groupby('lon').apply(find_end_of_snowmelt))
# end_of_snowmelt = movMean.groupby('time').groupby('lat').apply(find_end_of_snowmelt)
# end_of_snowmelt = movMean.groupby(['time']).apply(find_end_of_snowmelt)

print('asdasdasd')

# 10일 이동평균이 0이고, 원래의 눈 덮인 지역이 0인 마지막 날 찾기
# condition = (movMean == 0)
# last_true = condition.where(condition).resample(time='AS').map(lambda x: x.idxmax(dim='time'))

# last_true.isel(time=0)
# last_true.isel(time=0).plot()
# plt.show()



