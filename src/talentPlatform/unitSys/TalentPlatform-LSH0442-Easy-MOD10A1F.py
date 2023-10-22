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
import matplotlib.colors as mcolors

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
import numpy as np
from scipy.interpolate import Rbf

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
# end of snowmelt는 각 격자별로 daily snow cover 자료가 0이 되는 날짜 (end of snowmelt)를 구하는데 10일 이동평균을 계산하여 snow cover fracction이 0이 연속적으로 며칠간 0이 되는 날짜의 마지막날 (idl 코드 참고)을 계산하여 아웃풋이 각 년도별로 day of year 로 산출되도록 코딩.
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
        'lonMin': -0
        , 'lonMax': 360
        , 'lonInv': 0.25

        # 위도 최소/최대/간격
        , 'latMin': -90
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


# Appendix B) Coordinate conversion for the MODIS sinusoidal projection (MODIS_C61_BA_User_Guide_1.0.pdf)
def invGeoMODIS(H, V, size):

    R = 6371007.181
    T = 1111950.0
    xmin = -20015109.0
    ymax = 10007555.0

    w = T / size

    j, i = np.meshgrid(np.arange(size), np.arange(size))
    data = pd.DataFrame({'i': i.flatten(), 'j': j.flatten()})

    x = ((data['j'] + 0.5) * w) + H * T + xmin
    y = ymax - ((data['i'] + 0.5) * w) - (V * T)

    ll = y / R
    data['lat'] = np.rad2deg(ll)
    data['lon'] = np.rad2deg(x / (R * np.cos(ll)))

    data = data.where((-90 < data['lat']) & (data['lat'] < 90))
    data = data.where((-180 < data['lon']) & (data['lon'] < 360))

    return data

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

    if fileList is None or len(fileList) < 1: continue

    dataL5 = pd.DataFrame()
    for fileInfo in fileList:

        fileNameNoExt = os.path.basename(fileInfo).split('.hdf')[0]

        # NetCDF 파일 읽기
        data = xr.open_dataset(fileInfo, engine='pynio')

        match = re.search(r"h(\d{2})v(\d{2})", fileInfo)
        if not match: continue
        geoH = int(match.group(1))
        geoV = int(match.group(2))
        getoSize = 2400
        geoData = invGeoMODIS(geoH, geoV, getoSize)

        # 북반구 영역 선택
        geoDataL1 = geoData[geoData['lat'] >= 0]
        if len(geoDataL1) < 1: continue

        # qc-flag에서 best 선택
        dataL1 = data.where(
            (data['Basic_QA'] == 0)
        )

        # 0-100=NDSI snow, 200=missing data, 201=no decision, 211=night, 237=inland water, 239=ocean, 250=cloud, 254=detector saturated, 255=fill
        # data['CGF_NDSI_Snow_Cover'].values

        # 0=best, 1=good, 2=ok, 3=poor-not used, 4=other- not used, 211=night, 239=ocean, 255=unusable L1B or no data
        # data['Basic_QA'].values

        dataL2 = dataL1['CGF_NDSI_Snow_Cover'].to_dataframe().reset_index(drop=False).dropna().rename(
            columns={'XDim_MOD_Grid_Snow_500m': 'i', 'YDim_MOD_Grid_Snow_500m': 'j'}
        )

        print(f'[CHECK] fileInfo : {fileInfo} : {len(dataL2)}')
        if len(dataL2) < 1: continue

        dataL3 = dataL2.merge(geoData, on=['i', 'j'], how='left')

        dataL4 = dataL3[dataL3['lat'] >= 0]
        if len(dataL4) < 1: continue

        # 경도 변환 (-180~180 to 0~360)
        dataL4['lon'] = (dataL4['lon']) % 360

        #  2. 0.25x0.25도 해상도로 재격자화 한 후 북반구 영역만 nc 파일로 생성
        dataL5 = pd.concat([dataL5, dataL4], ignore_index=True)

        # ==============================================================
        # 규칙 격자를 통해 공간 내삽 (RBF) 수행
        # ==============================================================
        # posLon = dataL4['lon'].values
        # posLat = dataL4['lat'].values
        # posVar = dataL4['CGF_NDSI_Snow_Cover'].values

        # # lon1D = np.arange(sysOpt['grid']['lonMin'], sysOpt['grid']['lonMax'], sysOpt['grid']['lonInv'])
        # # lat1D = np.arange(sysOpt['grid']['latMin'], sysOpt['grid']['latMax'], sysOpt['grid']['latInv'])
        # lon1D = np.arange(int(posLon.min()), int(posLon.max() + 1), sysOpt['grid']['lonInv'])
        # lat1D = np.arange(int(posLat.min()), int(posLat.max() + 1), sysOpt['grid']['latInv'])
        #
        # print('[CHECK] len(lon1D) : {}'.format(len(lon1D)))
        # print('[CHECK] len(lat1D) : {}'.format(len(lat1D)))
        #
        # lon2D, lat2D = np.meshgrid(lon1D, lat1D)
        #
        # # Radial basis function (RBF) interpolation in N dimensions.
        # try:
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='multiquadric')
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='inverse')
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='gaussian')
        #     rbfModel = Rbf(posLon, posLat, posVar, function='linear')
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='cubic')
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='quintic')
        #     # rbfModel = Rbf(posLon, posLat, posVar, function='thin_plate')
        #
        #     rbfRes = rbfModel(lon2D, lat2D)
        # except Exception as e:
        #     print("Exception : {}".format(e))
        #
        # posData = xr.Dataset(
        #     {
        #         'CGF_NDSI_Snow_Cover': (('lat', 'lon'), (rbfRes).reshape(len(lat1D), len(lon1D)))
        #     }
        #     , coords={
        #         'lat': lat1D
        #         , 'lon': lon1D
        #     }
        # )
        #
        # posData['CGF_NDSI_Snow_Cover'].plot()
        # plt.show()

        # ***********************************************************
        # 시각화
        # ***********************************************************
        # lon1D = dataL3['lon'].values
        # lat1D = dataL3['lat'].values
        # val1D = dataL3['CGF_NDSI_Snow_Cover'].values
        #
        # cateList = {
        #     0: 'NDSI Snow', 200: 'Missing Data', 201: 'No Decision',
        #     211: 'Night', 237: 'Inland Water', 239: 'Ocean',
        #     250: 'Cloud', 254: 'Detector Saturated', 255: 'Fill'
        # }
        # colors = ['blue', 'gray', 'orange', 'yellow', 'green', 'cyan', 'white', 'red', 'black']
        # bounds = list(cateList.keys())
        #
        # # Create a custom colormap and norm based on your categories
        # cmap = mcolors.ListedColormap(colors)
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)
        #
        # # Create a scatter plot
        # fig, ax = plt.subplots()
        # sc = ax.scatter(lon1D, lat1D, c=val1D, cmap=cmap, norm=norm, s=1.0)
        #
        # # Create a colorbar with the custom colormap
        # cb = plt.colorbar(sc, ticks=bounds)
        # cb.set_ticklabels(list(cateList.values()))
        #
        # # saveImg = '{}/{}-{}.png'.format('/home/sbpark/analysis/python_resources/4satellites/20230723/figs', satType, obsDateTime)
        # saveImg = '{}/{}-{}-{}.png'.format('/DATA/FIG/LSH0442', fileNameNoExt, 'dataL3', 'best')
        # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        # print(f'[CHECK] saveImg : {saveImg}')
        #
        # cb.set_label(None)
        # plt.xlabel(None)
        # plt.ylabel(None)
        # plt.grid(True)
        # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        # plt.tight_layout()
        # plt.show()
        # plt.close()