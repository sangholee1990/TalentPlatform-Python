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
import geopandas as gpd
import os
import cartopy.crs as ccrs
import glob
import geopandas as gpd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib as mpl


# ============================================
# 요구사항
# ============================================
# Python을 이용한 온실가스 위성 (OCO2, OCO3, TROPOMI) 자료 처리 및 시각화


# OCO2, OCO3 위성의 변수 (XCO2, SIF) TROPOMI 위성의 변수 (NO2) 라는 변수들 중 good flag(0)인 자료만 mapping 하기.
# 사례선택을 위해 픽셀이 station 중심에 얼마나 많이 분포되어 있는지를 찾는 그림 (2번 슬라이드 같은 그림이 나와야함)
#
# Station 중심반경을 0.8도로 한것에 대한 map을 그려야함
# Station 마다 영역 설정이 다르게 들어가줘야 함
# 각 그림마다 어떤위성, 어떤 operational 모드이며 영역내의 good pixel 개수가 얼마인지, 최대최소값 범위는 얼마인지 같이 적어주고 아웃풋도 따로 txt로 출력해야함
#
# 한번에 돌릴수 있는 방법? 파이썬코드를 5개로 분리한 후 shell로 돌림


# 샘플자료 1개씩 보내기
# Xarray 사용하여 작성
# 적용잘짜는 2019.08.01~2022.12.31
# 파일에서 어느부분이 날짜인지
# 파이썬은 3개로 만들고, 쉘은 csh, bash모두 가능
# 변수명 정보
# 다음주 금요일까지 초안작업 완료

# ============================================
# 보조
# ============================================
def makeMapPlot(sysOpt, lon2D, lat2D, val2D, mainTitle, subTitle, saveImg, isRoi=False):
    result = None

    try:
        plt.figure(dpi=600)

        if isRoi:
            map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c', llcrnrlon=sysOpt['roi']['minLon'],
                          urcrnrlon=sysOpt['roi']['maxLon'], llcrnrlat=sysOpt['roi']['minLat'], urcrnrlat=sysOpt['roi']['maxLat'])
        else:
            map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c')

        cs = map.scatter(lon2D, lat2D, c=val2D, s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))

        plt.plot(sysOpt['roi']['cenLon'], sysOpt['roi']['cenLat'], marker='*', markersize=10, color='black', linestyle='none')

        # GADM shp 정보
        shpFile = '{}*'.format(sysOpt['metaInfo'][sysOpt['shpInfo']]['filePath'])
        fileList = sorted(glob.glob(shpFile))
        if len(fileList) > 0:
            map.readshapefile(sysOpt['metaInfo'][sysOpt['shpInfo']]['filePath'], sysOpt['metaInfo'][sysOpt['shpInfo']]['fileName'])

        # 기본적인 shp 정보
        # map.drawcoastlines()
        # map.drawmapboundary()
        # map.drawcountries(linewidth=1, linestyle='solid', color='k')
        map.drawmeridians(np.arange(-180, 180, 0.4), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(np.arange(-90, 90, 0.4), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])

        plt.ylabel(None)
        plt.xlabel(None)
        cbar = map.colorbar(cs)
        # cbar = plt.colorbar(cs, orientation='horizontal')
        cbar.set_label(None)
        plt.suptitle(mainTitle)
        plt.title(subTitle)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.tight_layout()
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        print(f'Exception : {e}')

        return result


# ============================================
# 주요
# ============================================
serviceName = 'LSH0455'

# ****************************************************************************
# 전역변수 설정
# ****************************************************************************
# 입력, 출력, 그림 파일 정보
globalVar = {
    'inpPath': '/DATA/INPUT'
    , 'outPath': '/DATA/OUTPUT'
    , 'figPath': '/DATA/FIG'
}

# globalVar = {
#     #  'inpPath': '/home/sbpark/data/Satellite/OCO2/L2/OCO2_L2_Lite_FP.11r'
#     #, 'outPath': '/home/sbpark/analysis/python_resources/4satellites/20230723/output'
#     'figPath': '/home/sbpark/analysis/python_resources/4satellites/20230723/figs'
# }


# globalVar['inpPath'] = '/DATA/INPUT'
# globalVar['outPath'] = '/DATA/OUTPUT'
# globalVar['figPath'] = '/DATA/FIG'

# ****************************************************************************
# 초기 전달인자 설정
# 쉘스크립트에서 line 47~에서 날짜 범위선택 또는 위성자료 또는 shapefile 선택
# ****************************************************************************
parser = argparse.ArgumentParser()

for i, argv in enumerate(sys.argv[1:]):
    if not argv.__contains__('--'): continue
    parser.add_argument(argv)

inParInfo = vars(parser.parse_args())
# print(f'[CHECK] inParInfo : {inParInfo}')

# 전역 변수에 할당
for key, val in inParInfo.items():
    if val is None: continue
    globalVar[key] = val
# print(f'[CHECK] globalVar : {globalVar}')

# ****************************************************************************
# 사용자 설정
# ****************************************************************************
# 옵션 설정
sysOpt = {

    # ****************************************************************************
    # 동적 설정
    # ****************************************************************************
    # 시작/종료 시간
    'srtDate': '2022-07-01'
    , 'endDate': '2022-12-31'
    # 'srtDate': globalVar['srtDate']
    # , 'endDate': globalVar['endDate']

    # 위성 목록
    , 'satList': ['OCO2-CO2']
    # , 'satList': ['OCO3-CO2']
    # , 'satList': ['OCO2-SIF']
    # , 'satList': ['OCO3-SIF']
    # , 'satList': ['TROPOMI']
    # , 'satList': ['OCO2-CO2', 'OCO3-CO2', 'OCO2-SIF', 'OCO3-SIF', 'TROPOMI']
    # , 'satList': [globalVar['satList']]

    # shp 파일 정보
    # , 'shpInfo': 'gadm36_KOR_0'
    , 'shpInfo': 'gadm36_KOR_1'
    # , 'shpInfo': 'gadm36_KOR_2'
    # , 'shpInfo': globalVar['shpInfo']

    # ****************************************************************************
    # 정적 설정
    # ****************************************************************************
    # 중심 반경
    # 2도 = 약 200 km
    , 'res': 0.8

    # 관측모드 설정 (OCO2 operational mode; Nadir0 & Target 3)
    # 관측모드 설정 (OCO3 operational mode; Nadir0 & Target 3 & SAM 4)
    , 'obsModeList': {0: 'Nadir', 1: 'Glint', 2: 'Target', 3: 'Transition', 4: 'SAMs'}

    # 관심영역 설정
    , 'roi': {}

    # 관측소 목록
    , 'stnList': [
        {'name': 'fossil0001', 'abbr': None, 'lon': 126.97, 'lat': 37.58}
        , {'name': 'fossil0049', 'abbr': None, 'lon': 129.15, 'lat': 35.45}
        , {'name': 'fossil0112', 'abbr': None, 'lon': 128.60, 'lat': 35.85}
        , {'name': 'fossil0122', 'abbr': None, 'lon': 127.20, 'lat': 36.15}
        , {'name': 'fossil0194', 'abbr': None, 'lon': 126.50, 'lat': 37.05}
        , {'name': 'fossil0225', 'abbr': None, 'lon': 129.15, 'lat': 35.65}
        , {'name': 'fossil0227', 'abbr': None, 'lon': 127.70, 'lat': 34.9}
        , {'name': 'tccon100', 'abbr': None, 'lon': 126.35, 'lat': 36.64}
    ]

    # 메타 정보
    , 'metaInfo': {
        # shp 관련 속성 정보
        'gadm36_KOR_0': {
            'filePath': '/DATA/INPUT/LSH0455/gadm36_KOR_shp/gadm36_KOR_0'
            , 'fileName': 'gadm36_KOR_0'
        }
        , 'gadm36_KOR_1': {
            'filePath': '/DATA/INPUT/LSH0455/gadm36_KOR_shp/gadm36_KOR_1'
            , 'fileName': 'gadm36_KOR_1'
        }
        , 'gadm36_KOR_2': {
            'filePath': '/DATA/INPUT/LSH0455/gadm36_KOR_shp/gadm36_KOR_2'
            , 'fileName': 'gadm36_KOR_2'
        }

        # 위성 관련 속성 정보
        , 'OCO2-CO2': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtCO2_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sounding'
            , 'var': {
                'id': 'sounding_id', 'lon': 'longitude', 'lat': 'latitude', 'flag': 'xco2_quality_flag', 'val': 'xco2', 'obsMode': 'operation_mode'
            }
            # , 'flag': {'val': 0, 'obsMode': 2}
            , 'flag': {'val': 0, 'obsMode': 0}
        }
        , 'OCO2-SIF': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtSIF_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sequences'
            , 'var': {
                'id': 'sounding_dim', 'lon': 'Longitude', 'lat': 'Latitude', 'flag': 'Quality_Flag', 'val': 'SIF_740nm', 'obsMode': 'SequencesMode'
            }
            , 'flag': {'val': 0}
        }
        , 'OCO3-CO2': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco3_LtCO2_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sounding'
            , 'var': {
                'id': 'sounding_id', 'lon': 'longitude', 'lat': 'latitude', 'flag': 'xco2_quality_flag', 'val': 'xco2', 'obsMode': 'operation_mode'
            }
            , 'flag': {'val': 0, 'obsMode': 4}
        }
        , 'OCO3-SIF': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtSIF_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sequences'
            , 'var': {
                'id': 'sounding_dim', 'lon': 'Longitude', 'lat': 'Latitude', 'flag': 'Quality_Flag', 'val': 'SIF_740nm', 'obsMode': 'SequencesMode'
            }
            , 'flag': {'val': 0}
        }
        , 'TROPOMI': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'S5P_RPRO_L2__NO2____%Y%m%dT*.nc'
            , 'group': 'PRODUCT'
            , 'groupObs': 'Sounding'
            , 'var': {
                'id': '', 'lon': 'longitude', 'lat': 'latitude', 'flag': 'qa_value', 'val': 'nitrogendioxide_tropospheric_column', 'obsMode': 'operation_mode'
            }
            , 'flag': {'val': 0}
        }
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
for dtDayIdx, dtDayInfo in enumerate(dtDayList):
    # print(f'[CHECK] dtDayInfo : {dtDayInfo}')

    # 수행 목록 기준으로 반복문
    for satIdx, satType in enumerate(sysOpt['satList']):
        # print(f'[CHECK] satInfo : {satInfo}')

        satInfo = sysOpt['metaInfo'][satType]

        inpFile = '{}/{}'.format(satInfo['filePath'], satInfo['fileName'])
        inpFileDate = dtDayInfo.strftime(inpFile)
        fileList = sorted(glob.glob(inpFileDate))

        if fileList is None or len(fileList) < 1:
            continue

        # NetCDF 파일 읽기
        fileInfo = fileList[0]

        data = xr.open_dataset(fileInfo, group=satInfo['group'])
        print(f'[CHECK] satInfo : {satInfo}')
        print(f'[CHECK] dtDayInfo : {dtDayInfo}')
        print(f'[CHECK] fileInfo : {fileInfo}')

        # 관측소 목록에 따른 반복문
        for j, stnInfo in enumerate(sysOpt['stnList']):
            print(f'[CHECK] stnInfo : {stnInfo}')

            # 영역 설정
            sysOpt['roi'] = {}
            sysOpt['roi']['minLon'] = stnInfo['lon'] - sysOpt['res']
            sysOpt['roi']['maxLon'] = stnInfo['lon'] + sysOpt['res']
            sysOpt['roi']['minLat'] = stnInfo['lat'] - sysOpt['res']
            sysOpt['roi']['maxLat'] = stnInfo['lat'] + sysOpt['res']
            sysOpt['roi']['cenLon'] = stnInfo['lon']
            sysOpt['roi']['cenLat'] = stnInfo['lat']

            # 관측소 위경도 및 flag 문턱값 설정
            dataL1 = data.where(
                (sysOpt['roi']['minLon'] <= data[satInfo['var']['lon']]) & (data[satInfo['var']['lon']] <= sysOpt['roi']['maxLon'])
                & (sysOpt['roi']['minLat'] <= data[satInfo['var']['lat']]) & (data[satInfo['var']['lat']] <= sysOpt['roi']['maxLat'])
                & (data[satInfo['var']['flag']] == satInfo['flag']['val'])
            )
            dataL3 = dataL1

            # 관측시간 계산
            obsDateTime = dtDayInfo.strftime('%Y-%m-%d')

            # 관측모드 계산
            obsMode = None
            # OCO2-CO2 또는 OCO2-CO2를 대상으로 관측모드 계산
            if re.search('OCO2-CO2|OCO3-CO2', satType):
                obsData = xr.open_dataset(fileInfo, group=satInfo['groupObs'])[satInfo['var']['obsMode']]

                dataL2 = xr.merge([dataL1, obsData])
                dataL3 = dataL2.where(
                    (dataL2[satInfo['var']['obsMode']] == satInfo['flag']['obsMode'])
                )

            # 값
            val1D = dataL3[satInfo['var']['val']]

            # 자료 개수
            cnt = np.count_nonzero(~np.isnan(val1D))

            if cnt < 1: continue

            # 위경도 정보
            lon1D = dataL3[satInfo['var']['lon']]
            lat1D = dataL3[satInfo['var']['lat']]
            # lon1D = dataL3['vertex_longitude']
            # lat1D = dataL3['vertex_latitude']

            minVal = np.nanmin(val1D)
            maxVal = np.nanmax(val1D)

            mainTitle = f'{satType} ({obsMode}, {stnInfo["name"]}) {obsDateTime}'
            subTitle = f'N = {cnt} / range = {minVal:.1f} ~ {maxVal:.1f}'

            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, mainTitle, sysOpt['shpInfo'])
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # result = makeMapPlot(sysOpt, lon1D, lat1D, val1D, mainTitle, subTitle, saveImg, isRoi=True)
            # print(f'[CHECK] result : {result}')

            # dataL3['xco2']

            aaa = dataL3
            #    sysOpt['roi']['minLon'] = stnInfo['lon'] - sysOpt['res']
            #             sysOpt['roi']['maxLon'] = stnInfo['lon'] + sysOpt['res']
            #             sysOpt['roi']['minLat'] = stnInfo['lat'] - sysOpt['res']
            #             sysOpt['roi']['maxLat'] = stnInfo['lat'] + sysOpt['res']
            #             sysOpt['roi']['cenLon'] = stnInfo['lon']
            #             sysOpt['roi']['cenLat'] = stnInfo['lat']

            # lonList = np.arange(sysOpt['roi']['minLon'], sysOpt['roi']['maxLon'], 0.1)
            # latList = np.arange(sysOpt['roi']['minLat'], sysOpt['roi']['maxLat'], 0.1)
            # print(f'[CHECK] lonList : {lonList}')
            # print(f'[CHECK] latList : {latList}')



            # dataL4 = dataL3.interp(lon=lonList, lat=latList, method='linear')

            dd = dataL3[['longitude', 'latitude', 'vertex_longitude', 'vertex_latitude', 'xco2']]
            dd2 = dd.to_dataframe().reset_index(drop=False).dropna()

            reshaped_df = dd2.pivot(index=['sounding_id', 'longitude', 'latitude', 'xco2'], columns='vertices')
            reshaped_df.columns = [f'{x}_{y}' for x, y in reshaped_df.columns]
            reshaped_df = reshaped_df.reset_index()

            # 37.12205

            # 37.11563,37.13560,37.12841,37.10845
            for check in ind_0_list:
                lat_target = lat_corner[check, [0, 1, 3, 2]].reshape(2, 2)
                lon_target = lon_corner[check, [0, 1, 3, 2]].reshape(2, 2)
                SIF_target = SIF[check]  ###해당 값이 지도로 표출되는 값

                al = ax.pcolormesh(lon_target, lat_target, SIF_target.reshape(1, 1), transform=ccrs.PlateCarree(), cmap='YlOrRd')
                al.set_clim([xco2_min_, xco2_max_])

            # import numpy as np
            # import matplotlib.pyplot as plt
            # import cartopy.crs as ccrs
            # import xarray as xr
            #
            # # Define the coordinates
            # latitudes = np.array([37.11563, 37.13560, 37.12841, 37.10845])
            # longitudes = np.array([127.04228, 127.03479, 127.04388, 127.04980])  # replace with your actual longitudes
            #
            # # Create a 2x2 grid from the coordinates
            # # The x and y coordinates need to be 2D arrays defining the vertices of each grid cell
            # # Since we only have one cell, we can use np.meshgrid to generate these from the min and max coordinates
            # lat_grid, lon_grid = np.meshgrid([latitudes.min(), latitudes.max()],
            #                                  [longitudes.min(), longitudes.max()], indexing='ij')
            #
            # # Define the data values
            # # These also need to be a 2D array
            # # Since we only have one value, we can create an array of that value
            # value = np.array([[411.80087, 411.80087], [411.80087, 411.80087]])  # replace with your actual value
            #
            # # Create an xarray DataArray
            # data = xr.DataArray(data=value, coords=[('latitude', lat_grid[:, 0]), ('longitude', lon_grid[0, :])])
            #
            # # Plot using pcolormesh and cartopy
            # ax = plt.axes(projection=ccrs.PlateCarree())
            # data.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='YlOrRd', add_colorbar=True)
            # ax.set_global()
            # ax.coastlines()
            #
            # plt.show()
            #

            # This is how to get the unique 'xco2' values
            reshaped_df['xco2'] = df.groupby('sounding_id')['xco2'].first().values


            # dd2 = dd.to_dataframe().reset_index(drop=False)
            dd3 = dd2.drop('sounding_id', axis='columns').set_index(['latitude', 'longitude'])
            # dd3 = dd2.drop('sounding_id', axis='columns').set_index(['vertex_latitude', 'vertex_longitude'])
            dd4 = dd3.to_xarray()

            dd4['xco2'].plot()
            plt.show()


            # dataL3['vertices'].values

            ds = dataL3
            # Convert 1D arrays to 2D
            lon, lat = np.meshgrid(dd2.vertex_longitude, dd2.vertex_latitude)

            # Replace nan values with 0
            solar_zenith_angle = dd2.xco2

            # Create the pcolormesh plot
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(lon, lat, solar_zenith_angle, cmap='viridis', shading='auto')
            plt.colorbar(label='Solar Zenith Angle')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Solar Zenith Angle on Earth')
            plt.show()

            # print('asdfasdf')

            # ax = plt.axes(projection=ccrs.PlateCarree())
            # ax.set_extent([sysOpt['roi']['minLon'], sysOpt['roi']['maxLon'], sysOpt['roi']['minLat'], sysOpt['roi']['maxLat']])

            # dataL3
            #
            # val1D.plot(transform=ccrs.PlateCarree(), cmap='Spectral_r')
            # # plt.show()
            #
            # ax.plot(sysOpt['roi']['cenLon'], sysOpt['roi']['cenLat'], marker='*', markersize=10, color='black', linestyle='none', transform=ccrs.PlateCarree())
            #
            # shpFile = '{}*'.format(sysOpt['metaInfo'][sysOpt['shpInfo']]['filePath'])
            # fileList = sorted(glob.glob(shpFile))
            #
            # if len(fileList) > 0:
            #     gdf = gpd.read_file(fileList[3])
            #     gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
            #
            # ax.gridlines(draw_labels=True)
            #
            # plt.ylabel(None)
            # plt.xlabel(None)
            # plt.suptitle(mainTitle)
            # plt.title(subTitle)
            # # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.show()
            # # plt.close()

            # lon_min, lon_max, lat_min, lat_max = extent_area

            fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
            # ax.set_extent([sysOpt['roi']['minLon'], sysOpt['roi']['maxLon'], sysOpt['roi']['minLat'], sysOpt['roi']['maxLat']])

            # Map Gridline and formatting xticks and yticks
            gl = ax.gridlines(draw_labels=False, linewidth=0.1)
            g1 = plt.gca().gridlines(color='dimgrey', linestyle='--', linewidth=0.4)
            line_scale = 0.2  # 간격
            xticks = np.arange(125, 130, line_scale)
            yticks = np.arange(30, 40, line_scale)

            g1.xformatter = LONGITUDE_FORMATTER
            g1.yformatter = LATITUDE_FORMATTER
            g1.xlocator = mticker.FixedLocator(xticks)
            g1.ylocator = mticker.FixedLocator(yticks)
            g1.xlabels_top = True
            g1.ylabels_left = True
            plt.show()

            # Suppose ds is your xarray.Dataset

            # satInfo['var']
            dataL3['']

            # cs = map.scatter(lon2D, lat2D, c=val2D, s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
            # plt.plot(sysOpt['roi']['cenLon'], sysOpt['roi']['cenLat'], marker='*', markersize=10, color='black', linestyle='none')

            # 추출한 픽셀 인덱스에 대해서 루프 실행 : 루프 하나마다 pcolormesh로 색칠하는 구조
            # for check in ind_0_list:
            #     lat_target = lat_corner[check, [0, 1, 3, 2]].reshape(2, 2)
            #     lon_target = lon_corner[check, [0, 1, 3, 2]].reshape(2, 2)
            #     SIF_target = SIF[check]  ###해당 값이 지도로 표출되는 값
            #
            #     al = ax.pcolormesh(lon_target, lat_target, SIF_target.reshape(1, 1), transform=ccrs.PlateCarree(), cmap='YlOrRd')
            #     al.set_clim([xco2_min_, xco2_max_])
            #
            # smaplegend = plt.colorbar(al, ticks=np.arange(xco2_min_, xco2_max_, 0.1), extend='both')
            # ax.scatter(stn_lon_list[stn_ind], stn_lat_list[stn_ind], c='r', s=200, marker='*')  # Center of site Location
            #
            # # Title or legend value
            # smaplegend.set_label('SIF 740nm', fontsize=15)
            # plt.title('OCO-3 SIF ' + str(stn_namelist[stn_ind]) + ' ' + str(target_list[stn_ind]), fontsize=14)
            # # plt.text(.0001, .0001, 'range = ('+str(round(float(xco2_min),3))+', '+str(round(float(xco2_max),3))+')', ha='left', va='top', transform=ax.transAxes, fontsize = 12)
            #
            # # Shapefile boundaries
            # shapefile = "/gadm36_KOR_1.shp"  # shapefile 시도단위
            # shapefile_1 = "/gadm36_KOR_2.shp"  # shapefile 시군구 단위
            # shp1 = gpd.read_file(shapefile)
            # shp2 = gpd.read_file(shapefile_1)
            # shp1.plot(ax=ax, color='None', edgecolor='black', linewidth=3)
            # shp2.plot(ax=ax, color='None', edgecolor='black')
            #
            # # 그림 저장, 용량이 크면 dpi=200으로 변환
            # fig.savefig(graph_path + 'OCO-3_SIF_' + str(stn_namelist[stn_ind]) + '_' + str(target_list[stn_ind]) + '.png', bbox_inches='tight', dpi=400)
