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
        # plt.show()
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

# globalVar['inpPath'] = '/DATA/INPUT'
# globalVar['outPath'] = '/DATA/OUTPUT'
# globalVar['figPath'] = '/DATA/FIG'

# ****************************************************************************
# 초기 전달인자 설정
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
    # 'srtDate': '2022-07-01'
    # , 'endDate': '2022-12-31'
    'srtDate': globalVar['srtDate']
    , 'endDate': globalVar['endDate']

    # 위성 목록
    # , 'satList': ['OCO2-CO2']
    # , 'satList': ['OCO3-CO2']
    # , 'satList': ['OCO2-SIF']
    # , 'satList': ['OCO3-SIF']
    # , 'satList': ['TROPOMI']
    # , 'satList': ['OCO2-CO2', 'OCO3-CO2', 'OCO2-SIF', 'OCO3-SIF', 'TROPOMI']
    , 'satList': [globalVar['satList']]

    # shp 파일 정보
    # , 'shpInfo': 'gadm36_KOR_0'
    # , 'shpInfo': 'gadm36_KOR_1'
    # , 'shpInfo': 'gadm36_KOR_2'
    , 'shpInfo': globalVar['shpInfo']

    # ****************************************************************************
    # 정적 설정
    # ****************************************************************************
    # 중심 반경
    # 2도 = 약 200 km
    , 'res': 0.8

    # 관측모드 설정
    , 'obsModeList': {0: 'Nadir', 1: 'Glint', 2: 'Target', 3: 'Transition'}

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
        }
        , 'OCO2-SIF': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtSIF_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sequences'
            , 'var': {
                'id': 'sounding_dim', 'lon': 'Longitude', 'lat': 'Latitude', 'flag': 'Quality_Flag', 'val': 'SIF_740nm', 'obsMode': 'SequencesMode'
            }
        }
        , 'OCO3-CO2': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco3_LtCO2_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sounding'
            , 'var': {
                'id': 'sounding_id', 'lon': 'longitude', 'lat': 'latitude', 'flag': 'xco2_quality_flag', 'val': 'xco2', 'obsMode': 'operation_mode'
            }
        }
        , 'OCO3-SIF': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtSIF_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'groupObs': 'Sequences'
            , 'var': {
                'id': 'sounding_dim', 'lon': 'Longitude', 'lat': 'Latitude', 'flag': 'Quality_Flag', 'val': 'SIF_740nm', 'obsMode': 'SequencesMode'
            }
        }
        , 'TROPOMI': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'S5P_RPRO_L2__NO2____%Y%m%dT*.nc'
            , 'group': 'PRODUCT'
            , 'groupObs': 'Sounding'
            , 'var': {
                'id': '', 'lon': 'longitude', 'lat': 'latitude', 'flag': 'qa_value', 'val': 'nitrogendioxide_tropospheric_column', 'obsMode': 'operation_mode'
            }
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
                & (data[satInfo['var']['flag']] == 0)
            )

            # 값
            val1D = dataL1[satInfo['var']['val']]

            # 자료 개수
            cnt = np.count_nonzero(~np.isnan(val1D))

            if cnt < 1: continue

            # ************************************************************************
            # 관측 모드 및 관측 시간 계산
            # ************************************************************************
            obsMode = None
            obsDateTime = None

            try:
                if re.search('OCO2-SIF|OCO3-SIF', satType):
                    idxData = val1D.reset_index(drop=True).reset_index(drop=False)
                    idxDataL1 = idxData.dropna()

                    obsData = xr.open_dataset(fileInfo, group=satInfo['groupObs'])
                    obsDataL1 = obsData[satInfo['var']['obsMode']].values
                    obsMode = pd.DataFrame(obsDataL1).mode().iloc[0, 0]

                    obsDateTime = pd.to_datetime(data['Delta_Time'], unit='s', origin='1990-01-01')[idxDataL1['index']].mean().strftime('%Y-%m-%d %H:%M:%S')

                elif re.search('OCO2-CO2|OCO3-CO2', satType):
                    idxData = data[satInfo['var']['id']].to_dataframe().reset_index(drop=True).reset_index(drop=False)
                    idxDataL1 = idxData[idxData[satInfo['var']['id']].isin(dataL1[satInfo['var']['id']].values)].reset_index(drop=True)

                    obsData = xr.open_dataset(fileInfo, group=satInfo['groupObs'])
                    obsDataL1 = obsData[satInfo['var']['obsMode']].sel(sounding_id=idxDataL1['index']).values

                    # 가장 최빈값 가져오기
                    obsDataL2 = int(stats.mode(obsDataL1, keepdims=False).mode)
                    obsMode = sysOpt['obsModeList'][obsDataL2]

                    obsDateTime = pd.to_datetime(str(int(np.nanmedian(idxDataL1[satInfo['var']['id']]))), format='%Y%m%d%H%M%S%f').strftime('%Y-%m-%d %H:%M:%S')

                else:
                    idxData = val1D.to_dataframe().reset_index(drop=False).reset_index(drop=False)
                    idxDataL1 = idxData.dropna()

                    scanList = list(set(idxDataL1['scanline']))
                    obsDateTime = pd.to_datetime(data['time_utc'].sel(scanline=scanList).to_dataframe().reset_index(drop=True)['time_utc']).mean().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(f'Exception : {e}')

            # 위경도 정보
            lon1D = dataL1[satInfo['var']['lon']]
            lat1D = dataL1[satInfo['var']['lat']]

            minVal = np.nanmin(val1D)
            maxVal = np.nanmax(val1D)

            mainTitle = f'{satType} ({obsMode}, {stnInfo["name"]}) {obsDateTime}'
            subTitle = f'N = {cnt} / range = {minVal:.1f} ~ {maxVal:.1f}'

            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, mainTitle, sysOpt['shpInfo'])
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            result = makeMapPlot(sysOpt, lon1D, lat1D, val1D, mainTitle, subTitle, saveImg, isRoi=True)
            print(f'[CHECK] result : {result}')
