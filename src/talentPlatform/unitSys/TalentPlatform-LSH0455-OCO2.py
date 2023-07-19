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


# ============================================
# 요구사항
# ============================================
# Python을 이용한 온실가스 위성 (OCO2, OCO3, TROPOMI) 자료 처리 및 시각화

# ============================================
# 보조
# ============================================
def makeMapPlot(sysOpt, lon2D, lat2D, val2D, mainTitle, saveImg, isLandUse=False, isKor=False):
    result = None

    plt.figure(dpi=600)

    if (isKor):
        map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c', llcrnrlon=sysOpt['roi']['ko']['minLon'],
                      urcrnrlon=sysOpt['roi']['ko']['maxLon'], llcrnrlat=sysOpt['roi']['ko']['minLat'],
                      urcrnrlat=sysOpt['roi']['ko']['maxLat'])
    else:
        map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c')

    # 육해상 분류
    if (isLandUse):
        makePlot = sysOpt['data']['landUse']['band_data'].plot.contourf(levels=np.linspace(1, 17, 17),
                                                                        add_colorbar=False)
        cbar = map.colorbar(makePlot, ticks=np.linspace(1, 17, 17))
        cbar.set_ticklabels(sysOpt['data']['landList'])

    # 관측소 지점
    for j, stnInfo in enumerate(sysOpt['data']['stnList']):
        plt.plot(stnInfo['lon'], stnInfo['lat'], markersize=3, marker='o', color='red')
        plt.annotate(text=stnInfo['abbr'], size=3, xy=(stnInfo['lon'], stnInfo['lat']), color='black')

    cs = map.scatter(lon2D, lat2D, c=val2D, s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))

    map.drawcoastlines()
    map.drawmapboundary()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')
    # map.drawmeridians(range(-180, 180, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
    # map.drawparallels(range(-90, 90, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
    map.drawmeridians(range(-180, 180, 5), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
    map.drawparallels(range(-90, 90, 2), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])

    plt.ylabel(None, fontsize=15, labelpad=35)
    plt.xlabel(None, fontsize=15, labelpad=20)
    # cbar2 = map.colorbar(cs, orientation='horizontal', location='right')
    # cbar2 = map.colorbar(cs, orientation='horizontal')
    cbar2 = plt.colorbar(cs, orientation='horizontal')
    cbar2.set_label(None, fontsize=13)
    plt.title(mainTitle, fontsize=15)
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


# ============================================
# 주요
# ============================================
serviceName = 'LSH0455'

# 입력, 출력, 그림 파일 정보
globalVar = {
    'inpPath': '/DATA/INPUT'
    , 'outPath': '/DATA/OUTPUT'
    , 'figPath': '/DATA/FIG'
}

# globalVar['inpPath'] = '/DATA/INPUT'
# globalVar['outPath'] = '/DATA/OUTPUT'
# globalVar['figPath'] = '/DATA/FIG'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2022-07-01'
    , 'endDate': '2022-12-31'

    # 영역 설정 시 해상도
    # 2도 = 약 200 km
    , 'res': 2

    # 설정 정보
    # , 'data': {
    #     'landList': None
    #     , 'landUse': None
    #     , 'stnList': None
    # }

    # 특정 월 선택
    # , 'selMonth': [3, 4, 5, 12]

    # 특정 시작일/종료일 선택
    # , 'selSrtDate': '2018-01-21'
    # , 'selEndDate': '2018-12-24'

    # 관심영역 설정
    , 'roi': {
        'ko': {
            'minLon': 120
            , 'maxLon': 150
            , 'minLat': 30
            , 'maxLat': 40
        }
    }

    , 'satList': ['OCO2']
    # , 'satList': ['OCO3']
    # , 'satList': ['TROPOMI']

    , 'metaInfo': {
        'OCO2': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtCO2_%y%m%d_B*_*.nc4'
            , 'varList': {
                'id': 'sounding_id'
                , 'lon': 'longitude'
                , 'lat': 'latitude'
                , 'flag': 'xco2_quality_flag'
                , 'val': 'xco2'
            }
        }
        , 'OCO3': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco3_LtCO2_%y%m%d_B*_*.nc4'
            , 'selVar': ['XCO2']
        }
        , 'TROPOMI': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'S5P_RPRO_L2__NO2____%Y%m%dT*.nc'
            , 'selVar': ['NO2']
        }
    }
}

# ****************************************************************************
# 설정 정보
# ****************************************************************************
# 육/해상 분류
# sysOpt['data']['landList'] = [
#         'water'
#         , 'evergreen needleleaf forest'
#         , 'evergreen broadleaf forest'
#         , 'deciduous needleleaf forest'
#         , 'deciduous broadleaf forest'
#         , 'mixed forests'
#         , 'closed shrubland'
#         , 'open shrublands'
#         , 'woody savannas'
#         , 'savannas'
#         , 'grasslands'
#         , 'permanent wetlands'
#         , 'croplands'
#         , 'urban and built-up'
#         , 'cropland/natural vegetation mosaic'
#         , 'snow and ice'
#         , 'barren or sparsely vegetated'
# ]

# 영역 설정
# sysOpt['data']['stnList'] = [
#         {'name': 'Seoul_SNU', 'abbr': 'SNU', 'lat': 37.458, 'lon': 126.951}
#         , {'name': 'Korea_University', 'abbr': 'OKU', 'lat': 37.585, 'lon': 127.025}
#         , {'name': 'KORUS_Iksan', 'abbr': 'Iksan', 'lat': 35.962, 'lon': 127.005}
#         , {'name': 'KORUS_NIER', 'abbr': 'NIER', 'lat': 37.569, 'lon': 126.640}
#         , {'name': 'KORUS_Taehwa', 'abbr': 'Taehwa', 'lat': 37.312, 'lon': 127.310}
#         , {'name': 'KORUS_UNIST_Ulsan', 'abbr': 'UNIST', 'lat': 35.582, 'lon': 129.190}
#         , {'name': 'KORUS_Olympic_Park', 'abbr': 'Olympic', 'lat': 37.522, 'lon': 127.124}
#         , {'name': 'PKU_PEK', 'abbr': 'PKU', 'lat': 39.593, 'lon': 116.184}
#         , {'name': 'Anmyon', 'abbr': 'Anmyon', 'lat': 36.539, 'lon': 126.330}
#         , {'name': 'Incheon', 'abbr': 'Incheon', 'lat': 37.569, 'lon': 126.637}
#         , {'name': 'KORUS_Songchon', 'abbr': 'Songchon', 'lat': 37.338, 'lon': 127.489}
#         , {'name': 'KORUS_Mokpo_NU', 'abbr': 'MNU', 'lat': 34.913, 'lon': 126.437}
#         , {'name': 'KORUS_Daegwallyeong', 'abbr': 'Daegwallyeong', 'lat': 37.687, 'lon': 128.759}
#         , {'name': 'KIOST_Ansan', 'abbr': 'KIOST', 'lat': 37.286, 'lon': 126.832}
#         , {'name': 'KORUS_Baeksa', 'abbr': 'Baeksa', 'lat': 37.412, 'lon': 127.569}
#         , {'name': 'KORUS_Kyungpook_NU', 'abbr': 'KNU', 'lat': 35.890, 'lon': 128.606}
#         , {'name': 'Gosan_NIMS_SNU', 'abbr': 'NIMS', 'lat': 33.300, 'lon': 126.206}
# ]

# ****************************************************************************
# 시작/종료일 설정
# ****************************************************************************
dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

# ****************************************************************************
# 자료 처리
# ****************************************************************************
for dtDayIdx, dtDayInfo in enumerate(dtDayList):
    # print(f'[CHECK] dtDayInfo : {dtDayInfo}')

    for satIdx, satType in enumerate(sysOpt['satList']):
        # print(f'[CHECK] satInfo : {satInfo}')

        satInfo = sysOpt['metaInfo'][satType]

        inpFile = '{}/{}'.format(satInfo['filePath'], satInfo['fileName'])
        inpFileDate = dtDayInfo.strftime(inpFile)
        fileList = sorted(glob.glob(inpFileDate))

        if fileList is None or len(fileList) < 1:
            # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
            continue

        # NetCDF 파일 읽기
        fileInfo = fileList[0]
        # data = xr.open_dataset(fileInfo)
        data = xr.open_dataset(fileInfo, engine='pynio')
        print(f'[CHECK] fileInfo : {fileInfo}')

        satInfo['varList']['lon']
        satInfo['varList'][1]
        satInfo['varList'][2]
        satInfo['varList'][3]
        satInfo['varList'][4]

        print(data)
        data['PRODUCT/nitrogendioxide_tropospheric_column']
        data['PRODUCT/latitude']
        data['PRODUCT/longitude']
        data['PRODUCT/qa_value']

        # 위경도 및 Flag 마스킹
        selData = data.where(
            (sysOpt['roi']['ko']['minLon'] <= data[satInfo['varList'][1]]) & (
                        data[satInfo['varList'][1]] <= sysOpt['roi']['ko']['maxLon'])
            & (sysOpt['roi']['ko']['minLat'] <= data[satInfo['varList'][2]]) & (
                        data[satInfo['varList'][2]] <= sysOpt['roi']['ko']['maxLat'])
            # & (data['xco2_quality_flag'] == 0)
        ).dropna(dim=satInfo['varList'][0], how='any')

        selData = data
        #
        # if (selData['xco2'].size < 1): continue

        # selData['xco2_quality_flag'].values

        # plt.scatter(data['longitude'], data['latitude'], c=data['xco2'], s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        # plt.scatter(selData['longitude'], selData['latitude'], c=selData['xco2'], s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        plt.scatter(selData[satInfo['varList'][1]], selData[satInfo['varList'][2]], c=selData[satInfo['varList'][4]],
                    s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        plt.scatter(selData['PRODUCT/longitude'], selData['PRODUCT/latitude'], c=selData['PRODUCT/qa_value'], s=10,
                    marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        plt.colorbar()
        plt.show()

        nx1D = data['sounding_id'].values
        lat1D = data['latitude'].values
        lon1D = data['longitude'].values
        time1D = data['time'].values
        val1D = data['xco2'].values
        flag1D = data['xco2_quality_flag'].values

        dataL1 = xr.Dataset(
            {
                'xco2': (('time', 'nx'), (val1D).reshape(1, len(nx1D)))
                , 'xco2Flag': (('time', 'nx'), (flag1D).reshape(1, len(nx1D)))
                , 'lon': (('time', 'nx'), (lon1D).reshape(1, len(nx1D)))
                , 'lat': (('time', 'nx'), (lat1D).reshape(1, len(nx1D)))
                , 'scanTime': (('time', 'nx'), (time1D).reshape(1, len(nx1D)))
            }
            , coords={
                'time': pd.date_range(dtDayInfo, periods=1)
                , 'nx': nx1D
            }
        )

        # # Flag 마스킹
        # gosatDataL2 = gosatDataL2.where((gosatDataL2['xco2Flag'] == 0), drop=True)
        #

        # selData = gosatDataL1.where(
        #     (sysOpt['roi']['ko']['minLon'] <= gosatDataL1['lon'])
        #     & (gosatDataL1['lon'] <= sysOpt['roi']['ko']['maxLon'])
        #     & (sysOpt['roi']['ko']['minLat'] <= gosatDataL1['lat'])
        #     & (gosatDataL1['lat'] <= sysOpt['roi']['ko']['maxLat'])
        # ).dropna(dim='nx', how='any')
        #
        # if (selData['xco2'].size < 1): continue
        #
        # gosatDataL2 = xr.merge([gosatDataL2, selData])
        # #        print(gosatDataL2)

# dtDayInfo = dtDayList[356]
# dataL2 = xr.Dataset()
# for i, dtDayInfo in enumerate(dtDayList):
#     # dtYmd = dtDayInfo.strftime('%Y%m%d')
#     dt2Ymd = dtDayInfo.strftime('%y%m%d')
#
# #       inpFile = '{}/{}/{}/oco2_*_{}*_*.nc4'.format(globalVar['inpPath'], serviceName, 'OCO2', dt2Ymd)
# # 파일이름 패턴 조회, 파일유무와 상관없음
# inpFile = '{}/oco2_*_{}*_*.nc4'.format(globalVar['inpPath1'], dt2Ymd)
# # print(inpFile)
# # 실제 파일 이름 출력
# fileList = sorted(glob.glob(inpFile))
# # print(fileList)
#
# if (fileList is None) or (len(fileList) < 1):
#     continue
# print("[CHECK] dtDayInfo : {}".format(dtDayInfo))
#
# gosatData = xr.open_dataset(fileList[0])
# # print(gosatData)
#
# # tmpData = gosatData[['latitude', 'longitude', 'time', 'xco2', 'xco2_quality_flag']].to_dataframe().reset_index()
#
# nx1D = gosatData['sounding_id'].values
# lat1D = gosatData['latitude'].values
# lon1D = gosatData['longitude'].values
# time1D = gosatData['time'].values
# val1D = gosatData['xco2'].values
# flag1D = gosatData['xco2_quality_flag'].values
#
# gosatDataL1 = xr.Dataset(
#     {
#         'xco2': (('time', 'nx'), (val1D).reshape(1, len(nx1D)))
#         , 'xco2Flag': (('time', 'nx'), (flag1D).reshape(1, len(nx1D)))
#         , 'lon': (('time', 'nx'), (lon1D).reshape(1, len(nx1D)))
#         , 'lat': (('time', 'nx'), (lat1D).reshape(1, len(nx1D)))
#         , 'scanTime': (('time', 'nx'), (time1D).reshape(1, len(nx1D)))
#     }
#     , coords={
#         'time': pd.date_range(dtDayInfo, periods=1)
#         , 'nx': nx1D
#         # 'time' : pd.to_datetime(time1D)
#     }
# )
#
# selData = gosatDataL1.where(
#     (sysOpt['roi']['ko']['minLon'] <= gosatDataL1['lon'])
#     & (gosatDataL1['lon'] <= sysOpt['roi']['ko']['maxLon'])
#     & (sysOpt['roi']['ko']['minLat'] <= gosatDataL1['lat'])
#     & (gosatDataL1['lat'] <= sysOpt['roi']['ko']['maxLat'])
# ).dropna(dim='nx', how='any')
#
# if (selData['xco2'].size < 1): continue
#
# gosatDataL2 = xr.merge([gosatDataL2, selData])
# #        print(gosatDataL2)
#
# # Flag 마스킹
# gosatDataL2 = gosatDataL2.where((gosatDataL2['xco2Flag'] == 0), drop=True)
#
# # CSV 파일 생성
# timeList = gosatDataL2['time'].values
# csvData = gosatDataL2.to_dataframe().reset_index()[['time', 'lon', 'lat', 'scanTime', 'xco2', 'xco2Flag']]
# saveCsvFile = '{}/OCO2-PROP-{}-{}.csv'.format(globalVar['outPath'], pd.to_datetime(timeList.min()).strftime('%Y%m%d'),
#                                               pd.to_datetime(timeList.max()).strftime('%Y%m%d'))
# os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
# csvData.to_csv(saveCsvFile, index=False)
# print('[CHECK] saveCsvFile : {}'.format(saveCsvFile))
#
# # ****************************************************************************
# # 특정일을 기준으로 시각화
# # 특정 월을 기준으로 시각화
# # 특정 시작일/종료일을 기준으로 시각화
# # ****************************************************************************
# # 특정일을 기준으로 시각화
# timeList = gosatDataL2['time'].values
# timeInfo = timeList[1]
# for i, timeInfo in enumerate(timeList):
#     print("[CHECK] timeInfo : {}".format(timeInfo))
#
#     dsData = gosatDataL2.sel(time=timeInfo)
#
#     # 스캔 영역
#     # mainTitle = 'OCO2-Day-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
#     # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
#     # makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=False)
#
#     # 스캔 영역 + 육해상 분류
#     # mainTitle = 'OCO2-Day-LandUse-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
#     # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
#     # makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=True)
#
#     dsDataL1 = dsData.where(
#         (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
#         & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
#         & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
#         & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
#     ).to_dataframe().reset_index()[['lon', 'lat', 'xco2']].dropna()
#
#     if len(dsDataL1['xco2']) > 0:
#         mainTitle = 'OCO2-Day-Kor-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
#         saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#         makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False,
#                     isKor=True)
#
#         # 스캔 영역 + 육해상 분류
#         # mainTitle = 'OCO2-Day-Kor-LandUse-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
#         # saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#         # makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True)
#
# # 특정 월을 기준으로 시각화
# selData = gosatDataL2.copy().sel(time=gosatDataL2.time.dt.month.isin(sysOpt['selMonth']))
# if (len(selData['time'])):
#     selDataL2 = selData.groupby('time.month').mean(skipna=True)
#
#     monthList = selDataL2['month'].values
#     monthInfo = monthList[0]
#     for i, monthInfo in enumerate(monthList):
#         print("[CHECK] monthInfo : {}".format(monthInfo))
#
#         dsData = selDataL2.sel(month=monthInfo)
#
#         dsDataL1 = dsData.where(
#             (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
#             & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
#             & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
#             & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
#         ).to_dataframe().reset_index()[['lon', 'lat', 'xco2']].dropna()
#
#         if len(dsDataL1['xco2']) > 0:
#             mainTitle = 'OCO2-Month-Kor-{:02d}'.format(monthInfo)
#             saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#             makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False,
#                         isKor=True)
#
#             # 스캔 영역 + 육해상 분류
#             # mainTitle = 'OCO2-Month-Kor-LandUse-{:02d}'.format(monthInfo)
#             # saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#             # makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True, isKor=True)
#
# # 특정 시작일/종료일을 기준으로 시각화
# selData = gosatDataL2.copy().sel(time=slice(sysOpt['selSrtDate'], sysOpt['selEndDate']))
# if (len(selData['time'])):
#
#     dsData = selData.to_dataframe().reset_index().dropna().groupby(by=['lon', 'lat'], dropna=False).mean().reset_index()
#
#     dsDataL1 = dsData.loc[
#         (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
#         & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
#         & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
#         & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
#         ].reset_index()[['lon', 'lat', 'xco2']].dropna()
#
#     if len(dsDataL1['xco2']) > 0:
#         # mainTitle = 'OCO2-All-Kor-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
#         # saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#         # makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False, isKor=True)
#
#         # 스캔 영역 + 육해상 분류
#         mainTitle = 'OCO2-All-Kor-LandUse-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'),
#                                                         pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
#         saveImg = '{}/{}.png'.format(globalVar['figPath'], mainTitle)
#         makeMapPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True,
#                     isKor=True)
