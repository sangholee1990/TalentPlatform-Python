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

    , 'metaInfo': {
        'OCO2': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'oco2_LtCO2_%y%m%d_B*_*.nc4'
            , 'group': None
            , 'var': {
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
            , 'group': None
            , 'var': {
                'id': 'sounding_id'
                , 'lon': 'longitude'
                , 'lat': 'latitude'
                , 'flag': 'xco2_quality_flag'
                , 'val': 'xco2'
            }
        }
        , 'TROPOMI': {
            'filePath': '/DATA/INPUT/LSH0455'
            , 'fileName': 'S5P_RPRO_L2__NO2____%Y%m%dT*.nc'
            , 'group': 'PRODUCT'
            , 'var': {
                'id': 'sounding_id'
                , 'lon': 'longitude'
                , 'lat': 'latitude'
                , 'flag': 'qa_value'
                , 'val': 'nitrogendioxide_tropospheric_column'
            }
        }
    }


    # 수행 목록
    , 'satList': ['OCO2']
    # , 'satList': ['OCO3']
    # , 'satList': ['TROPOMI']

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
            continue

        # NetCDF 파일 읽기
        fileInfo = fileList[0]
        print(f'[CHECK] fileInfo : {fileInfo}')

        data = xr.open_dataset(fileInfo, group=satInfo['group'])
        print(f'[CHECK] satInfo : {satInfo}')

        soundData = xr.open_dataset(fileInfo, group='Sounding')
        # soundData['operation_mode'].plot()
        # plt.show()


        # indices = list(range(len(data['sounding_id'].values)))

        # indices = np.where(np.isin(data['sounding_id'].values, values_to_find))[0]


        # data['operation_mode']

        selData[satInfo['var']['id']].values


        idxData = data[satInfo['var']['id']].to_dataframe().reset_index(drop=True).reset_index(drop=False)
        idxDataL1 = idxData[idxData['sounding_id'].isin(selData[satInfo['var']['id']].values)].reset_index(drop=True)

        optData = soundData['operation_mode'].sel(sounding_id =  idxDataL1['index']).values

        # calculate mode
        mode = stats.mode(optData)

        mode.mode[0]

        # create a dictionary to map numbers to categories
        category_dict = {0: "Nadir", 1: "Glint", 2: "Target", 3: "Transition"}

        # get the category that corresponds to the mode
        mode_category = category_dict[mode]



        cateDict = {0.0: "Nadir", 1.0: "Glint", 2.0: "Target", 3.0: "Transition"}
        # categories = [cateDict[val] for val in optData]

        from collections import Counter

        # mode = stats.mode(categories)

        # arr = np.vectorize(cateDict.get)(optData)
        counter = Counter(optData)
        most_common_category = counter.most_common(1)
        # mode = stats.mode(arr)
        most_common_category[0][0]
        categories = [cateDict[val] for val in most_common_category]


        # 위경도 및 Flag 마스킹
        selData = data.where(
            (sysOpt['roi']['ko']['minLon'] <= data[satInfo['var']['lon']]) & (data[satInfo['var']['lon']] <= sysOpt['roi']['ko']['maxLon'])
            & (sysOpt['roi']['ko']['minLat'] <= data[satInfo['var']['lat']]) & (data[satInfo['var']['lat']] <= sysOpt['roi']['ko']['maxLat'])
            # & (data['xco2_quality_flag'] == 0)
        ).dropna(dim=satInfo['var']['id'], how='any')

        if (selData[satInfo['var']['val']].size < 1): continue

        plt.scatter(selData[satInfo['var']['lon']], selData[satInfo['var']['lat']], c=selData[satInfo['var']['val']], s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        plt.colorbar()
        plt.show()

        # nx1D = data['sounding_id'].values
        # lat1D = data['latitude'].values
        # lon1D = data['longitude'].values
        # time1D = data['time'].values
        # val1D = data['xco2'].values
        # flag1D = data['xco2_quality_flag'].values

        # dataL1 = xr.Dataset(
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
        #     }
        # )

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
