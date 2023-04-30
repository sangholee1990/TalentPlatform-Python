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

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램 :부 프로그램을 호출
# 4. 부 프로그램 : 자료 처리를 위한 클래스로서 내부 함수 (초기 변수, 비즈니스 로직, 수행 프로그램 설정)
# 4.1. 환경 변수 설정 (로그 설정) : 로그 기록을 위한 설정 정보 읽기
# 4.2. 환경 변수 설정 (초기 변수) : 입력 경로 (inpPath) 및 출력 경로 (outPath) 등을 설정
# 4.3. 초기 변수 (Argument, Option) 설정 : 파이썬 실행 시 전달인자 설정 (pyhton3 *.py argv1 argv2 argv3 ...)
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리 또는 비즈니스 로직 구현

# =================================================
# 1. 초기 설정
# =================================================

warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 맵 시각화
def makeMapPlot(sysOpt, lon2D, lat2D, val2D, mainTitle, saveImg, isLandUse=False):

    print('[START] {}'.format('makeMapPlot'))

    result = None

    try:
        # import cartopy.crs as crs
        # import cartopy.feature as cfeature
        # from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
        # # proj = ccrs.PlateCarree()
        #
        #
        # mapProj = crs.LambertCylindrical(central_longitude=0.0)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
        #
        # # 육해상 분류
        # # if (isLandUse):
        # #     makePlot = sysOpt['data']['landUse']['band_data'].plot.contourf(levels=np.linspace(1, 17, 17), add_colorbar=False)
        # #     cbar = plt.colorbar(makePlot, ticks=np.linspace(1, 17, 17))
        # #     cbar.set_ticklabels(sysOpt['data']['landList'])
        # #     cbar.set_label(None, fontsize=11)
        #
        # cs = plt.scatter(lon2D, lat2D, c=val2D, s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))
        # plt.colorbar(cs, orientation='horizontal', ticklocation='auto')
        #
        # # ax.stock_img()
        # # ax.add_feature(cfeature.GSHHSFeature(edgecolor='k'))
        # ax.coastlines()
        # # ax.gridlines(xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 30), draw_labels=True, color="black", linestyle="dotted")
        # ax.gridlines(xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 30), color="black", linestyle="dotted")
        # ax.set_title(mainTitle)
        #
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        plt.figure(dpi=600)
        # map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c', llcrnrlon=120, llcrnrlat=30, urcrnrlon=150, urcrnrlat=40)
        map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c')

        # 육해상 분류
        if (isLandUse):
            makePlot = sysOpt['data']['landUse']['band_data'].plot.contourf(levels=np.linspace(1, 17, 17), add_colorbar=False)
            cbar = map.colorbar(makePlot, ticks=np.linspace(1, 17, 17))
            cbar.set_ticklabels(sysOpt['data']['landList'])

        # 관측소 지점
        for j, stnInfo in enumerate(sysOpt['data']['stnList']):
            map.text(stnInfo['lon'], stnInfo['lat'], stnInfo['name'])


        cs = map.scatter(lon2D, lat2D, c=val2D, s=10, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))

        map.drawcoastlines()
        map.drawmapboundary()
        map.drawcountries(linewidth=1, linestyle='solid', color='k')
        map.drawmeridians(range(-180, 180, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
        # map.drawmeridians(range(-180, 180, 5), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        # map.drawparallels(range(-90, 90, 2), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])

        plt.ylabel(None, fontsize=15, labelpad=35)
        plt.xlabel(None, fontsize=15, labelpad=20)
        # cbar2 = map.colorbar(cs, orientation='horizontal', location='right')
        # cbar2 = map.colorbar(cs, orientation='horizontal')
        cbar2 = plt.colorbar(cs, orientation='horizontal')
        cbar2.set_label(None, fontsize=13)
        plt.title(mainTitle, fontsize=15)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        print("Exception : {}".format(e))
        return result
    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        print('[END] {}'.format('makeMapPlot'))

# 맵 시각화
def makeMapKorPlot(sysOpt, lon2D, lat2D, val2D, mainTitle, saveImg, isLandUse=False):

    print('[START] {}'.format('makeMapKorPlot'))

    result = None

    try:
        plt.figure(dpi=600)
        map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c', llcrnrlon=120, llcrnrlat=30, urcrnrlon=150, urcrnrlat=40)
        # map = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0, resolution='c')

        # 육해상 분류
        if (isLandUse):
            makePlot = sysOpt['data']['landUse']['band_data'].plot.contourf(levels=np.linspace(1, 17, 17), add_colorbar=False)
            cbar = map.colorbar(makePlot, ticks=np.linspace(1, 17, 17))
            cbar.set_ticklabels(sysOpt['data']['landList'])

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
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        print("Exception : {}".format(e))
        return result
    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        print('[END] {}'.format('makeMapKorPlot'))


if __name__ == '__main__':

    global prjName, serviceName, globalVar

    prjName = 'test'
    serviceName = 'LSH0359'
    globalVar = {}

    if (platform.system() == 'Windows'):

        # 옵션 설정
        sysOpt = {
            # 시작/종료 시간
            # 'srtDate': globalVar['srtDate']
            # , 'endDate': globalVar['endDate']
            'srtDate': '2018-01-01'
            , 'endDate': '2018-12-31'

            # 영역 설정 시 해상도
            # 2도 = 약 200 km
            , 'res' : 2

            # 특정 월 선택
            , 'selMonth' : [3, 4, 5, 12]

            # 특정 시작일/종료일 선택
            , 'selSrtDate' : '2018-04-21'
            , 'selEndDate' : '2018-12-24'

            , 'data' : {
                'landList' : None
                , 'landUse' : None
                , 'stnList' : None
            }

            # 관심영역 설정
            , 'roi' : {
                'ko' : {
                    'minLon' : 120
                    , 'maxLon' : 150
                    , 'minLat' : 30
                    , 'maxLat' : 40
                }
            }
        }

    else:

        # 옵션 설정
        sysOpt = {
            # 시작/종료 시간
            # 'srtDate': globalVar['srtDate']
            # , 'endDate': globalVar['endDate']
            'srtDate': '2018-01-01'
            , 'endDate': '2018-12-31'

            # 영역 설정 시 해상도
            # 2도 = 약 200 km
            , 'res' : 2

            # 특정 월 선택
            , 'selMonth' : [3, 4, 5, 12]

            # 특정 시작일/종료일 선택
            , 'selSrtDate' : '2018-01-21'
            , 'selEndDate' : '2018-12-24'

            # 설정 정보
            , 'data' : {
                'landList' : None
                , 'landUse' : None
                , 'stnList' : None
            }

            # 관심영역 설정
            , 'roi' : {
                'ko' : {
                    'minLon' : 120
                    , 'maxLon' : 150
                    , 'minLat' : 30
                    , 'maxLat' : 40
                }
            }
        }

    globalVar['inpPath'] = '/DATA/INPUT'
    globalVar['outPath'] = '/DATA/OUTPUT'
    globalVar['figPath'] = '/DATA/FIG'

    # ****************************************************************************
    # 설정 정보
    # ****************************************************************************
    # 육/해상 분류
    sysOpt['data']['landList'] = [
        'water'
        , 'evergreen needleleaf forest'
        , 'evergreen broadleaf forest'
        , 'deciduous needleleaf forest'
        , 'deciduous broadleaf forest'
        , 'mixed forests'
        , 'closed shrubland'
        , 'open shrublands'
        , 'woody savannas'
        , 'savannas'
        , 'grasslands'
        , 'permanent wetlands'
        , 'croplands'
        , 'urban and built-up'
        , 'cropland/natural vegetation mosaic'
        , 'snow and ice'
        , 'barren or sparsely vegetated'
    ]

    # 영역 설정
    sysOpt['data']['stnList'] = [
        {'name': 'Seoul_SNU', 'lat': 37.458, 'lon': 126.951}
        , {'name': 'Korea_University', 'lat': 37.585, 'lon': 127.025}
        , {'name': 'KORUS_Iksan', 'lat': 35.962, 'lon': 127.005}
        , {'name': 'KORUS_NIER', 'lat': 37.569, 'lon': 126.640}
        , {'name': 'KORUS_Taehwa', 'lat': 37.312, 'lon': 127.310}
        , {'name': 'KORUS_UNIST_Ulsan', 'lat': 35.582, 'lon': 129.190}
        , {'name': 'KORUS_Olympic_Park', 'lat': 37.522, 'lon': 127.124}
        , {'name': 'PKU_PEK', 'lat': 39.593, 'lon': 116.184}
        , {'name': 'Anmyon', 'lat': 36.539, 'lon': 126.330}
        , {'name': 'Incheon', 'lat': 37.569, 'lon': 126.637}
        , {'name': 'KORUS_Songchon', 'lat': 37.338, 'lon': 127.489}
        , {'name': 'KORUS_Mokpo_NU', 'lat': 34.913, 'lon': 126.437}
        , {'name': 'KORUS_Daegwallyeong', 'lat': 37.687, 'lon': 128.759}
        , {'name': 'KIOST_Ansan', 'lat': 37.286, 'lon': 126.832}
        , {'name': 'KORUS_Baeksa', 'lat': 37.412, 'lon': 127.569}
        , {'name': 'KORUS_Kyungpook_NU', 'lat': 35.890, 'lon': 128.606}
        , {'name': 'Gosan_NIMS_SNU', 'lat': 33.300, 'lon': 126.206}
    ]

    # ****************************************************************************
    # 시작/종료일 설정
    # ****************************************************************************
    dtKst = timedelta(hours=9)

    dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
    dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
    dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

    # ****************************************************************************
    # 육해상 분류
    # ****************************************************************************
    inpLandFile = '{}/{}/MCD12C1.A{}*.tif'.format(globalVar['inpPath'], serviceName, '*')
    fileLandList = sorted(glob.glob(inpLandFile))

    if fileLandList is None or len(fileLandList) < 1:
        print('[ERROR] inpFile : {} / {}'.format(inpLandFile, '입력 자료를 확인해주세요.'))

    landData = xr.open_dataset(fileLandList[0])
    landDataL1 = landData.idxmax(dim='band')
    sysOpt['data']['landUse'] = landDataL1

    # ****************************************************************************
    # GOSAT 자료 처리
    # ****************************************************************************
    dtDayInfo = dtDayList[356]
    gosatDataL2 = xr.Dataset()
    for i, dtDayInfo in enumerate(dtDayList):

        dtYmd = dtDayInfo.strftime('%Y%m%d')
        dt2Ymd = dtDayInfo.strftime('%y%m%d')

        inpFile = '{}/{}/{}/oco2_*_{}*_*.nc4'.format(globalVar['inpPath'], serviceName, 'OCO2', dt2Ymd)
        fileList = sorted(glob.glob(inpFile))

        if (fileList is None) or (len(fileList) < 1): continue
        print("[CHECK] dtDayInfo : {}".format(dtDayInfo))

        gosatData = xr.open_dataset(fileList[0])

        # 자료 설명
        # for key in list(gosatData['Data'].keys()):
        #     for key2 in list(gosatData['Data'][key].keys()):
        #         print('[CHECK] Data/{}/{} : {}'.format(key, key2, gosatData['Data'][key][key2].shape))

        nx1D = gosatData['sounding_id'].values
        lat1D = gosatData['latitude'].values
        lon1D = gosatData['longitude'].values
        time1D = gosatData['time'].values
        val1D = gosatData['xco2'].values
        flag1D = gosatData['xco2_quality_flag'].values

        gosatDataL1 = xr.Dataset(
            {
                'xco2': (('time', 'nx'), (val1D).reshape(1, len(nx1D)))
                , 'xco2Flag': (('time', 'nx'), (flag1D).reshape(1, len(nx1D)))
                , 'lon': (('time', 'nx'), (lon1D).reshape(1, len(nx1D)))
                , 'lat': (('time', 'nx'), (lat1D).reshape(1, len(nx1D)))
                , 'scanTime': (('time', 'nx'), (time1D).reshape(1, len(nx1D)))
            }
            , coords={
                'time' : pd.date_range(dtDayInfo, periods=1)
                , 'nx' : nx1D
                # 'time' : pd.to_datetime(time1D)
            }
        )

        gosatDataL2 = xr.merge([gosatDataL2, gosatDataL1])

    # Flag 마스킹
    gosatDataL2 = gosatDataL2.where((gosatDataL2['xco2Flag'] == 0), drop=True)

    # CSV 파일 생성
    timeList = gosatDataL2['time'].values
    csvData = gosatDataL2.to_dataframe().reset_index()[['time', 'lon', 'lat', 'scanTime', 'xco2', 'xco2Flag']]
    saveCsvFile = '{}/{}/OCO2-PROP-{}-{}.csv'.format(globalVar['outPath'], serviceName, pd.to_datetime(timeList.min()).strftime('%Y%m%d'), pd.to_datetime(timeList.max()).strftime('%Y%m%d'))
    os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
    csvData.to_csv(saveCsvFile, index=False)
    print('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

    # ****************************************************************************
    # 특정일을 기준으로 시각화
    # 특정 월을 기준으로 시각화
    # 특정 시작일/종료일을 기준으로 시각화
    # ****************************************************************************
    # 특정일을 기준으로 시각화
    # timeList = gosatDataL2['time'].values
    # timeInfo = timeList[0]
    # for i, timeInfo in enumerate(timeList):
    #     print("[CHECK] timeInfo : {}".format(timeInfo))
    #
    #     dsData = gosatDataL2.sel(time = timeInfo)
    #
    #     # 스캔 영역
    #     # mainTitle = 'OCO2-Day-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
    #     # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    #     # makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=False)
    #
    #     # 스캔 영역 + 육해상 분류
    #     mainTitle = 'OCO2-Day-LandUse-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
    #     saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    #     makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=True)
    #
    #     dsDataL1 = dsData.where(
    #         (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
    #         & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
    #         & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
    #         & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
    #     ).to_dataframe().reset_index()[['lon', 'lat', 'xco2']].dropna()
    #
    #     if  len(dsDataL1['xco2']) > 0:
    #         # mainTitle = 'OCO2-Day-Kor-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
    #         # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    #         # makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False)
    #
    #         # 스캔 영역 + 육해상 분류
    #         mainTitle = 'OCO2-Day-Kor-LandUse-{}'.format(pd.to_datetime(timeInfo).strftime('%Y%m%d'))
    #         saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    #         makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True)


    # 특정 월을 기준으로 시각화
    selData = gosatDataL2.copy().sel(time=gosatDataL2.time.dt.month.isin(sysOpt['selMonth']))
    if (len(selData['time'])):
        selDataL2 = selData.groupby('time.month').mean(skipna=True)

        monthList = selDataL2['month'].values
        monthInfo = monthList[0]
        for i, monthInfo in enumerate(monthList):
            print("[CHECK] monthInfo : {}".format(monthInfo))

            dsData = selDataL2.sel(month=monthInfo)

            # 스캔 영역
            # mainTitle = 'OCO2-Month-{:02d}'.format(monthInfo)
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=False)

            # 스캔 영역 + 육해상 분류
            mainTitle = 'OCO2-Month-LandUse-{:02d}'.format(monthInfo)
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=True)

            dsDataL1 = dsData.where(
                (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
                & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
                & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
                & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
            ).to_dataframe().reset_index()[['lon', 'lat', 'xco2']].dropna()

            if len(dsDataL1['xco2']) > 0:
                # mainTitle = 'OCO2-Month-Kor-{:02d}'.format(monthInfo)
                # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                # makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False)

                # 스캔 영역 + 육해상 분류
                mainTitle = 'OCO2-Month-Kor-LandUse-{:02d}'.format(monthInfo)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True)

    # 특정 시작일/종료일을 기준으로 시각화
    selData = gosatDataL2.copy().sel(time=slice(sysOpt['selSrtDate'], sysOpt['selEndDate']))
    if (len(selData['time'])):

        dsData = selData.to_dataframe().reset_index().dropna().groupby(by = ['lon', 'lat'], dropna=False).mean().reset_index()

        # 스캔 영역
        # mainTitle = 'OCO2-All-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
        # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
        # makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=False)

        # 스캔 영역 + 육해상 분류
        mainTitle = 'OCO2-All-LandUse-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
        saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
        makeMapPlot(sysOpt, dsData['lon'], dsData['lat'], dsData['xco2'], mainTitle, saveImg, isLandUse=True)

        dsDataL1 = dsData.loc[
            (sysOpt['roi']['ko']['minLon'] <= dsData['lon'])
            & (dsData['lon'] <= sysOpt['roi']['ko']['maxLon'])
            & (sysOpt['roi']['ko']['minLat'] <= dsData['lat'])
            & (dsData['lat'] <= sysOpt['roi']['ko']['maxLat'])
        ].reset_index()[['lon', 'lat', 'xco2']].dropna()

        if len(dsDataL1['xco2']) > 0:
            # mainTitle = 'OCO2-All-Kor-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=False)

            # 스캔 영역 + 육해상 분류
            mainTitle = 'OCO2-All-Kor-LandUse-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeMapKorPlot(sysOpt, dsDataL1['lon'], dsDataL1['lat'], dsDataL1['xco2'], mainTitle, saveImg, isLandUse=True)

    # ==================================================================================
    # 특정일, 특정 영역을 기준으로 화소 개수 및 위성 개수
    # ==================================================================================
    # 자료처리
    timeList = gosatDataL2['time'].values
    timeInfo = timeList[0]
    gosatDataL3 = xr.Dataset()
    for i, timeInfo in enumerate(timeList):
        for j, stnInfo in enumerate(sysOpt['data']['stnList']):

            print("[CHECK] {} : {}".format(stnInfo['name'], timeInfo))

            areaData = gosatDataL2.sel(time=timeInfo)
            areaDataL1 = areaData.where(
                ((stnInfo['lon'] - sysOpt['res']) <= areaData['lon'])
                & (areaData['lon'] <= (stnInfo['lon'] + sysOpt['res']))
                & ((stnInfo['lat'] - sysOpt['res']) <= areaData['lat'])
                & (areaData['lat'] <= (stnInfo['lat'] + sysOpt['res']))
                , drop=True
            )

            # 단일 위성 자료를 기준으로 영역 내의 화소 개수
            statData = areaDataL1.count(dim=['nx'])

            # 위성 자료 개수
            cnt = 1 if (statData['xco2'].values > 0) else 0

            dsData = xr.Dataset(
                {
                    'pixelCnt': (('name', 'time'), (statData['xco2'].values).reshape(1, 1))
                    , 'cnt': (('name', 'time'), (np.array(cnt)).reshape(1, 1))
                }
                , coords={
                    'time' :  pd.date_range(timeInfo, periods=1)
                    , 'name' : [stnInfo['name']]
                }
            )

            gosatDataL3 = xr.merge([gosatDataL3, dsData])

    # ==================================================================================
    # 특정일, 특정 영역을 기준으로 위성 개수 (cnt) ,화소 개수 (pixelCnt) 시각화
    # ==================================================================================
    # 특정일, 특정 영역을 기준으로 화소 개수 시각화
    timeList = gosatDataL3['time'].values
    mainTitle = 'OCO2-Day-pixelCnt-{}-{}'.format(pd.to_datetime(timeList).min().strftime('%Y%m%d'), pd.to_datetime(timeList).max().strftime('%Y%m%d'))
    saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    for j, timeInfo in enumerate(timeList):
        selData = gosatDataL3.sel(time = timeInfo)
        plt.plot(selData['pixelCnt'], selData['name'], marker='o', label = pd.to_datetime(timeInfo).strftime('%Y.%m.%d'))
    plt.legend()
    plt.title(mainTitle)
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    # 특정 월을 기준으로 화소 개수
    selData = gosatDataL3.copy().sel(time = gosatDataL3.time.dt.month.isin(sysOpt['selMonth']))
    if (len(selData['time'])):
        selDataL2 = selData.groupby('time.month').sum(skipna = True)

        monthList = selDataL2['month'].values
        mainTitle = 'OCO2-Month-pixelCnt-{:02d}-{:02d}'.format(monthList.min(), monthList.max())
        saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
        for j, monthInfo in enumerate(monthList):
            selDataL3 = selDataL2.sel(month = monthInfo)
            plt.plot(selDataL3['pixelCnt'], selDataL3['name'], marker='o', label = monthInfo)
        plt.legend()
        plt.title(mainTitle)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    # 특정 시작일/종료일을 기준으로 화소 개수
    selData = gosatDataL3.copy().sel(time=slice(sysOpt['selSrtDate'], sysOpt['selEndDate']))
    if (len(selData['time'])):
        selDataL2 = selData.to_dataframe().reset_index().groupby(by=['name'], dropna=False).sum().reset_index()

        mainTitle = 'OCO2-All-pixelCnt-{}-{}'.format(pd.to_datetime(sysOpt['selSrtDate']).strftime('%Y%m%d'), pd.to_datetime(sysOpt['selEndDate']).strftime('%Y%m%d'))
        saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
        plt.plot(selDataL2['pixelCnt'], selDataL2['name'], marker='o', label='All')
        plt.legend()
        plt.title(mainTitle)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()