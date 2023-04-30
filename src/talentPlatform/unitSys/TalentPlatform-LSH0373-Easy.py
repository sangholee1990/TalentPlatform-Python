# -*- coding: utf-8 -*-
import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

# ============================================
# 보조
# ============================================


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0373'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2017-01-01'
    , 'endDate': '2018-12-31'

    # 관심 지점/영역 설정
    , 'roi': {
        'pos': {
            'lon': 140
            , 'lat': 40
        }

        , 'area': {
            'minLon': 120
            , 'maxLon': 150
            , 'invLon': 1

            , 'minLat': 30
            , 'maxLat': 50
            , 'invLat': 1
        }
    }
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

# ****************************************************************************
# 시작/종료일 설정
# ****************************************************************************
dtKst = timedelta(hours=9)

dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
# dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

# ****************************************************************************
# 신규 위도/경도 설정
# ****************************************************************************
lonList = np.arange(sysOpt['roi']['area']['minLon'], sysOpt['roi']['area']['maxLon'], sysOpt['roi']['area']['invLon'])
latList = np.arange(sysOpt['roi']['area']['minLat'], sysOpt['roi']['area']['maxLat'], sysOpt['roi']['area']['invLat'])

print('[CHECK] len(lonList) : {}'.format(len(lonList)))
print('[CHECK] len(latList) : {}'.format(len(latList)))

# ****************************************************************************
# 플럭스 변수 (GPP, NEE, RECO) 자료 처리 (특정 지점, 영역별 시계열 출력)
# ****************************************************************************
# dtMonthInfo = dtMonthList[0]
posData = xr.Dataset()
areaData = xr.Dataset()
for i, dtMonthInfo in enumerate(dtMonthList):

    dtYmd = dtMonthInfo.strftime('%Y%m%d')
    dtYm = dtMonthInfo.strftime('%Y%m')
    dt2Ymd = dtMonthInfo.strftime('%y%m%d')

    # inpFile = '{}/{}/{}/oco2_*_{}*_*.nc4'.format(globalVar['inpPath'], serviceName, 'OCO2', dt2Ymd)
    inpFile = '{}/{}/hrly_mean_GPP_Reco_NEE_easternAsia_{}.nc4'.format(globalVar['inpPath'], serviceName, dtYm)
    fileList = sorted(glob.glob(inpFile))

    if (fileList is None) or (len(fileList) < 1): continue
    print("[CHECK] dtMonthInfo : {}".format(dtMonthInfo))

    data = xr.open_dataset(fileList[0])
    dataL1 = data[['time', 'lon', 'lat', 'GPP_mean', 'Reco_mean', 'NEE_mean']]

    try:
        # 특정 지점에서 최근접 화소 가져오기
        selPosData = dataL1.interp(lon=sysOpt['roi']['pos']['lon'], lat=sysOpt['roi']['pos']['lat'], method='nearest')

        posData = xr.merge([posData, selPosData])
    except Exception as e:
        print("Exception : {}".format(e))

    try:
        # 특정 영역에서 최근접 화소 가져오기
        selAreaData = dataL1.interp(lon=lonList, lat=latList, method='nearest')

        # 특정 영역에서 위/경도 기준으로 평균
        selAreaDataL1 = selAreaData.mean(dim=['lon', 'lat'], skipna=True)

        areaData = xr.merge([areaData, selAreaDataL1])
    except Exception as e:
        print("Exception : {}".format(e))

# 특정 지점 NetCDF 생성
saveNcFile = '{}/{}/hrly-pos-prop-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((posData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((posData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
posData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 특정 영역 NetCDF 생성
saveNcFile = '{}/{}/hrly-area-prop-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((areaData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((areaData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
areaData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# ****************************************************************************
# 특정 지점 NetCDF 파일에서 일,월,계절,연 누적 NetCDF 생산
# ****************************************************************************
# 연 누적
posStatYearData = posData.resample(time='1Y').sum(skipna=True)

saveNcFile = '{}/{}/hrly-pos-stat-year-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((posStatYearData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((posStatYearData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
posStatYearData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 월 누적
posStatMonthData = posData.resample(time='1M').sum(skipna=True)

saveNcFile = '{}/{}/hrly-pos-stat-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((posStatMonthData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((posStatMonthData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
posStatMonthData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 계절 누적
posStatSeasonData = xr.Dataset()

# 봄 : 3, 4, 5월
posStatSpringData = posStatMonthData.sel(time=posStatMonthData.time.dt.month.isin([3, 4, 5])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'spring-GPP_mean', 'Reco_mean': 'spring-Reco_mean', 'NEE_mean': 'spring-NEE_mean'}
)

# 여름 : 6, 7, 8월
posStatSumerData = posStatMonthData.sel(time=posStatMonthData.time.dt.month.isin([6, 7, 8])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'sumer-GPP_mean', 'Reco_mean': 'sumer-Reco_mean', 'NEE_mean': 'sumer-NEE_mean'}
)

# 가을 : 9, 10, 11월
posStatFallData = posStatMonthData.sel(time=posStatMonthData.time.dt.month.isin([9, 10, 11])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'fall-GPP_mean', 'Reco_mean': 'fall-Reco_mean', 'NEE_mean': 'fall-NEE_mean'}
)

# 겨울 : 12, 1, 2월
posStatWntrData = posStatMonthData.sel(time=posStatMonthData.time.dt.month.isin([12, 1, 2])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'wntr-GPP_mean', 'Reco_mean': 'wntr-Reco_mean', 'NEE_mean': 'wntr-NEE_mean'}
)

posStatSeasonData = xr.merge([posStatSeasonData, posStatSpringData, posStatSumerData, posStatFallData, posStatWntrData])

saveNcFile = '{}/{}/hrly-pos-season-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((posStatSeasonData['year'].values).min(), format='%Y').strftime('%Y%m%d'), pd.to_datetime((posStatSeasonData['year'].values).max(), format='%Y').strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
posStatSeasonData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 일 누적
posStatDayData = posData.resample(time='1D').sum(skipna=True)

saveNcFile = '{}/{}/hrly-pos-stat-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((posStatDayData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((posStatDayData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
posStatDayData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# ****************************************************************************
# 특정 영역 NetCDF 파일에서 일,월,계절,연 누적 NetCDF 생산
# ****************************************************************************
# 연 누적
areaStatYearData = areaData.resample(time='1Y').sum(skipna=True)

saveNcFile = '{}/{}/hrly-area-stat-year-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((areaStatYearData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((areaStatYearData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
areaStatYearData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 월 누적
areaStatMonthData = areaData.resample(time='1M').sum(skipna=True)

saveNcFile = '{}/{}/hrly-area-stat-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((areaStatMonthData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((areaStatMonthData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
areaStatMonthData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 계절 누적
areaStatSeasonData = xr.Dataset()

# 봄 : 3, 4, 5월
areaStatSpringData = areaStatMonthData.sel(time=areaStatMonthData.time.dt.month.isin([3, 4, 5])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'spring-GPP_mean', 'Reco_mean': 'spring-Reco_mean', 'NEE_mean': 'spring-NEE_mean'}
)

# 여름 : 6, 7, 8월
areaStatSumerData = areaStatMonthData.sel(time=areaStatMonthData.time.dt.month.isin([6, 7, 8])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'sumer-GPP_mean', 'Reco_mean': 'sumer-Reco_mean', 'NEE_mean': 'sumer-NEE_mean'}
)

# 가을 : 9, 10, 11월
areaStatFallData = areaStatMonthData.sel(time=areaStatMonthData.time.dt.month.isin([9, 10, 11])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'fall-GPP_mean', 'Reco_mean': 'fall-Reco_mean', 'NEE_mean': 'fall-NEE_mean'}
)

# 겨울 : 12, 1, 2월
areaStatWntrData = areaStatMonthData.sel(time=areaStatMonthData.time.dt.month.isin([12, 1, 2])).groupby('time.year').sum(skipna=True).rename(
    {'GPP_mean': 'wntr-GPP_mean', 'Reco_mean': 'wntr-Reco_mean', 'NEE_mean': 'wntr-NEE_mean'}
)

areaStatSeasonData = xr.merge([areaStatSeasonData, areaStatSpringData, areaStatSumerData, areaStatFallData, areaStatWntrData])

saveNcFile = '{}/{}/hrly-area-season-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((areaStatSeasonData['year'].values).min(), format='%Y').strftime('%Y%m%d'), pd.to_datetime((areaStatSeasonData['year'].values).max(), format='%Y').strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
areaStatSeasonData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))

# 일 누적
areaStatDayData = areaData.resample(time='1D').sum(skipna=True)

saveNcFile = '{}/{}/hrly-area-stat-month-{}-{}.nc'.format(globalVar['outPath'], serviceName, pd.to_datetime((areaStatDayData['time'].values).min()).strftime('%Y%m%d'), pd.to_datetime((areaStatDayData['time'].values).max()).strftime('%Y%m%d'))
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
areaStatDayData.to_netcdf(saveNcFile)
print('[CHECK] saveNcFile : {}'.format(saveNcFile))