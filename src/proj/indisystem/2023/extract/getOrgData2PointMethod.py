# -*- coding: utf-8 -*-

import pandas as pd
import psycopg2
import re
import yaml
import os
import xarray as xr
import pandas as pd
import glob
from pandas.tseries.offsets import Hour, Day

# ===========================================================
# 실행 방법
# ===========================================================
# /home/hanul/SYSTEMS/KIER/LIB/py38/bin/python3 /home/hanul/SYSTEMS/KIER/PROG/PYTHON/extract/getOrgData2PointMethod.py
# /wind_home/jinyoung/SYSTEMS/KIER/LIB/py38/bin/python3 /wind_home/jinyoung/SYSTEMS/KIER/PROG/PYTHON/extract/getOrgData2PointMethod.py

# ===========================================================
# 입력 정보
# ===========================================================
ctxPath = os.getcwd()

# 특정 지점 정보
lon = 126
lat = 35
print(f'[CHECK] lon : {lon}')
print(f'[CHECK] lat : {lat}')

# 시작일/종료일
# srtDate = '2023-06-30 00:00'
# endDate = '2023-07-01 00:00'
srtDate = '2010-01-01 00:00'
endDate = '2010-01-01 00:00'

# 년월시시분초 변환
srtDt = pd.to_datetime(srtDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")
endDt = pd.to_datetime(endDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")
print(f'[CHECK] srtDt : {srtDt}')
print(f'[CHECK] endDt : {endDt}')

# 변수 선택
colList = ['U', 'V', 'U10', 'V10']
print(f'[CHECK] colList : {colList}')

# 내삽 방법
method = 'nearest' # 최근접
# method = 'linear' # 선형
print(f'[CHECK] method : {method}')

# 파일 정보 패턴
filePattern = '/thermal1/Rawdata/rawdata/wrf%Y_%m/wrfout_d04s_%Y-%m-%d*.nc'
# filePattern = '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_%Y-%m-%d*.nc'
print(f'[CHECK] filePattern : {filePattern}')

# ===========================================================
# 위경도 파일 가공
# ===========================================================
# 가장 근접한 위경도 가져오기
geoData = xr.open_dataset('/thermal1/Rawdata/rawdata/wrf2010_01/geo_em.d04.nc')
# geoData = xr.open_dataset('/DATA/INPUT/INDI2023/MODEL/KIER-WIND/geo_em.d04.nc')
geoDataL1 = geoData[['XLONG_M', 'XLAT_M']].isel(Time=0).to_dataframe().reset_index(drop=False)
geoDataL1['dist'] = ((geoDataL1['XLAT_M'] - lat) ** 2 + (geoDataL1['XLONG_M'] - lon) ** 2) ** 0.5
cloIdx = geoDataL1.loc[geoDataL1['dist'].idxmin()]
print(f'[CHECK] cloIdx : {cloIdx}')

# ===========================================================
# KIER-WIND 가공
# ===========================================================
# 파일 검색
# 시작일/종료일 설정
dtSrtDate = pd.to_datetime(srtDt, format='%Y%m%d%H%M%S')
dtEndDate = pd.to_datetime(endDt, format='%Y%m%d%H%M%S')
# dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))

for dtDateIdx, dtDateInfo in enumerate(dtDateList):
    print(f'[CHECK] dtDateInfo : {dtDateInfo}')

    fileDate = dtDateInfo.strftime(filePattern)
    fileList = sorted(glob.glob(fileDate))
    if fileList is None or len(fileList) < 1: continue

    for fileInfo in fileList:
        print(f'[CHECK] fileInfo : {fileInfo}')

        # NetCDF 파일 읽기
        orgData = xr.open_mfdataset(fileInfo)
        print(f'[CHECK] orgData : {orgData.keys()}')

        # 분석시간
        anaDt = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')

        # 시간 인덱스를 예보 시간 기준으로 변환
        timeByteList = orgData['Times'].values
        timeList = [timeInfo.decode('UTF-8').replace('_', ' ') for timeInfo in timeByteList]
        orgData['Time'] = pd.to_datetime(timeList)

        forDtList = orgData['Time'].values
        for idx, forDtInfo in enumerate(forDtList):
            forDt = pd.to_datetime(forDtInfo, format='%Y-%m-%d_%H:%M:%S')

            print(f'[CHECK] anaDt : {anaDt} / forDt : {forDt}')

            # 특정 위경도를 기준으로 내삽
            data = orgData[colList].sel(Time = forDtInfo).interp(south_north = cloIdx['south_north'], west_east_stag = cloIdx['west_east'], method = method)
            # data = orgData.sel(Time = forDtInfo).interp(south_north = cloIdx['south_north'], west_east_stag = cloIdx['west_east'], method = method)

            # 데이터프레임 변환
            dataL1 = data.to_dataframe().reset_index(drop=False)

            # csv 저장
            if len(dataL1) > 0:
                saveFile = f'{ctxPath}/CSV/getOrgData2PointMethod_{anaDt.strftime("%Y%m%d%H%M")}_{forDt.strftime("%Y%m%d%H%M")}_{lon}_{lat}_{srtDt}_{endDt}_{method}.csv'
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL1.to_csv(saveFile, index=False)
                print(f'[CHECK] saveFile : {saveFile}')