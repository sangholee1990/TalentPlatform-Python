# -*- coding: utf-8 -*-

###########################
# load packages
###########################
import glob
import sys

import xarray as xr
from datetime import datetime
import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

print('[START] {}'.format("TalentPlatform-LSH0297-example-FNL.py"))

###########################
# TEST
###########################
#inpFile = 'TMP2/SMAP_L4_MASK_*.nc'
#fileList = sorted(glob.glob(inpFile))

dtSrtDate = pd.to_datetime('2020-04-05', format='%Y-%m-%d')
dtEndDate = pd.to_datetime('2020-04-10', format='%Y-%m-%d')
# dtSrtDate = pd.to_datetime('2015-01-01', format='%Y-%m-%d')
# dtEndDate = pd.to_datetime('2015-12-31', format='%Y-%m-%d')
dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')

# 시작/종료 날짜
# srtYear = 2015
# endYear = 2015
srtYear = 2020
endYear = 2020

# 시작/종료 월
srtMonth = 4
endMonth = 10

# **********************************************
# 파일 읽기
# **********************************************
searchFileList = []
# dtIncDateList : 5개
for i, dtIncDateInfo in enumerate(dtIncDateList):
    dtYmd = dtIncDateInfo.strftime('%Y%m%d')
    iYear = int(dtIncDateInfo.strftime('%Y'))
    iMonth = int(dtIncDateInfo.strftime('%m'))

    if not ((srtYear <= iYear) and (iYear <= endYear)): continue
    if not ((srtMonth <= iMonth) and (iMonth <= endMonth)): continue

    # TMP2/SMAP_L4_MASK_20150601.nc
    inpFile = 'TMP2/SMAP_L4_MASK_{}.nc'.format(dtYmd)
    fileList = sorted(glob.glob(inpFile))
    if fileList is None or len(fileList) < 1:
        print('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        continue

    searchFileList.append(fileList[0])

print("[CHECK] searchFileList : ", searchFileList)

if searchFileList is None or len(searchFileList) < 1:
    print('[ERROR] inpFile : {} / {}'.format(searchFileList, '입력 자료를 확인해주세요.'))
    sys.exit(1)

# 파일 읽기
data = xr.open_mfdataset(searchFileList, engine = 'netcdf4')

# **********************************************
# 자료 전처리
# **********************************************
# # 날짜 행에서 열 변환
# # dataL1 = pd.pivot(data.to_dataframe().reset_index(), index=['y', 'x'], columns='time', values='maxNee')
# dataL1 = pd.pivot(data.to_dataframe().reset_index(), index=['y', 'x'], columns='time', values='maxNee')
#
# # 행 기준으로 최대값
# #maxData = dataL1.max(axis=1)
# maxData = dataL1.min(axis=1)
#
# # 행 기준으로 인덱스 및 Julian Day 변환
# #maxJdData = pd.to_numeric(pd.to_datetime(dataL1.idxmax(axis=1)).dt.strftime('%j'), errors='coerce')
# # maxJdData = pd.to_numeric(pd.to_datetime(dataL1.idxmin(axis=1)).dt.strftime('%j'), errors='coerce')


# 날짜 행에서 열 변환
pft2Data = pd.pivot(data.to_dataframe().reset_index(), index=['y', 'x'], columns='time', values='meanNeePft2')
# 행 기준으로 최대값
minPft2Data = pft2Data.min(axis=1)
# 행 기준으로 인덱스 및 Julian Day 변환
minPft2JdData = pd.to_numeric(pd.to_datetime(pft2Data.idxmin(axis=1)).dt.strftime('%j'), errors='coerce')

# 날짜 행에서 열 변환
pft5Data = pd.pivot(data.to_dataframe().reset_index(), index=['y', 'x'], columns='time', values='meanNeePft5')
# 행 기준으로 최대값
minPft5Data = pft5Data.min(axis=1)
# 행 기준으로 인덱스 및 Julian Day 변환
minPft5JdData = pd.to_numeric(pd.to_datetime(pft5Data.idxmin(axis=1)).dt.strftime('%j'), errors='coerce')




# **********************************************
# NetCDF 생성
# **********************************************
xEle = data['x']
yEle = data['y'][::-1]

saveData = xr.Dataset(
    {
        'lat': (('y', 'x'),  (data['lat'][0, :, :].values.reshape(len(yEle), len(xEle))))
        , 'lon': (('y', 'x'),  (data['lon'][0, :, :].values.reshape(len(yEle), len(xEle))))
        # , 'max': (('y', 'x'),  (maxData.values.reshape(len(yEle), len(xEle))))
        # , 'jd': (('y', 'x'), (maxJdData.values.reshape(len(yEle), len(xEle))))
        , 'minPft2': (('y', 'x'),  (minPft2Data.values.reshape(len(yEle), len(xEle))))
        , 'minPft2Jd': (('y', 'x'), (minPft2JdData.values.reshape(len(yEle), len(xEle))))
        , 'minPft5': (('y', 'x'),  (minPft5Data.values.reshape(len(yEle), len(xEle))))
        , 'minPft5Jd': (('y', 'x'), (minPft5JdData.values.reshape(len(yEle), len(xEle))))
    }
    , coords={
        'x': xEle
        , 'y': yEle
    }
)

saveNcFile = 'TMP2/SMAP_L4_FNL.nc'.format(dtYmd)
os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
saveData.to_netcdf(saveNcFile)

saveCsvFile = 'TMP2/SMAP_L4_FNL.csv'.format(dtYmd)
os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
saveData.to_dataframe().reset_index().to_csv(saveCsvFile)


# **********************************************
# 시각화
# **********************************************
xEle1D = saveData['x'].values
yEle1D = saveData['y'].values
val2D = saveData['minPft5'].values

saveImg = 'FIG/SMAP_L4_FNL_MIN_PFT5.png'
plt.contourf(xEle1D, yEle1D, val2D, cmap=plt.cm.RdBu_r)
plt.title( os.path.basename(saveImg))
plt.colorbar()
plt.savefig(saveImg, dpi=600, bbox_inches='tight')
plt.close()
plt.show()


xEle1D = saveData['x'].values
yEle1D = saveData['y'].values
val2D = saveData['minPft5Jd'].values

saveImg = 'FIG/SMAP_L4_FNL_JD_PFT5.png'
plt.contourf(xEle1D, yEle1D, val2D, cmap=plt.cm.RdBu_r)
plt.title( os.path.basename(saveImg))
plt.colorbar()
plt.savefig(saveImg, dpi=600, bbox_inches='tight')
plt.close()
plt.show()

print('[END] {}'.format("TalentPlatform-LSH0297-example-FNL.py"))