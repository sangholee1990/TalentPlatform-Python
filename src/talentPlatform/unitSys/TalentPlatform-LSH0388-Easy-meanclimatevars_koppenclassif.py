# -*- coding: utf-8 -*-
"""
September 17th, 2021. PyCharm Editor.
Script for NGEA12 project, Coffee.
The script averages NC files ts and pr data in a given X span, using a per month basis for later utilization in the koppen classification.
NC files (in_netcdf_ts & in_netcdf_pr) must match in institution, experiment, variant, table and resolution.
To use this script just assign NC files name strings, place them in the same folder as the script, change the yearSpan, starting year and finishing year integers, in a way it works for the correct span.
"""

import os
import numpy
# import writeMetadata
from netCDF4 import Dataset
from datetime import datetime
from sys import exit
from osgeo import gdal, osr
import glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================
# 보조
# ============================================
def writeMetadata(in_netcdf):
    attFile = in_netcdf[:-3] + "_metadata.txt"

    netcdf = Dataset(in_netcdf)
    variables = [i for i in netcdf.variables.keys()]
    with open(attFile, 'w') as f:
        for v in variables:
            f.write("{}\n{}\n".format(v, netcdf.variables[v]))
            f.write("--------------------------------------------------------\n\n")
    print("Metadata .txt written to:\n{}\n".format(attFile))

def koppen_beck(index: range) -> dict:
    """
    ts and pr global np.array w/ shape == (rows * cols, 12), not an argument to the function

    :param index: range iterable with the number of cells in the raster (rows * cols)
    :return dict:
    """

    # global koppenClass

    # pre-calculations
    MAT = ts[index].sum() / 12
    MAP = pr[index].sum()
    Pdry = pr[index].min()
    Tcold = ts[index].min()
    Thot = ts[index].max()

    # 2023.01.15 LSH
    if (pd.isna(MAT)): return
    if (pd.isna(MAP)): return
    if (pd.isna(Pdry)): return
    if (pd.isna(Tcold)): return
    if (pd.isna(Thot)): return

    Tmon10 = 0
    for temp in ts[index]:
        if temp > 10:
            Tmon10 += 1

    if index < rows * cols / 2:  # southern hemisphere, winter from the 3rd to 9th month
        he = "S"
        if pr[index, 3:9].sum() > 0.7 * MAP:
            Pth = 2 * MAT
        elif numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).sum() > 0.7 * MAP:  # summer
            Pth = 2 * MAT + 28
        else:
            Pth = 2 * MAT + 14
        Pwdry = pr[index, 3:9].min()
        Pwwet = pr[index, 3:9].max()
        Psdry = numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).min()
        Pswet = numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).max()
    else:  # northern hemisphere, summer from the 3rd to 9th month
        he = "N"
        if numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).sum() > 0.7 * MAP:
            Pth = 2 * MAT
        elif pr[index, 3:9].sum() > 0.7 * MAP:  # summer
            Pth = 2 * MAT + 28
        else:
            Pth = 2 * MAT + 14
        Psdry = pr[index, 3:9].min()
        Pswet = pr[index, 3:9].max()
        Pwdry = numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).min()
        Pwwet = numpy.concatenate((pr[index, 0:3], pr[index, 9:12])).max()


    # classification conditionals
    if MAP < 10 * Pth:
        koppenClass = "B"
        if MAP < 5 * Pth:
            koppenClass = koppenClass + "W"
        else:
            koppenClass = koppenClass + "S"
        if MAT >= 18:
            koppenClass = koppenClass + "h"
        else:
            koppenClass = koppenClass + "k"
    elif Tcold >= 18:
        koppenClass = "A"
        if Pdry >= 60:
            koppenClass = koppenClass + "f"
        else:
            if Pdry >= 100 - MAP / 25:
                koppenClass = koppenClass + "m"
            else:
                koppenClass = koppenClass + "w"

    elif Thot > 10 and 0 < Tcold < 18:
        koppenClass = "C"
        if Psdry < 40 and Psdry < Pwwet / 3:
            koppenClass = koppenClass + "s"
        elif Pwdry < Pswet / 10:
            koppenClass = koppenClass + "w"
        else:
            koppenClass = koppenClass + "f"
        if Thot >= 22:
            koppenClass = koppenClass + "a"
        else:
            if Tmon10 >= 4:
                koppenClass = koppenClass + "b"
            elif 1 <= Tmon10 < 4:
                koppenClass = koppenClass + "c"
    elif Thot > 10 and Tcold <= 0:
        koppenClass = "D"
        if Psdry < 40 and Psdry < Pwwet / 3:
            koppenClass = koppenClass + "s"
        elif Pwdry < Pswet / 10:
            koppenClass = koppenClass + "w"
        else:
            koppenClass = koppenClass + "f"
        if Thot >= 22:
            koppenClass = koppenClass + "a"
        else:
            if Tmon10 >= 4:
                koppenClass = koppenClass + "b"
            elif Tcold < -38:
                koppenClass = koppenClass + "d"
            else:
                koppenClass = koppenClass + "c"
    elif Thot <= 10:
        koppenClass = "E"
        if Thot > 0:
            koppenClass = koppenClass + "T"
        else:
            koppenClass = koppenClass + "F"

    koppenDict = {
        "Af": 1,
        "Am": 2,
        "Aw": 3,
        "BWh": 4,
        "BWk": 5,
        "BSh": 6,
        "BSk": 7,
        "Csa": 8,
        "Csb": 9,
        "Csc": 10,
        "Cwa": 11,
        "Cwb": 12,
        "Cwc": 13,
        "Cfa": 14,
        "Cfb": 15,
        "Cfc": 16,
        "Dsa": 17,
        "Dsb": 18,
        "Dsc": 19,
        "Dsd": 20,
        "Dwa": 21,
        "Dwb": 22,
        "Dwc": 23,
        "Dwd": 24,
        "Dfa": 25,
        "Dfb": 26,
        "Dfc": 27,
        "Dfd": 28,
        "ET": 29,
        "EF": 30
    }

    return koppenDict[koppenClass]

def coffee(index: range) -> dict:
    """
    ts and pr global np.array w/ shape == (rows * cols, 12), not an argument to the function

    :param index: range iterable with the number of cells in the raster (rows * cols)
    :return dict:
    """
    MAT = ts[index].sum() / 12
    Tcold = ts[index].min()
    if 17 < MAT < 24 and Tcold > 7:
        return 1
    else:
        return 0

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0388'

# 옵션 설정
sysOpt = {
    # NetCDF 파일에서 시작/종료 시간
    'srtDate': '2015-01'
    , 'endDate': '2055-01'

    # for 2015-2100 files use a 10 year span, 2091-2100, start = 76 (2091)
    # for 1850-2014 files use a 30 year span, 1971-2000, start = 121 (1971 = 1850 + 121), finish = 151 (2000)

    # 즉 0~360으로서 0 (2015.01), 1 (2015.02), ..., 360 (2044.01)
    # 인덱스 설정 (전체 0~360)
    , 'srtIdx' : 0
    , 'endIdx' : 30
    , 'yearSpan' : 30
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'


# scriptStopwatch = datetime.now()
############################# USER INPUT #############################
# path = "/home/salva/proyectos/netCDF4/"
path = os.path.dirname(__file__)
#in_netcdf = path + os.sep + "climate_copernicus.nc"
# https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r5i1p1f1/Amon/ts/gn/files/d20190222
# in_netcdf_ts = os.path.join(path, "ts_Amon_MRI-ESM2-0_historical_r5i1p1f1_gn_185001-201412.nc")

# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'mean-mean_tas_land.nc')
# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ts_Amon_MRI-ESM2-0_historical_r5i1p1f1_gn_185001-201412.nc')
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'tas_MME_land.nc')
fileList = sorted(glob.glob(inpFile))
in_netcdf_ts = fileList[0]

# https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r5i1p1f1/Amon/pr/gn/files/d20190222
# in_netcdf_pr = os.path.join(path, "pr_Amon_MRI-ESM2-0_historical_r5i1p1f1_gn_185001-201412.nc")

# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'mean-mean_pr_land.nc')
# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_Amon_MRI-ESM2-0_historical_r5i1p1f1_gn_185001-201412.nc')
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_MME_land.nc')
fileList = sorted(glob.glob(inpFile))
in_netcdf_pr = fileList[0]


# Dimensions:  (time: 361, lat: 301, lon: 721)
# prData = xr.open_dataset(in_netcdf_pr).sel(time = slice('2015-01', '2055-01'))
prData = xr.open_dataset(in_netcdf_pr).sel(time = slice(sysOpt['srtDate'], sysOpt['endDate']))

# Dimensions:  (time: 361, lat: 298, lon: 721)
# tsData = xr.open_dataset(in_netcdf_ts).sel(time = slice('2015-01', '2055-01'))
tsData = xr.open_dataset(in_netcdf_ts).sel(time = slice(sysOpt['srtDate'], sysOpt['endDate']))

# prData 및 tsData 간의 위도 불일치 존재 (3개)
# prData에서 위도를 기준으로 tasData 위도 내삽 수행
latList = prData['lat'].values
tsDataL1 = tsData.interp(lat = latList, method='linear')

# tsData.isel(time = 0)['tas'].plot()
# plt.show()

# prData.isel(time = 0)['pr'].plot()
# plt.show()

# pr 및 tas 자료 병합
# data = xr.merge([tsData, prData])
data = xr.merge([tsDataL1, prData])

# data.isel(time = 0)['pr'].plot()
# plt.show()

# data.isel(time = 0)['tas'].plot()
# plt.show()

# CMIP6: "ts": surface temperature, "pr": precipitation
# copernicus: "tp" for total precipitation, "swvl1" for soil water content, "t2m" for 2 metre temperature
# year_span = 30
# start = 121  # starting year, after the first year
# finish = 151  # finishing year, after the first year (use 0 to go til the end)
# for 2015-2100 files use a 10 year span, 2091-2100, start = 76 (2091)
# for 1850-2014 files use a 30 year span, 1971-2000, start = 121 (1971 = 1850 + 121), finish = 151 (2000)
# timeList[(121 - 1) * 12]
# timeList[(151 - 1) * 12]
# timeList[int((121 - 1) / 12)]


# 기존
# time: 1980, bnds: 2, lat: 160, lon: 320
# 1850-01-16T12:00:00 ... 2014-12-16T12:00:00

# 샘플 파일
# year_span = 30
# start = 5
# finish = 35

# MME-land 파일
# year_span = 30
# start = 0
# finish = 30

year_span = sysOpt['yearSpan']
start = sysOpt['srtIdx']
finish = sysOpt['endIdx']

# ts_netcdf = Dataset(in_netcdf_ts)
# pr_netcdf = Dataset(in_netcdf_pr)
#variables = list(ts_netcdf.variables.keys())

# rows = ts_netcdf.variables["lat"].size
# cols = ts_netcdf.variables["lon"].size
# frames = ts_netcdf.variables["time"].size
# pr_rows = ts_netcdf.variables["lat"].size
# pr_cols = ts_netcdf.variables["lon"].size
# pr_frames = ts_netcdf.variables["time"].size

rows = data["lat"].size
cols = data["lon"].size
frames = data["time"].size
pr_rows = data["lat"].size
pr_cols = data["lon"].size
pr_frames = data["time"].size

if not pr_cols == cols and pr_rows == rows and pr_frames == frames:
    exit("PR and TS dimensions don't match, aborting")

years = frames / 12
######################################################################

### ATTRIBUTE FILES WRITING AND PRE-PROCESSING
# print("TS NC File: {}".format(os.path.basename(in_netcdf_ts)))
# print("PR NC File: {}".format(os.path.basename(in_netcdf_pr)))

# writeMetadata.main(in_netcdf_ts)
# writeMetadata.main(in_netcdf_pr)
# writeMetadata(in_netcdf_ts)
# writeMetadata(in_netcdf_pr)
print("dimensions = {} x {} x {}, {} cells".format(rows, cols, frames, rows * cols))

# 평균 계산
# ts_array = data["ts"].values
ts_array = data["tas"].values

# pr_array = pr_netcdf.variables["pr"]
pr_array = data["pr"].values

monthly_ts = numpy.zeros((12, rows, cols))
monthly_pr = numpy.zeros((12, rows, cols))
if finish == 0:
    finish = int(years)

# 월별 평균 계산
for y in range(start * 12, finish * 12, 12):
    for m in range(12):
        # print('[CHECK] y : {}, m : {}'.format(y, m))
        print(f'[CHECK] y : {y}, m : {m}')
        monthly_ts[m] += ts_array[y + m]
        monthly_pr[m] += pr_array[y + m]

# [CHECK] max ts : 3168.990234375
# [CHECK] min ts : 2014.266342163086
# [CHECK] max ps : 0.005140622146427631
# [CHECK] min ps : 3.2449590125697136e-22

# [CHECK] max ts : 1214.0962932161297
# [CHECK] min ts : -1194.0259328498892
# [CHECK] max ps : 21589.959711847976
# [CHECK] min ps : 2.95024430859527
print('[CHECK] max ts : {}'.format(np.nanmax(monthly_ts)))
print('[CHECK] min ts : {}'.format(np.nanmin(monthly_ts)))
print('[CHECK] max ps : {}'.format(np.nanmax(monthly_pr)))
print('[CHECK] min ps : {}'.format(np.nanmin(monthly_pr)))

# monthly_ts = numpy.around((monthly_ts.reshape(12, -1) / year_span) - 273.15, 2)  # 0 Kelvin = -273.15 Celcius
# monthly_pr = numpy.around((monthly_pr.reshape(12, -1) / year_span) * 86400 * 30, 2)  # 1 kg m-2 s-1 = 86400 · 30 mm month-1
# ts = (monthly_ts.reshape(12, -1).T / year_span) - 273.15  # 0 Kelvin = -273.15 Celcius
# pr = (monthly_pr.reshape(12, -1).T / year_span) * 86400 * 30  # 1 kg m-2 s-1 = 86400 · 30 mm month-1

ts = (monthly_ts.reshape(12, -1).T / year_span)
pr = (monthly_pr.reshape(12, -1).T / year_span) * 30

# [CHECK] max ts : 43.7490234375
# [CHECK] min ts : -71.72336578369138
# [CHECK] max ps : 1332.449260354042
# [CHECK] min ps : 8.410933760580698e-17

# [CHECK] max ts : 40.469876440537654
# [CHECK] min ts : -39.80086442832964
# [CHECK] max ps : 21589.959711847976
# [CHECK] min ps : 2.95024430859527
print('[CHECK] ts : {} ~ {}'.format(np.nanmin(ts), np.nanmax(ts)))
print('[CHECK] ps : {} ~ {}'.format(np.nanmin(pr), np.nanmax(pr)))

print("Means done! Preparing the rasters...")


### CLASSES & NC FILE INDEX FIX LOOP
#y = lambda x: x[0:3]
# y(monthly_ts[0])
raster_classes = map(koppen_beck, range(rows * cols))
raster_array = numpy.zeros((rows, cols))


row = 0
for j in range(rows * cols - cols, -1, -cols):  # rows, bottoms up
    for i in range(0, cols):  # cols
        # print(f'[CHECK] i : {i}, j : {j}, int(i + cols / 2) : {int(i + cols / 2)}')
        if i < cols / 2:
            raster_array[row, int(i + cols / 2)] = next(raster_classes)
        else:
            raster_array[row, int(i - cols / 2)] = next(raster_classes)
    row += 1

# 자료 가공
lon1D = data['lon'].values
lat1D = data['lat'].values

dataL1 = xr.Dataset(
    {
        'koppen': (('lat', 'lon'), (raster_array).reshape(len(lat1D), len(lon1D)))
    }
    , coords={
        'lat': lat1D
        , 'lon': lon1D
    }
)

# 경도 변환 (0~360 to -180~180)
dataL2 = dataL1
dataL2.coords['lon'] = (dataL2.coords['lon'] + 180) % 360 - 180
dataL2 = dataL2.sortby(dataL2.lon)

# NetCDF 생성
saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'koppen_MME-land')
os.makedirs(os.path.dirname(saveFile), exist_ok=True)
dataL2.to_netcdf(saveFile)
print('[CHECK] saveFile : {}'.format(saveFile))

# 시각화
# mainTitle = '{}'.format('TEST-20230115')
mainTitle = '{}'.format('koppen_MME-land')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)

# dataL1['koppen'].plot()
dataL2['koppen'].plot.contourf(levels=np.linspace(1, 30, 30), add_colorbar=True)
plt.tight_layout()
plt.title(mainTitle)
plt.savefig(saveImg, dpi=600, bbox_inches='tight')
plt.show()
plt.close()

print('[CHECK] saveImg : {}'.format(saveImg))
