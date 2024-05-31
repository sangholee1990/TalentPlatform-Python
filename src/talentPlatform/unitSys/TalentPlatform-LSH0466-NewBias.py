import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
import faulthandler

faulthandler.enable()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.declarative import declarative_base
from xclim import sdba
from xclim.sdba.adjustment import QuantileDeltaMapping
from multiprocessing import Pool
import multiprocessing as mp
from dask.distributed import Client, LocalCluster

# 초기 설정
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

mpl.rcParams['axes.unicode_minus'] = False

Base = declarative_base()

# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile)

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    log.setLevel(level=logging.INFO)

    return log

# 초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    globalVar = {
        'prjName': prjName
        , 'sysOs': platform.system()
        , 'contextPath': contextPath
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar

# 초기 전달인자 설정
def initArgument(globalVar, inParams):
    if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
        parser = argparse.ArgumentParser()
        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)
        inParInfo = vars(parser.parse_args())

    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()
        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)
        inParInfo = vars(parser.parse_args())

    log.info("[CHECK] inParInfo : {}".format(inParInfo))

    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace('\\', '/')
        log.info("[CHECK] {} : {}".format(key, val))

    return globalVar

def add_leap_day_mrg(modDataL3, var_name='pr'):
    start_year = 1980
    end_year = 2014
    leap_years = [year for year in range(start_year, end_year + 1) if
                  (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
    existing_leap_days_years = set(
        modDataL3.time.dt.year.where((modDataL3.time.dt.month == 2) & (modDataL3.time.dt.day == 29), drop=True).values)
    missing_leap_years = [year for year in leap_years if year not in existing_leap_days_years]

    if not missing_leap_years:
        return modDataL3

    new_times = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            day_range = 30 if month in [4, 6, 9, 11] else 31
            if month == 2:
                day_range = 29 if year in missing_leap_years else 28
            for day in range(1, day_range + 1):
                new_times.append(f'{year}-{month:02d}-{day:02d}')

    new_times_sorted = sorted(pd.to_datetime(new_times))
    full_time_ds = modDataL3.reindex(time=new_times_sorted, method='nearest', tolerance='1D')

    for year in missing_leap_years:
        feb28 = f'{year}-02-28'
        mar01 = f'{year}-03-01'
        feb29 = f'{year}-02-29'
        if np.datetime64(feb28) in full_time_ds.time.values and np.datetime64(mar01) in full_time_ds.time.values:
            feb28_data = full_time_ds.sel(time=feb28)[var_name]
            mar01_data = full_time_ds.sel(time=mar01)[var_name]
            feb29_vals = (feb28_data + mar01_data) / 2
            full_time_ds[var_name].loc[{'time': np.datetime64(feb29)}] = feb29_vals

    return full_time_ds.sortby('time')

def filter_data_by_cont(data, cont_data):
    filtered_data = data.where(cont_data['contIdx'] == 200, drop=True)
    return filtered_data

def remove_leap_days_xarray(mod, obs_start_year=1980, obs_end_year=2014, sim_start_year=2015, sim_end_year=2100):
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    obs_leap_years = [year for year in range(obs_start_year, obs_end_year + 1) if is_leap_year(year)]
    sim_leap_years = [year for year in range(sim_start_year, sim_end_year + 1) if is_leap_year(year)]
    total_leap_years = obs_leap_years + sim_leap_years

    leap_days_to_remove = pd.to_datetime([f'{year}-02-29' for year in total_leap_years])
    mod_filtered = mod.sel(time=~mod.time.dt.date.isin(leap_days_to_remove.date))

    return mod_filtered


def debug_data(data, name):
    log.info(f"{name} - min: {np.nanmin(data)}, max: {np.nanmax(data)}, mean: {np.nanmean(data)}, dtype: {data.dtype}")

# 보정 프로세스 함수
def \
        makeSbckProc(method, contDataL4, mrgDataProcessed, simDataL3Processed, keyInfo):
    log.info('[START] {}'.format("makeSbckProc"))

    result = None

    try:
        procInfo = mp.current_process()
        log.info(f'[CHECK] method : {method} / pid : {procInfo.pid}')

        methodList = {
            'QDM': lambda: sdba.QuantileDeltaMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=50, group= sdba.Grouper("time", window=1), kind='*'),
            'EQM': lambda: sdba.EmpiricalQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=50, group=sdba.Grouper("time", window=1),kind='*'),
            'DQM': lambda: sdba.DetrendedQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=50, group=sdba.Grouper("time", window=1),kind='*'),
        }

        if method not in methodList:
            log.error("주어진 학습 모형 (QDM, EQM, DQM)을 선택해주세요.")
            return result

        prd = methodList[method]()

        lat_size = len(mrgDataProcessed['lat'])
        lon_size = len(mrgDataProcessed['lon'])
        time_size = len(mrgDataProcessed['time'])

        # 2024-06-01 01:08:40,342 [test | 217 | Z.py] [INFO ] [CHECK] lat_size : 68
        # 2024-06-01 01:08:40,342 [test | 218 | Z.py] [INFO ] [CHECK] lon_size : 47
        log.info(f'[CHECK] lat_size : {lat_size}')
        log.info(f'[CHECK] lon_size : {lon_size}')
        log.info(f'[CHECK] lon_size : {lon_size}')

        corrected_data = np.empty((lat_size, lon_size, time_size))
        corrected_data[:] = np.nan

        nan_count = 0
        for i in range(lat_size):
            for j in range(lon_size):
                ref_data = mrgDataProcessed['rain'].isel(lat=i, lon=j)
                hist_data = mrgDataProcessed['pr'].isel(lat=i, lon=j)

                if np.isnan(ref_data).any() or np.isnan(hist_data).any():
                    log.info(f"NaN found in input data at lat {i}, lon {j}. Skipping this cell.")
                    continue

                ref_data_log = np.log1p(ref_data + 1e-6)
                hist_data_log = np.log1p(hist_data + 1e-6)

                ref_data_log.attrs['units'] = 'mm/day'
                hist_data_log.attrs['units'] = 'mm/day'

                log.info(f"Cell at lat {i}, lon {j}")
                # debug_data(ref_data, "Reference data")
                # debug_data(hist_data, "Historical data")
                # debug_data(ref_data_log, "Reference data (log)")
                # debug_data(hist_data_log, "Historical data (log)")

                bias_correction = prd

                try:
                    corrected = bias_correction.adjust(sim=hist_data_log, extrapolation="constant", interp="linear")
                    corrected_exp = np.expm1(corrected) - 1e-6

                    corrected_exp = np.nan_to_num(corrected_exp, nan=0.0)
                    corrected_exp = np.clip(corrected_exp, 0, np.nanmax(ref_data))

                    # debug_data(corrected, "Corrected data (log)")
                    # debug_data(corrected_exp, "Corrected data (exp)")

                    corrected_exp = np.where(corrected_exp < 0.1, 0, corrected_exp)

                    current_nan_count = np.isnan(corrected_exp).sum()
                    nan_count += current_nan_count
                    log.info(f"NaN count for cell at lat {i}, lon {j}: {current_nan_count}")

                    # if corrected.size == time_size:
                    if corrected['time'].size == time_size:
                        # corrected_data[i, j, :] = corrected_exp
                        corrected_data[i, j, :] = corrected_exp.reshape(lat_size, lon_size, time_size)[i, j, :]
                    # else:
                    #     log.info(f"Skipping cell at lat {i}, lon {j} due to shape mismatch: corrected size {corrected.size}, expected size {time_size}")
                except Exception as e:
                    log.error(f"Error processing cell at lat {i}, lon {j}: {e}")
                    continue

        corrected_da = xr.DataArray(corrected_data, coords=[mrgDataProcessed['lat'], mrgDataProcessed['lon'], mrgDataProcessed['time']], dims=['lat', 'lon', 'time'])

        log.info(corrected_da)
        log.info(f"Total NaN count in corrected data: {nan_count}")

        max_value = corrected_da.max().values
        log.info(f"The maximum value in the corrected data array is: {max_value}")

        min_value = corrected_da.min().values
        log.info(f"The minimum value in the corrected data array is: {min_value}")

        df_corrected = corrected_da.to_dataframe(name='corrected_data').reset_index()
        df_corrected_pivot = df_corrected.pivot(index='time', columns=['lat', 'lon'], values='corrected_data')

        # 중간 확인: DataFrame의 크기와 일부 데이터 출력
        log.info(f"df_corrected shape: {df_corrected.shape}")
        log.info(f"df_corrected_pivot shape: {df_corrected_pivot.shape}")
        log.info(df_corrected_pivot.head())

        # 파일 경로 수정
        # csv_file_path = 'corrected_data_{}_{}.csv'.format(method, keyInfo)
        # csv_file_path = '{}/corrected_data_{}_{}.csv'.format(globalVar['outPath'], method, keyInfo)
        csv_file_path = '{}/{}/corrected_data_{}_{}.csv'.format(globalVar['outPath'], keyInfo, method, keyInfo)
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df_corrected_pivot.to_csv(csv_file_path)
        log.info(f"Data saved to {csv_file_path}")

    except Exception as e:
        log.error(f'Exception in makeSbckProc: {str(e)}')
        traceback.print_exc()
        return result

    finally:
        log.info(f'[CHECK] method : {method} / pid : {procInfo.pid}')
        log.info('[END] {}'.format("makeSbckProc"))

# 주 프로그램
class DtaProcess(object):
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'
    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0466'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            initArgument(globalVar, inParams)
        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    def exec(self):
        log.info('[START] {}'.format("exec"))

        try:
            if (platform.system() == 'Windows'):
                globalVar['inpPath'] = 'E:/Global bias/Regridding'
                globalVar['outPath'] = 'L:/Global bias results/OUTPUT'
                globalVar['figPath'] = 'L:/Global bias results/FIG'
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            sysOpt = {
                'srtDate': '1980-01-01',
                'endDate': '1982-12-31',
                'srtDate2': '2015-01-01',
                'endDate2': '2017-12-31',
                'lonMin': 0,
                'lonMax': 360,
                'lonInv': 1,
                'latMin': -90,
                'latMax': 90,
                'latInv': 1,
                'keyList': ['IPSL-CM6A-LR'],
                'methodList': ['QDM'],
                'contIdx': 100,
                'cpuCoreNum': 32,
                'cpuCoreDtlNum': 32
            }

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'TTL4.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1: raise Exception(
                '[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            contData = pd.read_csv(fileList[0]).rename(
                columns={'type': 'contIdx', 'Latitude': 'lat', 'Longitude': 'lon'})
            contDataL1 = contData[['lon', 'lat', 'contIdx']]
            contDataL1['lon'] = np.where(contDataL1['lon'] < 0, (contDataL1['lon']) % 360, contDataL1['lon'])
            contDataL1 = contDataL1[contDataL1['contIdx'] == int(sysOpt['contIdx'])].reset_index(drop=False)
            contDataL2 = contDataL1.set_index(['lat', 'lon'])
            contDataL4 = contDataL2.to_xarray()

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'ERA5_1979_2020.nc')
            fileList = sorted(glob.glob(inpFile))
            if fileList is None or len(fileList) < 1: raise Exception(
                '[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            obsData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            obsDataL1 = obsData
            obsDataL3 = remove_leap_days_xarray(obsDataL1)
            attrs = {"units": "mm d-1"}
            hist_t = xr.cftime_range("1980-01-01", "1982-12-31", freq="D", calendar="noleap")
            time_indexc = pd.Index([pd.Timestamp(date.isoformat()) for date in hist_t])
            lonList1 = contDataL4.lon.values
            latList1 = contDataL4.lat.values
            obsDataL3 = remove_leap_days_xarray(obsData.sel(lat=latList1, lon=lonList1))
            obsDataL3 = xr.DataArray(obsDataL3['rain'], dims=("time", "lat", "lon"),
                                      coords={'time': time_indexc, "lat": latList1, "lon": lonList1},
                                      attrs=attrs).transpose("time", "lat", "lon").to_dataset(name="rain")

            modDataL3CU_time_index = pd.DatetimeIndex(obsDataL3['time'].values)
            modDataL3CU_normalized_time = modDataL3CU_time_index.normalize()
            obsDataL3['time'] = ('time', modDataL3CU_normalized_time)
            obsDataL3 = xr.merge([obsDataL3['rain'], contDataL4])

            keyList = sysOpt['keyList']
            for keyInfo in keyList:
                log.info(f"[CHECK] keyInfo : {keyInfo}")

                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Historical', keyInfo, 'historical')
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1: raise Exception(
                    '[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                modDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    modData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                    modDataL1 = remove_leap_days_xarray(modData.sel(lat=latList1, lon=lonList1))
                    if (len(modData['time']) < 1): continue

                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']
                    for i, selInfo in enumerate(selList):
                        try:
                            modData = modData.drop([selInfo])
                        except Exception as e:
                            pass

                    modDataL1['pr'] = modDataL1['pr']
                    modDataL1['pr'].attrs["units"] = "mm d-1"
                    modDataL2 = xr.merge([modDataL2, modDataL1])

                modDataLU = remove_leap_days_xarray(modDataL2)
                modDataL3 = xr.merge([modDataLU['pr'], contDataL4])
                time_index = pd.to_datetime(obsDataL3.time.values)
                normalized_time_index = time_index.normalize()
                modDataL3['time'] = ('time', normalized_time_index)

                mrgData = xr.merge([obsDataL3, modDataL3])

                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Future', keyInfo, 'ssp126')
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1: raise Exception(
                    '[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                simDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    simData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate2'], sysOpt['endDate2']))
                    simDataL1 = remove_leap_days_xarray(simData.sel(lat=latList1, lon=lonList1))
                    if (len(simData['time']) < 1): continue

                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']
                    for i, selInfo in enumerate(selList):
                        try:
                            simData = simData.drop([selInfo])
                        except Exception as e:
                            pass

                    simDataL1['pr'] = simDataL1['pr']
                    simDataL1['pr'].attrs["units"] = "mm d-1"
                    simDataL2 = xr.merge([simDataL2, simDataL1])

                simDataL3 = xr.merge([simDataL2['pr'], contDataL4])
                simDataL3 = remove_leap_days_xarray(simDataL3)
                normalized_time_index2 = pd.to_datetime(simDataL3.time.values).normalize()
                simDataL3['time'] = ('time', normalized_time_index2)

                mrgDataProcessed = mrgData.drop_vars('contIdx')
                simDataL3Processed = simDataL3
                simDataL3Processed['time'] = ('time', normalized_time_index2)

                mrgDataProcessed = xr.merge([mrgDataProcessed, mrgData['contIdx']]).transpose('lat', 'lon', 'time')
                simDataL3Processed = xr.merge([simDataL3Processed, simDataL3['contIdx']]).transpose('lat', 'lon', 'time')

                mrgDataProcessed = mrgDataProcessed.convert_calendar("noleap")

                for method in sysOpt['methodList']:
                    log.info(f"[CHECK] method : {method}")
                    result = makeSbckProc(method, contDataL4, mrgDataProcessed, simDataL3Processed, keyInfo)
                    log.info(f"[CHECK] result : {result}")

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("exec"))

if __name__ == '__main__':
    print('[START] {}'.format("main"))

    try:
        inParams = {}
        print("[CHECK] inParams : {}".format(inParams))
        subDtaProcess = DtaProcess(inParams)
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))