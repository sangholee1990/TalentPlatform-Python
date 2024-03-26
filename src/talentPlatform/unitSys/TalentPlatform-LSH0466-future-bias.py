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
from sqlalchemy.ext.declarative import declarative_base
import xarray as xr
# from cmethods import adjust
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import gc
from distributed.protocol import serialize, deserialize
# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")
faulthandler.enable()
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

Base = declarative_base()

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info('스크립트 시작: 잔여 Dask 워커 프로세스 정리 중...')
    os.system('taskkill /f /im dask-worker*')

    print('[START] main')

    # Dask 클러스터 초기화
    client = cluster()

    try:
        # 클라이언트를 통해 모든 워커에서 가비지 컬렉션 실행
        client.run(gc.collect)

        # 작업 분배 확인
        check_dask_work_distribution(client)

        # 여기에 데이터 처리 및 분석 코드 추가
        # 예: result = client.submit(some_function, *args, **kwargs).result()

    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit(1)

    finally:
        # Dask 클러스터 종료
        shutdown_dask_cluster(client)
        logging.info('스크립트 종료: 잔여 Dask 워커 프로세스 정리 중...')
        os.system('taskkill /f /im dask-worker*')

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

    # logger instance 생성
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #log = logging.getLogger()
    log = logging.getLogger(prjName)


    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile)

    # logger instance에 format 설정
    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    # logger instance에 handler 설정
    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    # logger instance로 log 기록
    log.setLevel(level=logging.INFO)

    #Global logger 설정
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    return log

# 사용자 정의 직렬화 함수 등록
# @serialize.register(MyCustomClass)
# def serialize_myclass(x):
#     return {}, [pickle.dumps(x)]
#
# @deserialize.register(MyCustomClass)
# def deserialize_myclass(header, frames):
#     return pickle.loads(frames[0])

#  초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    # 환경 변수 (local, 그 외)에 따라 전역 변수 (입력 자료, 출력 자료 등)를 동적으로 설정
    # 즉 local의 경우 현재 작업 경로 (contextPath)를 기준으로 설정
    # 그 외의 경우 contextPath/resources/input/prjName와 같은 동적으로 구성
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


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    # 원도우 또는 맥 환경
    if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
        inParInfo = inParams

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

    log.info("[CHECK] inParInfo : {}".format(inParInfo))

    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

def adjust_chunk(obs_chunk, sim_hist_chunk, sim_future_chunk, method, n_quantiles, group, kind):
    # 청크 단위로 데이터를 조정하는 로직
    # 이 부분은 원래 `adjust_future_periods_dask` 함수의 내용을 기반으로 작성합니다.
    # 예를 들어, adjust() 함수 호출 등을 포함합니다.
    adjusted_data = adjust(obs=obs_chunk, simh=sim_hist_chunk, simp=sim_future_chunk, method=method, n_quantiles=n_quantiles, group=group, kind=kind)
    return adjusted_data


# Dask 클러스터 종료
def shutdown_dask_cluster(client):
    if client:
        client.close()
        print('Dask 클라이언트 종료됨.')

# 작업 분배 확인 (예시 함수)
def check_dask_work_distribution(client):
    workers_info = client.scheduler_info()['workers']
    for worker, info in workers_info.items():
        print(f"Worker {worker}: {info}")



def add_leap_day_mrg(modDataL3, var_name='pr'):
    start_year = 1980
    end_year = 2014

    # 윤년 확인
    leap_years = [year for year in range(start_year, end_year + 1) if
                  (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]

    # MrgData에 있는 윤년확인
    existing_leap_days_years = set(
        modDataL3.time.dt.year.where((modDataL3.time.dt.month == 2) & (modDataL3.time.dt.day == 29), drop=True).values)

    # mrgData에서 윤년이 누락된 경우만 처리
    missing_leap_years = [year for year in leap_years if year not in existing_leap_days_years]

    if not missing_leap_years:
        # 모든 윤년이 존재하면 mrgData 반환
        return modDataL3

    # 윤년이 누락된 경우, 윤년 추가
    # 전체 기간에 대한 날짜 추가
    new_times = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 기본 day_range 설정
            day_range = 30 if month in [4, 6, 9, 11] else 31

            # 2월의 경우 윤년 확인
            if month == 2:
                # 윤년에서 누락된 해에만 2월 29일을 추가
                day_range = 29 if year in missing_leap_years else 28

            for day in range(1, day_range + 1):
                new_times.append(f'{year}-{month:02d}-{day:02d}')

    # 새로운 시간 인덱스를 기반으로 데이터셋을 재색인합니다.
    new_times_sorted = sorted(pd.to_datetime(new_times))
    full_time_ds = modDataL3.reindex(time=new_times_sorted, method='nearest', tolerance='1D')

    # 누락된 윤년의 2월 29일에 대해 데이터 보간을 수행합니다.
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


def add_leap_day_L3(simDataL3, var_name='pr'):
    start_year = 2015
    end_year = 2100

    # 윤년 확인
    leap_years = [year for year in range(start_year, end_year + 1) if
                  (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]

    # MrgData에 있는 윤년확인
    existing_leap_days_years = set(
        simDataL3.time.dt.year.where((simDataL3.time.dt.month == 2) & (simDataL3.time.dt.day == 29), drop=True).values)

    # mrgData에서 윤년이 누락된 경우만 처리
    missing_leap_years = [year for year in leap_years if year not in existing_leap_days_years]

    if not missing_leap_years:
        # 모든 윤년이 존재하면 mrgData 반환
        return simDataL3

    # 윤년이 누락된 경우, 윤년 추가
    # 전체 기간에 대한 날짜 추가
    new_times = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 기본 day_range 설정
            day_range = 30 if month in [4, 6, 9, 11] else 31

            # 2월의 경우 윤년 확인
            if month == 2:
                # 윤년에서 누락된 해에만 2월 29일을 추가
                day_range = 29 if year in missing_leap_years else 28

            for day in range(1, day_range + 1):
                new_times.append(f'{year}-{month:02d}-{day:02d}')

    # 새로운 시간 인덱스를 기반으로 데이터셋을 재색인합니다.
    new_times_sorted = sorted(pd.to_datetime(new_times))
    full_time_ds = simDataL3.reindex(time=new_times_sorted, method='nearest', tolerance='1D')

    # 누락된 윤년의 2월 29일에 대해 데이터 보간을 수행합니다.
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

def run_adjustment(method, obs, simh, simp, kind, n_quantiles=None, group=None):
    # 각 편향 조정 메소드 실행에 필요한 매개변수에 따라 분기
    log.info(f"{method} 작업 시작...")
    # 각 편향 조정 메소드 실행에 필요한 매개변수에 따라 분기
    result = None
    if method in ['quantile_delta_mapping', 'quantile_mapping']:
        result = adjust(method=method, obs=obs, simh=simh, simp=simp, n_quantiles=n_quantiles, kind=kind)
    elif method in ['linear_scaling', 'delta_method']:
        result = adjust(method=method, obs=obs, simh=simh, simp=simp, group=group, kind=kind)

    # 메소드 완료 로깅
    log.info(f"{method} 작업 완료...")
    return result

def run_parallel_adjustments(tasks):
    delayed_tasks = [delayed(run_adjustment)(**task) for task in tasks]
    results = compute(*delayed_tasks)
    results_dict = {task['method']: result for task, result in zip(tasks, results)}
    return results_dict
# def adjust_future_periods_dask(client, obs, sim_hist, sim_future, methods, n_quantiles=100, group='time.month',
#                                kind='*'):
#     logging.basicConfig(level=logging.INFO)
#     future_periods = [('2015', '2049'), ('2050', '2084'), ('2085', '2100')]
#     combined_results = []

#     for start_year, end_year in future_periods:
#         logging.info(f"Processing future period: {start_year}-{end_year}")

#         simp_period = sim_future.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
#         total_time_points = len(simp_period.time)
#         chunk_size = 365  # 예시로 한 청크 당 365일을 가정합니다. 필요에 따라 조정하세요.
#         num_chunks = int(np.ceil(total_time_points / chunk_size))

#         for i in range(num_chunks):
#             start_idx = i * chunk_size
#             end_idx = min((i + 1) * chunk_size, total_time_points)
#             obs_chunk = obs.isel(time=slice(start_idx, end_idx))
#             sim_hist_chunk = sim_hist.isel(time=slice(start_idx, end_idx))
#             simp_period_chunk = simp_period.isel(time=slice(start_idx, end_idx))

#             tasks = []
#             for method in methods:
#                 logging.info(f"Submitting task for method: {method} for chunk {i + 1}/{num_chunks}")

#                 # 각 청크와 메소드에 대해 run_adjustment 함수를 비동기적으로 실행합니다.
#                 # client.submit 대신 delayed 함수를 사용하여 Dask 작업을 예약합니다.
#                 task = delayed(run_adjustment)(method, obs_chunk, sim_hist_chunk, simp_period_chunk, kind,
#                                                n_quantiles if method in ['quantile_mapping',
#                                                                          'quantile_delta_mapping'] else None,
#                                                group if method not in ['quantile_mapping',
#                                                                        'quantile_delta_mapping'] else None)
#                 tasks.append(task)

#             # 모든 작업이 예약된 후, compute 함수를 호출하여 병렬로 실행합니다.
#             results = compute(*tasks)
#             combined_results.extend(results)

#         # 모든 결과를 시간 차원을 기준으로 연결합니다.
#         # 결과 리스트에서 각각의 Dask future 결과를 추출하고 xarray 데이터셋으로 병합합니다.
#     combined_dataset = xr.concat([result[0] for result in combined_results], dim='time')

#     # 최종 결과 정렬
#     combined_sorted = combined_dataset.sortby('time')

#     logging.info("Adjustment process completed.")
#     return combined_sorted
def perform_adjustment(obs, sim_hist, sim_future_chunk, method, args):
    # 'adjust' 함수 호출
    adjusted_data = adjust(obs=obs, simh=sim_hist, simp=sim_future_chunk, method=method, **args)

    # 로그 메시지로 확인한 바에 따르면 adjust 함수는 이미 'xarray.Dataset' 객체를 반환합니다.
    # 따라서, 여기에서는 adjusted_data의 타입을 변경할 필요가 없습니다.

    return adjusted_data


def adjust_future_periods_dask(obs, sim_hist, sim_future, client):
    # Define adjustment methods with specific arguments
    method_args = {
        'quantile_mapping': {'n_quantiles': 1000, 'kind': '*'},
        'quantile_delta_mapping': {'n_quantiles': 1000, 'kind': '*'},
        'linear_scaling': {'group': 'time.month', 'kind': '*'},
        'delta_method': {'group': 'time.month', 'kind': '*'}
    }

    # Define time ranges for historical (obs and sim_hist) and future periods
    future_periods = [(start, start + 34) for start in range(2015, 2100, 35)]

    results = []
    for start_year, end_year in future_periods:
        sim_future_chunk = sim_future.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        tasks = []
        for method, arg in method_args.items():
            task = delayed(perform_adjustment)(obs, sim_hist, sim_future_chunk, method, arg)
            tasks.append(task)

        # Compute tasks in parallel and unwrap the results
        adjusted_chunks = [result[0] for result in compute(*tasks, scheduler='threads')]
        results.extend(adjusted_chunks)

    # Concatenate all adjusted future chunks
    adjusted_sim = xr.concat(results, dim='time')
    return adjusted_sim


# def parallel_bias_adjustments_future(obsDataL3U, modDataL3CU, simDataL3ProcessedU):
#      # 역사적 및 미래 보정 작업을 위한 태스크 정의
#      tasks = [
#          {'method': 'quantile_delta_mapping', 'obs': obsDataL3U['pr'], 'simh': modDataL3CU['pr'],
#           'simp': simDataL3ProcessedU['pr'], 'n_quantiles': 1000, 'kind': '+'},
#          {'method': 'linear_scaling', 'obs': obsDataL3U['pr'], 'simh': modDataL3CU['pr'],
#           'simp': simDataL3ProcessedU['pr'], 'group': 'time.dayofyear', 'kind': '+'},
#          {'method': 'variance_scaling', 'obs': obsDataL3U['pr'], 'simh': modDataL3CU['pr'],
#           'simp': simDataL3ProcessedU['pr'], 'group': 'time.dayofyear', 'kind': '+'},
#          {'method': 'delta_method', 'obs': obsDataL3U['pr'], 'simh': modDataL3CU['pr'],
#           'simp': simDataL3ProcessedU['pr'], 'group': 'time.dayofyear', 'kind': '+'},
#          {'method': 'quantile_mapping', 'obs': obsDataL3U['pr'], 'simh': modDataL3CU['pr'],
#           'simp': simDataL3ProcessedU['pr'], 'n_quantiles': 1000, 'kind': '+'},
#
#     ]
#      return run_parallel_adjustments(tasks)

import pandas as pd
import xarray as xr

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):


    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0466'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams, client):

        log.info("[START] __init__ : init")
        log.info("처리 시작...")
        self.client = client

        try:
            log.info("데이터 로드 중...")
            log.info("역사적 데이터에 대한 병렬 조정 시작...")
            log.info("미래 데이터에 대한 병렬 조정 시작...")
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))
        # check_dask_work_distribution(self.client)  # 작업 분배 확인

        try:

            if (platform.system() == 'Windows'):
                globalVar['inpPath'] = 'E:/Global bias/Regridding'
                globalVar['outPath'] = 'L:/Global bias results/OUTPUT'
                globalVar['figPath'] = 'L:/Global bias results/FIG'

            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 학습 시작/종료 시간
                'srtDate': '1980-01-01'
                , 'endDate': '2014-12-31'

                # 예측 시작/종료 시간
                , 'srtDate2': '2015-01-01'
                , 'endDate2': '2100-12-31'

                # 경도 최소/최대/간격
                , 'lonMin': 0
                , 'lonMax': 360
                , 'lonInv': 1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 1

                #, 'keyList' : ['GFDL-ESM4','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM','NorESM2-MM','TaiESM1']
                # , 'keyList': ['INM-CM5-0']
                , 'keyList': ['MRI-ESM2-0']

                #, 'keyList': ['MRI-ESM2-0','ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CanESM5','CESM2-WACCM','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3-Veg-LR']

            }

            # 날짜 설정
            #dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            #dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            #dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')
            # dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # 위경도 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')

            # ********************************************************************
            # 대륙별 분류 전처리
            # ********************************************************************
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'TT4.csv')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'TT4.csv')

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'TTL4.csv')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'TTL4.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))

            contData = pd.read_csv(fileList[0]).rename(columns={'type': 'contIdx', 'Latitude': 'lat', 'Longitude': 'lon'})
            # contDataL1 = contData[['lon', 'lat', 'isLand', 'contIdx']]
            contDataL1 = contData[['lon', 'lat', 'contIdx']]

            # 경도 변환 (-180~180 to 0~360)
            contDataL1['lon'] = np.where(contDataL1['lon'] < 0, (contDataL1['lon']) % 360, contDataL1['lon'])

            contDataL2 = contDataL1.set_index(['lat', 'lon'])
            contDataL3 = contDataL2.to_xarray()
            contDataL4 = contDataL3.interp({'lon': lonList, 'lat': latList}, method='nearest')

            # contDataL3['contIdx'].plot()
            # contDataL4['contIdx'].plot()
            # plt.show()

            # ********************************************************************
            # 강수량 데이터 전처리
            # ********************************************************************
            # 실측 데이터
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ERA5_1979_2020.nc')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'ERA5_1979_2020.nc')
            fileList = sorted(glob.glob(inpFile))
            obsData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

            # 경도 변환 (-180~180 to 0~360)
            obsDataL1 = obsData
            #obsDataL1.coords['lon'] = (obsDataL1.coords['lon']) % 360
            #obsDataL1 = obsDataL1.sortby(obsDataL1.lon)

            #obsDataL2 = obsDataL1.interp({'lon': lonList, 'lat': latList}, method='linear')
            #obsDataL3 = xr.merge([obsDataL2['rain'], contDataL4])
            obsDataL3 = xr.merge([obsDataL1['rain'], contDataL4])
            time_index = pd.to_datetime(obsDataL3.time.values)
            normalized_time_index = time_index.normalize()
            obsDataL3['time'] = ('time', normalized_time_index)

            # obsDataL2.attrs
            # obsDataL2['rain'].attrs

            keyList = sysOpt['keyList']
            for keyInfo in keyList:
                log.info(f"[CHECK] keyInfo : {keyInfo}")

                # 관측/학습 데이터
                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], serviceName, keyInfo, 'historical')
                # inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Historical', keyInfo, 'historical')
                fileList = sorted(glob.glob(inpFile))

                if not fileList:
                    log.info(f"No files found for {keyInfo}. Skipping this keyInfo.")
                    continue

                # fileInfo = fileList[0]
                # fileInfo = fileList[1]
                modDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    # fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                    modData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                    if (len(modData['time']) < 1):
                        log.info(f"No time data in file: {fileInfo}")
                        continue

                    # 필요없는 변수 삭제
                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']

                    for i, selInfo in enumerate(selList):
                        try:
                            modData = modData.drop([selInfo])
                        except Exception as e:
                            pass
                            # log.error("Exception : {}".format(e))

                    modDataL1 = modData.interp({'lon': lonList, 'lat': latList}, method='linear')

                    # 일 강수량 단위 환산 : 60 * 60 * 24
                    modDataL1['pr'] = modDataL1['pr']
                    modDataL1['pr'].attrs["units"] = "mm d-1"

                    modDataL2 = xr.merge([modDataL2, modDataL1])

                modDataL3 = xr.merge([modDataL2['pr'], contDataL4])
                modDataL3C = add_leap_day_mrg(modDataL3.drop_vars('contIdx'), var_name='pr')
                obsDataL3['time'] = ('time', normalized_time_index)

                # 병합 데이터 : 실측 + 관측/학습
                # mrgData = xr.merge([obsDataL3, modDataL3C])

                # 예측 데이터
                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], serviceName, keyInfo, 'ssp126')
                # inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Future', keyInfo, 'ssp126')
                fileList = sorted(glob.glob(inpFile))

                # fileInfo = fileList[0]
                simDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    # fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                    simData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate2'], sysOpt['endDate2']))
                    if (len(simData['time']) < 1): continue

                    # 필요없는 변수 삭제
                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']

                    for i, selInfo in enumerate(selList):
                        try:
                            simData = simData.drop([selInfo])
                        except Exception as e:
                            pass
                            # log.error("Exception : {}".format(e))

                    simDataL1 = simData.interp({'lon': lonList, 'lat': latList}, method='linear')

                    # 일 강수량 단위 환산 : 60 * 60 * 24
                    simDataL1['pr'] = simDataL1['pr']
                    simDataL1['pr'].attrs["units"] = "mm d-1"

                    simDataL2 = xr.merge([simDataL2, simDataL1])

                simDataL3 = xr.merge([simDataL2['pr'], contDataL4])
                attrs = {"units":"mm d-1"}

                simDataL3Processed = add_leap_day_L3(simDataL3.drop_vars('contIdx'), var_name='pr')
                simDataL3Processed.transpose('time','lat','lon')
                modDataL3C.transpose('time','lat','lon')
                obsDataL3.drop_vars('contIdx').transpose('time','lat','lon')

                obsDataL3U = xr.DataArray(obsDataL1['rain'], dims=("time", "lat", "lon"),
                                         coords={'time': obsDataL3['time'], "lat": latList, "lon": lonList},
                                         attrs=attrs).transpose("time","lat","lon").to_dataset(name="pr")
                modDataL3CU = xr.DataArray(modDataL3C['pr'], dims=("time", "lat", "lon"),
                                          coords={'time': modDataL3C['time'], "lat": latList, "lon": lonList},
                                          attrs=attrs).transpose("time", "lat", "lon").to_dataset(name="pr")
                simDataL3ProcessedU = xr.DataArray(simDataL3Processed['pr'], dims=("time", "lat", "lon"),
                                           coords={'time': simDataL3Processed['time'], "lat": latList, "lon": lonList},
                                           attrs=attrs).transpose("time", "lat", "lon").to_dataset(name="pr")
                methods = ['quantile_mapping', 'quantile_delta_mapping', 'linear_scaling', 'delta_method']

                # 병렬 처리를 위한 편향 조정 호출
                log.info("Starting adjust_future_periods_dask for future data.")
                future_results = adjust_future_periods_dask(obs=obsDataL3U, sim_hist=modDataL3CU, sim_future=simDataL3ProcessedU, client=client)
                log.info("Parallel adjust_future_periods_dask completed for future data.")

                QDM_Futresult = future_results['quantile_delta_mapping']
                LS_Futresult = future_results['linear_scaling']
                DM_Futresult = future_results['delta_method']
                QM_Futresult = future_results['quantile_mapping']

                # Future 결과에 대한 유사한 처리
                QDM_Futresult_Con = xr.merge([future_results['quantile_delta_mapping'], contDataL4]).rename({'pr': 'QDM'})
                LS_Futresult_Con = xr.merge([future_results['linear_scaling'], contDataL4]).rename({'pr': 'LS'})
                #VS_Futresult_Con = xr.merge([VS_Futresult['VS_Fut'], contDataL4]).rename({'pr': 'VS_Fut'})
                DM_Futresult_Con = xr.merge([future_results['delta_method'], contDataL4]).rename({'pr': 'DM'})
                QM_Futresult_Con = xr.merge([future_results['quantile_mapping'], contDataL4]).rename({'pr': 'QM'})


                saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', 'QDM', keyInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                QDM_Futresult.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', 'LS', keyInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                LS_Futresult.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')


                saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', 'DM', keyInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                DM_Futresult.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', 'QM', keyInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                QM_Futresult.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')


            lat1D = QDM_Futresult_Con['lat'].values
            lon1D = QDM_Futresult_Con['lon'].values
            time1D = QDM_Futresult_Con['time'].values

            Fut_QDMDataL1 = xr.Dataset(
                {
                    'SIM': (('time', 'lat', 'lon'),
                            (simDataL3Processed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'Porjected_pr': (('time', 'lat', 'lon'),
                               (QDM_Futresult_Con['QDM'].transpose('time', 'lat', 'lon').values).reshape(len(time1D),
                                                                                                  len(lat1D),
                                                                                                  len(lon1D)))
                    # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'contIdx': (('time', 'lat', 'lon'),
                                  np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(
                                      len(time1D), len(lat1D), len(lon1D)))
                }
                , coords={
                    'time': time1D
                    , 'lat': lat1D
                    , 'lon': lon1D
                }
            )



            Fut_LSDataL1 = xr.Dataset(
                {
                    'SIM': (('time', 'lat', 'lon'),
                            (simDataL3Processed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'Porjected_pr': (('time', 'lat', 'lon'),
                                       (LS_Futresult_Con['LS'].transpose('time', 'lat', 'lon').values).reshape(
                                           len(time1D),
                                           len(lat1D),
                                           len(lon1D)))
                    # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'contIdx': (('time', 'lat', 'lon'),
                                  np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(
                                      len(time1D), len(lat1D), len(lon1D)))
                }
                , coords={
                    'time': time1D
                    , 'lat': lat1D
                    , 'lon': lon1D
                }
            )


            Fut_DMDataL1 = xr.Dataset(
                {
                    'SIM': (('time', 'lat', 'lon'),
                            (simDataL3Processed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'Porjected_pr': (('time', 'lat', 'lon'),
                                       (DM_Futresult_Con['DM'].transpose('time', 'lat', 'lon').values).reshape(
                                           len(time1D),
                                           len(lat1D),
                                           len(lon1D)))
                    # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'contIdx': (('time', 'lat', 'lon'),
                                  np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(
                                      len(time1D), len(lat1D), len(lon1D)))
                }
                , coords={
                    'time': time1D
                    , 'lat': lat1D
                    , 'lon': lon1D
                }
            )

            Fut_QMDataL1 = xr.Dataset(
                {
                    'SIM': (('time', 'lat', 'lon'),
                            (simDataL3Processed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'Porjected_pr': (('time', 'lat', 'lon'),
                                       (QM_Futresult_Con['QM'].transpose('time', 'lat', 'lon').values).reshape(
                                           len(time1D),
                                           len(lat1D),
                                           len(lon1D)))
                    # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
                    , 'contIdx': (('time', 'lat', 'lon'),
                                  np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(
                                      len(time1D), len(lat1D), len(lon1D)))
                }
                , coords={
                    'time': time1D
                    , 'lat': lat1D
                    , 'lon': lon1D
                }
            )


            # ***********************************************************************************
            # 미래 기간 보정 NetCDF에서 세부 저장
            # ***********************************************************************************
            varList = ['SIM', 'Porjected_pr']
            contIdxList = np.unique(QDM_Futresult_Con['contIdx'])
            Fut_QDMDataL2 = Fut_QDMDataL1.to_dataframe().reset_index(drop=False)
            Fut_LSDataL2 = Fut_LSDataL1.to_dataframe().reset_index(drop=False)
            # Fut_VSDataL2 = Fut_VSDataL1.to_dataframe().reset_index(drop=False)
            Fut_DMDataL2 = Fut_DMDataL1.to_dataframe().reset_index(drop=False)
            Fut_QMDataL2 = Fut_QMDataL1.to_dataframe().reset_index(drop=False)
            # Fut_DQMDataL2 = Fut_DQMDataL1.to_dataframe().reset_index(drop=False)

            for varInfo in varList:
                selCol = ['time', 'lon', 'lat', varInfo]


                # 대륙별로 엑셀 저장
                for contIdx in contIdxList:
                    if np.isnan(contIdx): continue

                    Fut_QDMDataL3 = Fut_QDMDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    if len(Fut_QDMDataL3) < 0: continue
                    Fut_LSDataL3 = Fut_LSDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    if len(Fut_LSDataL3) < 0: continue
                    # Fut_VSDataL3 = Fut_VSDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    # if len(Fut_VSDataL3) < 0: continue
                    Fut_DMDataL3 = Fut_DMDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    if len(Fut_DMDataL3) < 0: continue
                    Fut_QMDataL3 = Fut_QMDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    if len(Fut_QMDataL3) < 0: continue
                    # Fut_DQMDataL3 = Fut_DQMDataL2.loc[Fut_QDMDataL2['contIdx'] == contIdx].reset_index(drop=True)
                    # if len(Fut_DQMDataL3) < 0: continue

                    sheetName = str(int(contIdx))

                    # mrgDataL6 = mrgDataL5[selCol].pivot(index=['lon', 'lat'], columns=['time'])
                    # mrgDataL6 = mrgDataL5[selCol].dropna().pivot(index=['lon', 'lat'], columns=['time'])
                    Fut_QDMDataL4 = Fut_QDMDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])
                    Fut_LSDataL4 = Fut_LSDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])
                    # Fut_VSDataL4 = Fut_VSDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])
                    Fut_DMDataL4 = Fut_DMDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])
                    Fut_QMDataL4 = Fut_QMDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])
                    #Fut_DQMDataL4 = Fut_DQMDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])



                    saveCsvFile1 = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC',
                                                                    varInfo, 'QDM', sheetName, keyInfo)
                    os.makedirs(os.path.dirname(saveCsvFile1), exist_ok=True)

                    csvData = Fut_QDMDataL4.reset_index(drop=False)
                    # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                    csvData.to_csv(saveCsvFile1, index=False)
                    log.info(f'[CHECK] saveCsvFile : {saveCsvFile1}')

                    saveCsvFile2 = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC',
                                                                    varInfo, 'LS', sheetName, keyInfo)
                    os.makedirs(os.path.dirname(saveCsvFile2), exist_ok=True)

                    csvData = Fut_LSDataL4.reset_index(drop=False)
                    # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                    csvData.to_csv(saveCsvFile2, index=False)
                    log.info(f'[CHECK] saveCsvFile : {saveCsvFile2}')

                    saveCsvFile3 = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC',
                                                                     varInfo, 'VS', sheetName, keyInfo)
                    os.makedirs(os.path.dirname(saveCsvFile3), exist_ok=True)

                    # csvData = Fut_VSDataL4.reset_index(drop=False)
                    # # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                    # csvData.to_csv(saveCsvFile3, index=False)
                    # log.info(f'[CHECK] saveCsvFile : {saveCsvFile3}')

                    saveCsvFile4 = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC',
                                                                     varInfo, 'DM', sheetName, keyInfo)
                    os.makedirs(os.path.dirname(saveCsvFile4), exist_ok=True)

                    csvData = Fut_DMDataL4.reset_index(drop=False)
                    # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                    csvData.to_csv(saveCsvFile4, index=False)
                    log.info(f'[CHECK] saveCsvFile : {saveCsvFile4}')

                    saveCsvFile5 = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC',
                                                                     varInfo, 'QM', sheetName, keyInfo)
                    os.makedirs(os.path.dirname(saveCsvFile5), exist_ok=True)

                    csvData = Fut_QMDataL4.reset_index(drop=False)
                    # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                    csvData.to_csv(saveCsvFile5, index=False)
                    log.info(f'[CHECK] saveCsvFile : {saveCsvFile5}')

            del simDataL3ProcessedU  # 미래 데이터 관련 변수 삭제
            gc.collect()  # 가비지 컬렉션 실행하여 메모리 해제



        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("exec"))
            shutdown_dask_cluster(self.client)

# ================================================
# 3. 주 프로그램
# ================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    mp.freeze_support()

    logging.info('스크립트 시작: 잔여 Dask 워커 프로세스 정리 중...')
    os.system('taskkill /f /im dask-worker*')

    try:
        logging.info('Dask 클러스터 생성 중...')

        # CPU 코어 수에 비례하여 워커와 스레드 수를 조정
        cluster = LocalCluster(# heartbeat_interval= 100000,
                               # n_workers=60,  # 적절한 워커 수 설정
                               n_workers=10,  # 적절한 워커 수 설정
                               processes=True,
                               threads_per_worker=2,
                               memory_target_fraction=0.6,  # 메모리 사용률 60%에 도달하면 관리 시작
                               memory_spill_fraction=0.7, # 메모리 사용률 70%에 도달하면 스왑 시작
                               )  # 적절한 메모리 제한 설정
        client = Client(cluster, timeout="3000s")

        # 클라이언트를 통해 모든 워커에서 가비지 컬렉션 실행
        client.run(gc.collect)

        inParams = {}
        logging.info(f"[CHECK] inParams : {inParams}")

        # DtaProcess 인스턴스 생성 시 client 인스턴스 전달
        subDtaProcess = DtaProcess(inParams, client)
        subDtaProcess.exec()

    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"An error occurred during the adjustment process: {e}")
        sys.exit(1)

    finally:
        if 'client' in locals():
            client.close()
            logging.info('Dask 클라이언트 종료됨.')
        if 'cluster' in locals():
            cluster.close()
            logging.info('Dask 클러스터 종료됨.')

        # 모든 작업이 완료된 후 잔여 프로세스를 정리
        print("Dask client and cluster have been closed.")
        logging.info('스크립트 종료: 잔여 Dask 워커 프로세스 정리 중...')
        os.system('taskkill /f /im dask-worker*')