# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
# from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import gzip
import shutil

import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from pyproj import Proj, Transformer
import re
# import xarray as xr
# from sklearn.neighbors import BallTree
import matplotlib.cm as cm
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# from pymap3d import enu2geodetic
import requests
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
import time
import pytz
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor

import requests
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
import time
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import concurrent.futures

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

kst = pytz.timezone("Asia/Seoul")

# plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# =================================================
# 2. 유틸리티 함수
# =================================================
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

    if not os.path.exists(os.path.dirname(saveLogFile)):
        os.makedirs(os.path.dirname(saveLogFile))

    # logger instance 생성
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

    return log

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
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

        # 글꼴 설정
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar

@retry(stop_max_attempt_number=1)
def cfgProc(sysOpt):
    try:
        res = requests.get(sysOpt['cfgUrl'])
        if not (res.status_code == 200): return None

        resJson = res.json()
        if resJson is None or len(resJson) < 1: return None

        resData = pd.DataFrame(resJson)
        if resData is None or len(resData) < 1: return None

        resDataL1 = resData[resData['symbol'].str.endswith('USDT')]

        return resDataL1

    except Exception as e:
        log.error(f"Exception : {str(e)}")
        return None

# 누적합 계산 후 리턴 함수
def calculate_sum_cv(data, start_time, end_time):
    # 1. 시작 시간과 끝 시간으로 데이터를 필터링
    filtered_data = data[(data['Open_time'] >= pd.to_datetime(start_time)) & (data['Open_time'] <= pd.to_datetime(end_time))]

    # 2. tb_quote_av - tb_quote_sell 차이 계산 및 누적합
    filtered_data['sum_cv'] = (filtered_data['tb_quote_av'] - filtered_data['tb_quote_sell']).cumsum()

    # 3. Close 및 sum_cv에 대해 0~1 사이 값으로 표준화 (정규화)
    scaler = MinMaxScaler()
    filtered_data[['normal_Close', 'normal_sum_cv']] = scaler.fit_transform(filtered_data[['Close', 'sum_cv']])

    # 4. Close 및 sum_cv의 최대-최소값 범위를 계산
    close_range = filtered_data['Close'].max() - filtered_data['Close'].min()
    sum_cv_range = filtered_data['sum_cv'].max() - filtered_data['sum_cv'].min()

    # 5. 계산된 범위를 모든 행에 추가
    filtered_data['close_range'] = close_range
    filtered_data['sum_cv_range'] = sum_cv_range

    # 6. 결과 반환 (Open_time, Close, sum_cv, normal_Close, normal_sum_cv, close_range, sum_cv_range)
    return filtered_data[
        ['Open_time', 'Close', 'sum_cv', 'normal_Close', 'normal_sum_cv', 'close_range', 'sum_cv_range']]


# 그래프를 그리는 함수 정의
def plot_close_and_sum_cv(sysOpt, modelInfo, symbol, result):

    dtDateInfo = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
    minDt = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M').strftime('%Y%m%d%H%M')
    maxDt = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M').strftime('%Y%m%d%H%M')
    saveImgPattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
    saveImg = dtDateInfo.strftime(saveImgPattern).format(symbol=symbol, minDt=minDt, maxDt=maxDt)
    # mainTitle = os.path.basename(saveImg).split(".")[0]
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)

    # 하나의 subplot을 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # normal_Close를 검은색 선으로
    ax.plot(result['Open_time'], result['normal_Close'], marker='o', color='black', label='normal_Close')

    # normal_sum_cv를 파란색 선으로
    ax.plot(result['Open_time'], result['normal_sum_cv'], marker='o', color='blue', label='normal_sum_cv')

    # 그래프 제목 및 축 레이블 설정
    ax.set_title('Normalized Close and Cumulative sum_cv Over Time')
    ax.set_xlabel('Open Time')
    ax.set_ylabel('Normalized Value')
    ax.tick_params(axis='x', rotation=45)

    # 범례 추가
    ax.legend(loc='upper left')

    # 그래프 우측 상단에 텍스트로 척도 정보 추가 (Close와 sum_cv 범위)
    close_range = result['close_range'].iloc[0]
    sum_cv_range = result['sum_cv_range'].iloc[0]
    textstr = f'Close Range: {close_range:.2f}\nsum_cv Range: {sum_cv_range:.2f}'

    # 텍스트를 그래프 안에 추가 (우측 상단)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5))

    # 그래프 간 여백 조정
    plt.tight_layout()

    # 그래프 출력
    # plt.show()

    plt.close()
    log.info(f"[CHECK] saveImg : {saveImg}")

# calculate_sum_cv 함수 최적화
def calculate_sum_cv_last(data, start_time, end_time):
    # 1. 시작 시간과 끝 시간으로 데이터를 필터링
    filtered_data = data.loc[
        (data['Open_time'] >= pd.to_datetime(start_time)) & (data['Open_time'] <= pd.to_datetime(end_time))].copy()

    # 2. tb_quote_av - tb_quote_sell 차이 계산 및 누적합
    filtered_data.loc[:, 'sum_cv'] = (filtered_data['tb_quote_av'] - filtered_data['tb_quote_sell']).cumsum()

    # 3. Close 및 sum_cv에 대해 0~1 사이 값으로 표준화 (정규화)
    scaler = MinMaxScaler()
    filtered_data[['normal_Close', 'normal_sum_cv']] = scaler.fit_transform(filtered_data[['Close', 'sum_cv']])

    # 4. Close 및 sum_cv의 최대-최소값 범위를 한 번만 계산
    close_range = filtered_data['Close'].max() - filtered_data['Close'].min()
    sum_cv_range = filtered_data['sum_cv'].max() - filtered_data['sum_cv'].min()

    # 5. 계산된 범위를 모든 행에 추가
    filtered_data['close_range'] = close_range
    filtered_data['sum_cv_range'] = sum_cv_range

    # 6. 마지막 행만 데이터프레임 형태로 반환
    return filtered_data[
               ['Open_time', 'Close', 'sum_cv', 'normal_Close', 'normal_sum_cv', 'close_range', 'sum_cv_range']].iloc[
           -1:]


# 병렬 처리용 함수 (calculate_sum_cv 호출)
def process_single_time(end_time, data_L1, time_deltas):
    temp_dfs = []
    for i, delta in enumerate(time_deltas):
        start_time = end_time - delta
        result_df = calculate_sum_cv_last(data_L1, start_time, end_time)
        result_df = result_df.add_suffix(f'_t{i + 1}')
        result_df.rename(columns={f'Open_time_t{i + 1}': 'Open_time'}, inplace=True)
        temp_dfs.append(result_df)

    # 열 병합 수행
    merged_df = pd.concat(temp_dfs, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    return merged_df


# 병렬 처리 적용
def process_in_parallel(data_L1, time_range, time_deltas):
    final_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_time, end_time, data_L1, time_deltas) for end_time in time_range]
        for future in concurrent.futures.as_completed(futures):
            final_results.append(future.result())

    # 행 병합
    final_data = pd.concat(final_results, axis=0).reset_index(drop=True)
    return final_data


# 6개의 그래프를 그리는 함수 정의
def plot_multiple_graphs(result):
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))  # 5x2 subplot 생성
    axes = axes.ravel()  # 2차원 배열을 1차원으로 변환하여 쉽게 접근할 수 있도록 함

    for i in range(6):
        # 각 t1 ~ t10에 맞는 데이터를 가져옴
        normal_close_col = f'normal_Close_t{i + 1}'
        normal_sum_cv_col = f'normal_sum_cv_t{i + 1}'
        close_range_col = f'close_range_t{i + 1}'
        sum_cv_range_col = f'sum_cv_range_t{i + 1}'

        # 그래프 그리기 (검은색 - normal_Close, 파란색 - normal_sum_cv)
        axes[i].plot(result['Open_time'], result[normal_close_col], color='black', label='normal_Close')
        axes[i].plot(result['Open_time'], result[normal_sum_cv_col], color='blue', label='normal_sum_cv')

        # 그래프 제목 및 축 레이블 설정
        axes[i].set_title(f'Time Range t{i + 1}')
        axes[i].set_xlabel('Open Time')
        axes[i].set_ylabel('Normalized Value')
        axes[i].tick_params(axis='x', rotation=45)

        # 범례 추가
        axes[i].legend(loc='upper left')

    # 그래프 간 여백 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()


# CSV 파일로 저장하는 함수
def save_daily_data_to_csv(sysOpt, modelInfo, symbol, data_L1, date, time_deltas):
    # 하루 단위의 시간을 설정 (00:00:00부터 23:59:59까지 1분 간격)
    start_of_day = pd.to_datetime(date)
    end_of_day = start_of_day + timedelta(days=1) - timedelta(minutes=1)
    time_range_day = pd.date_range(start=start_of_day, end=end_of_day, freq='1T')

    # 해당 날짜의 데이터 처리
    daily_data = process_in_parallel(data_L1, time_range_day, time_deltas)

    # 파일명 설정 (예: 2024-01-01.csv)
    # file_name = os.path.join(output_dir, f"{start_of_day.strftime('%Y-%m-%d')}.csv")

    # 데이터프레임을 CSV 파일로 저장
    # daily_data.to_csv(file_name, index=False)
    # print(f"Saved: {file_name}")

    saveFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
    saveFile = start_of_day.strftime(saveFilePattern).format(symbol=symbol)
    os.makedirs(os.path.dirname(saveFile), exist_ok=True)

    daily_data.to_csv(saveFile, index=False)
    log.info(f"[CHECK] saveFile : {saveFile}")

@retry(stop_max_attempt_number=5)
def colctTrendVideo(sysOpt, funName):
    print(f"[{datetime.now()}] {funName} : {sysOpt}")

    try:
        print(f"[{datetime.now()}] {funName} : {sysOpt}")
        raise Exception(f"예외 발생")
    except Exception as e:
        log.error(f"Exception : {str(e)}")
        raise e

async def asyncSchdl(sysOpt):

    scheduler = AsyncIOScheduler()
    scheduler.add_executor(AsyncIOExecutor(), 'default')

    # 정적 스케줄 등록
    # scheduler.add_job(colctTrendVideo, 'cron', second=0, args=[sysOpt, 'colctTrendVideo'])

    # 동적 스케줄 등록
    jobList = [
        (colctTrendVideo, 'cron', {'second': '0'}, {'args': [sysOpt, 'colctTrendVideo']})
        , (colctTrendVideo, 'cron', {'second': '30'}, {'args': [sysOpt, 'colctTrendVideo2']})
        , (colctTrendVideo, 'cron', {'second': '*/15'}, {'args': [None, 'colctTrendVideo3']})
    ]
    
    for fun, trigger, triggerArgs, kwargs in jobList:
        try:
            scheduler.add_job(fun, trigger, **triggerArgs, **kwargs)
        except Exception as e:
            log.error(f"Exception : {str(e)}")

    scheduler.start()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        asyncio.Event().close()
    finally:
        scheduler.shutdown()

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 비트코인 데이터 기반 AI 지능형 체계 구축

    # 0번 : 수집
    # 1번, 3번 : 전처리
    # 5번 : 모델생성 및 적용
    # 6번 : 시각화
    # 7번 : 백테스팅 코드

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0586'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                pass

            # globalVar['inpPath'] = '/DATA/INPUT'
            # globalVar['outPath'] = '/DATA/OUTPUT'
            # globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h, 분 1t)
                'srtDate': '2024-01-10 00:00'
                , 'endDate': '2024-01-20 00:00'
                # , 'endDate': '2024-01-10 06:00'
                , 'invDate': '1t'
                , 'timeDel': [
                    timedelta(days=7)   # 1주일 전
                    , timedelta(days=1) # 하루 전
                    , timedelta(hours=4)    # 4시간 전
                    , timedelta(hours=1)    # 1시간 전
                    , timedelta(minutes=15) # 15분 전
                    , timedelta(minutes=5)  # 5분 전
                ]

                # 비동기 다중 프로세스 개수
                , 'cpuCoreNum': '2'

                # 비동기 True, 동기 False
                # , 'isAsync': True
                , 'isAsync': False

                # 설정 정보
                , 'cfgUrl': 'https://api.binance.com/api/v3/ticker/price'

                # 수행 목록
                , 'modelList': ['USDT']
                # 세부 정보
                , 'USDT': {
                    # 'filePath': '/DATA/INPUT/LSH0579/uf'
                    # 'filePath': f'{contextPath}/uf'
                    # , 'fileName': 'RDR_{}_FQC_%Y%m%d%H%M.uf'

                    # 가공파일 시각화 여부
                    # , 'isProcVis': True
                    'isProcVis': False

                    # 수집 파일
                    , 'colctUrl': 'https://api.binance.com/api/v3/klines'
                    , 'colctColList': ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
                    , 'colctPath': '/DATA/OUTPUT/LSH0579/COLCT/%Y%m/%d/%H/{symbol}'
                    , 'colctName': '{symbol}_%Y%m%d%H%M.csv'

                    # 저장 영상
                    , 'figPath': '/DATA/OUTPUT/LSH0579/VIS/%Y%m/%d/%H/{symbol}'
                    , 'figName': '{symbol}_L1_{minDt}-{maxDt}.png'

                    # 가공 파일
                    , 'procPath': '/DATA/OUTPUT/LSH0579/PROC/%Y%m/%d/%H/{symbol}'
                    , 'procName': '{symbol}_%Y%m%d%H%M.csv'

                    # 엑셀 파일
                    # , 'xlsxPath': '/DATA/OUTPUT/LSH0579'
                    # , 'xlsxName': 'RDR_{}_FQC_{}-{}.xlsx'

                    # 누적 영상
                    # , 'cumPath': '/DATA/FIG/LSH0579'
                    # , 'cumName': 'RDR_{}_FQC-{}_%Y%m%d%H%M.png'
                }
            }

            # **********************************************************************************************************
            # 자동화
            # **********************************************************************************************************
            # asyncio.run(asyncSchdl(sysOpt))

            # **********************************************************************************************************
            # 수동화
            # **********************************************************************************************************
            # 시작일/종료일 설정
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # **************************************************************************************************************
            # 테스트
            # **************************************************************************************************************
            # endDate = datetime.now()

            cfgData = cfgProc(sysOpt)
            if cfgData is None or len(cfgData) < 1:
                log.error(f"[ERROR] cfgData['cfgUrl'] : {cfgData['cfgUrl']} / 설정 정보 URL을 확인해주세요.")
                raise Exception("오류 발생")

            # int(kst.localize(dtDateList[0]).timestamp() * 1000)

            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # symbol = 'BTCUSDT'
                for symbol in cfgData['symbol'].tolist():
                    # log.info(f'[CHECK] symbol : {symbol}')

                    # 테스트
                    if not 'BTCUSDT' == symbol: continue

                    # *******************************************************************************************
                    # 0번 : 수집
                    # Untitled.ipynb
                    # *******************************************************************************************
                    for dtDateInfo in dtDateList:
                        # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                        colctFilePattern = '{}/{}'.format(modelInfo['colctPath'], modelInfo['colctName'])
                        colctFile = dtDateInfo.strftime(colctFilePattern).format(symbol=symbol)
                        if os.path.exists(colctFile): continue

                        dtUnixTimeMs = int(dtDateInfo.timestamp() * 1000)
                        params = {
                            'symbol': symbol
                            , 'interval': '1m'
                            , 'limit': 1000
                            , 'startTime': dtUnixTimeMs
                            , 'endTime': dtUnixTimeMs
                        }

                        res = requests.get(modelInfo['colctUrl'], params=params)
                        if not (res.status_code == 200): return None

                        resJson = res.json()
                        if resJson is None or len(resJson) < 1: return None

                        resData = pd.DataFrame(resJson)
                        if resData is None or len(resData) < 1: return None

                        resDataL1 = resData
                        resDataL1.columns = modelInfo['colctColList']
                        resDataL1['Open_time'] = resDataL1.apply(lambda x: datetime.fromtimestamp(resDataL1['Open_time'] // 1000), axis=1)
                        resDataL1 = resDataL1.drop(columns=['Close_time', 'ignore'])
                        resDataL1['Symbol'] = symbol
                        resDataL1.loc[:, 'Open':'tb_quote_av'] = resDataL1.loc[:, 'Open':'tb_quote_av'].astype(float)
                        resDataL1['trades'] = resDataL1['trades'].astype(int)

                        # 파일 저장
                        os.makedirs(os.path.dirname(colctFile), exist_ok=True)
                        resDataL1.to_csv(colctFile, index=False)
                        log.info(f"[CHECK] colctFile : {colctFile}")

                    # *******************************************************************************************
                    # 1번, 3번 : 전처리
                    # Untitled1.ipynb
                    # *******************************************************************************************
                    data = pd.DataFrame()
                    for dtDateInfo in dtDateList:
                        colctFilePattern = '{}/{}'.format(modelInfo['colctPath'], modelInfo['colctName'])
                        colctFile = dtDateInfo.strftime(colctFilePattern).format(symbol=symbol)
                        orgData = pd.read_csv(colctFile)
                        if orgData is None or len(orgData) < 1: continue
                        data = pd.concat([data, orgData], ignore_index=True)

                    # 문자열을 datetime 형식으로 변환
                    data['Open_time'] = pd.to_datetime(data['Open_time'])

                    # KST TO UTC
                    data['Open_time'] = data['Open_time'] - timedelta(hours=9)

                    data_L1 = data.loc[:, ['Open_time', 'Close', 'trades', 'quote_av', 'tb_quote_av']]
                    data_L1['tb_quote_sell'] = data_L1['quote_av'] - data_L1['tb_quote_av']

                    # 예시 호출
                    # start_time = '2024-10-10 00:00:00'
                    # end_time = '2024-10-10 23:59:00'

                    result = calculate_sum_cv(data_L1, sysOpt['srtDate'], sysOpt['endDate'])

                    # 그림 생산
                    plot_close_and_sum_cv(sysOpt, modelInfo, symbol, result)

                    # 시간 범위 및 deltas 설정
                    # start_loop_date = pd.to_datetime('2024-10-28 00:00:00')
                    # end_loop_date = pd.to_datetime('2024-10-10 23:59:00')
                    # time_range = pd.date_range(start=start_loop_date, end=end_loop_date, freq='1T')
                    #
                    # time_deltas = [
                    #     timedelta(days=7),  # 1주일 전
                    #     timedelta(days=1),  # 하루 전
                    #     timedelta(hours=4),  # 4시간 전
                    #     timedelta(hours=1),  # 1시간 전
                    #     timedelta(minutes=15),  # 15분 전
                    #     timedelta(minutes=5)  # 5분 전
                    # ]

                    # 병렬 처리 함수 호출
                    # data_L2 = process_in_parallel(data_L1, time_range, time_deltas)
                    data_L2 = process_in_parallel(data_L1, dtDateList, sysOpt['timeDel'])
                    data_L2 = data_L2.sort_values(by='Open_time').reset_index(drop=True)

                    # CSV 저장 디렉토리 설정 (없으면 생성)
                    # output_dir = "L1OUT/BTC"
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)

                    # 시간 범위 및 deltas 설정
                    # start_loop_date = pd.to_datetime('2024-10-05')
                    # end_loop_date = pd.to_datetime('2024-10-29')
                    time_range = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')  # 1일 간격

                    # 날짜별로 데이터를 처리하고 CSV로 저장하는 루프
                    for single_date in time_range:
                        log.info(f"[CHECK] single_date : {single_date}")
                        save_daily_data_to_csv(sysOpt, modelInfo, symbol, data_L1, single_date, sysOpt['timeDel'])

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            # pool = Pool(int(sysOpt['cpuCoreNum'])) if sysOpt['isAsync'] else None
            #
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     for code in modelInfo['codeList']:
            #         log.info(f'[CHECK] code : {code}')
            #
            #         for dtDateInfo in dtDateList:
            #             if sysOpt['isAsync']:
            #                 # 비동기 자료 가공
            #                 pool.apply_async(radarProc, args=(modelInfo, code, dtDateInfo))
            #             else:
            #                 # 단일 자료 가공
            #                 radarProc(modelInfo, code, dtDateInfo)
            #         if pool:
            #             pool.close()
            #             pool.join()
            #
            #         # 지상관측소 및 레이더 간의 최근접 화소 찾기
            #         matchStnRadar(sysOpt, modelInfo, code, dtDateList)
            #
            #         # 자료 검증
            #         radarValid(sysOpt, modelInfo, code, dtDateList)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            # if pool:
            #     pool.terminate()
            raise e
        finally:
            # if pool:
            #     pool.close()
            #     pool.join()
            log.info('[END] {}'.format("exec"))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = {}
        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
