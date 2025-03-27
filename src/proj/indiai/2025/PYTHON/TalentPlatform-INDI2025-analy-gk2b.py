# ================================================
# 요구사항
# ================================================
# Python을 이용한 UMKR 수치모델 전처리

# ps -ef | grep "TalentPlatform-INDI2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2021-01-01'
# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '10' --srtDate '2019-01-01' --endDate '2021-01-01' &

import argparse
import glob
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# import cdsapi
import shutil

import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
from sklearn.neighbors import BallTree
import pygrib
from matplotlib import font_manager, rc
from metpy.units import units
from metpy.calc import wind_components, wind_direction, wind_speed

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

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')
dtKst = timedelta(hours=9)


# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        os.path.join(contextPath, 'log') if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

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
        # , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        # , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'INDI2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info(f"[END] __init__ : init")

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

        try:
            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 예보시간 시작일, 종료일, 시간 간격 (연 1y, 월 1m, 일 1d, 시간 1h, 분 1t, 초 1s)
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                'srtDate': '2024-11-01 00:45',
                'endDate': '2024-11-01 23:45',
                'invDate': '1h',

                # 수행 목록
                # 'modelList': [globalVar['modelList']],
                'modelList': ['UMKR'],

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',

                # 설정 파일
                'GK2B-O3T': {
                    'filePattern': '/DATA/GK2B/*/GK2_GEMS_L2_%Y%m%d_%H%M_O3T_*_DPRO_*.nc',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
                'GK2B-O3P': {
                    'filePattern': '/DATA/GK2B/*/GK2_GEMS_L2_*_*_O3P_*_DPRO_*.nc',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
                # 'ACT': {
                #     'ASOS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                #     'AWS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                # },
            }

            # **************************************************************************************************************
            # GK2B 환경위성 시계열
            # **************************************************************************************************************
            # GK2B O3T
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            target_lat = 36.0
            target_lon = 130.0

            dataL1 = pd.DataFrame()
            for dtDateInfo in dtDateList:

                filePattern = sysOpt['GK2B-O3T']['filePattern']
                fileList = sorted(glob.glob(dtDateInfo.strftime(filePattern)))
                if len(fileList) < 1: continue

                # data = xr.open_dataset(fileList[2], engine='pynio')
                # data.attrs
                # for fileInfo in fileList:

                fileInfo = fileList[0]

                geoData = xr.open_mfdataset(fileInfo, group='Geolocation Fields')
                # print(fileInfo, geoData['Latitude'].shape)

                # geoDataL1 = geoData[['Latitude', 'Longitude']]
                # geoDataL2 = geoDataL1.to_dataframe().reset_index(drop=False)

                # geoData['Latitude']
                # dist = ((geoDataL2['Latitude'] - target_lat) ** 2 + (geoDataL2['Longitude'] - target_lon) ** 2)
                # min_dist_idx = dist.idxmin()

                # 1. 위도, 경도 DataArray 가져오기
                lat_da = geoData['Latitude'].copy()
                lon_da = geoData['Longitude'].copy()

                # 2. 원하는 위경도 범위 설정 (예시)
                # lat_min, lat_max = 37.0, 38.0  # 예: 북위 37도 ~ 38도
                # lon_min, lon_max = 128.0, 129.0  # 예: 동경 128도 ~ 129도

                # 3. 불리언 마스크 생성 (Dask 연산, 아직 메모리에 다 로드되지 않음)
                mask = (lat_da >= target_lat - 0.07) & (lat_da <= target_lat + 0.07) & \
                       (lon_da >= target_lon - 0.07) & (lon_da <= target_lon + 0.07)

                # 4. 마스크를 Dataset에 적용
                # drop=False (기본값): 마스크가 False인 위치는 NaN으로 채워지고, Dataset 구조는 유지됨
                # subset_data_with_nan = geoData.where(mask)

                # drop=True: 마스크가 False인 위치의 데이터를 버림. 결과가 1차원으로 펼쳐질 수 있음
                subset_data_dropped = geoData.where(mask, drop=True)
                if subset_data_dropped['Latitude'].size < 1: continue
                if subset_data_dropped['Longitude'].size < 1: continue

                # print("--- .where(mask, drop=False) 결과 (일부 변수) ---")
                # 결과 확인 (NaN 값이 많이 포함될 수 있음)
                # print(subset_data_dropped[['Latitude', 'Longitude']])

                dd = subset_data_dropped.to_dataframe().reset_index(drop=False)

                dist = ((dd['Latitude'] - target_lat) ** 2 + (dd['Longitude'] - target_lon) ** 2)
                min_dist_idx = dist.idxmin()
                nearest_point = dd.loc[min_dist_idx]
                # print(nearest_point.spatial, nearest_point.image, nearest_point.Longitude, nearest_point.Latitude)

                valData = xr.open_dataset(fileInfo, group='Data Fields')
                # valDataL1 = valData['ColumnAmountO3']
                valDataL1 = valData['ColumnAmountO3'].sel(spatial = nearest_point.spatial, image = nearest_point.image)
                val = valDataL1.values
                # if np.isnan(val): continue
                val = None if np.isnan(val) else val

                print(dtDateInfo, fileInfo, geoData['Latitude'].shape, val, nearest_point.spatial, nearest_point.image, nearest_point.Longitude, nearest_point.Latitude)

                item = {
                    'date': [dtDateInfo],
                    'val': [val],
                }

                dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(item)], axis=0)


            # dataL1 = dataL1.sort_values(by='date').reset_index(drop=True)

            # dtDateList

            # df = df.where(pd.notnull(df), None)


            date_series = pd.Series(dtDateList, name='date')
            df = pd.DataFrame(date_series)

            dataL2 = pd.merge(df, dataL1, left_on=['date'], right_on=['date'], how='left')
            # dataL2['val'] = dataL2['val'].where(pd.notnull(dataL2['val']), None)
            dataL2['val'] = dataL2['val'].where(pd.notnull(dataL2['val']), np.nan)

            # dataL2['val'] = dataL2['val'].where(pd.notnull(dataL2['val']), np.nan)
            # filtered = dataL2.dropna(subset=['val'])
            dataL2['val'] = pd.to_numeric(dataL2['val'], errors='coerce')
            dataL2['date'] = dataL2['date'].astype(str)

            # dataL2['val'].to

            # dd = {
            #     "date": pd.date_range(start="2024-11-01 00:45:00", periods=24, freq="H"),
            #     "val": [None, 61.69367599487305, 67.14046478271484, 52.81095504760742, 45.49116897583008] + [None] * 19
            # }
            # dataL2 = pd.DataFrame(dd)

            # dataL2['date']


            import plotly.graph_objects as go
            import plotly.express as px
            # df.dtypes

            # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
            # fig = px.scatter(df, x='Date', y='AAPL.High', range_x=['2015-12-01', '2016-01-15'], title="Default Display with Gaps")
            # fig.show()
            # df.dtypes
            # df['Date']
            # df['AAPL.High']

            # print(dataL2['val'].notna().sum())

            # dataL2['date'] = dataL2['date'].astype(str)
            # dataL2['date'] = pd.to_datetime(dataL2['date'])
            # print(dataL2.info())

            # fig = px.scatter(dataL2, x='date', y='val', title="Default Display with Gaps")
            # fig.show()

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd  # pandas 예시를 위해 임포트

            # --- 데이터 준비 (Plotly 예시와 동일) ---
            # 1. Python 리스트 예시
            dates_list = pd.to_datetime(
                ['2024-11-01 01:00', '2024-11-01 02:00', '2024-11-01 03:00', '2024-11-01 04:00', '2024-11-01 05:00'])
            values_with_none = [10, 20, None, 40, None]  # None 포함 데이터

            # None을 np.nan으로 변환
            values_with_nan_list = [np.nan if v is None else v for v in values_with_none]

            # 2. pandas Series 예시
            s_dates = pd.to_datetime(
                ['2024-11-01 01:00', '2024-11-01 02:00', '2024-11-01 03:00', '2024-11-01 04:00', '2024-11-01 05:00'])
            s_values_with_none = pd.Series([10, 20, None, 40, None])

            # None을 np.nan으로 변환
            s_values_with_nan = s_values_with_none.fillna(np.nan)
            # 또는 pd.to_numeric 사용
            # s_values_with_nan = pd.to_numeric(s_values_with_none, errors='coerce')
            # --- 데이터 준비 완료 ---

            # --- Matplotlib 그래프 그리기 ---
            plt.figure(figsize=(10, 5))

            # np.nan으로 변환된 데이터 사용
            plt.plot(s_dates,  # x축 데이터 (pandas datetime Series)
                     s_values_with_nan,  # y축 데이터 (np.nan 포함된 pandas Series)
                     marker='o',  # 각 데이터 포인트에 마커 표시
                     linestyle='-',  # 선 스타일
                     label='Value')  # 범례 레이블

            # 그래프 제목 및 라벨 설정
            plt.title('Handling None/NaN in Matplotlib')
            plt.xlabel('Date')
            plt.ylabel('Value')

            # x축 눈금 레이블 회전 (선택 사항)
            plt.xticks(rotation=45)

            # 범례 표시
            plt.legend()

            # 그리드 추가 (선택 사항)
            plt.grid(True)

            # 레이아웃 조정
            plt.tight_layout()

            # 그래프 보여주기
            plt.show()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dataL2['date'],
                y=dataL2['val'],
                mode='lines+markers',
                # mode='markers',
                name='Value',
                connectgaps = True,
            ))

            fig.update_layout(
                # title="Time Series Data",
                xaxis_title="Date",
                yaxis_title="Value",
                xaxis=dict(tickangle=-45),
            )

            fig.show()


            # Scatter 트레이스 추가 (라인 플롯)
            fig.add_trace(go.Scatter(
                x=dataL2['date'],  # x축 데이터
                y=dataL2['val'],  # y축 데이터 (NaN 포함)
                # mode='lines+markers',  # 선과 마커 함께 표시
                # mode='markers',  # 선과 마커 함께 표시
                # mode='markers',  # 선과 마커 함께 표시
                name='값',  # 범례에 표시될 이름
                # connectgaps=False  # False (기본값): NaN 위치에서 선을 끊음
                # connectgaps=True
            ))

            # 레이아웃 업데이트 (제목, 축 라벨, rangeselector 등)
            fig.update_layout(
                title='시계열 데이터 플롯 (NaN 값을 포함한 원본)',
                xaxis=dict(
                    title='날짜',
                    tickformat="%Y-%m-%d %H:%M",  # 날짜/시간 표시 형식
                    # dtick=3600000, # 필요하다면 1시간 간격 눈금 설정
                    # rangeselector=dict(  # 시간 범위 선택 버튼
                    #     buttons=list([
                    #         dict(count=6, label="6h", step="hour", stepmode="backward"),
                    #         dict(count=12, label="12h", step="hour", stepmode="backward"),
                    #         dict(count=1, label="1d", step="day", stepmode="backward"),
                    #         dict(step="all")
                    #     ])
                    # ),
                    rangeslider=dict(visible=True),  # 하단 범위 슬라이더
                    type="date"
                ),
                yaxis_title='값',
                hovermode='x unified'
            )

            # 그래프 보이기
            fig.show()



            # **************************************************************************************************************
            # GK2B 환경위성 연직정보
            # **************************************************************************************************************
            # GK2B O3P
            filePattern = sysOpt['GK2B-O3P']['filePattern']
            fileList = sorted(glob.glob(filePattern))

            # data = xr.open_dataset(fileList[2], engine='pynio')
            # data.attrs

            # data.var()
            geoData = xr.open_dataset(fileList[0], group='Geolocation Fields')
            # geoDataL1 = geoData[['Latitude', 'Longitude', 'Pressure']]
            # geoDataL1 = geoData[['Latitude', 'Longitude', 'Pressure']]
            geoDataL1 = geoData[['Latitude', 'Longitude', 'Altitude']]
            geoDataL2 = geoDataL1.to_dataframe().reset_index(drop=False)
            geoDataL2['nlayer'] = geoDataL2['nlevel'] - 1

            valData = xr.open_dataset(fileList[0], group='Data Fields')
            # valDataL1 = valData['ColumnAmountO3']
            valDataL1 = valData['O3']
            valDataL2 = valDataL1.to_dataframe().reset_index(drop=False)

            # data = pd.merge(geoDataL2, valDataL2, left_on=['spatial', 'image'], right_on=['spatial', 'image'], how='left')
            data = pd.merge(valDataL2, geoDataL2, left_on=['spatial', 'image', 'nlayer'], right_on=['spatial', 'image', 'nlayer'], how='left')

            # target_lat = 36.0
            # # target_lon = 128.0
            # target_lon = 130.0
            #
            # dist = ((data['Latitude'] - target_lat) ** 2 + (data['Longitude'] - target_lon) ** 2)
            # min_dist_idx = dist.idxmin()
            #
            # nearest_point = data.loc[min_dist_idx]

            # spatial            71.000000
            # image              42.000000
            # Latitude           35.982388
            # Longitude         127.903641
            # ncolumns            0.000000
            # ColumnAmountO3    238.398359
            # distance_sq         0.009595

            # spatial            71.000000
            # image              35.000000
            # Latitude           35.980808
            # Longitude         130.136658
            # ncolumns            0.000000
            # ColumnAmountO3    239.266364
            # distance_sq         4.565674

            # selData = valDataL1.sel(spatial=71, image=slice(35, 42))

            # selData.plot()
            # plt.show()

            # sel_df = data[(data['spatial'] == 71) & (data['image'].between(35, 42))]
            sel_df = data[(data['spatial'].between(50, 71)) & (data['image'].between(35, 42))]
            sel_df

            # pip install PyBresenham
            # import pybresenham
            # start_point = (50, 35)
            # end_point = (71, 42)

            # line_coordinates = list(pybresenham(start_point[0], start_point[1], end_point[0], end_point[1]))

            # bresenham_points = list(pybresenham.line(50, 35, 71, 42))
            # image 행 spatial 열
            import pybresenham
            bresenham_points = list(pybresenham.line(35, 50, 42, 71))
            sel_df = data[data.apply(lambda row: (row['image'], row['spatial']) in bresenham_points, axis=1)]

            # sel_df = data[data.apply(lambda row: (row['image'], row['spatial']) in bresenham_points, axis=1)]



            import plotly.graph_objects as go
            #
            # fig = go.Figure(data=[go.Surface(
            #     x=sel_df['Longitude'],
            #     y=sel_df['Latitude'],
            #     z=sel_df['Pressure'],
            #     surfacecolor=sel_df['O3'],
            #     colorbar_title='O3 농도'
            # )])
            #
            # fig.update_layout(
            #     title='3D 오존 농도 Surface',
            #     scene=dict(
            #         xaxis_title='Longitude (경도)',
            #         yaxis_title='Latitude (위도)',
            #         zaxis_title='ncolumns (층)'
            #     )
            # )
            #
            # fig.show()

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # import numpy as np
            #
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(111, projection='3d')
            #
            # # 데이터 준비
            # x = sel_df['Longitude'].values
            # y = sel_df['Latitude'].values
            # z = sel_df['Altitude'].values
            # c = sel_df['O3'].values  # O3 농도 컬러 매핑
            #
            # # 3D Surface Mesh (Triangulation 기반)
            # surf = ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0.2, antialiased=True, shade=True)
            #
            # # 컬러바 추가
            # fig.colorbar(surf, ax=ax, label='O3 농도')
            #
            # # 라벨링
            # ax.set_title('3D 오존 농도 메쉬 플롯 (Matplotlib)')
            # ax.set_xlabel('Longitude (경도)')
            # ax.set_ylabel('Latitude (위도)')
            # ax.set_zlabel('Altitude (고도)')
            #
            # # 이미지로 저장
            # # plt.savefig('o3_mesh_matplotlib.png', dpi=300)
            #
            # plt.show()

            # import matplotlib.pyplot as plt
            # import numpy as np
            #
            # fig, ax = plt.subplots(figsize=(12, 8))
            #
            # # 데이터 추출
            # x = sel_df['Longitude'].values
            # y = sel_df['Latitude'].values
            # o3 = sel_df['O3'].values
            #
            # # 산점도로 그리면서 컬러맵 입히기 (한 판으로 O3 농도 시각화)
            # sc = ax.scatter(x, y, c=o3, cmap='viridis', s=50, marker='s')
            #
            # # 컬러바 추가
            # cbar = plt.colorbar(sc, ax=ax)
            # cbar.set_label('O3 농도')
            #
            # # 라벨링
            # ax.set_title('O3 농도 2D 평면 컬러맵')
            # ax.set_xlabel('Longitude (경도)')
            # ax.set_ylabel('Latitude (위도)')
            #
            # # 이미지 저장
            # plt.savefig('o3_2d_colormap.png', dpi=300)
            #
            # plt.show()

            # fig = go.Figure(data=[
            #     # Mesh trace (semi-transparent)
            #     go.Mesh3d(
            #         x=sel_df['Latitude'],
            #         y=sel_df['Longitude'],
            #         z=sel_df['Altitude'],
            #         intensity=sel_df['O3'],
            #         colorbar_title='O3 농도',
            #         alphahull=0,
            #         opacity=0.75,
            #         name='Mesh'  # Optional name
            #     ),
            #     # Scatter trace for points
            #     go.Scatter3d(
            #         x=sel_df['Latitude'],
            #         y=sel_df['Longitude'],
            #         z=sel_df['Altitude'],
            #         mode='markers',
            #         marker=dict(
            #             size=3,  # Adjust size as needed
            #             color=sel_df['O3'],  # Color points by O3
            #             # colorscale='Viridis',  # Match colorscale if desired
            #             opacity=0.8,  # Point opacity
            #             showscale=False  # Hide scatter's color bar if mesh one is sufficient
            #         ),
            #         name='Points'  # Optional name
            #     )
            # ])
            #
            # # Layout configuration (user's setup)
            # fig.update_layout(
            #     # title='3D 오존 농도 메쉬 + 포인트 플롯', # Optional: update title
            #     scene=dict(
            #         xaxis=dict(autorange='reversed'),  # Reverse Latitude axis
            #         xaxis_title='위도',
            #         yaxis_title='경도',
            #         zaxis_title='고도'
            #     ),
            #     legend_title_text='Trace Types'  # Add legend title if names are used
            # )
            #
            # fig.show()




            # Longitude, Latitude
            # Mesh3d 사용
            # 합격
            fig = go.Figure(data=[go.Mesh3d(
                x=sel_df['Latitude'],
                y=sel_df['Longitude'],
                # z=sel_df['Pressure'],
                z=sel_df['Altitude'],
                intensity=sel_df['O3'],
                # colorscale='Viridis',
                colorbar_title='O3 농도',
                alphahull=0,  # 알파 쉘 알고리즘을 사용하여 메쉬 생성
                opacity=0.7,
                name='Mesh'
            )])

            fig.update_layout(
                # title='3D 오존 농도 메쉬 플롯',
                scene=dict(
                    xaxis=dict(autorange='reversed'),
                    xaxis_title='위도',
                    yaxis_title='경도',
                    zaxis_title='고도'
                )
            )

            fig.show()


            # 3D 산점도 생성
            fig = go.Figure(data=[go.Scatter3d(
                x=sel_df['Longitude'],
                y=sel_df['Latitude'],
                z=sel_df['Pressure'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=sel_df['O3'],
                    colorscale='Viridis',
                    colorbar=dict(title='O3 농도'),
                    opacity=0.8
                )
            )])

            # 레이아웃 업데이트
            fig.update_layout(
                title='3D 오존 농도 산점도',
                scene=dict(
                    xaxis_title='Longitude (경도)',
                    yaxis_title='Latitude (위도)',
                    zaxis_title='ncolumns (층)'
                )
            )

            fig.show()

            import plotly.express as px

            # Plotly Express를 사용한 3D 산점도 생성
            fig = px.scatter_3d(
                sel_df,
                x='Longitude',
                y='Latitude',
                z='ncolumns',
                color='ColumnAmountO3',
                color_continuous_scale='Viridis',
                title='3D 오존 농도 산점도'
            )

            fig.show()

            import plotly.graph_objects as go

            # 빈 Figure 생성
            fig = go.Figure()

            # 고유한 ncolumns 값에 대해 반복
            for ncol in sel_df['ncolumns'].unique():
                # 해당 ncolumns 값에 대한 데이터 필터링
                filtered_df = sel_df[sel_df['ncolumns'] == ncol]

                # Scatter3d 트레이스 추가
                fig.add_trace(go.Scatter3d(
                    x=filtered_df['Longitude'],
                    y=filtered_df['Latitude'],
                    z=[ncol] * len(filtered_df),  # ncolumns 값을 z축에 사용
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=filtered_df['ColumnAmountO3'],  # 오존 농도를 색상에 매핑
                        colorscale='Viridis',
                        colorbar=dict(title='ColumnAmountO3')
                    ),
                    name=f'ncolumns={ncol}'
                ))

            # 레이아웃 업데이트
            fig.update_layout(
                title='각 ncolumns에 대한 3D 시각화',
                scene=dict(
                    xaxis_title='Longitude (경도)',
                    yaxis_title='Latitude (위도)',
                    zaxis_title='ncolumns (층)'
                )
            )

            # 그래프 표시
            fig.show()



            # aaa
            # Create contour plot using matplotlib
            # plt.figure(figsize=(10, 6))
            # # Use contourf for filled contours
            # # lons_2d = sel_df['Longitude'].squeeze('spatial').values  # Remove spatial dim, get numpy array
            # # lats_2d = sel_df['Latitude'].squeeze('spatial').values
            # # o3_data_2d = sel_df['ColumnAmountO3'].sel(ncolumns=ncolumn_index_to_plot).squeeze('spatial').values
            #
            #
            # contour = plt.contourf(lons_2d, lats_2d, o3_data_2d, levels=15, cmap='viridis', extend='both')
            # # Add contour lines
            # plt.contour(lons_2d, lats_2d, o3_data_2d, levels=contour.levels, colors='black', linewidths=0.5)
            #
            # # Add colorbar
            # cbar = plt.colorbar(contour, label=f'ColumnAmountO3 (ncolumns={ncolumn_index_to_plot}, DU)')
            #
            # # Labels and Title
            # plt.xlabel("Longitude (경도)")
            # plt.ylabel("Latitude (위도)")
            # plt.title(f"Contour Plot of ColumnAmountO3 (ncolumns={ncolumn_index_to_plot}, spatial=71)")
            # plt.grid(True, linestyle='--', alpha=0.6)
            # plt.axis('equal')  # Often useful for lat/lon plots
            # plt.show()




            # mrgData =
            # data['ncolumns'].values

            # data.plot()
            # plt.show()


            # data['Data_Fields/ColumnAmountO3']
            # data['ncolumns']

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print(f'[START] main')

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] main')
