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

import pandas as pd
import pybresenham
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
                # 'endDate': '2024-11-02 23:45',
                'invDate': '1h',

                # 수행 목록
                # 'modelList': [globalVar['modelList']],
                # 'modelList': ['UMKR'],

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                # 'cpuCoreNum': '5',

                # 설정 파일
                'GK2B-O3T': {
                    'filePattern': '/DATA/GK2B/*/GK2_GEMS_L2_%Y%m%d_%H%M_O3T_*_DPRO_*.nc',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
                'GK2B-O3P': {
                    'filePattern': '/DATA/GK2B/*/GK2_GEMS_L2_*_*_O3P_*_DPRO_*.nc',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
            }

            # **************************************************************************************************************
            # GK2B 환경위성 시계열
            # **************************************************************************************************************
            # GK2B O3T
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # target_lat = 36.0
            # target_lon = 130.0
            target_lat = 30.0
            target_lon = 120.0

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
                if np.isnan(val): continue
                # val = None if np.isnan(val) else val

                print(dtDateInfo, fileInfo, geoData['Latitude'].shape, val, nearest_point.spatial, nearest_point.image, nearest_point.Longitude, nearest_point.Latitude)

                item = {
                    'date': [dtDateInfo],
                    'val': [val],
                }

                dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(item)], axis=0)


            # dataL1 = dataL1.sort_values(by='date').reset_index(drop=True)

            # dtDateList

            # df = df.where(pd.notnull(df), None)


            # date_series = pd.to_datetime(dtDateList)
            dtDateData = pd.DataFrame(pd.to_datetime(dtDateList), columns=['date'])

            dataL2 = pd.merge(dtDateData, dataL1, left_on=['date'], right_on=['date'], how='left')
            dataL2['val'] = pd.to_numeric(dataL2['val'].astype(float), errors='coerce')

            # 지점 시계열
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dataL2['date'],
                y=dataL2['val'],
                mode='lines+markers',
                name='농도',
                connectgaps=True,
                marker=dict(size=6),
                line=dict(width=2),
                # hovertemplate = '<b>%{x|%Y-%m-%d %H:%M}</b><br>농도: %{y:.2f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="Date Time [%Y.%m.%d %H:%M]",
                yaxis_title="Value [DU]",
                xaxis=dict(
                    tickformat="%Y.%m.%d<br>%H:%M",
                    # tickangle=-45,
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(count=6, label="6h", step="hour", stepmode="backward"),
                            dict(count=12, label="12h", step="hour", stepmode="backward"),
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                hovermode="x unified",
            )

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

            # 지점 연직정보
            target_lat = 30.0
            target_lon = 120.0

            dist = ((data['Latitude'] - target_lat) ** 2 + (data['Longitude'] - target_lon) ** 2)
            posData = data.loc[dist.idxmin()]
            # posDataL1 = valDataL1.sel(spatial=int(posData.spatial), image=int(posData.image))

            sel_df = data[(data['spatial'] == posData.spatial) & (data['image'] == posData.image)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sel_df['O3'],
                y=sel_df['Altitude'],
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color=sel_df['O3'],
                    colorscale='Spectral',
                    colorbar=dict(title='', thickness=20),
                    line=dict(width=1, color='black')
                ),
                line=dict(color='grey', width=2),
                name='O3 Profile'
            ))

            fig.update_layout(
                # title='특정 지점 고도별 O₃ 농도 연직 프로파일',
                xaxis_title='Value [DU]',
                yaxis_title='Altitude [m]',
                # yaxis=dict(autorange='reversed', showgrid=True, gridwidth=1, gridcolor='lightgray'),
                # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                # plot_bgcolor='white',
                # width=600,
                # height=800,
                # template='plotly_white',
                # font=dict(size=14)
                # hovermode="x unified",
                hovermode="y unified",
                # hovermode="closest",
            )

            fig.show()


            bresenham_points = list(pybresenham.line(50, 35, 71, 42))
            bresenham_points_set = set(bresenham_points)
            idx = pd.MultiIndex.from_arrays([data['spatial'], data['image']])
            sel_df = data[idx.isin(bresenham_points_set)]

            theta_deg = 45*3
            theta_rad = np.deg2rad(theta_deg)
            r = 3
            eye_x = r * np.cos(theta_rad)
            eye_y = r * np.sin(theta_rad)
            eye_z = 1

            # 직선 연직정보
            # fig = go.Figure(data=[go.Mesh3d(
            fig = go.Figure()
            fig.add_trace(go.Mesh3d(
                x=sel_df['Longitude'],
                y=sel_df['Latitude'],
                # z=sel_df['Pressure'],
                z=sel_df['Altitude'],
                intensity=sel_df['O3'],
                colorscale='Spectral',
                colorbar=dict(title='', len=0.6),
                alphahull=0,
                opacity=1.0,
                # opacity=0.8,
            ))

            fig.update_layout(
                # title='3D 오존 농도 메쉬 플롯',
                scene=dict(
                    xaxis=dict(title='Longitude', autorange='reversed'),
                    yaxis=dict(title='Latitude', autorange='reversed'),
                    zaxis=dict(title='Altitude'),
                ),
                scene_camera=dict(
                    eye=dict(x=eye_x, y=eye_y, z=eye_z)
                )
            )

            fig.show()


            sel_df = data[(data['spatial'].between(50, 71)) & (data['image'].between(35, 42))]
            sel_df

            # 영역 연직정보
            fig = go.Figure()
            fig.add_trace(go.Mesh3d(
                x=sel_df['Longitude'],
                y=sel_df['Latitude'],
                # z=sel_df['Pressure'],
                z=sel_df['Altitude'],
                intensity=sel_df['O3'],
                colorscale='Spectral',
                colorbar=dict(title='', len=0.6),
                alphahull=0,
                opacity=1.0,
                # opacity=0.8,
            ))

            fig.update_layout(
                # title='3D 오존 농도 메쉬 플롯',
                scene=dict(
                    xaxis=dict(title='Longitude', autorange='reversed'),
                    yaxis=dict(title='Latitude', autorange='reversed'),
                    zaxis=dict(title='Altitude'),
                )
            )

            fig.show()

            # fig = go.Figure()
            # fig.add_trace(go.Mesh3d(
            #
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=sel_df['Longitude'],
            #     y=sel_df['Latitude'],
            #     z=sel_df['Pressure'],
            #     mode='markers',
            #     marker=dict(
            #         size=5,
            #         color=sel_df['O3'],
            #         colorscale='Viridis',
            #         colorbar=dict(title='O3 농도'),
            #         opacity=0.8
            #     )
            # )])
            #
            # fig.update_layout(
            #     title='3D 오존 농도 메쉬 플롯',
            #     scene=dict(
            #         xaxis_title='경도',
            #         yaxis_title='위도',
            #         zaxis_title='고도',
            #     ),
            #     # scene_camera=dict(
            #     #     eye=dict(x=eye_x, y=eye_y, z=eye_z)
            #     # )
            # )

            fig.show()

            #
            #
            #
            #
            # 3D 산점도 생성
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=data['Longitude'],
            #     y=data['Latitude'],
            #     # z=data['Pressure'],
            #     z=data['Altitude'],
            #     mode='markers',
            #     marker=dict(
            #         size=5,
            #         color=data['O3'],
            #         colorscale='Viridis',
            #         colorbar=dict(title='O3 농도'),
            #         opacity=0.8
            #     )
            # )])
            #
            # # 레이아웃 업데이트
            # fig.update_layout(
            #     title='3D 오존 농도 산점도',
            #     scene=dict(
            #         xaxis_title='Longitude (경도)',
            #         yaxis_title='Latitude (위도)',
            #         zaxis_title='ncolumns (층)'
            #     )
            # )
            #
            # fig.show()


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
