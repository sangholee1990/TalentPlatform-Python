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
                'srtDate': '2019-01-01',
                'endDate': '2011-01-01',
                'invDate': '1d',

                # 수행 목록
                # 'modelList': [globalVar['modelList']],
                'modelList': ['UMKR'],

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',

                # 설정 파일
                'GK2B-O3T': {
                    'filePattern': '/DATA/GK2B/*/GK2_GEMS_L2_*_*_O3T_*_DPRO_*.nc',
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
            # GK2B 환경위성
            # **************************************************************************************************************
            # GK2B O3T
            filePattern = sysOpt['GK2B-O3P']['filePattern']
            fileList = sorted(glob.glob(filePattern))

            # data = xr.open_dataset(fileList[2], engine='pynio')
            # data.attrs

            # data.var()
            geoData = xr.open_dataset(fileList[0], group='Geolocation Fields')
            # geoData = xr.open_mfdataset(fileList, group='Geolocation Fields')
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

            target_lat = 36.0
            # target_lon = 128.0
            target_lon = 130.0

            dist = ((data['Latitude'] - target_lat) ** 2 + (data['Longitude'] - target_lon) ** 2)
            min_dist_idx = dist.idxmin()

            nearest_point = data.loc[min_dist_idx]

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
            # sel_df = data[(data['spatial'].between(70, 70)) & (data['image'].between(42, 42))]
            sel_df

            # pip install PyBresenham
            import pybresenham
            # start_point = (50, 35)
            # end_point = (71, 42)

            # line_coordinates = list(pybresenham(start_point[0], start_point[1], end_point[0], end_point[1]))

            bresenham_points = list(pybresenham.line(50, 35, 71, 42))
            # sel_df = data[data.apply(lambda row: (row['spatial'], row['image']) in bresenham_points, axis=1)]

            bresenham_points_set = set(bresenham_points)
            # filtered_data = data[data.apply(lambda row: (row['spatial'], row['image']) in bresenham_points_set, axis=1)]
            idx = pd.MultiIndex.from_arrays([data['spatial'], data['image']])
            # sel_df = data[idx.isin(bresenham_points_set)].sort_values(by=['spatial', 'image'], ascending=[False, False])
            sel_df = data[idx.isin(bresenham_points_set)]

            # Latitude   Longitude   Altitude
            # from scipy.interpolate import griddata
            # import matplotlib.pyplot as plt
            #
            # data = sel_df
            # grid_lon = np.linspace(data['Longitude'].min(), data['Longitude'].max(), 100)
            # grid_lat = np.linspace(data['Latitude'].min(), data['Latitude'].max(), 100)
            # grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            #
            # # Altitude 값을 그리드에 보간
            # grid_z = griddata(
            #     (data['Longitude'], data['Latitude']),
            #     data['Altitude'],
            #     (grid_x, grid_y),
            #     method='cubic'  # linear, cubic, nearest 가능
            # )
            #
            # # 등치선 그리기
            # plt.figure(figsize=(8, 6))
            # contour = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis')
            # plt.colorbar(contour, label='Altitude (m)')
            # plt.xlabel('Longitude')
            # plt.ylabel('Latitude')
            # plt.title('2D Altitude Contour')
            # plt.show()

            # import matplotlib.pyplot as plt
            #
            # plt.plot(sel_df['O3'], sel_df['Altitude'], marker='o')
            # plt.xlabel('O3 농도 (ppb)')
            # plt.ylabel('고도 (m)')
            # plt.title('특정 지점 고도별 O3 농도 프로파일')
            # # plt.gca().invert_yaxis()  # 고도가 위로 높아지게
            # plt.show()

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sel_df['O3'],
                y=sel_df['Altitude'],
                mode='markers+lines',
                marker=dict(
                    size=14,
                    color=sel_df['O3'],  # 농도 컬러 매핑
                    colorscale='Plasma',  # 컬러맵
                    colorbar=dict(title='O3 농도 (ppb)', thickness=20),
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                line=dict(color='gray', width=2),
                name='O3 Profile'
            ))

            fig.update_layout(
                title='특정 지점 고도별 O₃ 농도 연직 프로파일',
                xaxis_title='O₃ 농도 (ppb)',
                yaxis_title='고도 (m)',
                # yaxis=dict(autorange='reversed', showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                plot_bgcolor='white',
                # width=600,
                # height=800,
                template='plotly_white',
                font=dict(size=14)
            )

            fig.show()




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

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # 데이터 준비
            x = sel_df['Longitude'].values
            y = sel_df['Latitude'].values
            z = sel_df['Altitude'].values
            c = sel_df['O3'].values  # O3 농도 컬러 매핑

            # 3D Surface Mesh (Triangulation 기반)
            surf = ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0.2, antialiased=True, shade=True)

            # 컬러바 추가
            fig.colorbar(surf, ax=ax, label='O3 농도')

            # 라벨링
            ax.set_title('3D 오존 농도 메쉬 플롯 (Matplotlib)')
            ax.set_xlabel('Longitude (경도)')
            ax.set_ylabel('Latitude (위도)')
            ax.set_zlabel('Altitude (고도)')

            # 이미지로 저장
            # plt.savefig('o3_mesh_matplotlib.png', dpi=300)

            plt.show()

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

            theta_deg = 45*3
            theta_rad = np.deg2rad(theta_deg)

            r = 3  # 카메라 거리 (조절 가능)
            eye_x = r * np.cos(theta_rad)
            eye_y = r * np.sin(theta_rad)
            eye_z = 1  # 고도는 원하는 대로


            # Mesh3d 사용
            fig = go.Figure(data=[go.Mesh3d(
                x=sel_df['Longitude'],
                y=sel_df['Latitude'],
                # z=sel_df['Pressure'],
                z=sel_df['Altitude'],
                intensity=sel_df['O3'],
                colorscale='Viridis',
                colorbar_title='농도',
                alphahull=0,  # 알파 쉘 알고리즘을 사용하여 메쉬 생성
                opacity=1,
            )])

            fig.update_layout(
                title='3D 오존 농도 메쉬 플롯',
                scene=dict(
                    xaxis_title='경도',
                    yaxis_title='위도',
                    zaxis_title='고도',
                ),
                scene_camera = dict(
                    eye=dict(x=eye_x, y=eye_y, z=eye_z)
                )
            )

            fig.show()

            fig = go.Figure(data=[go.Mesh3d(
                x=data['Longitude'],
                y=data['Latitude'],
                # z=sel_df['Pressure'],
                z=data['Altitude'],
                intensity=sel_df['O3'],
                colorscale='Viridis',
                colorbar_title='농도',
                alphahull=0,  # 알파 쉘 알고리즘을 사용하여 메쉬 생성
                opacity=1,
            )])

            fig.update_layout(
                title='3D 오존 농도 메쉬 플롯',
                scene=dict(
                    xaxis_title='경도',
                    yaxis_title='위도',
                    zaxis_title='고도',
                ),
                # scene_camera=dict(
                #     eye=dict(x=eye_x, y=eye_y, z=eye_z)
                # )
            )

            fig.show()

            #
            #
            #
            #
            # # 3D 산점도 생성
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



            import plotly.express as px
            import pandas as pd

            # Plotly Express를 사용한 3D 산점도 생성
            fig = px.scatter_3d(
                data,
                x='Longitude',
                y='Latitude',
                z='ncolumns',
                color='ColumnAmountO3',
                color_continuous_scale='Viridis',
                title='3D 오존 농도 산점도'
            )

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
