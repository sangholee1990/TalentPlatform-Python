# ================================================
# 요구사항
# ================================================
# Python을 이용한 기상청 AWS 다운로드

# ps -ef | grep "TalentPlatform-bdwide-colctAwsDbSave.py" | awk '{print $2}' | xargs kill -9
# pkill -f "TalentPlatform-bdwide-colctAwsDbSave.py"

# 프로그램 시작
# conda activate py38
# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT
# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/TalentPlatform-bdwide-colctAwsDbSave.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/TalentPlatform-bdwide-colctAwsDbSave.py > /dev/null 2>&1 &
# tail -f nohup.out

# tail -f /SYSTEMS/PROG/PYTHON/IDE/resources/log/test/Linux_x86_64_64bit_solarmy-253048.novalocal_test.log


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
import io
import uuid
import re
from datetime import datetime
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
import configparser
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RBFInterpolator
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import datetime
from matplotlib.ticker import FuncFormatter
from flaml import AutoML
import matplotlib.pyplot as plt
import numpy as np

import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import Point
import numpy as np

import matplotlib.pyplot as plt
import geojsoncontour
import geopandas as gpd
import numpy as np

import matplotlib.pyplot as plt
import geojsoncontour
import geopandas as gpd
import numpy as np
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geojsoncontour
import geopandas as gpd
import math

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

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
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
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
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

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'BDWIDE2026'

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
                'srtDate': '2025-08-01',
                'endDate': '2026-01-03',
                'cfgDbKey': 'mysql-iwin-dms01user01-DMS03',
                'cfgDb': None,

                # 설정 정보
                'stnFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/stnInfo/ALL_STN_INFO.csv',
                'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                'obsFile': '/HDD/DATA/INPUT/BDWIDE2026/OBS_ASOS_ANL_20260302205902.csv',
                'refFile': '/HDD/DATA/INPUT/BDWIDE2026/OBS_계절관측_20260128005918.csv',
            }

            # **********************************************************************************************************
            # 설정 정보
            # **********************************************************************************************************
            cfgData = pd.read_csv(sysOpt['stnFile'])
            cfgDataL1 = cfgData[['STN', 'LON', 'LAT']]


            obsData = pd.read_csv(sysOpt['obsFile'], encoding='EUC-KR')
            # obsData.columns
            obsData.columns = ['stnId', 'stnName', 'dt', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'sumSolarRad', 'avgWindSpeed']

            refData = pd.read_csv(sysOpt['refFile'], skiprows=2, encoding='EUC-KR')
            refDataL1 = refData.melt(id_vars=['지점', '년도'], var_name='구분', value_name='값')
            refDataL1[['구분1', '구분2']] = refDataL1['구분'].str.split('_', n=1, expand=True)

            obsData['year'] = obsData['dt'].astype(str)
            refDataL1['년도'] = refDataL1['년도'].astype(str)

            # dt를 datetime으로 변환 후 년도 추출
            # obsData['dt'] = pd.to_datetime(obsData['dt'])
            # obsData['년도'] = obsData['dt'].dt.year


            # 교집합 병합 (obsData: stnId/dt의 년도, refDataL1: 지점/년도)
            mergeData = pd.merge(obsData, refDataL1, left_on=['stnName', 'year'], right_on=['지점', '년도'], how='inner')
            mergeData = pd.merge(mergeData, cfgDataL1, left_on=['stnId'], right_on=['STN'], how='inner')

            # 속초 지역 & 개화 시기 필터링 (벚나무를 예시로 사용)
            # sokcho_df = mergeData[(mergeData['stnName'] == '속초')].copy()
            # sokcho_df = sokcho_df[sokcho_df['구분2'] == '개화'].copy()
            mergeData['구분1'].unique()
            mergeData['구분2'].unique()

            sokcho_df = mergeData[mergeData['구분2'] == '개화'].copy()
            sokcho_df['구분1'].unique()

            # 대상 식물(예: 벚나무)
            # sokcho_df = sokcho_df[sokcho_df['구분1'] == '개나리'].copy()  # 대상 식물(예: 벚나무)
            sokcho_df = sokcho_df[sokcho_df['구분1'] == '아까시나무'].copy()  # 대상 식물(예: 벚나무)
            # sokcho_df = sokcho_df[sokcho_df['구분1'] == '배나무'].copy()

            # 컬럼명 통일 및 시계열 처리 (결측치 등)
            sokcho_df = sokcho_df.rename(columns={'값': 'demand'})
            sokcho_df['demand'] = pd.to_datetime(sokcho_df['demand'], format='%Y-%m-%d', errors='coerce').dt.strftime('%j').astype('float')
            # sokcho_df['timeStamp'] = sokcho_df['timeStamp'].dt.to_period('Y').dt.to_timestamp()


            sokcho_df['timeStamp'] = pd.to_datetime(sokcho_df['year'], errors='coerce')
            #             # sokcho_df = sokcho_df.set_index('timeStamp')
            #             # # sokcho_df['temp'] = sokcho_df['temp'].fillna(method='ffill')
            #             # # sokcho_df['precip'] = sokcho_df['precip'].fillna(method='ffill')
            sokcho_df = sokcho_df.dropna(subset=['demand']).reset_index()
            # sokcho_df.columns

            # ---------------------------------------------------------
            # 모델 학습
            # ---------------------------------------------------------
            automl = AutoML()

            # 예측
            settings = {
                "time_budget": 60,
                # "time_budget": 600,
                "metric": "rmse",
                "task": "regression",
            }

            #  'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'sumSolarRad', 'avgWindSpeed'

            # 3. 예측에 사용할 피처에 시간 피처 추가
            features = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']

            X_train = sokcho_df[features].copy()
            y_train = sokcho_df['demand'].copy()

            automl.fit(X_train=X_train, y_train=y_train, **settings)

            # 5. 예측 수행 (가급적 X_test를 별도로 만들어 사용하는 것을 권장합니다)
            # print("====== 예측 결과 (Prediction) ======")
            predictions = automl.predict(X_train)
            # print(predictions)

            sokcho_df['prd'] = np.round(predictions).astype(int)

            # 1. 평가 지표 계산 (정답인 y_train과 모델이 찍은 predictions를 비교)
            rmse = np.sqrt(mean_squared_error(y_train,  sokcho_df['prd']))
            rrmse = (rmse / y_train.mean()) * 100
            nrmse = (rmse / (y_train.max() - y_train.min())) * 100
            mae = mean_absolute_error(y_train,  sokcho_df['prd'])
            r2 = r2_score(y_train,  sokcho_df['prd'])
            r = np.sqrt(r2)

            # print("====== 모델 검증 결과 (Metrics) ======")
            # print(f"RMSE (평균 오차): {rmse:.2f}")
            # print(f"MAE (절대 평균 오차): {mae:.2f}")
            # print(f"R2 Score (설명력, 1에 가까울수록 좋음): {r2:.2f}")

            # ---------------------------------------------------------
            # 웹 기반 가공 데이터 생산
            # ---------------------------------------------------------
            # 데이터 준비
            target_year = 2025
            year_df = sokcho_df[sokcho_df['dt'] == target_year].copy()
            year_df['prd'] = year_df['prd'].round().astype(int)

            lats = year_df['LAT'].values
            lons = year_df['LON'].values
            values = year_df['prd'].values

            obs_coords = np.column_stack((lons, lats))

            # 2. 고해상도 타겟 격자 생성 (500x500)
            grid_lon = np.linspace(124, 131, 1000)
            grid_lat = np.linspace(33, 39, 1000)
            Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            target_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))

            smoothing_candidates = np.arange(0, 100, 0.5)
            best_smoothing = 0.0
            min_error = float('inf')

            # LOOCV (Leave-One-Out Cross Validation) 방식으로 최적값 찾기
            for s in smoothing_candidates:
                errors = []
                # 관측소가 100개라면, 1개를 빼고 99개로 학습한 뒤 1개를 예측해보는 과정을 반복
                for i in range(len(obs_coords)):
                    # i번째 관측소만 제외 (검증용)
                    train_coords = np.delete(obs_coords, i, axis=0)
                    train_values = np.delete(values, i)
                    test_coord = obs_coords[i:i + 1]
                    test_value = values[i]

                    # RBF 보간
                    rbf = RBFInterpolator(train_coords, train_values, kernel='thin_plate_spline', smoothing=s)
                    pred_value = rbf(test_coord)[0]

                    errors.append((test_value - pred_value) ** 2)

                avg_rmse = np.sqrt(np.mean(errors))
                print(f" - Smoothing {s}: RMSE = {avg_rmse:.3f} 일")

                if avg_rmse < min_error:
                    min_error = avg_rmse
                    best_smoothing = s

            print(f"최적의 Smoothing 값 선택: {best_smoothing} (최소 오차: {min_error:.3f} 일)")

            print(f" {target_year}년 개화일 RBF 보간 중...")
            rbf = RBFInterpolator(
                y=obs_coords,
                d=values,
                kernel='thin_plate_spline',
                smoothing=best_smoothing,
            )
            grid_prd_smooth = rbf(target_coords).reshape(Grid_lon.shape)
            print(" 대한민국 영토 마스크 생성 중...")

            shpfilename = shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')
            reader = shpreader.Reader(shpfilename)

            # 'South Korea' (대한민국) 지형만 추출
            korea_geoms = []
            for record in reader.records():
                # ISO_A3 코드가 'KOR'인 국가(대한민국)를 찾습니다.
                if record.attributes['ISO_A3'] == 'KOR':
                    korea_geoms.append(record.geometry)

            # 추출한 대한민국 지형(본토 및 섬들)을 하나로 병합
            korea_geom = unary_union(korea_geoms)
            korea_prep = prep(korea_geom)

            # 격자점이 대한민국 영토 안에 있는지 확인하여 마스크 생성 (True: 한국, False: 바다/외국)
            mask = np.array([korea_prep.contains(Point(lon, lat))
                             for lon, lat in zip(Grid_lon.ravel(), Grid_lat.ravel())])
            mask = mask.reshape(Grid_lon.shape)

            grid_prd_final = np.where(mask, grid_prd_smooth, np.nan)

            # ---------------------------------------------------------
            # 웹 기반 가공 차트
            # ---------------------------------------------------------
            fig, ax = plt.subplots()
            levels = np.arange(int(np.nanmin(values)), int(np.nanmax(values)) + 2, 1)

            # 💡 변경점 1: contourf() 가 아닌 contour() 를 사용합니다! (f가 빠짐)
            contourf_plot = ax.contourf(Grid_lon, Grid_lat, grid_prd_final, levels=levels, cmap='YlGn')
            plt.close(fig)

            geojson_str = geojsoncontour.contourf_to_geojson(
                contourf=contourf_plot,
                ndigits=5,
                stroke_width=0
            )
            gdf_poly = gpd.read_file(geojson_str, driver='GeoJSON')
            gdf_poly = gdf_poly.set_crs(epsg=4326)
            gdf_poly['geometry'] = gdf_poly['geometry'].buffer(0)

            poly_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_poly_{target_year}.fgb"
            gdf_poly.to_file(poly_fgb, driver='FlatGeobuf')
            print(f"면(Polygon) 데이터 저장 완료: {poly_fgb}")

            point_geometries = [Point(xy) for xy in zip(year_df['LON'], year_df['LAT'])]

            gdf_points = gpd.GeoDataFrame(
                year_df,
                geometry=point_geometries,
                crs="EPSG:4326"
            )

            point_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_point_{target_year}.fgb"
            gdf_points.to_file(point_fgb, driver='FlatGeobuf')
            print(f"점(Point) 데이터 저장 완료: {point_fgb}")

            sys.exit(0)


            # 기후자료 예측 테스트
            fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/BDWIDE2026/AR6_SSP126_5ENSMN_skorea_*_gridraw_yearly_2021_2100.nc'))
            ds = xr.open_mfdataset(fileList)

            target_year = '2025'

            # ds['time']
            # # =========================================================
            # # 💡 1. xarray에서 2050년 3월~5월(봄) 데이터만 쏙 뽑아서 평균 내기
            # # =========================================================
            # # 특정 기간 슬라이싱 후, 시간(time) 차원에 대해 평균(mean)을 구합니다.
            # ds_spring = ds.sel(time=slice(f'{target_year}-01-01', f'{target_year}-01-01'))
            # ds_spring_mean = ds_spring.mean(dim='time')
            #
            # #       grid_lon = np.linspace(124, 131, 1000)
            # #             grid_lat = np.linspace(33, 39, 1000)
            # lonList = np.arange(124, 131, 0.01)
            # latList = np.arange(33, 39, 0.01)
            # ds_spring_mean = ds_spring_mean.interpolate_na({'longitude': lonList, 'latitude': latList}, method='linear', fill_value="extrapolate")
            #
            # ta_values = ds_spring_mean['TA'].values
            # lons = ds_spring_mean['longitude'].values
            # lats = ds_spring_mean['latitude'].values
            #
            # Grid_lon, Grid_lat = np.meshgrid(lons, lats)
            #
            # # =========================================================
            # # 2. 고해상도 등치면(Polygon) 생성 (전체 영역)
            # # =========================================================
            # print("등치면 생성 중... (마스크 없이 바다까지 포함)")
            #
            # fig, ax = plt.subplots()
            #
            # min_val = np.nanmin(ta_values)
            # max_val = np.nanmax(ta_values)
            # levels = np.arange(int(min_val) - 1, int(max_val) + 2, 1)
            # print(levels)
            #
            # # 기온 변화를 직관적으로 보여주는 컬러맵
            # contourf_plot = ax.contourf(Grid_lon, Grid_lat, ta_values, levels=levels, cmap='YlOrRd', extend='both')
            # # plt.show()
            # plt.close(fig)
            #
            # # =========================================================
            # # 3. GeoJSON 변환
            # # =========================================================
            # print("GeoJSON 변환 중...")
            # geojson_str = geojsoncontour.contourf_to_geojson(
            #     contourf=contourf_plot,
            #     ndigits=7,
            #     stroke_width=0
            # )
            #
            # # GeoPandas로 로드 및 꼬인 폴리곤(buffer 0) 복구
            # gdf = gpd.read_file(geojson_str, driver='GeoJSON')
            # gdf = gdf.set_crs(epsg=4326)
            # # gdf['geometry'] = gdf['geometry'].buffer(0)
            # # gdf['geometry'] = gdf.simplify(tolerance=0.005, preserve_topology=True)
            # gdf['year'] = int(target_year)
            #
            # gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            #
            # # =========================================================
            # # 4. FlatGeobuf 최종 저장 (클리핑 없이 즉시 저장)
            # # =========================================================
            # # output_fgb = f"future_climate_TA_{target_year}_nomask.fgb"
            # output_fgb = f"{globalVar['outPath']}/{serviceName}/future_climate_TA_{target_year}_nomask.fgb"
            #
            #
            # # 자르는 과정 없이 원본 gdf를 그대로 저장합니다.
            # gdf.to_file(output_fgb, driver='FlatGeobuf')
            #
            # print(f"마스크 없는 초고속 FGB 파일 저장 완료: {output_fgb}")
            #
            # import rasterio
            # # from rasterio.transform import from_bounds
            #
            # # 좌표계(CRS)를 위경도(EPSG:4326)로 설정
            # ds_spring_mean.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            # ds_spring_mean.rio.write_crs("epsg:4326", inplace=True)
            #
            # # 최종 COG(GeoTIFF) 파일로 저장
            # output_tif = f"{globalVar['outPath']}/{serviceName}/future_climate_TA_{target_year}.tif"
            # da_clipped.rio.to_raster(output_tif, driver="COG")
            #
            # print(f"✅ 초고속 렌더링용 COG 파일 저장 완료: {output_tif}")
            #
            # sys.exit(0)







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