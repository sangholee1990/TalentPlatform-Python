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

            automl = AutoML()

            # 예측
            settings = {
                "time_budget": 60,
                # "time_budget": 600,
                "metric": "rmse",
                "task": "regression",
            }

            # 3. 예측에 사용할 피처에 시간 피처 추가
            features = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']

            X_train = sokcho_df[features].copy()
            y_train = sokcho_df['demand'].copy()


            # 4. 모델 학습
            automl.fit(X_train=X_train, y_train=y_train, **settings)

            # 5. 예측 수행 (가급적 X_test를 별도로 만들어 사용하는 것을 권장합니다)
            # print("====== 예측 결과 (Prediction) ======")
            predictions = automl.predict(X_train)
            # print(predictions)

            sokcho_df['prd'] = np.round(predictions).astype(int)
            # sokcho_df['prd'] = predictions.astype(int)

            # sokcho_df['diff'] = sokcho_df['demand'] - sokcho_df['prd']
            # sokcho_df2 = sokcho_df.sort_values(by='diff', ascending=False)

            # sokcho_df = sokcho_df.drop(index=[1241])
            # sokcho_df = sokcho_df.reset_index()
            # y_train = sokcho_df['demand'].copy()

            # # 산점도
            # plt.figure(figsize=(8, 8))
            #
            # # 실제값(y_train)과 예측값(predictions)으로 산포도 그리기
            # plt.scatter(y_train, sokcho_df['prd'], color='dodgerblue', alpha=0.7, edgecolor='k', s=50, label='예측 데이터')
            #
            # # 예측이 100% 정확할 때를 나타내는 대각선 기준선 (y = x)
            # min_val = min(np.min(y_train), np.min(predictions))
            # max_val = max(np.max(y_train), np.max(predictions))
            #
            # # x, y축 범위를 약간 여유있게 조정
            # margin = (max_val - min_val) * 0.05
            # min_val -= margin
            # max_val += margin
            #
            # plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2,
            #          label='완벽한 예측 (y=x)')
            #
            # # 축 및 레이블 설정
            # plt.title('실제 개화일 vs 예측 개화일 산포도', fontsize=16, fontweight='bold', pad=15)
            # plt.xlabel('실제 개화일 (Day of year)', fontsize=12)
            # plt.ylabel('예측 개화일 (Day of year)', fontsize=12)
            #
            # # 축 범위 통일 (정사각형 형태 유지용)
            # plt.xlim(min_val, max_val)
            # plt.ylim(min_val, max_val)
            #
            # plt.grid(True, linestyle='--', alpha=0.6)
            # plt.legend(loc='upper left', fontsize=11)
            # plt.tight_layout()
            #
            # plt.show()


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


            # 1. 데이터 준비
            target_year = 2025
            year_df = sokcho_df[sokcho_df['dt'] == target_year].copy()
            year_df['prd'] = year_df['prd'].round().astype(int)

            lats = year_df['LAT'].values
            lons = year_df['LON'].values
            values = year_df['prd'].values

            obs_coords = np.column_stack((lons, lats))

            # 2. 고해상도 타겟 격자 생성 (500x500)
            # grid_lon = np.linspace(124, 131, 2000)
            # grid_lat = np.linspace(33, 39, 2000)
            grid_lon = np.linspace(124, 131, 1000)
            grid_lat = np.linspace(33, 39, 1000)
            Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            target_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))

            # 3. RBFInterpolator 적용
            # 테스트할 smoothing 후보군
            # smoothing_candidates = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 6]
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



            # 4. 육상 마스크 적용
            # land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')
            # land_geom = unary_union(list(land_feature.geometries()))
            # land_prep = prep(land_geom)
            #
            # mask = np.array([land_prep.contains(Point(lon, lat))
            #                  for lon, lat in zip(Grid_lon.ravel(), Grid_lat.ravel())])
            # mask = mask.reshape(Grid_lon.shape)


            # 4. 대한민국 영토 마스크 적용
            print(" 대한민국 영토 마스크 생성 중...")

            # Natural Earth에서 전 세계 국가 경계 데이터 가져오기 (10m 고해상도)
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
            # 📊 5. 시각화
            # ---------------------------------------------------------
            # plt.figure(figsize=(12, 12))
            # ax = plt.axes(projection=ccrs.PlateCarree())
            #
            # ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', zorder=5)
            # ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=4)
            #
            # # 정수형 범위를 고려한 레벨 설정 (예: 2일 간격으로 등고선 그리기)
            # # 날짜가 너무 촘촘하게 찍히면 보기 지저분하므로 간격(step)을 조절할 수 있습니다.
            # levels = np.arange(int(np.nanmin(values)), int(np.nanmax(values)) + 2, 2)
            #
            # # 등고선 채우기 (개화일은 봄 분위기가 나도록 'spring'이나 'YlGn' 컬러맵도 예쁩니다)
            # cf = ax.contourf(Grid_lon, Grid_lat, grid_prd_final, levels=levels,
            #                  cmap='YlGn', alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)
            #
            # # 등고선 라인
            # cl = ax.contour(Grid_lon, Grid_lat, grid_prd_final, levels=levels,
            #                 colors='black', linewidths=0.3, alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)
            #
            # # 💡 [핵심] 등고선 라벨에 날짜 변환 함수(lambda) 적용
            # # x는 등고선 라인의 값(예: 88)이며, 이를 day_to_date_str 함수에 통과시켜 출력합니다.
            # ax.clabel(cl, inline=True, fontsize=9, fmt=lambda x: day_to_date_str(x, target_year))
            #
            # # 💡 [핵심] 컬러바 틱 라벨에 날짜 변환 포맷터 적용
            # # FuncFormatter를 사용하면 컬러바의 숫자들도 모두 날짜로 변환됩니다.
            # date_formatter = FuncFormatter(lambda x, pos: day_to_date_str(x, target_year))
            # cbar = plt.colorbar(cf, shrink=0.6, format=date_formatter)
            # cbar.set_label(f'Blooming Date ({target_year})', fontsize=12, labelpad=15)
            #
            # plt.title(f'{target_year}년 전국 개화일 예측 등고선', fontsize=16, pad=20, fontweight='bold')
            #
            # # (선택) 관측소 위치 표시
            # ax.scatter(lons, lats, c='black', s=10, zorder=6)
            #
            # plt.show()

            # ---------------------------------------------------------
            # 📊 5. 웹 기반 반응형 지도 (Folium) 시각화
            # ---------------------------------------------------------
            # import folium
            # import branca.colormap as cm
            # from matplotlib.colors import Normalize
            # import matplotlib.pyplot as plt
            #
            # print("️ 반응형 웹 지도(Folium) 생성 중...")
            #
            # # 1. 색상 매핑 준비 (Cartopy에서 썼던 YlGn 컬러맵 사용)
            # cmap = plt.get_cmap('YlGn')
            # vmin = np.nanmin(grid_prd_final)
            # vmax = np.nanmax(grid_prd_final)
            # norm = Normalize(vmin=vmin, vmax=vmax)
            #
            # # 2. 격자 데이터(grid_prd_final)를 RGBA(빨강,초록,파랑,투명도) 이미지 배열로 변환
            # rgba_img = cmap(norm(grid_prd_final))
            #
            # # 3. 육지 마스크 밖의 바다 부분(NaN)을 완전 투명하게 처리 (Alpha=0)
            # rgba_img[np.isnan(grid_prd_final), 3] = 0.0
            #
            # # 4. 이미지 배열 상하 반전 (중요 ⭐️)
            # # Folium은 배열의 [0,0]을 북쪽으로 인식하지만, 우리 데이터는 33도(남쪽)부터 시작하므로 뒤집어야 합니다.
            # rgba_img_flipped = np.flipud(rgba_img)
            #
            # # 5. 지도가 표시될 경계 좌표 설정 [South, West], [North, East]
            # # (앞서 np.linspace로 설정한 33~39, 124~131 범위와 동일)
            # bounds = [[33.0, 124.0], [39.0, 131.0]]
            #
            # # 6. Folium 기본 지도 생성 (중심 좌표: 대한민국, 타일: 오픈스트리트맵)
            # m = folium.Map(location=[36.0, 127.5], zoom_start=7, tiles='OpenStreetMap')
            #
            # # 7. 예측 데이터 이미지 오버레이 추가
            # folium.raster_layers.ImageOverlay(
            #     image=rgba_img_flipped,
            #     bounds=bounds,
            #     opacity=0.75,  # 투명도 조절 (배경 지도가 살짝 보이게)
            #     name=f'{target_year}년 개화일 예측',
            #     interactive=True
            # ).add_to(m)
            #
            # # 8. 컬러바(범례) 추가
            # # YlGn 컬러맵과 유사한 헥사코드로 선형 컬러바 생성
            # colormap = cm.LinearColormap(
            #     colors=['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837'],
            #     vmin=vmin,
            #     vmax=vmax
            # )
            # colormap.caption = f'Blooming Date (Day of Year) - {target_year}'
            # m.add_child(colormap)
            #
            # # 9. 관측소 위치 마커 찍기 (마우스를 올리면 날짜 정보가 뜹니다)
            # for lon, lat, val in zip(lons, lats, values):
            #     date_str = day_to_date_str(val, target_year)  # 위에 만들어둔 날짜 변환 함수 활용
            #     folium.CircleMarker(
            #         location=[lat, lon],
            #         radius=5,
            #         color='black',
            #         weight=1,
            #         fill=True,
            #         fill_color='crimson',
            #         fill_opacity=0.9,
            #         tooltip=f"<b>위치:</b> 관측소<br><b>예측 개화일:</b> {date_str} ({int(val)}일)"
            #     ).add_to(m)
            #
            # # 레이어 컨트롤 추가 (우측 상단 겹쳐보기 On/Off 버튼)
            # folium.LayerControl().add_to(m)
            #
            # # 10. HTML 파일로 저장
            # output_html = f"{globalVar['outPath']}/{serviceName}/Blooming_Prediction_Map_{target_year}.html"
            # m.save(output_html)
            #
            # print(f" 완료! 웹 브라우저에서 '{output_html}' 파일을 열어보세요!")

            # print(" FlatGeobuf (.fgb) 선(Line) 파일 생성 중...")
            # 1. Matplotlib으로 등고선(Line) 생성
            fig, ax = plt.subplots()
            levels = np.arange(int(np.nanmin(values)), int(np.nanmax(values)) + 2, 1)

            # 💡 변경점 1: contourf() 가 아닌 contour() 를 사용합니다! (f가 빠짐)
            contourf_plot = ax.contourf(Grid_lon, Grid_lat, grid_prd_final, levels=levels, cmap='YlGn')
            plt.close(fig)
            # #
            # # 2. 등고선 객체를 GeoJSON 텍스트로 변환
            # # 💡 변경점 2: contourf_to_geojson() 대신 contour_to_geojson() 을 사용합니다!
            # geojson_str = geojsoncontour.contourf_to_geojson(
            #     contourf=contourf_plot,
            #     ndigits=5,
            #     stroke_width=0
            # )
            #
            # # 3. GeoPandas를 이용해 메모리에 로드
            # gdf = gpd.read_file(geojson_str, driver='GeoJSON')
            # gdf = gdf.set_crs(epsg=4326)
            # gdf['geometry'] = gdf['geometry'].buffer(0)
            # # 💡 변경점 3: 선(LineString) 데이터이므로 gdf['geometry'].buffer(0) 코드는 삭제합니다!
            # # (선 데이터에 buffer(0)을 주면 선이 증발해버릴 수 있습니다.)
            #
            # # 4. FlatGeobuf 포맷으로 내보내기 (Export)
            # fgb_filename = f"{globalVar['outPath']}/{serviceName}/blooming_{target_year}.fgb"
            # gdf.to_file(fgb_filename, driver='FlatGeobuf')
            #
            # print(f" FlatGeobuf (Line) 저장 완료: {fgb_filename}")


            # ==========================================================
            # 💡 [추가] 4. 관측소(Point) 데이터를 GeoDataFrame으로 생성
            # =========================================================
            # 1. 면(Polygon) 데이터 생성 및 저장
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

            # 2. 관측소 지점(Point) 데이터 생성 및 저장
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



            # =========================================================
            # 💾 2. 면(Polygon) 데이터를 FGB로 저장
            # =========================================================
            # geojson_poly = geojsoncontour.contourf_to_geojson(
            #     contourf=contourf_plot,
            #     ndigits=5,
            #     stroke_width=0  # 선은 따로 그릴 것이므로, 면 데이터의 자체 테두리는 없앱니다.
            # )
            # gdf_poly = gpd.read_file(geojson_poly, driver='GeoJSON')
            # gdf_poly = gdf_poly.set_crs(epsg=4326)
            # gdf_poly['geometry'] = gdf_poly['geometry'].buffer(0)  # 꼬인 폴리곤 복구
            #
            # poly_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_{target_year}_poly.fgb"
            # gdf_poly.to_file(poly_fgb, driver='FlatGeobuf')

            # =========================================================
            # 💾 3. 선(Line) 데이터를 FGB로 저장
            # =========================================================
            # geojson_line = geojsoncontour.contour_to_geojson(
            #     contour=contour_lines,
            #     ndigits=5,
            #     stroke_width=2  # 경계선 두께 설정
            # )
            # gdf_line = gpd.read_file(geojson_line, driver='GeoJSON')
            # gdf_line = gdf_line.set_crs(epsg=4326)
            # # 선 데이터이므로 buffer(0)은 생략합니다.
            #
            # line_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_{target_year}_line.fgb"
            # gdf_line.to_file(line_fgb, driver='FlatGeobuf')
            #
            # print(f" 면 파일 저장 완료: {poly_fgb}")
            # print(f" 선 파일 저장 완료: {line_fgb}")








            # 1. 데이터 준비
            target_year = 2025
            year_df = sokcho_df[sokcho_df['dt'] == target_year].copy()
            year_df['prd'] = year_df['prd'].round().astype(int)

            lats = year_df['LAT'].values
            lons = year_df['LON'].values
            values = year_df['prd'].values

            obs_coords = np.column_stack((lons, lats))

            # 2. 고해상도 타겟 격자 생성 (500x500)
            grid_lon = np.linspace(124, 131, 500)
            grid_lat = np.linspace(33, 39, 500)
            Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            target_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))

            # 3. RBFInterpolator 적용
            print(f" {target_year}년 개화일 RBF 보간 중...")
            rbf = RBFInterpolator(
                y=obs_coords,
                d=values,
                kernel='thin_plate_spline',
            )
            grid_prd_smooth = rbf(target_coords).reshape(Grid_lon.shape)

            # 4. 육상 마스크 적용
            land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')
            land_geom = unary_union(list(land_feature.geometries()))
            land_prep = prep(land_geom)

            mask = np.array([land_prep.contains(Point(lon, lat))
                             for lon, lat in zip(Grid_lon.ravel(), Grid_lat.ravel())])
            mask = mask.reshape(Grid_lon.shape)

            grid_prd_final = np.where(mask, grid_prd_smooth, np.nan)

            # ---------------------------------------------------------
            # 📊 5. 시각화
            # ---------------------------------------------------------
            plt.figure(figsize=(12, 12))  # 컬러바 글씨가 길어지므로 가로 크기를 살짝 늘렸습니다.
            ax = plt.axes(projection=ccrs.PlateCarree())

            ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', zorder=5)
            ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=4)

            # 정수형 범위를 고려한 레벨 설정 (예: 2일 간격으로 등고선 그리기)
            # 날짜가 너무 촘촘하게 찍히면 보기 지저분하므로 간격(step)을 조절할 수 있습니다.
            levels = np.arange(int(np.nanmin(values)), int(np.nanmax(values)) + 2, 2)

            # 등고선 채우기 (개화일은 봄 분위기가 나도록 'spring'이나 'YlGn' 컬러맵도 예쁩니다)
            cf = ax.contourf(Grid_lon, Grid_lat, grid_prd_final, levels=levels,
                             cmap='YlGn', alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)

            # 등고선 라인
            cl = ax.contour(Grid_lon, Grid_lat, grid_prd_final, levels=levels,
                            colors='black', linewidths=0.3, alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)

            # 💡 [핵심] 등고선 라벨에 날짜 변환 함수(lambda) 적용
            # x는 등고선 라인의 값(예: 88)이며, 이를 day_to_date_str 함수에 통과시켜 출력합니다.
            ax.clabel(cl, inline=True, fontsize=9, fmt=lambda x: day_to_date_str(x, target_year))

            # 💡 [핵심] 컬러바 틱 라벨에 날짜 변환 포맷터 적용
            # FuncFormatter를 사용하면 컬러바의 숫자들도 모두 날짜로 변환됩니다.
            date_formatter = FuncFormatter(lambda x, pos: day_to_date_str(x, target_year))
            cbar = plt.colorbar(cf, shrink=0.6, format=date_formatter)
            cbar.set_label(f'Blooming Date ({target_year})', fontsize=12, labelpad=15)

            plt.title(f'{target_year}년 전국 개화일 예측 등고선', fontsize=16, pad=20, fontweight='bold')

            # (선택) 관측소 위치 표시
            ax.scatter(lons, lats, c='black', s=10, zorder=6)

            plt.show()


























            # import seaborn as sns
            # import matplotlib.pyplot as plt
            #
            # # 1. 상관계수를 계산하기 위해 X_train과 y_train을 임시로 하나로 합칩니다.
            # train_data = X_train.copy()
            # train_data['demand'] = y_train
            #
            # # 2. 상관계수 계산 (Pearson)
            # correlation_matrix = train_data.corr()
            #
            # # 3. 'demand'와 다른 변수들 간의 상관계수만 내림차순으로 텍스트 출력
            # print("====== 📊 타겟(demand)과의 상관계수 ======")
            # print(correlation_matrix['demand'].sort_values(ascending=False))
            #
            # # 4. 보기 좋게 히트맵(Heatmap)으로 시각화
            # plt.figure(figsize=(10, 8))
            # # annot=True: 숫자 표시, cmap='coolwarm': 파란색(음) ~ 붉은색(양) 컬러맵
            # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            # plt.title("Feature vs Demand Correlation Heatmap")
            # plt.tight_layout()
            # plt.show()
            #
            #
            #
            #
            #
            #
            #
            #
            #
            # import numpy as np
            # import matplotlib.pyplot as plt
            # from scipy.interpolate import RBFInterpolator
            # import cartopy.crs as ccrs
            # import cartopy.feature as cfeature
            # from shapely.prepared import prep
            #
            # # 1. 데이터 준비 (기존 코드와 동일)
            # np.random.seed(42)
            # obs_lon = np.random.uniform(126.0, 129.0, 20)
            # obs_lat = np.random.uniform(34.0, 38.0, 20)
            # obs_temp = np.random.uniform(10, 25, 20)
            # obs_coords = np.column_stack((obs_lon, obs_lat))
            #
            # # 2. 격자 생성 (기존 코드와 동일)
            # grid_lon = np.arange(126.0, 129.1, 0.05)  # 해상도를 조금 더 높였습니다
            # grid_lat = np.arange(34.0, 38.1, 0.05)
            # Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            # grid_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))
            #
            # # 3. RBF 보간 계산
            # rbf_model = RBFInterpolator(y=obs_coords, d=obs_temp, kernel='thin_plate_spline', smoothing=0.1)
            # grid_temp_flat = rbf_model(grid_coords)
            # Grid_temp = grid_temp_flat.reshape(Grid_lon.shape)
            #
            # # ---------------------------------------------------------
            # # 🛠️ 4. 해상 마스크(Sea Mask) 적용 로직
            # # ---------------------------------------------------------
            # print("🌊 해상 마스크 생성 중...")
            #
            # # Cartopy의 육지 데이터를 가져옵니다 (10m 또는 50m 해상도)
            # land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m')
            # land_geom = list(land_feature.geometries())
            #
            # # 속도를 위해 준비된 기하구조(prepared geometry)를 사용합니다.
            # from shapely.ops import unary_union
            # land_polygons = prep(unary_union(land_geom))
            #
            # # 각 격자점이 육지인지 바다인지 판별하는 함수
            # from shapely.geometry import Point
            #
            # # 격자 크기만큼의 마스크 배열 생성 (True=육지, False=바다)
            # mask = np.array([land_polygons.contains(Point(lon, lat)) for lon, lat in grid_coords])
            # mask = mask.reshape(Grid_lon.shape)
            #
            # # 바다 영역(mask == False)을 NaN으로 처리하여 시각화에서 가립니다.
            # Grid_temp_masked = np.where(mask, Grid_temp, np.nan)
            # # ---------------------------------------------------------
            #
            # # 5. 시각화
            # plt.figure(figsize=(10, 8), dpi=100)
            # ax = plt.axes(projection=ccrs.PlateCarree())
            #
            # # 해안선 및 지형 추가
            # ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', zorder=3)
            # ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, zorder=3)
            #
            # # 마스킹된 데이터로 등고선 그리기
            # cf = ax.contourf(Grid_lon, Grid_lat, Grid_temp_masked, levels=20, cmap='YlOrRd',
            #                  transform=ccrs.PlateCarree())
            # plt.colorbar(cf, label='Interpolated Temp (Land Only, °C)', shrink=0.6)
            #
            # # 관측점 표시
            # ax.scatter(obs_lon, obs_lat, c='blue', edgecolors='white', s=30, label='Stations',
            #            transform=ccrs.PlateCarree(), zorder=4)
            #
            # plt.title("RBF Interpolation with Land Masking (Sea Masked Out)")
            # plt.legend()
            # plt.show()
            #
            #
            #
            #
            # import intake
            # import xarray as xr
            #
            # # 1. CMIP6 데이터 카탈로그 열기 (Pangeo 클라우드 활용)
            # col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
            #
            # # 2. 원하는 시나리오 조건 검색 (예: 2050년 포함된 SSP5-8.5 고탄소 시나리오)
            # query = dict(
            #     experiment_id='ssp585',
            #     table_id='Amon',
            #     variable_id='tas',  # 지표 기온
            #     member_id='r1i1p1f1'
            # )
            #
            # search = col.search(**query)
            # dataset_dict = search.to_dataset_dict()
            #
            # # 3. 데이터 로드 및 2050년 데이터 슬라이싱
            # # 이 데이터를 pySCM의 'state'에 넣어 시뮬레이션을 시작할 수 있습니다.
            # key = list(dataset_dict.keys())[0]
            # ds = dataset_dict[key]
            # future_ta = ds.sel(time='2050-08')







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