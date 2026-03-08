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
import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
import configparser
from urllib.parse import urlparse, parse_qs
from lxml import etree
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
import pymysql
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
                'stnFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/asosInfo/ALL_STN_INFO.csv',
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
            refData.columns
            refDataL1 = refData.melt(id_vars=['지점', '년도'], var_name='구분', value_name='값')
            refDataL1[['구분1', '구분2']] = refDataL1['구분'].str.split('_', n=1, expand=True)

            obsData['year'] = obsData['dt'].astype(str)
            refDataL1['년도'] = refDataL1['년도'].astype(str)

            # dt를 datetime으로 변환 후 년도 추출
            # obsData['dt'] = pd.to_datetime(obsData['dt'])
            # obsData['년도'] = obsData['dt'].dt.year


            # 교집합 병합 (obsData: stnId/dt의 년도, refDataL1: 지점/년도)
            mergeData = pd.merge(obsData, refDataL1, left_on=['stnName', 'year'], right_on=['지점', '년도'], how='inner')

            # 속초 지역 & 개화 시기 필터링 (벚나무를 예시로 사용)
            # sokcho_df = mergeData[(mergeData['stnName'] == '속초')].copy()
            # sokcho_df = sokcho_df[sokcho_df['구분2'] == '개화'].copy()

            sokcho_df = mergeData[mergeData['구분2'] == '개화'].copy()  # 대상 식물(예: 벚나무)
            sokcho_df = sokcho_df[sokcho_df['구분1'] == '개나리'].copy()  # 대상 식물(예: 벚나무)

            # 컬럼명 통일 및 시계열 처리 (결측치 등)
            sokcho_df = sokcho_df.rename(columns={'값': 'demand'})
            sokcho_df['demand'] = pd.to_datetime(sokcho_df['demand'], format='%Y-%m-%d', errors='coerce').dt.strftime('%j').astype('float')
            # sokcho_df['timeStamp'] = sokcho_df['timeStamp'].dt.to_period('Y').dt.to_timestamp()


            sokcho_df['timeStamp'] = pd.to_datetime(sokcho_df['year'], errors='coerce')
            # sokcho_df = sokcho_df.set_index('timeStamp')
            # # sokcho_df['temp'] = sokcho_df['temp'].fillna(method='ffill')
            # # sokcho_df['precip'] = sokcho_df['precip'].fillna(method='ffill')
            sokcho_df = sokcho_df.dropna(subset=['demand']).reset_index()
            # sokcho_df.columns

            # Using temperature values create categorical values
            # # where 1 denotes daily tempurature is above monthly average and 0 is below.
            # def get_monthly_avg(data):
            #     data["month"] = data["timeStamp"].dt.month
            #     data_grp = data[["month", "temp"]].groupby("month")
            #     data_grp = data_grp.agg({"temp": "mean"})
            #     return data_grp
            #
            # monthly_avg = get_monthly_avg(sokcho_df).to_dict().get("temp", {})
            #
            # def above_monthly_avg(date, temp):
            #     month = date.month
            #     if temp > monthly_avg.get(month, temp):
            #         return 1
            #     else:
            #         return 0
            #
            # sokcho_df["temp_above_monthly_avg"] = sokcho_df.apply(
            #     lambda x: above_monthly_avg(x["timeStamp"], x["temp"]), axis=1
            # )
            #
            # if "month" in sokcho_df.columns:
            #     del sokcho_df["month"]  # remove month column to reduce redundancy

            # # split data into train and test
            # num_samples = sokcho_df.shape[0]
            # time_horizon = 30  # 예시: 30일 예측 (사용자 코드는 180일)
            # split_idx = num_samples - time_horizon
            
            # if split_idx > 0:
                # multi_train_df = sokcho_df[:split_idx]
                # multi_test_df = sokcho_df[split_idx:]
                #
                # multi_X_test = multi_test_df[["timeStamp", "precip", "temp", "temp_above_monthly_avg"]]
                # multi_y_test = multi_test_df["demand"]

            # initialize AutoML instance
            from flaml import AutoML
            automl = AutoML()


            # 예측
            settings = {
                "time_budget": 10,
                "metric": "rmse",
                "task": "regression",
            }

            # 3. 예측에 사용할 피처에 시간 피처 추가
            features = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']

            X_train = sokcho_df[features].copy()
            y_train = sokcho_df['demand'].copy()  # demand가 숫자형 데이터라고 가정


            # 4. 모델 학습
            automl.fit(X_train=X_train, y_train=y_train, **settings)

            # 5. 예측 수행 (가급적 X_test를 별도로 만들어 사용하는 것을 권장합니다)
            print("====== 예측 결과 (Prediction) ======")
            predictions = automl.predict(X_train)
            print(predictions)



            # 검증
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import matplotlib.pyplot as plt
            import numpy as np

            # 1. 평가 지표 계산 (정답인 y_train과 모델이 찍은 predictions를 비교)
            rmse = np.sqrt(mean_squared_error(y_train, predictions))
            mae = mean_absolute_error(y_train, predictions)
            r2 = r2_score(y_train, predictions)

            print("====== 📈 모델 검증 결과 (Metrics) ======")
            print(f"RMSE (평균 오차): {rmse:.2f}")
            print(f"MAE (절대 평균 오차): {mae:.2f}")
            print(f"R2 Score (설명력, 1에 가까울수록 좋음): {r2:.2f}")

            # RMSE (평균 오차): 5.38
            # MAE (절대 평균 오차): 4.21
            # R2 Score (설명력, 1에 가까울수록 좋음): 0.64

            # RMSE (평균 오차): 3.09
            # MAE (절대 평균 오차): 2.34
            # R2 Score (설명력, 1에 가까울수록 좋음): 0.88

            # RMSE (평균 오차): 3.19
            # MAE (절대 평균 오차): 2.39
            # R2 Score (설명력, 1에 가까울수록 좋음): 0.87

            # 2. 시각화 (눈으로 직접 확인하기)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_train, predictions, alpha=0.3, color='blue')
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # 빨간 점선 = 완벽한 정답선
            plt.xlabel('Actual Demand (실제 수요)')
            plt.ylabel('Predicted Demand (예측 수요)')
            plt.title('Actual vs. Predicted Demand')
            plt.grid(True)
            plt.show()


            import seaborn as sns
            import matplotlib.pyplot as plt

            # 1. 상관계수를 계산하기 위해 X_train과 y_train을 임시로 하나로 합칩니다.
            train_data = X_train.copy()
            train_data['demand'] = y_train

            # 2. 상관계수 계산 (Pearson)
            correlation_matrix = train_data.corr()

            # 3. 'demand'와 다른 변수들 간의 상관계수만 내림차순으로 텍스트 출력
            print("====== 📊 타겟(demand)과의 상관계수 ======")
            print(correlation_matrix['demand'].sort_values(ascending=False))

            # 4. 보기 좋게 히트맵(Heatmap)으로 시각화
            plt.figure(figsize=(10, 8))
            # annot=True: 숫자 표시, cmap='coolwarm': 파란색(음) ~ 붉은색(양) 컬러맵
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title("Feature vs Demand Correlation Heatmap")
            plt.tight_layout()
            plt.show()









            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.interpolate import RBFInterpolator
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from shapely.prepared import prep

            # 1. 데이터 준비 (기존 코드와 동일)
            np.random.seed(42)
            obs_lon = np.random.uniform(126.0, 129.0, 20)
            obs_lat = np.random.uniform(34.0, 38.0, 20)
            obs_temp = np.random.uniform(10, 25, 20)
            obs_coords = np.column_stack((obs_lon, obs_lat))

            # 2. 격자 생성 (기존 코드와 동일)
            grid_lon = np.arange(126.0, 129.1, 0.05)  # 해상도를 조금 더 높였습니다
            grid_lat = np.arange(34.0, 38.1, 0.05)
            Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            grid_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))

            # 3. RBF 보간 계산
            rbf_model = RBFInterpolator(y=obs_coords, d=obs_temp, kernel='thin_plate_spline', smoothing=0.1)
            grid_temp_flat = rbf_model(grid_coords)
            Grid_temp = grid_temp_flat.reshape(Grid_lon.shape)

            # ---------------------------------------------------------
            # 🛠️ 4. 해상 마스크(Sea Mask) 적용 로직
            # ---------------------------------------------------------
            print("🌊 해상 마스크 생성 중...")

            # Cartopy의 육지 데이터를 가져옵니다 (10m 또는 50m 해상도)
            land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m')
            land_geom = list(land_feature.geometries())

            # 속도를 위해 준비된 기하구조(prepared geometry)를 사용합니다.
            from shapely.ops import unary_union
            land_polygons = prep(unary_union(land_geom))

            # 각 격자점이 육지인지 바다인지 판별하는 함수
            from shapely.geometry import Point

            # 격자 크기만큼의 마스크 배열 생성 (True=육지, False=바다)
            mask = np.array([land_polygons.contains(Point(lon, lat)) for lon, lat in grid_coords])
            mask = mask.reshape(Grid_lon.shape)

            # 바다 영역(mask == False)을 NaN으로 처리하여 시각화에서 가립니다.
            Grid_temp_masked = np.where(mask, Grid_temp, np.nan)
            # ---------------------------------------------------------

            # 5. 시각화
            plt.figure(figsize=(10, 8), dpi=100)
            ax = plt.axes(projection=ccrs.PlateCarree())

            # 해안선 및 지형 추가
            ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', zorder=3)
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, zorder=3)

            # 마스킹된 데이터로 등고선 그리기
            cf = ax.contourf(Grid_lon, Grid_lat, Grid_temp_masked, levels=20, cmap='YlOrRd',
                             transform=ccrs.PlateCarree())
            plt.colorbar(cf, label='Interpolated Temp (Land Only, °C)', shrink=0.6)

            # 관측점 표시
            ax.scatter(obs_lon, obs_lat, c='blue', edgecolors='white', s=30, label='Stations',
                       transform=ccrs.PlateCarree(), zorder=4)

            plt.title("RBF Interpolation with Land Masking (Sea Masked Out)")
            plt.legend()
            plt.show()




            import intake
            import xarray as xr

            # 1. CMIP6 데이터 카탈로그 열기 (Pangeo 클라우드 활용)
            col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

            # 2. 원하는 시나리오 조건 검색 (예: 2050년 포함된 SSP5-8.5 고탄소 시나리오)
            query = dict(
                experiment_id='ssp585',
                table_id='Amon',
                variable_id='tas',  # 지표 기온
                member_id='r1i1p1f1'
            )

            search = col.search(**query)
            dataset_dict = search.to_dataset_dict()

            # 3. 데이터 로드 및 2050년 데이터 슬라이싱
            # 이 데이터를 pySCM의 'state'에 넣어 시뮬레이션을 시작할 수 있습니다.
            key = list(dataset_dict.keys())[0]
            ds = dataset_dict[key]
            future_ta = ds.sel(time='2050-08')







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