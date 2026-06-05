# ================================================
# 요구사항
# ================================================
# Python을 이용한 기상청 AWS 다운로드

# ps -ef | grep "TalentPlatform-bdwide-bloomAsosAnlPrd.py" | awk '{print $2}' | xargs kill -9
# pkill -f "TalentPlatform-bdwide-bloomAsosAnlPrd.py"

# 프로그램 시작
# conda activate py39
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/TalentPlatform-bdwide-bloomAsosAnlPrd.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/TalentPlatform-bdwide-bloomAsosAnlPrd.py &
# tail -f nohup.out

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
import pandas as pd
import matplotlib.pyplot as plt
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
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
from datetime import datetime
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
import configparser
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

import geopandas as gpd
# import contextily as ctx

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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

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


def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):
    # log.info(f'[START] makeFlamlModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srv=subOpt['srv'])), reverse=True)

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)
            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # 전체 학습 데이터
            # trainData = dataL1

            fnlModel = AutoML(
                task='regression',
                metric='rmse',
                # ensemble=True,
                ensemble=False,
                seed=int(datetime.datetime.now().timestamp()),
                time_budget=60,
                verbose=1,
            )

            # 각 모형에 따른 자동 머신러닝
            fnlModel.fit(X_train=trainDataL1[xCol], y_train=trainDataL1[yCol], X_val=testDataL1[xCol], y_val=testDataL1[yCol])

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srv=subOpt['srv'])
            log.info(f"saveModel : {saveModel}")
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            with open(saveModel, 'wb') as file:
                pickle.dump(fnlModel, file, pickle.HIGHEST_PROTOCOL)
        else:
            saveModel = saveModelList[0]
            log.info(f"[CHECK] saveModel : {saveModel}")

            with open(saveModel, 'rb') as f:
                fnlModel = pickle.load(f)

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error(f"Exception : {e}")
        return result

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
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'
        contextPath = os.getcwd() if env in 'local' else '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT'

    prjName = 'bloomAsosAnlPrd'
    serviceName = 'BDWIDE2025'

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
                'stnFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/stnInfo/ALL_STN_INFO.csv',
                'obsFile': '/DATA/INPUT/BDWIDE2026/OBS_ASOS_ANL_20260302205902.csv',
                'refFile': '/DATA/INPUT/BDWIDE2026/OBS_계절관측_20260128005918.csv',
                'saveFile': '/DATA/OUTPUT/BDWIDE2026/BDWIDE2025_output_%Y%m%d.csv',
                'saveFile2': '/DATA/OUTPUT/BDWIDE2026/BDWIDE2025_output2_%Y%m%d.csv',

                'flaml': {
                    'saveModelList': "/DATA/OUTPUT/BDWIDE2026/AI/*/*/BDWIDE2025_{srv}_flaml_*.model",
                    'saveModel': "/DATA/OUTPUT/BDWIDE2026/AI/%Y%m/%d/BDWIDE2025_{srv}_flaml_%Y%m%d.model",
                    'isOverWrite': False,
                    # 'isOverWrite': True,
                    'srv': None,
                    'preDt': datetime.datetime.now(),
                },
            }





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

            # 교집합 병합 (obsData: stnId/dt의 년도, refDataL1: 지점/년도)
            mergeData = pd.merge(obsData, refDataL1, left_on=['stnName', 'year'], right_on=['지점', '년도'], how='inner')
            mergeData = pd.merge(mergeData, cfgDataL1, left_on=['stnId'], right_on=['STN'], how='inner')




            # # **********************************************************************************************************
            # # 기후자료 미래예측
            # # **********************************************************************************************************
            # # 기후자료 예측 테스트
            # fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/BDWIDE2026/AR6_SSP126_5ENSMN_skorea_*_gridraw_yearly_2021_2100.nc'))
            # ds = xr.open_mfdataset(fileList, chunks='auto')
            # # ds = xr.open_mfdataset(fileList, combine='by_coords')
            #
            # timeList = ds['time'].values
            # for timeInfo in timeList:
            #
            #     # ds['TA'].sel(time = timeInfo).plot()
            #     # plt.show()
            #
            #     # AI 모델 학습(AutoML 등)에 활용하기 위해 Pandas DataFrame으로 변환
            #     df_climate = ds.sel(time = timeInfo).to_dataframe().reset_index()
            #     # df_climate = ds.to_dataframe().reset_index()
            #
            #     # 결측치(해양 격자 등 데이터가 없는 곳) 제거
            #     df_climate = df_climate.dropna().reset_index(drop=True)
            #
            #     df_climate = df_climate.rename(columns={
            #         'TA': 'avgTemp',  # 연평균 기온
            #         'TAMIN': 'avgMinTemp',  # 연평균 최저기온
            #         'TAMAX': 'avgMaxTemp',  # 연평균 최고기온
            #         'RN': 'sumPrecip',  # 연 누적 강수량
            #         'RHM': 'avgRh',  # 연평균 상대습도
            #         'WS': 'avgWindSpeed'  # 연평균 풍속
            #     })
            #
            #     # time 객체에서 예측 기준이 될 year 컬럼 독립적 추출
            #     # df_climate['year'] = df_climate['time'].dt.year
            #     # df_climate['time'] = pd.to_datetime(df_climate['time'])
            #     df_climate['year'] = df_climate['time'].apply(lambda x: x.year)
            #
            #     # 2. 이제 정상적으로 .dt 접근자를 사용하여 연도(year)를 추출할 수 있습니다.
            #     # df_climate['year'] = df_climate['time'].dt.year
            #
            #     # 데이터 확인
            #     print(df_climate[['time', 'year']].head())
            #
            #     # 최종 사용할 피처만 선택
            #     # features = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']
            #     final_test_df = df_climate.copy()
            #
            #     print(final_test_df.head())
            #
            #     xCol = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']
            #     yCol = 'demand'
            #
            #     trainData = None
            #     testData = None
            #     prdData = final_test_df
            #
            #     for (gubun1, gubun2), target_df in mergeData.groupby(['구분1', '구분2']):
            #         target_df = target_df.groupby('stnName').filter(lambda x: len(x) >= 30)
            #         if gubun1 not in ['아까시나무', '매화', '벚나무', '복숭아', '배나무']: continue
            #         if gubun2 not in ['개화']: continue
            #         log.info(f"gubun1 : {gubun1}, gubun2 : {gubun2}")
            #
            #         sysOpt['flaml']['srv'] = f"{gubun1}-{gubun2}-전체"
            #         resFlaml = makeFlamlModel(sysOpt['flaml'], xCol, yCol, trainData, testData)
            #         # log.info(f'resFlaml : {resFlaml}')
            #
            #         if resFlaml:
            #             prdVal = resFlaml['mlModel'].predict(prdData[xCol])
            #             prdData['ai'] = prdVal
            #
            #
            #
            #         # 2026.05.29 수정
            #         # fig = plt.figure(figsize=(9,9))
            #         fig = plt.figure(figsize=(2200 / 300, 2000 / 300))
            #         ax = fig.add_subplot(111)
            #
            #         # 2026.05.29 수정
            #         #   m = Basemap(projection='lcc', resolution='i',
            #         #                          lat_0=38, lon_0=126,
            #         #                          llcrnrlat=11.308528, urcrnrlat=53.303712,
            #         #                          llcrnrlon=101.395259, urcrnrlon=175.188166,
            #         #                          lat_1=30, lat_2=60)
            #
            #         m = Basemap(projection='cyl',
            #                     llcrnrlat=28, urcrnrlat=46,
            #                     llcrnrlon=120, urcrnrlon=135,
            #                     resolution='h', ax=ax)
            #
            #         # 등온선 설정
            #         VMIN, VMAX, DLEV = 60, 150, 10
            #         LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)
            #
            #         # 2026.05.29 수정
            #         VMIN2, VMAX2, DLEV2 = 60, 150, 10
            #         LEVELS2 = np.arange(VMIN2, VMAX2 + 1e-6, DLEV2)
            #
            #         lon = prdData['longitude']
            #         lat = prdData['latitude']
            #         data = prdData['ai']
            #
            #         x, y = m(lon, lat)
            #         pcm = m.pcolormesh(x, y, data, cmap='jet', shading='auto', vmin=VMIN, vmax=VMAX)
            #         m.drawcoastlines(color='k', linewidth=0.5)
            #         m.drawcountries(color='gray', linewidth=0.5)
            #         m.fillcontinents(color='lightgray', lake_color='white')
            #
            #         # 2026.05.29 수정
            #         #   m.drawparallels(np.arange(float(np.nanmin(lat)),float(np.nanmax(lat)) + 1, 5), labels=[1,0,0,0], fontsize=15, fontname='arial', fmt='%d')
            #         #   m.drawmeridians(np.arange(float(np.nanmin(lon)),float(np.nanmax(lon)) + 1, 10), labels=[0,0,0,1], fontsize=15, fontname='arial', fmt='%d')
            #         m.drawparallels(np.arange(15, 56, 10), labels=[1, 0, 0, 0], fontsize=10, fontname='DejaVu Sans',
            #                         fmt='%d', fontweight='bold')
            #         m.drawmeridians(np.arange(110, 151, 10), labels=[0, 0, 0, 1], fontsize=10, fontname='DejaVu Sans',
            #                         fmt='%d', fontweight='bold')
            #
            #         # 2026.05.29 수정
            #         # smoothed_data = gaussian_filter(data, sigma=3)
            #
            #         # 가우시안 필터
            #         weight = np.ones_like(data.data)
            #         weight[data.mask] = 0.0
            #         data_zeroed = data.filled(0.0)
            #         sigma_val = 15
            #         smoothed_data = gaussian_filter(data_zeroed, sigma=sigma_val)
            #         smoothed_weight = gaussian_filter(weight, sigma=sigma_val)
            #         with np.errstate(invalid='ignore', divide='ignore'):
            #             smoothed_corrected = smoothed_data / smoothed_weight
            #         smoothed_ma = np.ma.masked_array(smoothed_corrected, mask=data.mask)
            #
            #         # 2026.05.29 수정
            #         # c = m.contour(x,y,smoothed_data,levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8)
            #         c = m.contour(x, y, smoothed_ma, levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8)
            #
            #         # 2026.05.29 수정
            #         # plt.clabel(c, inline=True, fontsize=10, fmt='%d', colors='black')
            #         # labels = plt.clabel(c, levels=LEVELS, inline=False, fontsize=8, fmt='%d', colors='black')
            #         labels = plt.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black')
            #         for label in labels:
            #             label.set_rotation(0)
            #
            #         # 2026.05.29 수정
            #         # cbar = plt.colorbar(pcm,shrink=0.8)
            #         cbar = plt.colorbar(pcm, shrink=0.8, extend='both', spacing='proportional')
            #         cbar.set_ticks(LEVELS2)
            #
            #         # cbar.set_label('Sea Surface Temperature (°C)', fontsize=10, fontname='DejaVu Sans',
            #         #                fontweight='bold')
            #         for lbl in cbar.ax.get_yticklabels():
            #             lbl.set_fontname('DejaVu Sans')
            #             lbl.set_fontsize(10)
            #             lbl.set_fontweight('bold')
            #
            #         # 2026.05.29 수정
            #         # plt.title(f'Monthly GK2A Sea Surface Temperature ({year_comp}.{month_comp})',fontsize=12, fontweight='bold', fontname='DejaVu Sans')
            #         # plt.title(f'Monthly Mean GK2A Sea Surface Temperature ({COMP_MONTH})', fontsize=12,
            #         #           fontweight='bold', fontname='DejaVu Sans')
            #         plt.tight_layout()
            #         plt.show()
            #
            #         # # 2026.05.29 수정
            #         # save_path = PNG_DIR
            #         #
            #         # # 2026.05.29 수정
            #         # # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #         # plt.savefig(save_path, dpi=300, transparent=True)
            #         # plt.close()
            #         # print(f"save_path : {save_path}")
            #
            #
            #
            #
            #
            #         # 데이터 준비
            #         x = prdData['longitude']
            #         y = prdData['latitude']
            #         z = prdData['ai']
            #
            #         # 1. 비어있는 2차원 바둑판(Grid) 좌표 만들기
            #         # 데이터를 500x500 픽셀의 해상도로 쪼갭니다. (숫자가 클수록 더 고해상도가 됩니다)
            #         grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
            #
            #         # 2. 흩어진 데이터를 바둑판 빈칸에 채워넣기 (보간법)
            #         # method='linear' (선형 보간), 'cubic' (부드럽게), 'nearest' (가장 가까운 값) 중 선택 가능
            #         grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
            #
            #         # 3. 그래프 그리기 시작
            #         plt.figure(figsize=(10, 10), dpi=100)
            #
            #         # 4. pcolormesh 그리기
            #         # shading='auto'는 격자의 색상을 부드럽게 채워주는 최신 표준 옵션입니다.
            #         mesh_map = plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap='Spectral_r')
            #
            #         # 5. 컬러바(범례) 추가
            #         cbar = plt.colorbar(mesh_map, shrink=0.8)
            #         cbar.set_label('AI Value', fontsize=12)
            #
            #         # 6. 축 라벨 및 타이틀 설정
            #         plt.xlabel('Longitude', fontsize=12)
            #         plt.ylabel('Latitude', fontsize=12)
            #         plt.title('AI Value Spatial Distribution (pcolormesh)', fontsize=15, pad=15)
            #
            #         # 7. 그리드 추가 및 출력
            #         plt.grid(True, linestyle='--', alpha=0.5)
            #         plt.tight_layout()
            #         plt.show()



                    # 1. 데이터 불러오기 (이미 df라는 데이터프레임에 데이터가 있다고 가정)
                    # df = pd.read_csv('data.csv') # 실제 파일 경로에 맞게 수정하세요.


                    # x = prdData['longitude']
                    # y = prdData['latitude']
                    # z = prdData['ai']
                    #
                    # # 2. 그래프 크기 및 해상도 설정
                    # plt.figure(figsize=(10, 10), dpi=100)
                    #
                    # # 3. 색상이 채워진 등고선 그리기 (배경색)
                    # # levels=20은 등고선의 단계를 20개로 나눈다는 의미입니다. 숫자를 키우면 더 부드러워집니다.
                    # contour_filled = plt.tricontourf(x, y, z, levels=20, cmap='Spectral_r')
                    #
                    # # 4. 등고선 라인(선) 그리기
                    # # colors='black'으로 검은색 선을 그리고, linewidths로 선 굵기를 조절합니다.
                    # contour_lines = plt.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.7)
                    #
                    # plt.clabel(contour_lines, inline=True, fontsize=8, colors='black')
                    #
                    # # 5. 컬러바(범례) 추가
                    # cbar = plt.colorbar(contour_filled, shrink=0.8)
                    # cbar.set_label('AI Value', fontsize=12)
                    #
                    # # 6. 축 라벨 및 타이틀 설정
                    # plt.xlabel('Longitude', fontsize=12)
                    # plt.ylabel('Latitude', fontsize=12)
                    # plt.title('AI Value Spatial Distribution with Contours', fontsize=15, pad=15)
                    #
                    # # 7. 그리드 추가 및 출력
                    # plt.grid(True, linestyle='--', alpha=0.5)
                    # plt.tight_layout()
                    # plt.show()
                    #
                    # print('asdasd')


                    # 1. 데이터 준비
                    # df = pd.read_csv('data.csv')

                    # # 2. 위/경도 데이터를 지도 좌표계(EPSG:3857)로 변환하기 위해 GeoDataFrame 생성
                    # gdf = gpd.GeoDataFrame(
                    #     prdData,
                    #     geometry=gpd.points_from_xy(prdData['longitude'], prdData['latitude']),
                    #     crs="EPSG:4326" # 원래 좌표계 (위경도)
                    # )
                    # # 배경 지도와 일치시키기 위해 좌표계 변환
                    # gdf = gdf.to_crs(epsg=3857)
                    #
                    # # 변환된 X, Y 좌표 추출
                    # x = gdf.geometry.x
                    # y = gdf.geometry.y
                    # z = gdf['ai']
                    #
                    # # 3. 그래프 그리기
                    # fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
                    #
                    # # 색상이 채워진 등고선 (alpha 값을 주어 배경 지도가 비치게 합니다)
                    # contour = ax.tricontourf(x, y, z, levels=20, cmap='Spectral_r', alpha=0.6)
                    # # 등고선 테두리선
                    # ax.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.5)
                    #
                    # # 4. 배경 지도 추가 (여기가 핵심!)
                    # # source 옵션으로 위성 지도, 네이버 스타일 지도 등 다양한 배경 선택 가능
                    # ctx.add_baseline_map(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                    #
                    # # 5. 컬러바 및 타이틀 설정
                    # cbar = plt.colorbar(contour, ax=ax, shrink=0.7)
                    # cbar.set_label('AI Value')
                    #
                    # ax.set_title('AI Map with Background', fontsize=16, pad=15)
                    # ax.set_axis_off() # 지도 좌표 숫자는 보기 지저분하므로 숨김 처리
                    #
                    # plt.tight_layout()
                    # plt.show()
                    # print('asdasd')

            # =============================================================
            # 학습모형 생산/예측
            # =============================================================
            # 최종 모든 결과를 누적할 빈 리스트
            all_metrics_list = []
            all_metrics_list2 = []
            for (gubun1, gubun2), target_df in mergeData.groupby(['구분1', '구분2']):
                try:
                    target_df = target_df.groupby('stnName').filter(lambda x: len(x) >= 30)
                    # if gubun1 not in ['아까시나무', '매화', '벚나무', '복숭아', '배나무']: continue
                    if gubun2 not in ['개화']: continue
                    log.info(f"gubun1 : {gubun1}, gubun2 : {gubun2}")

                    target_df = target_df.rename(columns={'값': 'demand'})
                    target_df['demand'] = pd.to_datetime(target_df['demand'], format='%Y-%m-%d', errors='coerce').dt.strftime('%j').astype('float')
                    target_df['timeStamp'] = pd.to_datetime(target_df['year'], errors='coerce')
                    target_df = target_df.dropna(subset=['demand']).reset_index(drop=True)

                    # 데이터가 너무 적으면 시계열 분할 및 모델 학습이 불가하므로 스킵 (최소 10건)
                    if len(target_df['demand']) < 30: continue

                    # ---------------------------------------------------------
                    # 2. 피처 설정 및 Train/Test 시계열 분할
                    # ---------------------------------------------------------
                    target_df = target_df.sort_values(by='year')

                    # ****************************************************************************
                    # 독립/종속 변수 설정
                    # ****************************************************************************
                    xCol = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']
                    yCol = 'demand'

                    # ---------------------------------------------------------
                    # 3. 개별 지역 모델 학습 및 예측
                    # ---------------------------------------------------------
                    yearList = target_df['year'].unique()
                    idx = int(len(yearList) * 0.8)

                    trainData = target_df[target_df['year'] <= yearList[idx - 1]]
                    testData = target_df[target_df['year'] > yearList[idx - 1]]
                    prdData = target_df

                    sysOpt['flaml']['srv'] = f"{gubun1}-{gubun2}-전체"
                    # resFlaml = makeFlamlModel(sysOpt['flaml'], xCol, yCol, trainData, testData)
                    resFlaml = makeFlamlModel(sysOpt['flaml'], xCol, yCol, trainData, prdData)
                    # log.info(f'resFlaml : {resFlaml}')

                    if resFlaml:
                        prdVal = resFlaml['mlModel'].predict(prdData[xCol])
                        prdData['ai'] = prdVal

                        # 예측된 결과를 바탕으로 지역별(stnName) 성능 계산 루프
                        for stn, stn_df in prdData.groupby('stnName'):

                            # 지역별 데이터가 2개 이상이어야 상관계수 계산 가능
                            if len(stn_df) > 1:
                                rmse = np.sqrt(mean_squared_error(stn_df['demand'], stn_df['ai']))

                                # 상관계수 계산 (예외 처리 포함)
                                try:
                                    r = np.corrcoef(stn_df['demand'], stn_df['ai'])[0, 1]
                                except Exception:
                                    r = np.nan
                            else:
                                rmse = np.nan
                                r = np.nan

                            # 지역별 결과를 리스트에 누적
                            all_metrics_list2.append({
                                '구분1': gubun1,
                                '구분2': gubun2,
                                'stnName': stn,  # '전체' 대신 실제 지역명 기록
                                'N': len(stn_df['demand']),
                                'RMSE': round(rmse, 4) if not np.isnan(rmse) else np.nan,
                                'R': round(r, 4) if not np.isnan(r) else np.nan
                            })

                    if len(prdData) > 1:
                        rmse = np.sqrt(mean_squared_error(prdData['demand'], prdData['ai']))
                        r = np.corrcoef(target_df['demand'], prdData['ai'])[0, 1]
                    else:
                        rmse = np.nan
                        r = np.nan

                    all_metrics_list.append({
                        '구분1': gubun1,
                        '구분2': gubun2,
                        'stnName': '전체',
                        'N': len(prdData['demand']),
                        'RMSE': round(rmse, 4) if not np.isnan(rmse) else np.nan,
                        'R': round(r, 4) if not np.isnan(r) else np.nan
                    })
                    log.info(pd.DataFrame(all_metrics_list))
                except Exception as e:
                    log.error(f"Exception : {e}")


            # # 최종 모든 결과를 누적할 빈 리스트
            # all_metrics_list = []
            # for (gubun1, gubun2, stn), target_df in mergeData.groupby(['구분1', '구분2', 'stnName']):
            #     try:
            #         if gubun1 not in ['아까시나무', '매화', '벚나무', '복숭아', '배나무']: continue
            #         if gubun2 not in ['개화']: continue
            #         log.info(f"gubun1 : {gubun1}, gubun2 : {gubun2}, stn : {stn}")
            #
            #         # ---------------------------------------------------------
            #         # 1. 컬럼명 통일 및 시계열 처리
            #         # ---------------------------------------------------------
            #         target_df = target_df.rename(columns={'값': 'demand'})
            #         target_df['demand'] = pd.to_datetime(target_df['demand'], format='%Y-%m-%d',
            #                                              errors='coerce').dt.strftime('%j').astype('float')
            #         target_df['timeStamp'] = pd.to_datetime(target_df['year'], errors='coerce')
            #         target_df = target_df.dropna(subset=['demand']).reset_index(drop=True)
            #
            #
            #         # 데이터가 너무 적으면 시계열 분할 및 모델 학습이 불가하므로 스킵 (최소 10건)
            #         if len(target_df['demand']) < 30: continue
            #
            #         # ---------------------------------------------------------
            #         # 2. 피처 설정 및 Train/Test 시계열 분할
            #         # ---------------------------------------------------------
            #         target_df = target_df.sort_values(by='year')
            #         # unique_years = target_df['year'].unique()
            #
            #         # ****************************************************************************
            #         # 독립/종속 변수 설정
            #         # ****************************************************************************
            #         xCol = ['year', 'avgTemp', 'avgMinTemp', 'avgMaxTemp', 'sumPrecip', 'avgRh', 'avgWindSpeed']
            #         yCol = 'demand'
            #
            #         # ---------------------------------------------------------
            #         # 3. 개별 지역 모델 학습 및 예측
            #         # ---------------------------------------------------------
            #         # from sklearn.model_selection import train_test_split
            #         # trainData, testData = train_test_split(target_df, test_size=0.2, random_state=int(datetime.datetime.now().timestamp()))
            #         yearList = target_df['year'].unique()
            #         idx = int(len(yearList) * 0.8)
            #
            #         trainData = target_df[target_df['year'] <= yearList[idx - 1]]
            #         testData = target_df[target_df['year'] > yearList[idx - 1]]
            #         prdData = target_df
            #
            #         sysOpt['flaml']['srv'] = f"{gubun1}-{gubun2}-{stn}"
            #         resFlaml = makeFlamlModel(sysOpt['flaml'], xCol, yCol, trainData, prdData)
            #         # log.info(f'resFlaml : {resFlaml}')
            #
            #         if resFlaml:
            #             prdVal = resFlaml['mlModel'].predict(prdData[xCol])
            #             prdData['ai'] = prdVal
            #
            #         if len(prdData) > 1:
            #             # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #             # mean_demand = y_test.mean()
            #
            #             rmse = np.sqrt(mean_squared_error(prdData['demand'], prdData['ai']))
            #             r = np.corrcoef(target_df['demand'], prdData['ai'])[0, 1]
            #         else:
            #             rmse = np.nan
            #             r = np.nan
            #
            #         # 결과 저장
            #         all_metrics_list.append({
            #             '구분1': gubun1,
            #             '구분2': gubun2,
            #             'stnName': stn,
            #             # 'N': len(y_test),
            #             'N': len(prdData['demand']),
            #             'RMSE': round(rmse, 2) if not np.isnan(rmse) else np.nan,
            #             'R': round(r, 2) if not np.isnan(r) else np.nan
            #         })
            #         log.info(pd.DataFrame(all_metrics_list))
            #     except Exception as e:
            #         log.error(f"Exception : {e}")

            # ---------------------------------------------------------
            # 5. 최종 결과 취합 및 정렬
            # ---------------------------------------------------------
            final_metrics_df = pd.DataFrame(all_metrics_list)
            final_metrics_df = final_metrics_df.sort_values(by=['구분1', '구분2', 'RMSE']).reset_index(drop=True)
            log.info(final_metrics_df)

            saveFile = datetime.datetime.now().strftime(sysOpt['saveFile'])
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            final_metrics_df.to_csv(saveFile, index=False)
            log.info(f'saveFile : {saveFile}')


            final_metrics_df2 = pd.DataFrame(all_metrics_list2)
            final_metrics_df2 = final_metrics_df2.sort_values(by=['구분1', '구분2', 'RMSE']).reset_index(drop=True)
            log.info(final_metrics_df2)

            saveFile = datetime.datetime.now().strftime(sysOpt['saveFile2'])
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            final_metrics_df2.to_csv(saveFile, index=False)
            log.info(f'saveFile : {saveFile}')


            # all_metrics_list2
            # # 구분1, 구분2 순으로 묶고, 오차율이 낮은 순으로 정렬
            # if not final_metrics_df.empty:
            #     final_metrics_df = final_metrics_df.sort_values(by=['구분1', '구분2', 'RMSE']).reset_index(drop=True)
            #     print("🏆 [식물별/이벤트별/지역별 개별 모델 예측 성능 평가 결과]")
            #     print(final_metrics_df)
            # else:
            #     print("⚠️ 조건을 만족하여 평가가 완료된 모델이 없습니다. (데이터 부족 등)")





            #
            # # 그래프 사이즈 설정
            # plt.figure(figsize=(16, 6))
            #
            # # -------------------------------------------------------------
            # # [그래프 1] 연도별 실제 수요 vs 예측 수요 흐름 (Line Plot)
            # # -------------------------------------------------------------
            # plt.subplot(1, 2, 1)
            # sns.lineplot(data=result_df, x='year', y='demand', label='Actual (실제값)', marker='o')
            # sns.lineplot(data=result_df, x='year', y='predicted_demand', label='Predicted (예측값)', marker='s',
            #              linestyle='--')
            #
            # plt.title('연도별 실제 수요량 vs 예측 수요량 흐름')
            # plt.xlabel('연도 (Year)')
            # plt.ylabel('수요량 (Demand)')
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            #
            # # -------------------------------------------------------------
            # # [그래프 2] 1:1 산점도 (Scatter Plot)
            # # -------------------------------------------------------------
            # plt.subplot(1, 2, 2)
            # sns.scatterplot(x='demand', y='predicted_demand', data=result_df, alpha=0.7)
            #
            # # 1:1 기준선(y=x) 그리기 - 이 선에 점들이 모여있을수록 예측이 완벽하다는 뜻입니다.
            # min_val = min(result_df['demand'].min(), result_df['predicted_demand'].min())
            # max_val = max(result_df['demand'].max(), result_df['predicted_demand'].max())
            # plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line (완벽한 예측)')
            #
            # plt.title('실제값 vs 예측값 1:1 산점도')
            # plt.xlabel('실제 수요량 (Actual Demand)')
            # plt.ylabel('예측 수요량 (Predicted Demand)')
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            #
            # # 그래프 간격 조절 후 출력
            # plt.tight_layout()
            # plt.show()
            #
            #
            #
            #
            # # 5. 예측 수행 (가급적 X_test를 별도로 만들어 사용하는 것을 권장합니다)
            # # # print("====== 예측 결과 (Prediction) ======")
            # # predictions = automl.predict(X_train)
            # # # print(predictions)
            # #
            # # sokcho_df['prd'] = np.round(predictions).astype(int)
            # #
            # # # 1. 평가 지표 계산 (정답인 y_train과 모델이 찍은 predictions를 비교)
            # # rmse = np.sqrt(mean_squared_error(y_train,  sokcho_df['prd']))
            # # rrmse = (rmse / y_train.mean()) * 100
            # # nrmse = (rmse / (y_train.max() - y_train.min())) * 100
            # # mae = mean_absolute_error(y_train,  sokcho_df['prd'])
            # # r2 = r2_score(y_train,  sokcho_df['prd'])
            # # r = np.sqrt(r2)
            #
            # # print("====== 모델 검증 결과 (Metrics) ======")
            # # print(f"RMSE (평균 오차): {rmse:.2f}")
            # # print(f"MAE (절대 평균 오차): {mae:.2f}")
            # # print(f"R2 Score (설명력, 1에 가까울수록 좋음): {r2:.2f}")
            #
            # # ---------------------------------------------------------
            # # 웹 기반 가공 데이터 생산
            # # ---------------------------------------------------------
            # # 데이터 준비
            # target_year = 2025
            # year_df = sokcho_df[sokcho_df['dt'] == target_year].copy()
            # year_df['prd'] = year_df['prd'].round().astype(int)
            #
            # lats = year_df['LAT'].values
            # lons = year_df['LON'].values
            # values = year_df['prd'].values
            #
            # obs_coords = np.column_stack((lons, lats))
            #
            # # 2. 고해상도 타겟 격자 생성 (500x500)
            # grid_lon = np.linspace(124, 131, 1000)
            # grid_lat = np.linspace(33, 39, 1000)
            # Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
            # target_coords = np.column_stack((Grid_lon.ravel(), Grid_lat.ravel()))
            #
            # smoothing_candidates = np.arange(0, 100, 0.5)
            # best_smoothing = 0.0
            # min_error = float('inf')
            #
            # # LOOCV (Leave-One-Out Cross Validation) 방식으로 최적값 찾기
            # for s in smoothing_candidates:
            #     errors = []
            #     # 관측소가 100개라면, 1개를 빼고 99개로 학습한 뒤 1개를 예측해보는 과정을 반복
            #     for i in range(len(obs_coords)):
            #         # i번째 관측소만 제외 (검증용)
            #         train_coords = np.delete(obs_coords, i, axis=0)
            #         train_values = np.delete(values, i)
            #         test_coord = obs_coords[i:i + 1]
            #         test_value = values[i]
            #
            #         # RBF 보간
            #         rbf = RBFInterpolator(train_coords, train_values, kernel='thin_plate_spline', smoothing=s)
            #         pred_value = rbf(test_coord)[0]
            #
            #         errors.append((test_value - pred_value) ** 2)
            #
            #     avg_rmse = np.sqrt(np.mean(errors))
            #     print(f" - Smoothing {s}: RMSE = {avg_rmse:.3f} 일")
            #
            #     if avg_rmse < min_error:
            #         min_error = avg_rmse
            #         best_smoothing = s
            #
            # print(f"최적의 Smoothing 값 선택: {best_smoothing} (최소 오차: {min_error:.3f} 일)")
            #
            # print(f" {target_year}년 개화일 RBF 보간 중...")
            # rbf = RBFInterpolator(
            #     y=obs_coords,
            #     d=values,
            #     kernel='thin_plate_spline',
            #     smoothing=best_smoothing,
            # )
            # grid_prd_smooth = rbf(target_coords).reshape(Grid_lon.shape)
            # print(" 대한민국 영토 마스크 생성 중...")
            #
            # shpfilename = shpreader.natural_earth(resolution='10m',
            #                                       category='cultural',
            #                                       name='admin_0_countries')
            # reader = shpreader.Reader(shpfilename)
            #
            # # 'South Korea' (대한민국) 지형만 추출
            # korea_geoms = []
            # for record in reader.records():
            #     # ISO_A3 코드가 'KOR'인 국가(대한민국)를 찾습니다.
            #     if record.attributes['ISO_A3'] == 'KOR':
            #         korea_geoms.append(record.geometry)
            #
            # # 추출한 대한민국 지형(본토 및 섬들)을 하나로 병합
            # korea_geom = unary_union(korea_geoms)
            # korea_prep = prep(korea_geom)
            #
            # # 격자점이 대한민국 영토 안에 있는지 확인하여 마스크 생성 (True: 한국, False: 바다/외국)
            # mask = np.array([korea_prep.contains(Point(lon, lat))
            #                  for lon, lat in zip(Grid_lon.ravel(), Grid_lat.ravel())])
            # mask = mask.reshape(Grid_lon.shape)
            #
            # grid_prd_final = np.where(mask, grid_prd_smooth, np.nan)
            #
            # # ---------------------------------------------------------
            # # 웹 기반 가공 차트
            # # ---------------------------------------------------------
            # fig, ax = plt.subplots()
            # levels = np.arange(int(np.nanmin(values)), int(np.nanmax(values)) + 2, 1)
            #
            # # 💡 변경점 1: contourf() 가 아닌 contour() 를 사용합니다! (f가 빠짐)
            # contourf_plot = ax.contourf(Grid_lon, Grid_lat, grid_prd_final, levels=levels, cmap='YlGn')
            # plt.close(fig)
            #
            # geojson_str = geojsoncontour.contourf_to_geojson(
            #     contourf=contourf_plot,
            #     ndigits=5,
            #     stroke_width=0
            # )
            # gdf_poly = gpd.read_file(geojson_str, driver='GeoJSON')
            # gdf_poly = gdf_poly.set_crs(epsg=4326)
            # gdf_poly['geometry'] = gdf_poly['geometry'].buffer(0)
            #
            # poly_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_poly_{target_year}.fgb"
            # gdf_poly.to_file(poly_fgb, driver='FlatGeobuf')
            # print(f"면(Polygon) 데이터 저장 완료: {poly_fgb}")
            #
            # point_geometries = [Point(xy) for xy in zip(year_df['LON'], year_df['LAT'])]
            #
            # gdf_points = gpd.GeoDataFrame(
            #     year_df,
            #     geometry=point_geometries,
            #     crs="EPSG:4326"
            # )
            #
            # point_fgb = f"{globalVar['outPath']}/{serviceName}/blooming_point_{target_year}.fgb"
            # gdf_points.to_file(point_fgb, driver='FlatGeobuf')
            # print(f"점(Point) 데이터 저장 완료: {point_fgb}")
            #
            # sys.exit(0)
            #
            #
            # # 기후자료 예측 테스트
            # fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/BDWIDE2026/AR6_SSP126_5ENSMN_skorea_*_gridraw_yearly_2021_2100.nc'))
            # ds = xr.open_mfdataset(fileList)
            #
            # target_year = '2025'
            #
            # # ds['time']
            # # # =========================================================
            # # # 💡 1. xarray에서 2050년 3월~5월(봄) 데이터만 쏙 뽑아서 평균 내기
            # # # =========================================================
            # # # 특정 기간 슬라이싱 후, 시간(time) 차원에 대해 평균(mean)을 구합니다.
            # # ds_spring = ds.sel(time=slice(f'{target_year}-01-01', f'{target_year}-01-01'))
            # # ds_spring_mean = ds_spring.mean(dim='time')
            # #
            # # #       grid_lon = np.linspace(124, 131, 1000)
            # # #             grid_lat = np.linspace(33, 39, 1000)
            # # lonList = np.arange(124, 131, 0.01)
            # # latList = np.arange(33, 39, 0.01)
            # # ds_spring_mean = ds_spring_mean.interpolate_na({'longitude': lonList, 'latitude': latList}, method='linear', fill_value="extrapolate")
            # #
            # # ta_values = ds_spring_mean['TA'].values
            # # lons = ds_spring_mean['longitude'].values
            # # lats = ds_spring_mean['latitude'].values
            # #
            # # Grid_lon, Grid_lat = np.meshgrid(lons, lats)
            # #
            # # # =========================================================
            # # # 2. 고해상도 등치면(Polygon) 생성 (전체 영역)
            # # # =========================================================
            # # print("등치면 생성 중... (마스크 없이 바다까지 포함)")
            # #
            # # fig, ax = plt.subplots()
            # #
            # # min_val = np.nanmin(ta_values)
            # # max_val = np.nanmax(ta_values)
            # # levels = np.arange(int(min_val) - 1, int(max_val) + 2, 1)
            # # print(levels)
            # #
            # # # 기온 변화를 직관적으로 보여주는 컬러맵
            # # contourf_plot = ax.contourf(Grid_lon, Grid_lat, ta_values, levels=levels, cmap='YlOrRd', extend='both')
            # # # plt.show()
            # # plt.close(fig)
            # #
            # # # =========================================================
            # # # 3. GeoJSON 변환
            # # # =========================================================
            # # print("GeoJSON 변환 중...")
            # # geojson_str = geojsoncontour.contourf_to_geojson(
            # #     contourf=contourf_plot,
            # #     ndigits=7,
            # #     stroke_width=0
            # # )
            # #
            # # # GeoPandas로 로드 및 꼬인 폴리곤(buffer 0) 복구
            # # gdf = gpd.read_file(geojson_str, driver='GeoJSON')
            # # gdf = gdf.set_crs(epsg=4326)
            # # # gdf['geometry'] = gdf['geometry'].buffer(0)
            # # # gdf['geometry'] = gdf.simplify(tolerance=0.005, preserve_topology=True)
            # # gdf['year'] = int(target_year)
            # #
            # # gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            # #
            # # # =========================================================
            # # # 4. FlatGeobuf 최종 저장 (클리핑 없이 즉시 저장)
            # # # =========================================================
            # # # output_fgb = f"future_climate_TA_{target_year}_nomask.fgb"
            # # output_fgb = f"{globalVar['outPath']}/{serviceName}/future_climate_TA_{target_year}_nomask.fgb"
            # #
            # #
            # # # 자르는 과정 없이 원본 gdf를 그대로 저장합니다.
            # # gdf.to_file(output_fgb, driver='FlatGeobuf')
            # #
            # # print(f"마스크 없는 초고속 FGB 파일 저장 완료: {output_fgb}")
            # #
            # # import rasterio
            # # # from rasterio.transform import from_bounds
            # #
            # # # 좌표계(CRS)를 위경도(EPSG:4326)로 설정
            # # ds_spring_mean.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            # # ds_spring_mean.rio.write_crs("epsg:4326", inplace=True)
            # #
            # # # 최종 COG(GeoTIFF) 파일로 저장
            # # output_tif = f"{globalVar['outPath']}/{serviceName}/future_climate_TA_{target_year}.tif"
            # # da_clipped.rio.to_raster(output_tif, driver="COG")
            # #
            # # print(f"✅ 초고속 렌더링용 COG 파일 저장 완료: {output_tif}")
            # #
            # # sys.exit(0)







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