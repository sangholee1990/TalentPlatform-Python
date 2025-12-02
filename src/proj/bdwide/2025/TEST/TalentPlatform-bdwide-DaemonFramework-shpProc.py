# ================================================
# 요구사항
# ================================================
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0627-DaemonFramework-model.py
# 0 0 * * * cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys && /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py

import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import scipy.sparse
import pandas as pd
import numpy as np
import re
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import scipy.sparse
import geopandas as gpd
import matplotlib.font_manager as fm
from shapely.geometry import Point
import matplotlib.cm as cm
warnings.filterwarnings('ignore')

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

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')


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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'BDWIDE2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

        try:
            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                'fontInfo': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/fontInfo/malgun.ttf',
                'shpFilePattern': '/HDD/DATA/INPUT/BDWIDE2025/20251203_대축적_산림_임상도(2024)/*/*.shp',
                'saveImgPattern': '/HDD/DATA/FIG/BDWIDE2025/{type}_{key}.png',

                # 경기도 가평군 상면 축령로 99
                '가평군_농장': {'posLat': 37.7738688, 'posLon': 127.3618844},

                # 지점 대비 반경 1km
                'posDist': 1,
            }

            # 글꼴 설정
            fm.fontManager.addfont(sysOpt['fontInfo'])
            fontName = fm.FontProperties(fname=sysOpt['fontInfo']).get_name()
            plt.rcParams['font.family'] = fontName

            # shp 파일
            shpList = sorted(glob.glob(sysOpt['shpFilePattern']))
            for shpInfo in shpList:
                log.info(f"shpInfo : {shpInfo}")

                partList = re.split(r'[./]', shpInfo)
                type = partList[6]
                key = partList[7]
                log.info(f"type : {type}, key : {key}")

                # 지점, 반경 설정
                posInfo = sysOpt.get(type)
                if not posInfo: continue

                posLat = posInfo['posLat']
                posLon = posInfo['posLon']

                posDist = sysOpt['posDist']
                log.info(f"posLat : {posLat}, posLon : {posLon}, posDist : {posDist} km")
                centerPointGeom = Point(posLon, posLat)

                gdf = gpd.read_file(shpInfo, encoding='euc-kr')
                log.info(f"shape : {gdf.shape}")
                log.info(f"columns : {gdf.columns}")

                # 테스트
                # fig, ax = plt.subplots(figsize=(10, 10))
                # gdf.plot(column='KOFTR_NM', ax=ax, legend=True, cmap='Set3', legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1, 0.5)})
                # plt.axis('off')
                # plt.show()

                centerGdf = gpd.GeoDataFrame([{'geometry': centerPointGeom}], crs='EPSG:4326')
                centerConverted = centerGdf.to_crs(gdf.crs)

                bufferGeom = centerConverted.geometry.buffer(posDist * 1000)
                bufferGdf = gpd.GeoDataFrame([{'geometry': bufferGeom[0]}], crs=gdf.crs)
                radiusGdf = gpd.clip(gdf, bufferGdf)

                if radiusGdf.empty:
                    log.info(f"분석 자료 없음")
                    continue

                radiusGdf['newArea'] = radiusGdf.geometry.area
                radiusGdf = radiusGdf.sort_values(by='newArea', ascending=False)
                statsRadius = radiusGdf.groupby('KOFTR_NM')['newArea'].sum().sort_values(ascending=False)
                totalAreaRadius = statsRadius.sum()

                labelMapping = {}
                for name, area in statsRadius.items():
                    ratio = (area / totalAreaRadius) * 100
                    # labelText = f"{name} {int(area):,} m² ({ratio:.1f}%)"
                    labelText = f"{name} ({ratio:.1f}%)"
                    log.info(f"labelText : {labelText}")
                    labelMapping[name] = labelText

                fig, ax = plt.subplots(figsize=(12, 10))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                gdf.plot(ax=ax, color='lightgray', alpha=0.4)

                # radiusGdf.plot(column='KOFTR_NM', ax=ax, cmap='Set3', alpha=1.0, legend=True, legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1, 0.5)})
                # radiusGdf.plot(column='KOFTR_NM', ax=ax, cmap=cm.get_cmap('jet'), alpha=1.0, legend=True, legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1, 0.5)})
                # radiusGdf.plot(column='KOFTR_NM', ax=ax, cmap=cm.get_cmap('jet'), alpha=1.0, legend=True, legend_kwds={'loc': 'lower right', 'borderaxespad': 0})

                cmap = cm.get_cmap('jet')
                categories = statsRadius.index.tolist()
                count = len(categories)

                legendList = []
                for i, name in enumerate(categories):
                    subset = radiusGdf[radiusGdf['KOFTR_NM'] == name]
                    if count > 1:
                        color = cmap(i / (count - 1))
                    else:
                        color = cmap(0.5)
                    subset.plot(
                        ax=ax,
                        color=color,
                        alpha=1.0
                    )
                    patch = mpl.patches.Patch(color=color, label=labelMapping[name])
                    legendList.append(patch)

                leg = ax.legend(handles=legendList, loc='lower right', borderaxespad=0)
                if leg:
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_alpha(1.0)
                    leg.get_frame().set_edgecolor('white')

                # bufferGdf.boundary.plot(ax=ax, color='red', linestyle='--', linewidth=2, label='1km 반경')
                centerConverted.plot(ax=ax, color='grey', marker='*', markersize=150, zorder=5)
                # ax.set_title("지점 반경 1km 산림정보 분석")
                ax.margins(0)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])

                saveImg = sysOpt['saveImgPattern'].format(type=type, key=key)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
                log.info(f"saveImg : {saveImg}")
                plt.show()
                plt.close()

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))