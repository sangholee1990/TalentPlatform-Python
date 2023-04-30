# -*- coding: utf-8 -*-
import glob
# import seaborn as sns
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from scipy.stats import linregress
import pandas as pd
import cartopy.crs as ccrs
import cartopy as crt
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis
import matplotlib.cm as cm

os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW

# import pykrige.kriging_tools as kt

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

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

            log.info("[CHECK] {} / val : {}".format(key, val))

            # self 변수에 할당
            # setattr(self, key, val)

        return globalVar

def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
    # nohup bash RunShell-Python.sh "2020-10" &

    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
    # python3 ${contextPath}/TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "$1" --endDate "$2"
    # python3 TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "20210901" --endDate "20210902"
    # bash RunShell-Python.sh "20210901" "20210902"

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0255'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info("[START] __init__ : {}".format("init"))

        try:
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

        try:

            if (platform.system() == 'Windows'):

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2020-09-01'
                    , 'endDate': '2020-09-03'
                }
            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                }

            inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/GA_STN_INFO.xlsx')
            posData = pd.read_excel(inpPosFile)
            posDataL1 = posData[['id', 'lat', 'lon']]

            lat1D = np.array(posDataL1['lat'])
            lon1D = np.array(posDataL1['lon'])
            # lon2D, lat2D = np.meshgrid(lon1D, lat1D)


            # *******************************************************
            # GK2A
            # *******************************************************
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(10))

            cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'satInfo/gk2a_ami_le2_cld_ko020lc_202009010000.nc')
            log.info("[CHECK] cfgFile : {}".format(cfgFile))

            cfgDs = xr.open_dataset(cfgFile)

            # 위/경도 반환
            imgProjInfo = cfgDs['gk2a_imager_projection'].attrs

            # ccrs.LambertConformal()
            mapLccProj = ccrs.LambertConformal(
                central_longitude=imgProjInfo['central_meridian']
                , central_latitude=imgProjInfo['origin_latitude']
                , secant_latitudes=(imgProjInfo['standard_parallel1'], imgProjInfo['standard_parallel2'])
                , false_easting=imgProjInfo['false_easting']
                , false_northing=imgProjInfo['false_northing']
            )

            mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'

            try:
                mapLccProjInfo = mapLccProj.to_proj4()
            except Exception as e:
                log.error("Exception : {}".format(e))

            mapProj = pyproj.Proj(mapLccProjInfo)

            nx = imgProjInfo['image_width']
            ny = imgProjInfo['image_height']
            xOffset = imgProjInfo['lower_left_easting']
            yOffset = imgProjInfo['lower_left_northing']

            res = imgProjInfo['pixel_size']

            # 직교 좌표
            rowEle = (np.arange(0, nx, 1) * res) + xOffset
            colEle = (np.arange(0, ny, 1) * res) + yOffset
            # colEle = colEle[::-1]
            # colEle[0]

            # posLon = posInfo['lon']
            # posLat = posInfo['lat']
            posRow, posCol = mapProj(lon1D, lat1D, inverse=False)
            posDataL1['row'] = posRow
            posDataL1['col'] = posCol
            posDataL1['id'] = posDataL1['id'].astype(int)

            # getImg = 'http://nmsc.kma.go.kr/IMG/GK2A/AMI/PRIMARY/L1B/COMPLETE/KO/202203/24/12/gk2a_ami_le1b_rgb-s-true_ko020lc_202203241240.png'
            imgUrl = 'http://nmsc.kma.go.kr/IMG/GK2A/AMI/PRIMARY/L1B/COMPLETE/KO/202203/24/03/gk2a_ami_le1b_rgb-true_ko010lc_202203240352.png'

            from PIL import Image
            imgOrg = plt.imread(imgUrl)
            # img_color = Image.open(getImg)

            # img_color.size
            # 열, 행, rgba
            imgOrg.shape
            dim = imgOrg.shape
            height = dim[0]
            width = dim[1]
            labelWidth = 45
            plt.imshow(imgOrg)
            plt.show()

            # **********************************************************
            # 이미지 제목
            # **********************************************************
            imgLabel = imgOrg[0:labelWidth, :, :]
            imgLabel.shape
            plt.imshow(imgLabel)
            plt.axis('off')
            plt.show()

            # **********************************************************
            # 이미지 내용
            # **********************************************************
            imgCont = imgOrg[labelWidth:height, :, :]
            plt.imshow(imgCont, extent=[rowEle.min(), rowEle.max(), colEle.min(), colEle.max()])

            cbarList = cm.rainbow(np.linspace(0, 1, len(posDataL1)))
            for i, posInfo in posDataL1.iterrows():
                plt.scatter(posInfo['row'], posInfo['col'], color=cbarList[i],  marker="o", s=5, cmap='Set2', label=int(posInfo['id']))

            plt.legend(fontsize=6)
            plt.axis('off')

            import tempfile
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'RGB-TRUE')
            saveTmp = tempfile.NamedTemporaryFile(suffix='.png').name
            plt.savefig(saveTmp, width=10, height=10, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()


            imgContL1 = plt.imread(saveTmp)
            imgContL1.shape
            plt.imshow(imgContL1)
            plt.axis('off')
            plt.show()
            import cv2
            imgContResize = cv2.resize(imgContL1, dsize=(1800, 1800), interpolation=cv2.INTER_LINEAR)
            # plt.imshow(imgContResize)
            # plt.axis('off')
            # plt.show()

            imgPrep = cv2.vconcat([imgLabel, imgContResize])
            plt.imshow(imgPrep)
            plt.axis('off')
            plt.savefig(saveImg, width=10, height=10, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()
            #
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #
            # dsDataL2.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))


from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time
import os
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler


def tick():
    print('Tick! The time is: %s' % datetime.now())

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = { }

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

        # scheduler = BackgroundScheduler()
        # scheduler = AsyncIOScheduler()
        # scheduler.add_job(tick, 'interval', seconds=1)
        # scheduler.start()
        # print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
        #
        # try:
        #     # This is here to simulate application activity (which keeps the main thread alive).
        #     # while True:
        #     #     time.sleep(2)
        #     asyncio.get_event_loop().run_forever()
        # except (KeyboardInterrupt, SystemExit):
        #     # Not strictly necessary if daemonic mode is enabled but should be done if possible
        #     scheduler.shutdown()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
