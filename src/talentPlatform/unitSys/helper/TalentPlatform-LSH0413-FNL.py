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
from datetime import datetime

from osgeo_utils.gdal_merge import file_info
from sqlalchemy import Column, Float
from sqlalchemy.dialects.mysql import DOUBLE
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy import Column, Numeric
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pytesseract import pytesseract
import os
import re
import cv2
import re
import csv
import pytesseract
from PIL import Image
import glob
import zipfile
import cv2
import pytesseract
import io
from datetime import datetime, timedelta
import pytz
from matplotlib import font_manager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from datetime import timedelta
import pymysql
import requests
from urllib.parse import quote_plus
import configparser
import pymysql

from sqlalchemy.dialects.mysql import insert

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError

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

Base = declarative_base()


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
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
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

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


# 위도와 경도를 도로 환산하는 함수
def dmsToDecimal(dms):
    degrees, minutes, seconds = map(float, dms.split('/'))
    decimalDeg = degrees + minutes / 60 + seconds / 3600
    return decimalDeg


def sortKeyNum(filename):
    match = re.search(r'(\d+)\.txt$', filename)
    return int(match.group(1)) if match else float('inf')


def procFile(data, column_cnt, column_info, path_pattern, serviceName, fileNameNoExt):
    for j, row in data.iterrows():
        log.info(f'[CHECK] row : {row.idx}')

        # inpFile = os.path.join(globalVar['inpPath'], serviceName, path_pattern.format(fileNameNoExt, row.idx))
        inpFile = os.path.join(path_pattern.format(fileNameNoExt, row.idx))
        fileList = sorted(glob.glob(inpFile), key=sortKeyNum)

        data.loc[j, column_cnt] = None
        data.loc[j, column_info] = None

        if len(fileList) > 0:
            txtData = pd.read_csv(fileList[0], sep=' ', header=None)
            data.loc[j, column_cnt] = len(txtData)
            data.loc[j, column_info] = txtData.to_json(orient='records')

    return data


def getPubliIp():
    response = requests.get('https://api.ipify.org')
    return response.text


def initCfgInfo(sysPath):
    log.info('[START] {}'.format('initCfgInfo'))
    # log.info('[CHECK] sysPath : {}'.format(sysPath))

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')

        configKey = 'mysql-clova-dms02user01'
        dbUser = config.get(configKey, 'user')
        dbPwd = quote_plus(config.get(configKey, 'pwd'))
        dbHost = config.get(configKey, 'host')
        dbHost = 'localhost' if dbHost == getPubliIp() else dbHost
        dbPort = config.get(configKey, 'port')
        dbName = config.get(configKey, 'dbName')

        dbEngine = create_engine(f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}', echo=False)
        # dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessMake = sessionmaker(bind=dbEngine)
        session = sessMake()
        # session.execute("""SELECT * FROM TB_VIDEO_INFO""").fetchall()

        # API 정보
        # apiUrl = config.get('pv', 'url')
        # apiToken = config.get('pv', 'token')

        result = {
            'dbEngine': dbEngine
            , 'session': session
            # , 'apiUrl': apiUrl
            # , 'apiToken': apiToken
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))


# ================================================
# 4. 부 프로그램
# ================================================
# class VideoInfoDtl(Base):
#     __tablename__ = "TB_VIDEO_INFO_DTL"
#
#     VIEDO_PATH = Column(String(200), primary_key=True, index=True, comment="비디오 경로")
#     VIEDO_NAME = Column(String(200), primary_key=True, index=True, comment="비디오 이름")
#     DATE_TIME = Column(DateTime, comment="날짜")
#     IDX = Column(Integer, comment="인덱스")
#     LAT = Column(Float, comment="위도")
#     LON = Column(Float, comment="경도")
#     CNT = Column(Integer, comment="인덱스")
#     CNT2 = Column(Integer, comment="인덱스")
#     DATE_TIME_DIFF = Column(Integer, comment="시간 간격")


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 뇌파 측정 자료 처리 및 스펙트럼 및 푸리에 변환 시각화

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0413'

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

                # 옵션 설정
                sysOpt = {
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 'srtRow': 430
                    # , 'endRow': 1800
                    # , 'srcCol': 1000
                    # , 'endCol': 1060
                    'srtRow': 290
                    , 'endRow': 1030
                    , 'srcCol': 670
                    , 'endCol': 710

                    , 'pyTesCmd': '/SYSTEMS/anaconda3/envs/py38/bin/tesseract'
                    , 'pyTesData': '/SYSTEMS/anaconda3/envs/py38/share/tessdata'
                    , 'pyHumanCnt': '/SYSTEMS/PROG/PYTHON/HUMAN-CNT'

                    , 'minLat': 30
                    , 'maxLat': 40
                    , 'minLon': 120
                    , 'maxLon': 135

                    # 시간 임계값
                    , 'timeThres': 60

                    # , 'videoPath' : '202305/29/1840'
                    # , 'videoName' : '20230504_output.mp4'
                    # , 'videoPath': '202305/31/2333'
                    # , 'videoName': 'test2.mp4'
                    # , 'videoPath': '202306/04/1225'
                    # , 'videoName': '20230604.mp4'
                    # , 'videoPath': '202306/04/1306'
                    # , 'videoName': '20230519100347_095553.MP4'
                    , 'videoPath': globalVar['videoPath']
                    , 'videoName': globalVar['videoName']
                }

                globalVar['updPath'] = '/DATA/VIDEO'
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ********************************************************************************************
            #
            # ********************************************************************************************
            # DB 정보
            cfgInfo = initCfgInfo(f"{globalVar['cfgPath']}/system.cfg")
            dbEngine = cfgInfo['dbEngine']
            session = cfgInfo['session']

            metadata = MetaData()

            makeTable = Table(
                'TB_VIDEO_INFO_DTL'
                , metadata
                , Column('VIDEO_PATH', String(500), primary_key=True, index=True, comment="비디오 경로")
                , Column('VIDEO_NAME', String(500), primary_key=True, index=True, comment="비디오 이름")
                , Column('DATE_TIME', DateTime, comment="날짜")
                , Column('IDX', Integer, primary_key=True, index=True, comment="인덱스")
                , Column('LAT', Float, comment="위도")
                , Column('LON', Float, comment="경도")
                , Column('CNT', Integer, comment="AI인파")
                , Column('CNT2', Integer, comment="AI인파2")
                , Column('DATE_TIME_DIFF', Float, comment="시간 간격")
                , Column('DOWN', String(500), comment="다운로드")
                , extend_existing=True
            )

            metadata.create_all(bind=dbEngine)

            x1, x2, y1, y2 = sysOpt['srtRow'], sysOpt['endRow'], sysOpt['srcCol'], sysOpt['endCol']
            pytesseract.pytesseract.tesseract_cmd = sysOpt['pyTesCmd']
            os.environ['TESSDATA_PREFIX'] = sysOpt['pyTesData']

            patterns = {
                # "lat": r".(\d{2}\/\d{2}\/\d{2})",
                # "lon": r".(\d{3}\/\d{2}\/\d{2})",
                # "dateTime": r"(\d{4}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2})"
                "lat": r'[-+]?[0-9]*\.?[0-9]+',
                "lon": r'[-+]?[0-9]*\.?[0-9]+',
                "dateTime": r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
            }

            #     pattern = r"(N|S)*(\d+\.*\d*) (E|W|£)(\d+\.*\d*) (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20230504_output.mp4')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20230501122318_000005.mp4')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'KakaoTalk_20230519_112032690.mp4')
            inpFile = '{}/{}/{}'.format(globalVar['updPath'], sysOpt['videoPath'], sysOpt['videoName'])

            fileList = sorted(glob.glob(inpFile))

            if (len(fileList) < 1):
                raise Exception(f'[ERROR] inpFile : {inpFile} : 입력 자료를 확인해주세요.')

            for i, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo : {fileInfo}')

                # fileNameNoExt = os.path.basename(fileInfo).split('.mp4')[0]
                fileNameNoExt = os.path.splitext(os.path.basename(fileInfo))[0]

                cap = cv2.VideoCapture(fileInfo)

                # 프레임 수
                cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                # 프레임 레이트 확인
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 재생 시간 초 단위
                playTime = cnt / fps

                # idx = 0
                # dataL1 = pd.DataFrame()
                # while cap.isOpened():
                #     try:
                #         # cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
                #         # cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
                #         # cap.set(cv2.CAP_PROP_POS_FRAMES, 8000)
                #         # cap.set(cv2.CAP_PROP_POS_FRAMES, 9500)
                #         # cap.set(cv2.CAP_PROP_POS_FRAMES, 10500)
                #         ret, frame = cap.read()
                #         if not ret: break
                #
                #         cropImg = frame[y1:y2, x1:x2]
                #         newImg = Image.fromarray(cropImg)
                #
                #         # plt.imshow(newImg)
                #         # plt.show()
                #
                #         # plt.imshow(Image.fromarray(cropImg[:, 20:180]))
                #         # plt.show()
                #         #
                #         # plt.imshow(Image.fromarray(cropImg[:, 210:390]))
                #         # plt.show()
                #         #
                #         # plt.imshow(Image.fromarray(cropImg[:, 400:740]))
                #         # plt.show()
                #
                #         grayImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
                #         thresVal, thresImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                #         # newImg = Image.fromarray(thresImg)
                #
                #         # plt.imshow(newImg)
                #         # plt.show()
                #
                #         # plt.imshow(Image.fromarray(thresImg))
                #         # plt.show()
                #         #
                #         # plt.imshow(Image.fromarray(thresImg[:, 210:265]))
                #         # plt.show()
                #         #
                #         # plt.imshow(Image.fromarray(thresImg[5:35, 280:385]))
                #         # plt.show()
                #
                #         idx += 1
                #         # log.info(f'[CHECK] idx : {idx}')
                #
                #         extInfo = {}
                #
                #         # if len(data) < 1: continue
                #         # data = pytesseract.image_to_string(newImg)
                #         # log.info(f'[CHECK] data : {data}')
                #
                #         # 패턴에 따라 문자열에서 데이터 추출
                #         # extInfo = {key: re.search(pattern, data).group(1) if re.search(pattern, data) else None for key, pattern in patterns.items()}
                #         # log.info(f'[CHECK] extInfo : {extInfo}')
                #
                #         # getLat = pytesseract.image_to_string(Image.fromarray(thresImg[:, 0:180]))
                #         extInfo['lat'] = None
                #         try:
                #             extInfo['lat-imgToStr'] = pytesseract.image_to_string(Image.fromarray(thresImg[:, 20:55]))
                #             extInfo['lat-imgToStr2'] = pytesseract.image_to_string(Image.fromarray(thresImg[:, 70:180]))
                #
                #             if len(extInfo['lat-imgToStr']) > 0 and len(extInfo['lat-imgToStr2']) > 0:
                #                 extInfo['lat'] = pd.to_numeric(''.join(re.findall(patterns['lat'], f"{extInfo['lat-imgToStr']}.{extInfo['lat-imgToStr2']}")))
                #         except Exception as e:
                #             log.error("Exception : {}".format(e))
                #
                #         # getLon = pytesseract.image_to_string(Image.fromarray(thresImg[:, 210:390]))
                #         extInfo['lon'] = None
                #         try:
                #             extInfo['lon-imgToStr'] = pytesseract.image_to_string(Image.fromarray(thresImg[:, 210:265]))
                #             extInfo['lon-imgToStr2'] = pytesseract.image_to_string(Image.fromarray(thresImg[:, 280:390]))
                #
                #             if len(extInfo['lon-imgToStr']) > 0 and len(extInfo['lon-imgToStr2']) > 0:
                #                 extInfo['lon'] = pd.to_numeric(''.join(re.findall(patterns['lon'], f"{extInfo['lon-imgToStr']}.{extInfo['lon-imgToStr2']}")))
                #         except Exception as e:
                #             log.error("Exception : {}".format(e))
                #
                #         extInfo['dateTime'] = None
                #         try:
                #             extInfo['dateTime-imgToStr'] = pytesseract.image_to_string(Image.fromarray(thresImg[:, 400:740]))
                #
                #             if len(extInfo['dateTime-imgToStr']) > 0:
                #                 extInfo['dateTime'] = re.search(patterns['dateTime'], extInfo['dateTime-imgToStr']).group(1)
                #         except Exception as e:
                #             log.error("Exception : {}".format(e))
                #
                #         dict = {
                #             'videoInfo': [fileInfo]
                #             , 'idx': [idx]
                #             # , 'frame': [frame]
                #             # , 'lat': [dmsToDecimal(extInfo['lat'])]
                #             # , 'lon': [dmsToDecimal(extInfo['lon'])]
                #             # , 'dateTime': [pd.to_datetime(extInfo['dateTime'])]
                #             # , 'data': [data]
                #             , 'lat': [extInfo['lat']]
                #             , 'lon': [extInfo['lon']]
                #             , 'dateTime': [pd.to_datetime(extInfo['dateTime'])]
                #             , 'extInfo': [extInfo]
                #         }
                #
                #         # log.info(f'[CHECK] dict : {dict}')
                #
                #         dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dict)], ignore_index=True)
                #     except Exception as e:
                #         log.error("Exception : {}".format(e))
                #
                # # ********************************************************************************************
                # # 자료 병합
                # # **************************************************************************************
                # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, 'dataL1')
                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['updPath'], sysOpt['videoPath'], fileNameNoExt, 'dataL1')
                # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # dataL1.to_csv(saveFile, index=False)
                # log.info(f'[CHECK] saveFile : {saveFile}')

                dataL1 = pd.read_csv(saveFile)

                dataL1 = dataL1[
                    (dataL1['lat'] >= sysOpt['minLat']) &
                    (dataL1['lat'] <= sysOpt['maxLat']) &
                    (dataL1['lon'] >= sysOpt['minLon']) &
                    (dataL1['lon'] <= sysOpt['maxLon'])
                    ]

                statData = dataL1.groupby(['videoInfo']).agg(lambda x: x.value_counts().index[0])
                # log.info(f'[CHECK] statData : {statData}')

                # 데이터 필터링
                dataL2 = dataL1[
                    (abs(dataL1['lat'] - statData['lat'][0]) <= 0.05) &
                    (abs(dataL1['lon'] - statData['lon'][0]) <= 0.05) &
                    (abs(pd.to_datetime(dataL1['dateTime']) - pd.to_datetime(statData['dateTime'][0])) <= timedelta(seconds=playTime))
                    ]

                dataL3 = dataL2.groupby(['videoInfo', 'dateTime']).agg(lambda x: x.value_counts().index[0]).reset_index()

                # yolov7-object-tracking
                dataL3 = procFile(dataL3, 'NEW-cnt', 'NEW-info', f"{sysOpt['pyHumanCnt']}/yolov7-object-tracking/result/{sysOpt['videoPath']}/labels/" + "{}_{}.txt", serviceName, fileNameNoExt)

                # YOLOv7-DeepSORT-Object-Tracking
                dataL3 = procFile(dataL3, 'NEW2-cnt', 'NEW2-info',   f"{sysOpt['pyHumanCnt']}/YOLOv7-DeepSORT-Object-Tracking/result/{sysOpt['videoPath']}/labels/" + "{}_{}.txt", serviceName, fileNameNoExt)

                # for j, row in dataL3.iterrows():
                #     saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, str(row.idx).zfill(10))
                #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                #     plt.imshow(row.frame)
                #     plt.axis("off")
                #     plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                #     # plt.show()
                #     log.info(f'[CHECK] saveImg : {saveImg}')

                dataL3['dateTimeDiff'] = pd.to_datetime(dataL3['dateTime']).diff().dt.total_seconds()

                # 특정 임계값 60초 이상
                dataL4 = dataL3[dataL3['dateTimeDiff'] <= sysOpt['timeThres']]

                # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, 'FNL')
                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['updPath'], sysOpt['videoPath'], fileNameNoExt, 'FNL')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL4.to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # saveFile = '{}/{}/{}_{}.xlsx'.format(globalVar['outPath'], serviceName, fileNameNoExt, 'FNL')
                saveFile = '{}/{}/{}_{}.xlsx'.format(globalVar['updPath'], sysOpt['videoPath'], fileNameNoExt, 'FNL')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL4.to_excel(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # 알집 압축
                # inpFile = '{}/{}/**/*'.format(globalVar['updPath'], sysOpt['videoPath'], fileNameNoExt)
                inpFile = '{}/{}/**/*'.format(globalVar['updPath'], sysOpt['videoPath'])
                orgFileList = sorted(glob.glob(inpFile, recursive=True))
                fileList = sorted(filter(re.compile('.*\.(csv|xlsx|mp4|MP4)$').match, orgFileList))

                zipFile = '{}/{}/{}.zip'.format(globalVar['updPath'], sysOpt['videoPath'], fileNameNoExt)
                with zipfile.ZipFile(zipFile, 'w', compresslevel=9) as zipf:
                    for fileInfo in fileList:
                        zipf.write(fileInfo, arcname=fileInfo.replace(globalVar['updPath'], '').replace(sysOpt['videoPath'], ''))

                log.info(f'[CHECK] zipFile : {zipFile}')

                # DB 저장
                dbData = dataL4[['dateTime', 'idx', 'lat', 'lon', 'NEW-cnt', 'NEW2-cnt', 'dateTimeDiff']].rename(
                    {
                        'dateTime': 'DATE_TIME'
                        , 'idx': 'IDX'
                        , 'lat': 'LAT'
                        , 'lon': 'LON'
                        , 'NEW-cnt': 'CNT'
                        , 'NEW2-cnt': 'CNT2'
                        , 'dateTimeDiff': 'DATE_TIME_DIFF'
                    }
                    , axis=1
                )

                dbData['VIDEO_PATH'] = sysOpt['videoPath']
                dbData['VIDEO_NAME'] = sysOpt['videoName']
                # dbData['DOWN'] = f"http://{getPubliIp()}:9000/file/down?file={sysOpt['videoPath']}/{fileNameNoExt}.zip"
                dbData['DOWN'] = f"http://{getPubliIp()}:9000/VIDEO/{sysOpt['videoPath']}/{fileNameNoExt}.zip"

                try:
                    dbData.to_sql(name=makeTable.name, con=dbEngine, if_exists='append', index=False)
                    session.commit()
                except SQLAlchemyError as e:
                    session.rollback()
                    log.error(f'Exception : {e}')

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
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
