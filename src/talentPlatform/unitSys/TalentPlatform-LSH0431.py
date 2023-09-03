# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import urllib
import warnings
from datetime import datetime
import pytz
import googlemaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict
from sklearn.neighbors import BallTree
import requests
import json
import pymysql
import requests
from urllib.parse import quote_plus
import configparser
import pymysql
import zipfile

from sqlalchemy.dialects.mysql import insert
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float
from sqlalchemy.dialects.mysql import DOUBLE
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy import Column, Numeric

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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

def getPubliIp():
    response = requests.get('https://api.ipify.org')
    return response.text

def initCfgInfo(sysPath):
    log.info('[START] {}'.format('initCfgInfo'))

    result = None

    try:
        # DB 연결 정보
        # pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='UTF-8')

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

        # 테이블 정보
        metaData = MetaData()

        # 예보 모델 테이블
        tbSaleDown = Table('TB_SALE_DOWN', metaData, autoload_with=dbEngine, schema=dbName)

        # 기본 위경도 테이블
        tbSaleInfo = Table('TB_SALE_INFO', metaData, autoload_with=dbEngine, schema=dbName)

        result = {
            'dbEngine': dbEngine
            , 'session': session
            , 'tbSaleDown': tbSaleDown
            , 'tbSaleInfo': tbSaleInfo
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))

def getRentType(row):
    if row['거래가'] > 0:
        return '매매'
    elif row['월세가'] > 0:
        return '월세'
    else:
        return '전세'

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 공공데이터포털 실거래가 API (아파트, 단독다가구, 연립다세대, 오피스텔) 수집 및 구글 스튜디오 시각화

    # 그런데 이번에 법이 좀 바뀌어서 항목에 등기여부가 들어가 있습니다. 그부분도 추가할 수 있을까요?
    # 그리고 부동산 종류도 상업용과 토지도 있는데 이부분도 포함해서 할 수 있을까요?

    # [국토교통부] 실거래 관련 API 추가 항목 안내 (총2종)
    # 2023-07-25
    # 등기일자
    # 계약체결일이 '23.1.1. 이후인 아파트 거래신고 건
    # 국토교통부_아파트매매 실거래자료
    # 국토교통부_아파트매매 실거래 상세 자료

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
    serviceName = 'LSH0431'

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2000-01-01'
                , 'endDate': datetime.now().strftime("%Y-%m-%d")

                # 공공데이터포털 API
                , 'apiKey': 'bf9fH0KLgr65zXKT5D/dcgUBIj1znJKnUPrzDVZEe6g4gquylOjmt65R5cjivLPfOKXWcRcAWU0SN7KKXBGDKA=='

                # 매매 실거래
                , 'apiList': {
                    '아파트': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade'
                    , '오피스텔': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcOffiTrade'
                    , '단독다가구': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcSHTrade'
                    , '연립다세대': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHTrade'
                }

                # 전월세
                , 'apiList2': {
                    '아파트': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent'
                    , '오피스텔': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcOffiRent'
                    , '단독다가구': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcSHRent'
                    , '연립다세대': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHRent'
                }

                # 검색 목록
                # , 'addrList': ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원특별자치도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도']
                # , 'addrList': ['제주특별자치도']
                # , 'addrList': ['부산광역시']
                , 'addrList':  [globalVar['addrList']]
            }

            # 변수 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # 구글 API 설정
            # gmap = googlemaps.Client(key=sysOpt['googleApiKey'])

            # DB 정보
            cfgInfo = initCfgInfo(f"{globalVar['cfgPath']}/system.cfg")
            dbEngine = cfgInfo['dbEngine']
            session = cfgInfo['session']
            tbSaleDown = cfgInfo['tbSaleDown']
            tbSaleInfo = cfgInfo['tbSaleInfo']

            # metadata = MetaData()
            #
            # tbSaleDown = Table(
            #     'TB_SALE_DOWN'
            #     , metadata
            #     , Column('ID', Integer, primary_key=True, index=True, autoincrement=True, comment="번호")
            #     , Column('TYPE', String(500), comment="시군구")
            #     , Column('CSV_INFO', String(500), comment="가공 파일")
            #     , Column('ZIP_INFO', String(500), comment="원본 파일")
            #     , Column('REG_DATE', DateTime, default=datetime.now(pytz.timezone('Asia/Seoul')), onupdate=datetime.now(pytz.timezone('Asia/Seoul')), comment="등록일")
            #     , extend_existing=True
            # )
            #
            # tbSaleInfo = Table(
            #     'TB_SALE_INFO'
            #     , metadata
            #     , Column('ID', Integer, primary_key=True, index=True, autoincrement=True, comment="번호")
            #     , Column('TYPE', String(500), comment="종류")
            #     , Column('NAME', String(500), comment="이름")
            #     , Column('ADDR', String(500), comment="주소")
            #     , Column('SI_DONG', String(500), comment="시도 법정동")
            #     , Column('DONG', String(500), comment="법정동")
            #     , Column('SALE_DATE', DateTime, comment="계약 날짜")
            #     , Column('CONV_YEAR', Integer, comment="건축년도")
            #     , Column('YEAR', Integer, comment="연도")
            #     , Column('AREA', Integer, comment="평형")
            #     , Column('SALE_PRICE', Float, comment="매매가")
            #     , Column('SALE_PRICE2', Float, comment="보증가")
            #     , Column('SALE_PRICE3', Float, comment="월세가")
            #     , Column('SALE_PRICE_CONV', Float, comment="거래금액 억원")
            #     , Column('SALE_TYPE', String(500), comment="거래가 분류")
            #     , Column('FLOOR', Integer, comment="층")
            #     , Column('PYEONG', String(500), comment="평수")
            #     , Column('LAT', Float, comment="위도")
            #     , Column('LON', Float, comment="경도")
            #     , Column('RENT_TYPE', String(500), comment="거래 유형")
            #     , Column('DONG', String(500), comment="법정동")
            #     , Column('REG_DATE', DateTime, default=datetime.now(pytz.timezone('Asia/Seoul')), onupdate=datetime.now(pytz.timezone('Asia/Seoul')), comment="등록일")
            #     , extend_existing=True
            # )
            #
            # metadata.create_all(bind=dbEngine)

            # *********************************************************************************
            # 법정동 코드 읽기
            # *********************************************************************************
            inpFile = '{}/{}'.format(globalVar['mapPath'], 'admCode/법정동코드_전체자료.txt')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')

            admData = pd.read_csv(fileList[0], encoding='EUC-KR', sep='\t')
            admData[['d1', 'd2', 'd3', 'd4', 'd5']] = admData['법정동명'].str.split(expand=True)
            admData['sigunguCd'] = admData['법정동코드'].astype('str').str.slice(0, 5)
            admData['bjdongCd'] = admData['법정동코드'].astype('str').str.slice(5, 10)
            admData = admData[(admData['폐지여부'] == '존재')]

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용한 아파트/오피스텔/단독다가구/연립다세대 매매 실거래
            # # *********************************************************************************
            # for i, apiInfo in enumerate(sysOpt['apiList']):
            #     log.info(f'[CHECK] apiInfo : {apiInfo}')
            #
            #     apiUrl = sysOpt['apiList'][apiInfo]
            #
            #     # addrInfo = sysOpt['addrList'][0]
            #     for ii, addrInfo in enumerate(sysOpt['addrList']):
            #         log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #         # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #         admDataL1 = admData[
            #             (admData['법정동명'].str.contains(addrInfo))
            #             & (admData['d1'].notna())
            #             & (admData['d2'].notna())
            #             & (admData['d3'].isna())
            #             & (admData['d4'].isna())
            #             & (admData['d5'].isna())
            #         ]
            #         if (len(admDataL1) < 1): continue
            #
            #         # 시군구 목록
            #         sigunguCdList = set(admDataL1['sigunguCd'])
            #         log.info(f'[CHECK] sigunguCdList : {sigunguCdList}')
            #
            #         # 페이지 설정
            #         pageList = np.arange(0, 9999, 1)
            #
            #         for i, sigunguCdInfo in enumerate(sigunguCdList):
            #
            #             selAddrInfo = admData[
            #                 (admData['sigunguCd'] == sigunguCdInfo)
            #                 & (admData['bjdongCd'] == '00000')
            #             ]['법정동명'].values[0]
            #
            #             log.info(f'[CHECK] sigunguCdInfo : {sigunguCdInfo}')
            #             log.info(f'[CHECK] selAddrInfo : {selAddrInfo}')
            #
            #             for j, dtMonthInfo in enumerate(dtMonthList):
            #                 # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')
            #
            #                 dtYearMonth = dtMonthInfo.strftime('%Y%m')
            #
            #                 saveFile = '{}/{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '매매', apiInfo, selAddrInfo, apiInfo, selAddrInfo, dtYearMonth)
            #                 os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #
            #                 dataL1 = pd.DataFrame()
            #
            #                 fileChkList = glob.glob(saveFile)
            #                 # if (len(fileChkList) > 0): continue
            #
            #                 for k, pageInfo in enumerate(pageList):
            #
            #                     try:
            #                         # reqUrl = f'{apiUrl}?serviceKey={apiKey}&pageNo=1&numOfRows=100&LAWD_CD={sigunguCdInfo}&DEAL_YMD={dtYearMonth}'
            #                         #
            #                         # res = urllib.request.urlopen(reqUrl)
            #                         # resCode = res.getcode()
            #
            #                         inParams = {'serviceKey': sysOpt['apiKey'], 'pageNo': 1, 'numOfRows': 100, 'LAWD_CD': sigunguCdInfo, 'DEAL_YMD': dtYearMonth}
            #                         apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')
            #
            #                         res = requests.get(apiUrl, params=apiParams)
            #
            #                         resCode = res.status_code
            #                         if resCode != 200: continue
            #
            #                         # json 읽기
            #                         # resData = json.loads(res.read().decode('utf-8'))
            #                         # resData = json.loads(res.content.decode('utf-8'))
            #
            #                         # xml to json 읽기
            #                         # resData = xmltodict.parse(res.read().decode('utf-8'))
            #                         resData = xmltodict.parse(res.content.decode('utf-8'))
            #                         resultCode = resData['response']['header']['resultCode']
            #                         if (resultCode != '00'): continue
            #
            #                         resBody = resData['response']['body']
            #                         totalCnt = int(resBody['totalCount'])
            #                         if (totalCnt < 1): break
            #
            #                         itemList = resBody['items']['item']
            #                         if (len(itemList) < 1): break
            #
            #                         if (totalCnt == 1):
            #                             data = pd.DataFrame.from_dict([itemList])
            #                         else:
            #                             data = pd.DataFrame.from_dict(itemList)
            #
            #                         data['addrInfo'] = selAddrInfo
            #                         data['sigunguCdInfo'] = sigunguCdInfo
            #                         data['dtYearMonth'] = dtYearMonth
            #                         data['pageInfo'] = pageInfo
            #                         data['type'] = apiInfo
            #
            #                         dataL1 = pd.concat([dataL1, data], ignore_index=True)
            #
            #                         if (totalCnt < (pageInfo * 100)): break
            #
            #                     except Exception as e:
            #                         log.error("Exception : {}".format(e))
            #
            #                 # 자료 저장
            #                 if (len(dataL1) > 0):
            #                     data.to_csv(saveFile, index=False)
            #                     log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용한 아파트/오피스텔/단독다가구/연립다세대 전월세
            # *********************************************************************************
            # for i, apiInfo in enumerate(sysOpt['apiList2']):
            #     log.info(f'[CHECK] apiInfo : {apiInfo}')
            #
            #     apiUrl = sysOpt['apiList2'][apiInfo]
            #
            #     # addrInfo = sysOpt['addrList'][0]
            #     for ii, addrInfo in enumerate(sysOpt['addrList']):
            #         log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #         # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #         admDataL1 = admData[
            #             (admData['법정동명'].str.contains(addrInfo))
            #             & (admData['d1'].notna())
            #             & (admData['d2'].notna())
            #             & (admData['d3'].isna())
            #             & (admData['d4'].isna())
            #             & (admData['d5'].isna())
            #         ]
            #         if (len(admDataL1) < 1): continue
            #
            #         # 시군구 목록
            #         sigunguCdList = set(admDataL1['sigunguCd'])
            #         log.info(f'[CHECK] sigunguCdList : {sigunguCdList}')
            #
            #         # 페이지 설정
            #         pageList = np.arange(0, 9999, 1)
            #
            #         for i, sigunguCdInfo in enumerate(sigunguCdList):
            #
            #             selAddrInfo = admData[
            #                 (admData['sigunguCd'] == sigunguCdInfo)
            #                 & (admData['bjdongCd'] == '00000')
            #             ]['법정동명'].values[0]
            #
            #             log.info(f'[CHECK] sigunguCdInfo : {sigunguCdInfo}')
            #             log.info(f'[CHECK] selAddrInfo : {selAddrInfo}')
            #
            #             for j, dtMonthInfo in enumerate(dtMonthList):
            #                 # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')
            #
            #                 dtYearMonth = dtMonthInfo.strftime('%Y%m')
            #
            #                 saveFile = '{}/{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전월세', apiInfo, selAddrInfo, apiInfo, selAddrInfo, dtYearMonth)
            #                 os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #
            #                 dataL1 = pd.DataFrame()
            #
            #                 fileChkList = glob.glob(saveFile)
            #                 # if (len(fileChkList) > 0): continue
            #
            #                 for k, pageInfo in enumerate(pageList):
            #
            #                     try:
            #                         # reqUrl = f'{apiUrl}?serviceKey={apiKey}&pageNo=1&numOfRows=100&LAWD_CD={sigunguCdInfo}&DEAL_YMD={dtYearMonth}'
            #                         #
            #                         # res = urllib.request.urlopen(reqUrl)
            #                         # resCode = res.getcode()
            #
            #                         inParams = {'serviceKey': sysOpt['apiKey'], 'pageNo': 1, 'numOfRows': 100, 'LAWD_CD': sigunguCdInfo, 'DEAL_YMD': dtYearMonth}
            #                         apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')
            #
            #                         res = requests.get(apiUrl, params=apiParams)
            #
            #                         resCode = res.status_code
            #                         if resCode != 200: continue
            #
            #                         # json 읽기
            #                         # resData = json.loads(res.read().decode('utf-8'))
            #                         # resData = json.loads(res.content.decode('utf-8'))
            #
            #                         # xml to json 읽기
            #                         # resData = xmltodict.parse(res.read().decode('utf-8'))
            #                         resData = xmltodict.parse(res.content.decode('utf-8'))
            #                         resultCode = resData['response']['header']['resultCode']
            #                         if (resultCode != '00'): continue
            #
            #                         resBody = resData['response']['body']
            #                         totalCnt = int(resBody['totalCount'])
            #                         if (totalCnt < 1): break
            #
            #                         itemList = resBody['items']['item']
            #                         if (len(itemList) < 1): break
            #
            #                         if (totalCnt == 1):
            #                             data = pd.DataFrame.from_dict([itemList])
            #                         else:
            #                             data = pd.DataFrame.from_dict(itemList)
            #
            #                         data['addrInfo'] = selAddrInfo
            #                         data['sigunguCdInfo'] = sigunguCdInfo
            #                         data['dtYearMonth'] = dtYearMonth
            #                         data['pageInfo'] = pageInfo
            #                         data['type'] = apiInfo
            #
            #                         dataL1 = pd.concat([dataL1, data], ignore_index=True)
            #
            #                         if (totalCnt < (pageInfo * 100)): break
            #
            #                     except Exception as e:
            #                         log.error("Exception : {}".format(e))
            #
            #                 # 자료 저장
            #                 if (len(dataL1) > 0):
            #                     data.to_csv(saveFile, index=False)
            #                     log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 전처리] 실거래가, 전월세가
            # *********************************************************************************
            colList = ['종류', '이름', 'addrDtlInfo', 'addrInfo', '계약일', '면적', '거래가', '보증가', '층', '월세가', '분류', '건축년도', '년', '평형', 'lat', 'lon', '거래유형', '법정동', '등기일']
            for ii, addrInfo in enumerate(sysOpt['addrList']):
                log.info(f'[CHECK] addrInfo : {addrInfo}')

                posDataL1 = pd.DataFrame()
                for i, apiInfo in enumerate(sysOpt['apiList']):
                    log.info(f'[CHECK] apiInfo : {apiInfo}')

                    inpFile = '{}/{}/*/{}/*{}*/*{}*.csv'.format(globalVar['outPath'], serviceName, apiInfo, addrInfo, apiInfo)
                    # inpFile = '{}/{}/*/{}/*{}*/{}*.csv'.format(globalVar['outPath'], serviceName, apiInfo, addrInfo, apiInfo)
                    fileList = sorted(glob.glob(inpFile))
                    if fileList is None or len(fileList) < 1:
                        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                        # raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                    posData = pd.DataFrame()
                    fileInfo = fileList[0]
                    for fileInfo in fileList:
                        tmpData = pd.read_csv(fileInfo)
                        posData = pd.concat([posData, tmpData], ignore_index=True)

                    colInfo = ['보증금액', '월세금액']
                    if (apiInfo in '아파트'):
                        # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'] + ' ' + posData['지번'].astype(str)
                        posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'].astype(str) + ' ' + posData['지번'].astype(str)
                        posData['면적'] = posData['전용면적']
                        posData['이름'] = posData['아파트']
                    elif (apiInfo in '오피스텔'):
                        # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['단지'] + ' ' + posData['지번'].astype(str)
                        posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['단지'].astype(str) + ' ' + posData['지번'].astype(str)
                        posData['면적'] = posData['전용면적']
                        posData['이름'] = posData['단지']
                        colInfo = ['보증금', '월세']
                    elif (apiInfo in '단독다가구'):
                        posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동']
                        posData['면적'] = posData['대지면적']
                        posData['이름'] = None
                        posData['층'] = None
                    elif (apiInfo in '연립다세대'):
                        # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['연립다세대'] + ' ' + posData['지번'].astype(str)
                        posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['연립다세대'].astype(str) + ' ' + posData['지번'].astype(str)
                        posData['면적'] = posData['대지권면적']
                        posData['이름'] = posData['연립다세대']
                    else:
                        continue

                    posData['종류'] = apiInfo
                    posData['계약일'] = pd.to_datetime(posData['년'].astype(str) + '-' + posData['월'].astype(str) + '-' + posData['일'].astype(str), format='%Y-%m-%d')

                    binList = [(posData['면적'] >= 330), (posData['면적'] >= 264), (posData['면적'] >= 198), (posData['면적'] >= 114), (posData['면적'] >= 80), (posData['면적'] >= 60), (posData['면적'] >= 40), (posData['면적'] >= 20), (posData['면적'] >= 10)]
                    labelList = ["100평형", "80평형", "60평형", "43평형", "32평형", "24평형", "18평형", "9평형", "5평형"]
                    posData['평형'] = np.select(binList, labelList, default=None)

                    # 등기일
                    if not posData.get('등기일자') is None:
                        posData['등기일'] =  pd.to_datetime(posData['등기일자']).replace({pd.NaT: None})
                    else:
                        posData['등기일'] = None

                    # 매매가
                    if not posData.get('거래금액') is None:
                        posData['거래금액'] = pd.to_numeric(posData['거래금액'].astype(str).str.replace(',', ''), errors='coerce').astype(float)
                        posData['거래가'] = posData['거래금액'] * 10000
                        posData['val'] = round(posData['거래금액'] / 10000, 1)
                        binList = [(posData['val'] > 15), (posData['val'] >= 9), (posData['val'] >= 6), (posData['val'] > 3), (posData['val'] <= 3)]
                        labelList = ["15억 초과", "9-15억", "6-9억", "3-6억", "3억 이하"]
                        posData['분류'] = np.select(binList, labelList, default=None)
                    else:
                        posData['거래가'] = None
                        posData['val'] = None
                        posData['분류'] = None

                    # 보증가
                    if not posData.get(colInfo[0]) is None:
                        posData['보증금액'] = pd.to_numeric(posData[colInfo[0]].astype(str).str.replace(',', ''), errors='coerce').astype(float)
                        posData['보증가'] = posData['보증금액'] * 10000
                        posData['val2'] = round(posData['보증금액'] / 10000, 1)
                    else:
                        posData['보증가'] = None
                        posData['val2'] = None

                    # 월세가
                    if not posData.get(colInfo[1]) is None:
                        posData['월세금액'] = pd.to_numeric(posData[colInfo[1]].astype(str).str.replace(',', ''), errors='coerce').astype(float)
                        posData['월세가'] = posData['월세금액'] * 10000
                        posData['val3'] = round(posData['월세금액'] / 10000, 1)
                    else:
                        posData['월세가'] = None
                        posData['val3'] = None

                    posData['lat'] = None
                    posData['lon'] = None
                    posData['거래유형'] = posData.apply(getRentType, axis=1)

                    posDataL1 = pd.concat([posDataL1, posData], ignore_index=True)

                    # 중복 제거
                    posDataL2 = posData.drop_duplicates(subset=colList, inplace=False)


                    # # DB 저장
                    dbData = posDataL2[colList].rename(
                        {
                            '종류': 'TYPE'
                            , '이름': 'NAME'
                            , 'addrDtlInfo': 'ADDR'
                            , 'addrInfo': 'SI_DONG'
                            , '계약일': 'SALE_DATE'
                            , '면적': 'AREA'
                            , '거래가': 'SALE_PRICE'
                            , '보증가': 'SALE_PRICE2'
                            , '월세가': 'SALE_PRICE3'
                            , '분류': 'SALE_TYPE'
                            , '층': 'FLOOR'
                            , '건축년도': 'CONV_YEAR'
                            , '년': 'YEAR'
                            , '평형': 'PYEONG'
                            , 'lat': 'LAT'
                            , 'lon': 'LON'
                            , '거래유형': 'RENT_TYPE'
                            , '법정동': 'DONG'
                            , '등기일': 'RGSTR_DATE'
                        }
                        , axis=1
                    )

                    # DB 내 데이터 삭제
                    # TRUNCATE TABLE TB_SALE_INFO;
                    # SELECT SI_DONG, COUNT(SI_DONG) FROM TB_SALE_INFO GROUP BY SI_DONG;

                    try:
                        # dbData.to_sql(name=tbSaleInfo.name, con=dbEngine, if_exists='replace', index=False)
                        dbData.to_sql(name=tbSaleInfo.name, con=dbEngine, if_exists='append', index=False)
                        session.commit()
                    except SQLAlchemyError as e:
                        session.rollback()
                        log.error(f'Exception : {e}')

                # 중복 제거
                posDataL3 = posDataL1.drop_duplicates(subset=colList, inplace=False)

                # 알집 압축
                zipFile = '{}/{}/{}.zip'.format(globalVar['updPath'], addrInfo, addrInfo)
                zipInpFile = '{}/{}/*/*/*{}*/*{}*.csv'.format(globalVar['outPath'], serviceName, addrInfo, addrInfo)
                zipFileList = sorted(glob.glob(zipInpFile))

                os.makedirs(os.path.dirname(zipFile), exist_ok=True)
                with zipfile.ZipFile(zipFile, 'w', compresslevel=9) as zipf:
                    for zipFileInfo in zipFileList:
                        zipf.write(zipFileInfo, arcname=zipFileInfo.replace(globalVar['outPath'], '').replace(serviceName, ''))
                log.info(f'[CHECK] zipFile : {zipFile}')

                # 자료 저장
                saveFile = '{}/{}/{}.csv'.format(globalVar['updPath'], addrInfo, addrInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                posDataL3.to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')
                #
                # dbData = pd.DataFrame(
                #     {
                #         'TYPE': [addrInfo],
                #         'ZIP_INFO': [f"http://{getPubliIp()}:9000/CSV{zipFile.replace(globalVar['updPath'], '')}"],
                #         'CSV_INFO': [f"http://{getPubliIp()}:9000/CSV{saveFile.replace(globalVar['updPath'], '')}"]
                #     }
                # )
                #
                # try:
                #     dbData.to_sql(name=tbSaleDown.name, con=dbEngine, if_exists='append', index=False)
                #     session.commit()
                # except SQLAlchemyError as e:
                #     session.rollback()
                #     log.error(f'Exception : {e}')

            # *********************************************************************************
            # 주소를 통해 위경도 환산
            # *********************************************************************************
            # addDtlrList = set(posDataL1['addrDtlInfo'].dropna())
            # posDataL2 = pd.DataFrame()

            # 구글 지오코딩
            # for i, addrDtlInfo in enumerate(addDtlrList):
            #     log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
            #
            #     posDataL2.loc[i, 'addrDtlInfo'] = addrDtlInfo
            #     posDataL2.loc[i, 'lat'] = None
            #     posDataL2.loc[i, 'lon'] = None
            #
            #     try:
            #         rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
            #         if (len(rtnGeo) < 1): continue
            #
            #         # 위/경도 반환
            #         posDataL2.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
            #         posDataL2.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
            #
            #     except Exception as e:
            #         log.error("Exception : {}".format(e))

            # 네이버 지오코딩
            # for i, addrInfo in enumerate(addrList):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     addUrlenc = parse.quote(addrInfo)
            #     url = f'{sysOpt["naverApi"]}?query={addUrlenc}'
            #     request = Request(url)
            #     request.add_header('X-NCP-APIGW-API-KEY-ID', sysOpt['naverId'])
            #     request.add_header('X-NCP-APIGW-API-KEY', sysOpt['naverPw'])
            #
            #     posDataL2.loc[i, '주소'] = addrInfo
            #     posDataL2.loc[i, 'lat'] = None
            #     posDataL2.loc[i, 'lon'] = None
            #
            #     try:
            #         response = urlopen(request)
            #         resCode = response.getcode()
            #
            #         if (resCode != 200): continue
            #
            #         responseBody = response.read().decode('utf-8')
            #         responseBody = json.loads(responseBody)
            #
            #         resCnt = responseBody['meta']['count']
            #         if (resCnt < 1): continue
            #
            #         log.info("[CHECK] [{}] {}".format(i, addrInfo))
            #
            #         posDataL2.loc[i, 'lat'] = responseBody['addresses'][0]['y']
            #         posDataL2.loc[i, 'lon'] = responseBody['addresses'][0]['x']
            #
            #     except HTTPError as e:
            #         log.error("Exception : {}".format(e))
            #
            # posDataL3 = pd.merge(left=posDataL1, right=posDataL2, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
            # posDataL3 = posDataL1
            #
            # colList = ['종류', '이름', 'addrDtlInfo', '계약일', '면적', '거래가', '보증가', '월세가', '분류', '층', '건축년도', '년', '평형', 'lat', 'lon', '거래유형', '법정동']
            #
            # # 중복 제거
            # posDataL3  = posDataL1.drop_duplicates(subset=colList, inplace=False)
            #
            # # DB 저장
            # dbData = posDataL3[colList].rename(
            #     {
            #         '종류': 'TYPE'
            #         , '이름': 'NAME'
            #         , 'addrDtlInfo': 'ADDR'
            #         , '계약일': 'SALE_DATE'
            #         , '면적': 'AREA'
            #         , '거래가': 'SALE_PRICE'
            #         , '보증가': 'SALE_PRICE2'
            #         , '월세가': 'SALE_PRICE3'
            #         , '분류': 'SALE_TYPE'
            #         , '층': 'FLOOR'
            #         , '건축년도': 'CONV_YEAR'
            #         , '년': 'YEAR'
            #         , '평형': 'PYEONG'
            #         , 'lat': 'LAT'
            #         , 'lon': 'LON'
            #         , '거래유형': 'RENT_TYPE'
            #         , '법정동': 'DONG'
            #     }
            #     , axis=1
            # )
            #
            # try:
            #     # dbData.to_sql(name=tbSaleInfoDtl.name, con=dbEngine, if_exists='replace', index=False)
            #     dbData.to_sql(name=tbSaleInfoDtl.name, con=dbEngine, if_exists='append', index=False)
            #     session.commit()
            # except SQLAlchemyError as e:
            #     session.rollback()
            #     log.error(f'Exception : {e}')

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
