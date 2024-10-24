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

        tbRprDown = Table('TB_RPR_DOWN', metaData, autoload_with=dbEngine, schema=dbName)
        tbRprInfo = Table('TB_RPR_INFO', metaData, autoload_with=dbEngine, schema=dbName)

        result = {
            'dbEngine': dbEngine
            , 'session': session
            , 'tbRprDown': tbRprDown
            , 'tbRprInfo': tbRprInfo
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
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 공공데이터포털 매매 신고 API (토지, 상업) 수집 및 구글 스튜디오 시각화

    # 공통 컬럼
    # addrInfo	sigunguCdInfo	dtYearMonth	pageInfo	type
    # 거래금액
    # 거래면적
    # 해제여부
    # 거래유형
    # 용도지역
    # 년월일 -> 신고일
    # 년 -> 연도
    # 해제사유발생일
    # 중개사소재지
    # 시군구	법정동		지번 -> 주소, 시도

    # 타 컬럼
    # 유형
    # 지목
    # 건물주용도
    # 건축년도
    # 대지면적
    # 층
    # 구분
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
    serviceName = 'LSH0465'

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
                'srtDate': '2005-01-01'
                , 'endDate': datetime.now().strftime("%Y-%m-%d")

                # 공공데이터포털 API
                , 'apiKey': 'bf9fH0KLgr65zXKT5D/dcgUBIj1znJKnUPrzDVZEe6g4gquylOjmt65R5cjivLPfOKXWcRcAWU0SN7KKXBGDKA=='
                #
                # 매매 신고
                , 'apiList': {
                    '토지': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcLandTrade'
                    , '상업': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcNrgTrade'
                }

                # 검색 목록
                # , 'addrList': ['세종특별자치시']
                # , 'addrList': ['제주특별자치도']
                # , 'addrList': ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원특별자치도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도']
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
            tbRprDown = cfgInfo['tbRprDown']
            tbRprInfo = cfgInfo['tbRprInfo']

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
            # [자료 수집] 오픈API를 이용한 토지/상업 매매 신고
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
            #                 if (len(fileChkList) > 0): continue
            #
            #                 for k, pageInfo in enumerate(pageList):
            #                     # log.info(f'[CHECK] pageInfo : {pageInfo}')
            #
            #                     try:
            #                         inParams = {'serviceKey': sysOpt['apiKey'], 'pageNo': 1, 'numOfRows': 100, 'LAWD_CD': sigunguCdInfo, 'DEAL_YMD': dtYearMonth}
            #                         apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')
            #
            #                         res = requests.get(apiUrl, params=apiParams)
            #
            #                         resCode = res.status_code
            #                         if resCode != 200: break
            #
            #                         # json 읽기
            #                         # resData = json.loads(res.read().decode('UTF-8'))
            #                         # resData = json.loads(res.content.decode('UTF-8'))
            #
            #                         # xml to json 읽기
            #                         # resData = xmltodict.parse(res.read().decode('UTF-8'))
            #                         resData = xmltodict.parse(res.content.decode('UTF-8'))
            #                         resultCode = resData['response']['header']['resultCode']
            #                         if (resultCode != '00'): break
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

            # # *********************************************************************************
            # # [자료 전처리] 토지/상업 매매 신고
            # # *********************************************************************************
            colList = ['시도', '시군구', '법정동', '주소', '분류', '거래가', '거래가 분류', '면적', '면적 분류', '해제여부', '거래유형', '용도지역', '연도', '해제사유발생일', '중개사소재지', '등록일', '신고일']
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

                    posData['주소'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['지번'].astype(str)
                    # 세종특별자치시 시군구 없음
                    posData['시군구'] = None if posData.get('시군구') is None else posData['시군구']
                    posData['시도'] = posData['addrInfo'].apply(lambda x: x.split(' ')[0])
                    posData['신고일'] = pd.to_datetime(posData['년'].astype(str) + '-' + posData['월'].astype(str) + '-' + posData['일'].astype(str), format='%Y-%m-%d')
                    posData['연도'] = posData['년']
                    posData['분류'] = posData['type']
                    posData['등록일'] =  pd.to_datetime(posData['dtYearMonth'].astype(str), format='%Y%m')

                    # 거래금액
                    posData['거래금액'] = pd.to_numeric(posData['거래금액'].astype(str).str.replace(',', ''), errors='coerce').astype(float)
                    posData['거래가'] = posData['거래금액'] * 10000
                    convVal = round(posData['거래금액'] / 10000, 1)
                    binList = [(convVal > 15), (convVal >= 9), (convVal >= 6), (convVal > 3), (convVal <= 3)]
                    labelList = ["15억 초과", "9-15억", "6-9억", "3-6억", "3억 이하"]
                    posData['거래가 분류'] = np.select(binList, labelList, default=None)

                    # 면적
                    if (apiInfo in '토지'):
                        posData['면적'] = posData['거래면적']
                    elif (apiInfo in '상업'):
                        posData['면적'] = posData['대지면적']
                    else:
                        posData['면적'] = None

                    convArea = posData['면적']
                    binList = [(convArea >= 33000000), (convArea >= 3300000), (convArea >= 330000), (convArea >= 33000), (convArea >= 3300), (convArea >= 330), (convArea >= 264), (convArea >= 198), (convArea >= 114), (convArea >= 80), (convArea >= 60), (convArea >= 40), (convArea >= 20), (convArea >= 0)]
                    labelList = ["10,000,000평형", "1,000,000평형", "100,000평형", "10,000평형", "1,000평형", "100평형", "80평형", "60평형", "43평형", "32평형", "24평형", "18평형", "9평형", "5평형"]
                    posData['면적 분류'] = np.select(binList, labelList, default=None)

                    # 중복 제거
                    posDataL2 = posData.drop_duplicates(subset=colList, inplace=False)

                    # posDataL2[colList].columns

                    # # DB 저장
                    dbData = posDataL2[colList].rename(
                        {
                            '시도': 'SIDO'
                            , '시군구': 'SIGUNGU'
                            , '법정동': 'DONG'
                            , '주소': 'ADDR'
                            , '분류': 'TYPE'
                            , '거래가': 'SALE_PRICE'
                            , '거래가 분류': 'SALE_PRICE_TYPE'
                            , '면적': 'AREA'
                            , '면적 분류': 'AREA_TYPE'
                            , '해제여부': 'IS_CANC'
                            , '거래유형': 'SALE_TYPE'
                            , '용도지역': 'USAGE_AREA'
                            , '연도': 'YEAR'
                            , '중개사소재지': 'AGE_LOC'
                            , '신고일': 'DEC_DATE'
                            , '해제사유발생일': 'CAN_DATE'
                            , '등록일': 'REG_DATE'
                        }
                        , axis=1
                    )

                    posDataL1 = pd.concat([posDataL1, dbData], ignore_index=True)

                    # DB 내 데이터 삭제
                    # TRUNCATE TABLE TB_RPR_INFO;
                    # TRUNCATE TABLE TB_RPR_DOWN;
                    # SELECT SI_DONG, COUNT(SI_DONG) FROM TB_SALE_INFO GROUP BY SI_DONG;

                    try:
                        dbData.to_sql(name=tbRprInfo.name, con=dbEngine, if_exists='append', index=False)
                        session.commit()
                    except SQLAlchemyError as e:
                        session.rollback()
                        log.error(f'Exception : {e}')

                # 중복 제거
                posDataL3 = posDataL1

                # 알집 압축
                zipFile = '{}/{}/{}/{}.zip'.format(globalVar['updPath'], serviceName, addrInfo, addrInfo)
                zipInpFile = '{}/{}/*/*/*{}*/*{}*.csv'.format(globalVar['outPath'], serviceName, addrInfo, addrInfo)
                zipFileList = sorted(glob.glob(zipInpFile))

                os.makedirs(os.path.dirname(zipFile), exist_ok=True)
                with zipfile.ZipFile(zipFile, 'w', compresslevel=9) as zipf:
                    for zipFileInfo in zipFileList:
                        zipf.write(zipFileInfo, arcname=zipFileInfo.replace(globalVar['outPath'], '').replace(serviceName, ''))
                log.info(f'[CHECK] zipFile : {zipFile}')

                # 자료 저장
                saveFile = '{}/{}/{}/{}.csv'.format(globalVar['updPath'], serviceName, addrInfo, addrInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                posDataL3.to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                dbData = pd.DataFrame(
                    {
                        'TYPE': [addrInfo]
                        , 'ZIP_INFO': [f"http://{getPubliIp()}:9000/CSV{zipFile.replace(globalVar['updPath'], '')}"]
                        , 'CSV_INFO': [f"http://{getPubliIp()}:9000/CSV{saveFile.replace(globalVar['updPath'], '')}"]
                        , 'REG_DATE': [datetime.now()]
                    }
                )

                try:
                    dbData.to_sql(name=tbRprDown.name, con=dbEngine, if_exists='append', index=False)
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
