# ================================================
# 요구사항
# ================================================
# Python을 이용한 부동산 데이터 분석 및 가격 예측 고도화 및 구글 스튜디오 시각화

# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-DataCollectToPrep.py

# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
from datetime import datetime

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

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
# warnings.filterwarnings('ignore')

# plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font='Malgun Gothic', rc={'axes.unicode_minus': False}, style='darkgrid')

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

    if len(log.handlers) > 0: return log

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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

def getFloorArea(size):
  """
  주어진 크기(m²)에 따라 평수를 반환합니다.

  Args:
    size: 주거 면적 (m²)

  Returns:
    평수 (문자열)
  """
  if size >= 114:
    return "43평"
  elif size >= 80:
    return "32평"
  elif size >= 60:
    return "24평"
  elif size >= 40:
    return "18평"
  elif size >= 20:
    return "9평"
  elif size >= 10:
    return "5평"
  else:
    return "5평 미만"


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

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0454'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] __init__ : {}'.format('init'))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error('Exception : {}'.format(e))
            raise e
        finally:
            log.info('[END] __init__ : {}'.format('init'))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):
        log.info('[START] {}'.format('exec'))

        try:
            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            sysOpt = {
                # 빅쿼리 설정 정보
                'jsonFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/iconic-ruler-239806-7f6de5759012.json',

                '예측': {
                    # 'propFile': '/DATA/OUTPUT/LSH0613/예측/수익률_{addrInfo}_{d2}.csv',
                    'propFile': '/DATA/OUTPUT/LSH0613/예측/수익률_*_*.csv',
                    'saveFile': '/DATA/OUTPUT/LSH0613/통합/수익률.csv',
                },
                '아파트실거래': {
                    # 'propFile': '/DATA/OUTPUT/LSH0613/전처리/아파트실거래_{addrInfo}_{d2}.csv',
                    'propFile': '/DATA/OUTPUT/LSH0613/전처리/아파트실거래_*_*.csv',
                    'saveFile': '/DATA/OUTPUT/LSH0613/통합/아파트실거래.csv',
                }
            }

            # *********************************************************************************
            # 코드 정보 읽기
            # *********************************************************************************
            colNameList = ['아파트(도로명)', '면적', '건축연도', '연도', '날짜', '위도', '경도', '인허가', '건축년도', '매매가', '전세가',
                           '예측 딥러닝 매매가', '예측 딥러닝 전세가', '예측 머신러닝 매매가', '예측 머신러닝 전세가', '실측 갭투자',
                           '예측 머신러닝 갭투자', '예측 딥러닝 갭투자', '실측 수익금', '예측 딥러닝 수익금', '예측 머신러닝 수익금',
                           '실측 수익률', '예측 딥러닝 수익률', '예측 머신러닝 수익률']

            colCodeList = ['name', 'capacity', 'construction_year', 'year', 'date', 'lat', 'lon', 'inhuga', 'conYear', 'realPrice', 'realBjprice',
                           'realPriceDL', 'realBjPriceDL', 'realPriceML', 'realBjPriceML', 'gapReal', 'gapML', 'gapDL',
                           'gapDiffReal', 'gapDiffDL', 'gapDiffML', 'gapPctReal', 'gapPctDL', 'gapPctML']

            renameDict = {colName: colCode for colName, colCode in zip(colNameList, colCodeList)}

            # *********************************************************************************
            # 파일 읽기
            # *********************************************************************************
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '예측/수익률 테이블_*.xlsx')
            inpFile = sysOpt['예측']['propFile']
            fileList = sorted(glob.glob(inpFile), reverse=True)
            if fileList is None or len(fileList) < 1:
                log.error(f'파일 없음 : {inpFile}')
                sys.exit(1)

            dataL2 = pd.DataFrame()
            for fileInfo in fileList:
                log.info(f'[CHECK] fileInfo : {fileInfo}')

                # data = pd.read_excel(fileInfo, engine='openpyxl')
                data = pd.read_csv(fileInfo)

                data["area"] = data["면적"].apply(getFloorArea)
                data['geo'] = data["위도"].astype('str') + ", " + data["경도"].astype('str')

                splitData = data['아파트(도로명)'].str.split(" ")
                data['sgg'] = splitData.str[0] + " " + splitData.str[1]
                data['town'] = splitData.str[2]
                data['apt'] = splitData.str[3]

                dataL1 = data.rename(columns=renameDict, inplace=False)
                dataL2 = pd.concat([dataL2, dataL1], axis=0)

            # =================================================================
            # CSV 통합파일
            # =================================================================
            # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, datetime.now().strftime("%Y%m%d"), 'TB_PRD')
            saveFile = sysOpt['예측']['saveFile']
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL2.to_csv(saveFile, index=False)
            log.info(f'[CHECK] saveFile : {saveFile}')

            # =================================================================
            # 빅쿼리 업로드
            # =================================================================
            jsonFile = sysOpt['jsonFile']
            jsonList = sorted(glob.glob(jsonFile))
            if jsonList is None or len(jsonList) < 1:
                log.error(f'jsonFile : {jsonFile} / 설정 파일 검색 실패')
                exit(1)

            jsonInfo = jsonList[0]

            try:
                credentials = service_account.Credentials.from_service_account_file(jsonInfo)
                client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            except Exception as e:
                log.error(f'Exception : {e} / 빅쿼리 연결 실패')
                exit(1)

            jobCfg = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                autodetect=True,
                # autodetect=False,
                # schema=[  # BigQuery 테이블 스키마 정의 (열 이름, 데이터 타입)
                #     bigquery.SchemaField("Y_NO", "INTEGER"),
                #     bigquery.SchemaField("DEPART", "STRING"),
                #     bigquery.SchemaField("DEPART_NO", "STRING"),
                #     bigquery.SchemaField("SECTION", "STRING"),
                #     bigquery.SchemaField("SUBJECT", "STRING"),
                #     bigquery.SchemaField("NAME", "STRING"),
                #     bigquery.SchemaField("YEAR", "STRING"),
                #     bigquery.SchemaField("YEAR_DATE", "STRING"),
                #     bigquery.SchemaField("PUBLIC", "STRING"),
                #     bigquery.SchemaField("RC_DATE", "DATE"),
                #     bigquery.SchemaField("REG_DATE", "DATE"),
                #     bigquery.SchemaField("SIZE", "INTEGER"),
                # ],
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                max_bad_records=1000,
            )

            tableId = f"{credentials.project_id}.DMS01.TB_PRD"
            # with open(fileInfo, "rb") as file:
            with open(saveFile, "rb") as file:
                job = client.load_table_from_file(file, tableId, job_config=jobCfg)
            job.result()
            log.info(f"[CHECK] tableId : {tableId}")

        except Exception as e:
            log.error(f'Exception : {e}')
            raise e
        finally:
            log.info('[END] {}'.format('exec'))

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