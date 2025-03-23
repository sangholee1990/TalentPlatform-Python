# ================================================
# 요구사항
# ================================================
# Python, Looker Studio을 이용한 19년간 공개 생산문서 대시보드

# rm -f 20250320_ydg2007-2025.csv
# cat *.csv > 20250320_ydg2007-2025.csv

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
from dbfread import DBF, FieldParser
import csv
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re

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

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
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
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0610'

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

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                # 'srtDate': '2019-01-01'
                # , 'endDate': '2023-01-01'

                # 빅쿼리 설정 정보
                'jsonFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/iconic-ruler-239806-7f6de5759012.json',
            }

            # =================================================================
            # csv 파일 변환
            # =================================================================
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20250316_ydgDBF/ydg*.dbf')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20250316_ydgDBF/ydg2013.dbf')
            fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[0]
            # dataL3 = pd.DataFrame()
            # for fileInfo in fileList:
            for i, fileInfo in enumerate(fileList):
                log.info(f"[CHECK] fileInfo : {fileInfo}")

                data = DBF(fileInfo, encoding='euc-kr', char_decode_errors='ignore', ignore_missing_memofile=True)
                dataL1 = pd.DataFrame(data)
                # dataL2 =  dataL1.drop(['_NullFlags'], axis=1, errors='ignore')
                dataL2 =  dataL1.drop(['_NullFlags', 'MEMO'], axis=1, errors='ignore')
                # dataL3 = pd.concat([dataL3, dataL2], ignore_index=True)

                fileName = os.path.basename(fileInfo)
                fileNameNotExt = fileName.split(".")[0]
                isHeader = True if i == 0 else False

                dataL2['YEAR_DATE'] = pd.to_datetime(dataL2['RC_DATE'], format='%Y-%m-%d').dt.strftime('%Y')
                dataL2['DEPART'] = dataL2['DEPART'].str.replace(r'[\n\r\x0f\x06\x14\x0e\|]', '', regex=True).str.strip()

                # dataL2['DEPART'].unique()

                saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, fileNameNotExt)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL2.to_csv(saveFile, index=False, header=isHeader)
                log.info(f"[CHECK] saveFile : {saveFile}")

            # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, datetime.now().strftime("%Y%m%d"), 'ydg_2007_2025')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL3.to_csv(saveFile, index=False)
            # log.info(f"[CHECK] saveFile : {saveFile}")

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

            inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, '20250320_ydg2007-2025.csv')
            fileList = sorted(glob.glob(inpFile))
            if fileList is None or len(fileList) < 1:
                log.error(f'inpFile : {inpFile} / 파일 검색 실패')
                exit(1)

            fileInfo = fileList[0]
            # data = pd.read_csv(fileInfo)

            # data['DEPART'].unique()
            # data['YEAR_DATE'].unique()

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
                max_bad_records=1000, # 최대 오류 허용
            )

            tableId = f"{credentials.project_id}.DMS01.TB_YDG"
            with open(fileInfo, "rb") as file:
                job = client.load_table_from_file(file, tableId, job_config=jobCfg)
            job.result()
            log.info(f"[CHECK] tableId : {tableId}")

            # dataL1 = data.astype(str)
            # dataL1.dtypes

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

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