# ================================================
# 요구사항
# ================================================
# Python을 이용한 부동산 데이터 분석 및 가격 예측 고도화 및 구글 스튜디오 시각화

# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-MergeRealData.py

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
import json
import requests
import time
from concurrent.futures import ProcessPoolExecutor

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

def getAmountType(amount):
    if amount > 15:
        return "15억 초과"
    elif amount >= 9:
        return "9-15억"
    elif amount >= 6:
        return "6-9억"
    elif amount > 3:
        return "3-6억"
    else:
        return "3억 이하"

def fetchApi(apiUrl, payload, apiType, recAptData):

    result = pd.DataFrame(columns=['idx', 'score'])

    try:
        response = requests.post(apiUrl, data=payload, verify=False, timeout=30)
        response.raise_for_status()
        resJson = response.json().get('recommends')
        resData = pd.DataFrame(resJson[apiType], columns=['idx', 'score'])
        result = pd.merge(resData, recAptData, how='left', left_on=['idx'], right_on=['idx'])
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

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
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
                },
                '추천': {
                    'apiCfUrl': 'http://125.251.52.42:9010/recommends_cf',
                    'apiSimUrl': 'http://125.251.52.42:9010/recommends_simil',
                    'propAptFile': '/DATA/OUTPUT/LSH0613/추천/20250526_tbl_apts.xlsx',
                    'propUserFile': '/DATA/OUTPUT/LSH0613/추천/20250526_tbl_users.xlsx',
                },
            }

            # *********************************************************************************
            # 파일 읽기
            # *********************************************************************************
            # 사용자 설정 정보
            inpFile = sysOpt['추천']['propUserFile']
            fileList = sorted(glob.glob(inpFile), reverse=True)
            if fileList is None or len(fileList) < 1:
                log.error(f'파일 없음 : {inpFile}')
                sys.exit(1)

            recUserData = pd.read_excel(fileList[0])

            # 아파트 설정 정보
            inpFile = sysOpt['추천']['propAptFile']
            fileList = sorted(glob.glob(inpFile), reverse=True)
            if fileList is None or len(fileList) < 1:
                log.error(f'파일 없음 : {inpFile}')
                sys.exit(1)

            recAptData = pd.read_excel(fileList[0])



            # 검색어
            gender = '1'
            minAge, maxAge = '20-39'.split('-')
            minPrice, maxPrice = '3-6'.split('-')
            minArea, maxArea = '58-100'.split('-')
            debtRat = '0.25'

            # prefer = '편의시설'
            apt = '두산(가산로 99)'
            rcmdCnt = 10

            recUserDataL1 = recUserData.loc[
                (recUserData['gender'] == int(gender))
                & (recUserData['age'] >= float(minAge)) & (recUserData['age'] <= float(maxAge))
                & (recUserData['price_from'] >= float(minPrice)) & (recUserData['price_to'] <= float(maxPrice))
                & (recUserData['area_from'] >= float(minArea)) & (recUserData['area_to'] <= float(maxArea))
                & (recUserData['debt_ratio'] >= float(debtRat))
                # & (recUserData['prefer'] == prefer)
            ]

            # userIdx = recUserDataL1.iloc[0]['idx']

            # len(recUserDataL1) < 1


            recAptDataL1 = recAptData.loc[
                (recAptData['apt'] == apt)
                & (recAptData['area'] >= float(minArea)) & (recAptData['area'] <= float(maxArea))
                & (recAptData['price'] >= float(minPrice)) & (recAptData['price'] <= float(maxPrice))
            ]
            # len(recAptDataL1) < 1

            # CF기반 아파트 추천
            # 사용자 id, 아파트 idx, 추천 개수
            payload = {
                'user_id': recUserDataL1.iloc[0]['idx'],
                'apt_idx': recAptDataL1.iloc[0]['idx'],
                'rcmd_count': 10,
            }

            #
            # response = requests.post(sysOpt['추천']['apiCfUrl'], data=payload, verify=False)
            #
            # res_json = response.json().get('recommends')
            # log.info(f'[CHECK] res_json : {res_json}')
            #
            # resData = pd.DataFrame(res_json['cf'], columns=['idx', 'score'])
            # resDataL1 = pd.merge(resData, recAptData, how='left', left_on=['idx'], right_on=['idx'])
            #
            # # 유사도 기반 아파트 추천
            # payload = {
            #     'user_id': recUserDataL1.iloc[0]['idx'],
            #     'apt_idx': recAptDataL1.iloc[0]['idx'],
            #     'rcmd_count': rcmdCnt,
            # }
            #
            # response = requests.post(sysOpt['추천']['apiSimUrl'], data=payload, verify=False)
            # res_json = response.json().get('recommends')
            # log.info(f'[CHECK] res_json : {res_json}')
            #
            # resData = pd.DataFrame(res_json['simil'], columns=['idx', 'score'])
            # resDataL1 = pd.merge(resData, recAptData, how='left', left_on=['idx'], right_on=['idx'])

            # CF기반 및 유사도 기반 아파트 추천
            with ProcessPoolExecutor(max_workers=2) as executor:
                futureCf = executor.submit(fetchApi,sysOpt['추천']['apiCfUrl'], payload, 'cf', recAptData)
                futureSim = executor.submit(fetchApi,sysOpt['추천']['apiSimUrl'], payload, 'simil', recAptData)

                cfData = futureCf.result()
                simData = futureSim.result()


        #
            # def fetch_recommendations(payload):
            #     try:
            #         # verify=False는 SSL 인증서 검증 비활성화
            #         response = requests.post(sysOpt['apiCfUrl'], data=payload, verify=False)
            #         response.raise_for_status()
            #         res_json = response.json().get('recommends')
            #         if res_json and 'cf' in res_json:
            #             return res_json['cf']
            #         else:
            #             print(f"경고: 페이로드 {payload}에 대한 응답에서 'recommends' 또는 'cf' 키를 찾을 수 없습니다.")
            #             return None
            #     except requests.exceptions.RequestException as e:
            #         print(f"페이로드 {payload}에 대한 요청 실패: {e}")
            #         return None
            #     except ValueError as e:  # JSONDecodeError 포함
            #         print(f"페이로드 {payload}에 대한 JSON 디코딩 실패: {e}")
            #         print(f"응답 내용: {response.text}")
            #         return None




            # recAptDataL1


            #     if price < user_info.get('price_from').values[0] or price > user_info.get('price_to').values[0] * 1.2:
            #                             continue
            #                         if area < user_info.get('area_from').values[0] or area > user_info.get('area_to').values[0]:
            #                             continue


            # dataL2 = pd.DataFrame()
            # for fileInfo in fileList:
            #     log.info(f'[CHECK] fileInfo : {fileInfo}')
            #
            #     fileName = os.path.basename(fileInfo)
            #     fileNameNoExt = fileName.split('.')[0]
            #     # ssg = fileNameNoExt.split('_')[1]

                # data = pd.read_excel(fileInfo, sheet_name=sysOpt['sheetName'], engine='openpyxl')
            
                # try:
                #     data = pd.read_csv(fileInfo, encoding='EUC-KR', low_memory=False)
                # except UnicodeDecodeError:
                #     data = pd.read_csv(fileInfo, encoding='UTF-8', low_memory=False)
                # data = data.rename(columns=sysOpt['아파트실거래']['renameDict'])
                #
                # data['거래 금액'] = pd.to_numeric(data['거래금액'].str.replace(',', ''), errors='coerce') * 10000
                # data['거래 금액(억원)'] = data['거래 금액'] / 100000000
                # data['계약날짜'] = pd.to_datetime(data['dtYearMonth'].astype(str) + data['일'].apply(lambda x: f'{x:02d}').astype(str), format='%Y%m%d')
                # data['건축날짜'] = pd.to_datetime(data['건축년도'].astype(str), format='%Y')
                # data["평수"] = pd.to_numeric(data["전용면적"], errors='coerce').apply(getFloorArea)
                # data["거래가 분류"] = data["거래 금액(억원)"].apply(getAmountType)
                # # data['sgg'] = data['addrInfo']
                # data['sgg'] = data['addrInfo'] + ' ' + data['d2'].astype(str)
                # data['계약연도'] = data['계약날짜'].dt.strftime('%Y').astype(int)
                # data['건축연도'] = data['건축날짜'].dt.strftime('%Y').astype(int)
                # data['층'] = pd.to_numeric(data["층"], errors='coerce').astype('Int64')
                #
                # if 'lat' in data.columns and 'lon' in data.columns:
                #     data['geo'] = data["lat"].astype('str') + ", " + data["lon"].astype('str')
                # else:
                #     data['geo'] = data["latitude"].astype('str') + ", " + data["longitude"].astype('str')

                # 지번주소
                # 서울특별시 중랑구 상봉동 484 엘지,쌍용아파트
                # data['keyDtl'] = data['addrInfo'] + ' ' + data['d2'].astype(str) + ' ' + data['법정동'] + ' ' + data['아파트'] + '(' + data['지번'] + ')'
                # data['key'] = data['아파트'] + '(' + data['지번'] + ')'
                # data['keyDtl'] = data['addrDtlInfo']
                #
                # # 도로명주소
                # # 서울특별시 중랑구 봉화산로 130.0 엘지,쌍용아파트
                # data['apt'] = data['아파트'] + '(' + data['도로명']  + ' ' + data['도로명건물본번호코드'].astype('Int64').astype(str) + ')'
                # data['aptDtl'] = data['addrInfo'].astype(str) + ' ' + data['d2'].astype(str) + ' ' + data['도로명'].astype(str) + ' ' + data['도로명건물본번호코드'].astype('Int64').astype(str) + ' ' + data['아파트'].astype(str)
                #
                # dataL1 = data[colNameList].rename(columns=renameDict, inplace=False)
                # # dataL1 = data.rename(columns=renameDict, inplace=False)
                #
                # dataL2 = pd.concat([dataL2, dataL1], axis=0)

            # dataL2.loc[dataL2['apt'] == '라움(태평로)'].iloc[0]
            # data.iloc[0]

            # =================================================================
            # CSV 통합파일
            # =================================================================
            # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, datetime.now().strftime("%Y%m%d"), 'TB_REAL')
            # saveFile = sysOpt['아파트실거래']['saveFile']
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL2.to_csv(saveFile, index=False)
            # log.info(f'[CHECK] saveFile : {saveFile}')

            # =================================================================
            # 빅쿼리 업로드
            # =================================================================
            # jsonFile = sysOpt['jsonFile']
            # jsonList = sorted(glob.glob(jsonFile))
            # if jsonList is None or len(jsonList) < 1:
            #     log.error(f'설정 파일 없음 : {jsonFile}')
            #     raise Exception(f'설정 파일 없음 : {jsonFile}')
            #     # exit(1)
            #
            # jsonInfo = jsonList[0]
            #
            # try:
            #     credentials = service_account.Credentials.from_service_account_file(jsonInfo)
            #     client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            # except Exception as e:
            #     log.error(f'빅쿼리 연결 실패 : {e}')
            #     raise Exception(f'빅쿼리 연결 실패 : {e}')
            #     # exit(1)
            #
            # jobCfg = bigquery.LoadJobConfig(
            #     source_format=bigquery.SourceFormat.CSV,
            #     skip_leading_rows=1,
            #     autodetect=True,
            #     # autodetect=False,
            #     # schema=[  # BigQuery 테이블 스키마 정의 (열 이름, 데이터 타입)
            #     #     bigquery.SchemaField("Y_NO", "INTEGER"),
            #     #     bigquery.SchemaField("DEPART", "STRING"),
            #     #     bigquery.SchemaField("DEPART_NO", "STRING"),
            #     #     bigquery.SchemaField("SECTION", "STRING"),
            #     #     bigquery.SchemaField("SUBJECT", "STRING"),
            #     #     bigquery.SchemaField("NAME", "STRING"),
            #     #     bigquery.SchemaField("YEAR", "STRING"),
            #     #     bigquery.SchemaField("YEAR_DATE", "STRING"),
            #     #     bigquery.SchemaField("PUBLIC", "STRING"),
            #     #     bigquery.SchemaField("RC_DATE", "DATE"),
            #     #     bigquery.SchemaField("REG_DATE", "DATE"),
            #     #     bigquery.SchemaField("SIZE", "INTEGER"),
            #     # ],
            #     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            #     max_bad_records=1000,
            # )
            #
            # tableId = f"{credentials.project_id}.DMS01.TB_REAL"
            # with open(saveFile, "rb") as file:
            #     job = client.load_table_from_file(file, tableId, job_config=jobCfg)
            # job.result()
            # log.info(f"[CHECK] tableId : {tableId}")

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