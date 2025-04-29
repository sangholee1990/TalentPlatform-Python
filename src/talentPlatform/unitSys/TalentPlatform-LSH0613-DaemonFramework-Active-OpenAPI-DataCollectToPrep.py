# ================================================
# 요구사항
# ================================================
#  Python을 이용한 부동산 데이터 분석 및 가격 예측 고도화 개선

# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-DataCollectToPrep.py

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-DataCollectToPrep.py &

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

import googlemaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict
from sklearn.neighbors import BallTree
import requests

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
        # , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
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

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    # serviceName = 'LSH0454'
    serviceName = 'LSH0613'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2014-01-01'
                , 'endDate': datetime.now().strftime("%Y-%m-%d")

                # 공공데이터포털 API 디코딩
                , 'apiKey': ''

                # 구글 API 정보
                , 'googleApiKey': ''

                , '건축인허가': {
                    # 국토교통부_건축HUB_주택인허가정보 서비스 > 건축HUB 주택인허가 기본개요 조회
                    # 'apiUrl': 'https://apis.data.go.kr/1613000/ArchPmsService_v2/getApBasisOulnInfo',
                    'apiUrl': 'http://apis.data.go.kr/1613000/HsPmsHubService/getHpBasisOulnInfo?serviceKey={apiKey}&sigunguCd={sigunguCd}&bjdongCd={bjdongCd}&numOfRows=100&pageNo={pageInfo}',
                    'colctFile': '/DATA/OUTPUT/LSH0613/건축인허가/{addrInfo}/건축인허가_{addrInfo}.csv',
                    'colctFilePattern': '/DATA/OUTPUT/LSH0613/건축인허가/{addrInfo}/건축인허가_{addrInfo}.csv',
                    'propFile': '/DATA/OUTPUT/LSH0613/건축인허가/전처리/건축인허가_{addrInfo}_prop.csv',
                }

                , '아파트실거래': {
                    # 국토교통부_아파트 매매 실거래가 상세 자료 > 아파트 매매 신고 상세자료
                    # 'apiUrl': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev',
                    'apiUrl': 'http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev?serviceKey={apiKey}&pageNo={pageInfo}&numOfRows=100&LAWD_CD={sigunguCd}&DEAL_YMD={dtYearMonth}',
                    'colctFile': '/DATA/OUTPUT/LSH0613/아파트실거래/{addrInfo}/아파트실거래_{addrInfo}_{dtYearMonth}.csv',
                    'colctFilePattern': '/DATA/OUTPUT/LSH0613/아파트실거래/{addrInfo}/아파트실거래_{addrInfo}_*.csv',
                    'propFile': '/DATA/OUTPUT/LSH0613/아파트실거래/전처리/아파트실거래_{addrInfo}_prop.csv',
                }
                , '아파트전월세': {
                    # 국토교통부_아파트 매매 실거래가 상세 자료 > 아파트 전월세 실거래가 자료
                    # 'apiUrl': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent',
                    'apiUrl': 'http://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent?serviceKey={apiKey}&pageNo={pageInfo}&numOfRows=100&LAWD_CD={sigunguCd}&DEAL_YMD={dtYearMonth}',
                    'colctFile': '/DATA/OUTPUT/LSH0613/아파트전월세/{addrInfo}/아파트전월세_{addrInfo}_{dtYearMonth}.csv',
                    'colctFilePattern': '/DATA/OUTPUT/LSH0613/아파트전월세/{addrInfo}/아파트전월세_{addrInfo}_*.csv',
                    'propFile': '/DATA/OUTPUT/LSH0613/아파트전월세/전처리/아파트전월세_{addrInfo}_prop.csv',
                }

                # 검색 목록
                #, 'addrList': ['서울특별시 강북구', '서울특별시 송파구', '서울특별시 강남구', '서울특별시 양천구', '서울특별시 서초구']
                #, 'addrList': ['서울특별시 강남구', '서울특별시 서초구', '서울특별시 송파구', '서울특별시 양천구', '서울특별시 용산구']
                #, 'addrList': ['서울특별시 강서구', '서울특별시 구로구', '서울특별시 동작구', '서울특별시 영등포구']
                , 'addrList': ['서울특별시', '경기도']
                # , 'addrList': ['서울특별시']
                # , 'addrList': [globalVar['addrList']]

                # 설정 정보
                , 'inpFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/mapInfo/admCode/법정동코드_전체자료.txt'
            }

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')
            # dt6HourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))
            # dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            # dt3HourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(3))

            # 구글 API 설정
            gmap = googlemaps.Client(key=sysOpt['googleApiKey'])

            # *********************************************************************************
            # 법정동 코드 읽기
            # *********************************************************************************
            # inpFile = '{}/{}'.format(globalVar['mapPath'], 'admCode/법정동코드_전체자료.txt')
            inpFile = sysOpt['inpFile']
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')

            admData = pd.read_csv(fileList[0], encoding='EUC-KR', sep='\t')
            admData[['d1', 'd2', 'd3', 'd4', 'd5']] = admData['법정동명'].str.split(expand=True)
            admData['sigunguCd'] = admData['법정동코드'].astype('str').str.slice(0, 5)
            admData['bjdongCd'] = admData['법정동코드'].astype('str').str.slice(5, 10)

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용하여 건축 인허가
            # *********************************************************************************
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #     admDataL1 = admData[
            #         admData['법정동명'].str.contains(addrInfo) &
            #         admData['폐지여부'].str.contains('존재')
            #         # admData['d4'].isna() &
            #         # admData['d5'].isna()
            #         # admData['d1'].notna() &
            #         # admData['d2'].notna() &
            #         # admData['d3'].notna()
            #     ].reset_index(drop=False)
            #
            #     if (len(admDataL1) < 1): continue
            #
            #     # saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '건축인허가', addrInfo, '건축인허가', addrInfo, dtYearMonth)
            #     # saveFile = '{}/{}/{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, '건축인허가', addrInfo, '건축인허가', addrInfo)
            #     saveFile = sysOpt['건축인허가']['colctFile'].format(addrInfo=addrInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     if len(glob.glob(saveFile)) > 0: continue
            #
            #     dataL1 = pd.DataFrame()
            #     for i, rowData in admDataL1.iterrows():
            #
            #         pageList = np.arange(1, 9999, 1)
            #         for k, pageInfo in enumerate(pageList):
            #
            #             # log.info(f"[CHECK] {addrInfo} : {dtMonthInfo} : {rowData['sigunguCd']} : {rowData['bjdongCd']} : {pageInfo}")
            #
            #             try:
            #                 # reqUrl = f"{sysOpt['apiUrl']}?serviceKey={sysOpt['apiKey']}&sigunguCd={rowData['sigunguCd']}&bjdongCd={rowData['bjdongCd']}&numOfRows=100&pageNo={pageInfo}&startDate={dtSrtYmd}&endDate={dtEndYmd}"
            #                 # reqUrl = f"{sysOpt['apiUrl']}?serviceKey={sysOpt['apiKey']}&sigunguCd={rowData['sigunguCd']}&bjdongCd={rowData['bjdongCd']}&numOfRows=100&pageNo={pageInfo}"
            #                 reqUrl = sysOpt['건축인허가']['apiUrl'].format(apiKey=sysOpt['apiKey'], sigunguCd=rowData['sigunguCd'], bjdongCd=rowData['bjdongCd'], pageInfo=pageInfo)
            #
            #                 # requests 라이브러리
            #                 res = requests.get(reqUrl)
            #                 if res.status_code != 200: break
            #
            #                 resData = res.content.decode('utf-8')
            #                 if resData is None or len(resData) < 1: break
            #
            #                 # urlopen 라이브러리
            #                 # res = urllib.request.urlopen(reqUrl)
            #                 # if res.getcode() != 200: continue
            #
            #                 # resData = res.read().decode('utf-8')
            #                 # if resData is None or len(resData) < 1: continue
            #
            #                 # xml to json 읽기
            #                 resDataL1 = xmltodict.parse(resData)
            #                 if resDataL1.get('response') is None: break
            #
            #                 resultCode = resDataL1['response']['header']['resultCode']
            #                 if resultCode != '00': break
            #
            #                 resBody = resDataL1['response']['body']
            #                 totalCnt = int(resBody['totalCount'])
            #                 if totalCnt < 1: break
            #                 if resBody['items'] is None: break
            #
            #                 itemList = resBody['items']['item']
            #                 if len(itemList) < 1: break
            #
            #                 data = pd.DataFrame.from_dict(itemList if isinstance(itemList, list) else [itemList])
            #                 data['addrInfo'] = addrInfo
            #                 data['sigunguCdInfo'] = rowData['sigunguCd']
            #                 data['bjdongCdInfo'] = rowData['bjdongCd']
            #                 data['pageInfo'] = pageInfo
            #
            #                 dataL1 = pd.concat([dataL1, data], ignore_index=False)
            #
            #                 if (totalCnt < (pageInfo * 100)): break
            #
            #             except Exception as e:
            #                 log.error(f"Exception : {str(e)}")
            #
            #     # log.info(f"[CHECK] addrInfo {dtSrtYmd}~{dtEndYmd} : {len(dataL1)} : {addrInfo} {dtYearMonth}")
            #     log.info(f"[CHECK] addrInfo : {len(dataL1)} : {addrInfo}")
            #
            #     # 자료 저장
            #     if len(dataL1) > 0:
            #         dataL1.to_csv(saveFile, index=False)
            #         log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용하여 아파트매매 실거래
            # *********************************************************************************
            # # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #     admDataL1 = admData[
            #         admData['법정동명'].str.contains(addrInfo) &
            #         admData['폐지여부'].str.contains('존재')
            #         # admData['d1'].notna() &
            #         # admData['d2'].notna()
            #         # admData['d3'].notna()
            #         # admData['d4'].isna() &
            #         # admData['d5'].isna()
            #         ].drop_duplicates(subset=['sigunguCd'], inplace=False).reset_index(drop=False)
            #
            #     if (len(admDataL1) < 1): continue
            #
            #     for j, dtMonthInfo in enumerate(dtMonthList):
            #         # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')
            #
            #         dtYearMonth = dtMonthInfo.strftime('%Y%m')
            #
            #         # saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '아파트실거래', addrInfo, '아파트실거래', addrInfo, dtYearMonth)
            #         saveFile = sysOpt['아파트실거래']['colctFile'].format(addrInfo=addrInfo, dtYearMonth=dtYearMonth)
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         # if len(glob.glob(saveFile)) > 0: continue
            #
            #         dataL1 = pd.DataFrame()
            #         for i, rowData in admDataL1.iterrows():
            #
            #             pageList = np.arange(1, 9999, 1)
            #             for k, pageInfo in enumerate(pageList):
            #                 # log.info(f"[CHECK] {addrInfo} : {dtYearMonth} : {rowData['sigunguCd']} : {pageInfo}")
            #
            #                 try:
            #                     # reqUrl = f"{sysOpt['apiUrl2']}?serviceKey={sysOpt['apiKey']}&pageNo={pageInfo}&numOfRows=100&LAWD_CD={rowData['sigunguCd']}&DEAL_YMD={dtYearMonth}"
            #                     reqUrl = sysOpt['아파트실거래']['apiUrl'].format(apiKey=sysOpt['apiKey'], sigunguCd=rowData['sigunguCd'], dtYearMonth=dtYearMonth, pageInfo=pageInfo)
            #
            #                     # requests 라이브러리
            #                     res = requests.get(reqUrl)
            #                     if res.status_code != 200: break
            #
            #                     resData = res.content.decode('utf-8')
            #                     if resData is None or len(resData) < 1: break
            #
            #                     # urlopen 라이브러리
            #                     # res = urllib.request.urlopen(reqUrl)
            #                     # if res.getcode() != 200: continue
            #
            #                     # resData = res.read().decode('utf-8')
            #                     # if resData is None or len(resData) < 1: continue
            #
            #                     # json 읽기
            #                     # resData = json.loads(res.read().decode('utf-8'))
            #
            #                     # xml to json 읽기
            #                     resDataL1 = xmltodict.parse(resData)
            #                     if resDataL1.get('response') is None: break
            #
            #                     resultCode = resDataL1['response']['header']['resultCode']
            #                     if resultCode != '000': break
            #
            #                     resBody = resDataL1['response']['body']
            #                     totalCnt = int(resBody['totalCount'])
            #                     if resBody.get('items') is None: break
            #                     if (totalCnt < 1): break
            #
            #                     itemList = resBody['items']['item']
            #                     if (len(itemList) < 1): break
            #
            #                     # data = pd.DataFrame.from_dict(itemList)
            #                     data = pd.DataFrame.from_dict(itemList if isinstance(itemList, list) else [itemList])
            #                     data['addrInfo'] = addrInfo
            #                     data['sigunguCdInfo'] = rowData['sigunguCd']
            #                     data['d2'] = rowData['d2']
            #                     data['dtYearMonth'] = dtYearMonth
            #                     data['pageInfo'] = pageInfo
            #
            #                     dataL1 = pd.concat([dataL1, data], ignore_index=False)
            #
            #                     if (totalCnt < (pageInfo * 100)): break
            #
            #                 except Exception as e:
            #                     log.error(f"Exception : {str(e)}")
            #
            #         log.info(f"[CHECK] addrInfo : {len(dataL1)} : {addrInfo} {dtYearMonth}")
            #
            #         # 자료 저장
            #         if len(dataL1) > 0:
            #             dataL1.to_csv(saveFile, index=False)
            #             log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용하여 아파트 전월세
            # *********************************************************************************
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #     # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #     admDataL1 = admData[
            #         admData['법정동명'].str.contains(addrInfo) &
            #         admData['폐지여부'].str.contains('존재')
            #         # admData['d1'].notna() &
            #         # admData['d2'].notna()
            #         # admData['d3'].notna()
            #         # admData['d4'].isna() &
            #         # admData['d5'].isna()
            #         ].drop_duplicates(subset=['sigunguCd'], inplace=False).reset_index(drop=False)
            #
            #     if (len(admDataL1) < 1): continue
            #
            #     for j, dtMonthInfo in enumerate(dtMonthList):
            #         # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')
            #
            #         dtYearMonth = dtMonthInfo.strftime('%Y%m')
            #
            #         # saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '아파트전월세', addrInfo, '아파트전월세', addrInfo, dtYearMonth)
            #         saveFile = sysOpt['아파트전월세']['colctFile'].format(addrInfo=addrInfo, dtYearMonth=dtYearMonth)
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         if len(glob.glob(saveFile)) > 0: continue
            #
            #         dataL1 = pd.DataFrame()
            #         for i, rowData in admDataL1.iterrows():
            #
            #             pageList = np.arange(1, 9999, 1)
            #             for k, pageInfo in enumerate(pageList):
            #                 # log.info(f"[CHECK] {addrInfo} : {dtYearMonth} : {rowData['sigunguCd']} : {pageInfo}")
            #
            #                 try:
            #                     # reqUrl = f"{sysOpt['apiUrl3']}?serviceKey={sysOpt['apiKey']}&pageNo={pageInfo}&numOfRows=100&LAWD_CD={rowData['sigunguCd']}&DEAL_YMD={dtYearMonth}"
            #                     reqUrl = sysOpt['아파트전월세']['apiUrl'].format(apiKey=sysOpt['apiKey'], sigunguCd=rowData['sigunguCd'], dtYearMonth=dtYearMonth, pageInfo=pageInfo)
            #
            #                     # requests 라이브러리
            #                     res = requests.get(reqUrl)
            #                     if res.status_code != 200: break
            #
            #                     resData = res.content.decode('utf-8')
            #                     if resData is None or len(resData) < 1: break
            #
            #                     # urlopen 라이브러리
            #                     # res = urllib.request.urlopen(reqUrl)
            #                     # if res.getcode() != 200: continue
            #
            #                     # json 읽기
            #                     # resData = res.read().decode('utf-8')
            #                     # if resData is None or len(resData) < 1: continue
            #
            #                     # xml to json 읽기
            #                     resDataL1 = xmltodict.parse(resData)
            #                     if resDataL1.get('response') is None: break
            #
            #                     resultCode = resDataL1['response']['header']['resultCode']
            #                     if resultCode != '000': continue
            #
            #                     resBody = resDataL1['response']['body']
            #                     totalCnt = int(resBody['totalCount'])
            #                     if resBody.get('items') is None: break
            #                     if (totalCnt < 1): break
            #
            #                     itemList = resBody['items']['item']
            #                     if (len(itemList) < 1): break
            #
            #                     # data = pd.DataFrame.from_dict(itemList)
            #                     data = pd.DataFrame.from_dict(itemList if isinstance(itemList, list) else [itemList])
            #                     data['addrInfo'] = addrInfo
            #                     data['sigunguCdInfo'] = rowData['sigunguCd']
            #                     data['d2'] = rowData['d2']
            #                     data['dtYearMonth'] = dtYearMonth
            #                     data['pageInfo'] = pageInfo
            #
            #                     dataL1 = pd.concat([dataL1, data], ignore_index=False)
            #
            #                     if (totalCnt < (pageInfo * 100)): break
            #
            #                 except Exception as e:
            #                     log.error("Exception : {}".format(e))
            #
            #         log.info(f"[CHECK] addrInfo : {len(dataL1)} : {addrInfo} {dtYearMonth}")
            #
            #         # 자료 저장
            #         if len(dataL1) > 0:
            #             dataL1.to_csv(saveFile, index=False)
            #             log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # 서울특별시 부동산 중개업소
            # *********************************************************************************
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 부동산 중개업소 정보.csv')
            # posData = pd.read_csv(inpFile, encoding='EUC-KR')
            #
            # for i, posInfo in posData.iterrows():
            #     add = posInfo['주소']
            #
            #     addUrlenc = parse.quote(add)
            #     url = apiUrl + addUrlenc
            #     request = Request(url)
            #     request.add_header('X-NCP-APIGW-API-KEY-ID', clientId)
            #     request.add_header('X-NCP-APIGW-API-KEY', clientPw)
            #
            #     posData.loc[i, 'lat'] = None
            #     posData.loc[i, 'lon'] = None
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
            #         log.info("[CHECK] [{}] ".format(i))
            #
            #         posData.loc[i, 'lat'] = responseBody['addresses'][0]['y']
            #         posData.loc[i, 'lon'] = responseBody['addresses'][0]['x']
            #
            #     except HTTPError as e:
            #         log.error("Exception : {}".format(e))
            #
            # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '서울특별시 부동산 중개업소 위경도 정보')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # posData.to_csv(saveFile, index=False)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # *********************************************************************************
            # [자료 전처리] 아파트 실거래
            # *********************************************************************************
            #inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '????1101*.xlsx')
            #inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '????1101*.xlsx')
            # addrInfo = sysOpt['addrList'][0]
            for ii, addrInfo in enumerate(sysOpt['addrList']):
                log.info(f'[CHECK] addrInfo : {addrInfo}')

                # inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '아파트실거래', addrInfo, '아파트실거래_*_*')
                inpFile = sysOpt['아파트실거래']['colctFilePattern'].format(addrInfo=addrInfo)
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                    # raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                orgData = pd.DataFrame()
                for fileInfo in fileList:
                    # tmpData = pd.read_excel(fileInfo, skiprows=16)
                    # posData = posData.append(tmpData)
                    tmpData = pd.read_csv(fileInfo)
                    orgData = pd.concat([orgData, tmpData], ignore_index=True)

                # 2025.04.21 컬럼 영문 변경
                # orgData.columns

                # 법정동 umdNm
                # 지번 jibun
                # 아파트 aptNm

                # 서울특별시 명일동 332 명일지에스아파트
                posData = orgData
                posData['sigunguCdInfo'] = posData['sigunguCdInfo'].astype(str)

                # posData = posData.reset_index()
                # posData['주소'] = addrInfo + ' ' + posData['도로명'] + ' ' + posData['단지명']
                # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'] + ' ' + posData['지번'].astype('string')
                # posData['addrDtlInfo'] = posData['addrInfo'].astype(str) + ' ' + posData['법정동'].astype(str) + ' ' + posData['지번'].astype(str) + ' ' + posData['아파트'].astype(str)
                posData['addrDtlInfo'] = posData['addrInfo'].astype(str).str.strip() + ' ' + posData['d2'].astype(str).str.strip() + ' ' + posData['umdNm'].astype(str).str.strip() + ' ' + posData['jibun'].astype(str).str.strip() + ' ' + posData['aptNm'].astype(str).str.strip()
                # posData['addrDtlInfo'] = posData['addrDtlInfo'].str.strip()

                addrDtlList = posData['addrDtlInfo'].unique()
                # addrDtlList = set(posData['addrDtlInfo'])

                propData = pd.DataFrame(addrDtlList, columns=['addr'])
                propData['addr'] = propData['addr'].astype(str)

                saveFile = sysOpt['아파트실거래']['propFile'].format(addrInfo=addrInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                propData.to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # posDataL1 = pd.DataFrame()
                # for i, addrDtlInfo in enumerate(addDtlrList):
                #     log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
                #
                #     posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
                #     posDataL1.loc[i, 'lat'] = None
                #     posDataL1.loc[i, 'lon'] = None
                #
                #     try:
                #         rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
                #         if (len(rtnGeo) < 1): continue
                #
                #         # 위/경도 반환
                #         posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
                #         posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
                #
                #     except Exception as e:
                #         log.error("Exception : {}".format(e))

                # 기존 네이버 주석처리
                # for i, addrInfo in enumerate(addrList):
                #     log.info(f'[CHECK] addrInfo : {addrInfo}')
                #
                #     addUrlenc = parse.quote(addrInfo)
                #     url = f'{sysOpt["naverApi"]}?query={addUrlenc}'
                #     request = Request(url)
                #     request.add_header('X-NCP-APIGW-API-KEY-ID', sysOpt['naverId'])
                #     request.add_header('X-NCP-APIGW-API-KEY', sysOpt['naverPw'])
                #
                #     # posDataL1.loc[i, '주소'] = addrInfo
                #     # posDataL1.loc[i, 'lat'] = None
                #     # posDataL1.loc[i, 'lon'] = None
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
                #         posDataL1.loc[i, 'lat'] = responseBody['addresses'][0]['y']
                #         posDataL1.loc[i, 'lon'] = responseBody['addresses'][0]['x']
                #
                #     except HTTPError as e:
                #         log.error("Exception : {}".format(e))

                # posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
                # # posDataL2.drop(['index'], axis=1, inplace=True)
                #
                # saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 실거래', addrInfo, datetime.now().strftime('%Y%m%d'))
                # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # posDataL2.to_csv(saveFile, index_label=False, index=False)
                # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # *********************************************************************************
            # [자료 전처리] 아파트 전월세
            # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '아파트(전월세)_????.xlsx')
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '아파트 전월세', addrInfo, '아파트 전월세_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     posData = pd.DataFrame()
            #     for fileInfo in fileList:
            #         # tmpData = pd.read_excel(fileInfo, skiprows=16)
            #         # posData = posData.append(tmpData)
            #         tmpData = pd.read_csv(fileInfo)
            #         posData = pd.concat([posData, tmpData], ignore_index=True)
            #
            #     # posData = posData.reset_index()
            #     # posData['주소'] = '서울특별시 강북구' + ' ' + posData['도로명'] + ' ' + posData['단지명']
            #     # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'] + ' ' + posData['지번']
            #     # posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['지번'].astype('string') + ' ' + posData['아파트']
            #     posData['addrDtlInfo'] = posData['addrInfo'].astype(str) + ' ' + posData['d2'].astype(str) + ' ' + posData['umdNm'].astype(str) + ' ' + posData['jibun'].astype(str) + ' ' + posData['aptNm'].astype(str)
            #
            #     # addrList = posData['주소'].unique()
            #     addrDtlList = set(posData['addrDtlInfo'])
            #
            #     propData = pd.DataFrame(addrDtlList, columns=['addr'])
            #     saveFile = sysOpt['아파트전월세']['propFile'].format(addrInfo=addrInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     propData.to_csv(saveFile, index=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')


            #     posDataL1 = pd.DataFrame()
            #     for i, addrDtlInfo in enumerate(addrDtlList):
            #         log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
            #
            #         posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
            #         posDataL1.loc[i, 'lat'] = None
            #         posDataL1.loc[i, 'lon'] = None
            #
            #         try:
            #             rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
            #             if (len(rtnGeo) < 1): continue
            #
            #             # 위/경도 반환
            #             posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
            #             posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
            #
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            #     # for i, addrInfo in enumerate(addrList):
            #     #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #     #
            #     #     addUrlenc = parse.quote(addrInfo)
            #     #     url = f'{sysOpt["naverApi"]}?query={addUrlenc}'
            #     #     request = Request(url)
            #     #     request.add_header('X-NCP-APIGW-API-KEY-ID', sysOpt['naverId'])
            #     #     request.add_header('X-NCP-APIGW-API-KEY', sysOpt['naverPw'])
            #     #
            #     #     # posDataL1.loc[i, '주소'] = addrInfo
            #     #     # posDataL1.loc[i, 'lat'] = None
            #     #     # posDataL1.loc[i, 'lon'] = None
            #     #
            #     #     try:
            #     #         response = urlopen(request)
            #     #         resCode = response.getcode()
            #     #
            #     #         if (resCode != 200): continue
            #     #
            #     #         responseBody = response.read().decode('utf-8')
            #     #         responseBody = json.loads(responseBody)
            #     #
            #     #         resCnt = responseBody['meta']['count']
            #     #         if (resCnt < 1): continue
            #     #
            #     #         log.info("[CHECK] [{}] {}".format(i, addrInfo))
            #     #
            #     #         posDataL1.loc[i, 'lat'] = responseBody['addresses'][0]['y']
            #     #         posDataL1.loc[i, 'lon'] = responseBody['addresses'][0]['x']
            #     #
            #     #     except HTTPError as e:
            #     #         log.error("Exception : {}".format(e))
            #
            #     posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
            #     # posDataL2.drop(['index'], axis=1, inplace=True)
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 전월세', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     posDataL2.to_csv(saveFile, index_label=False, index=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))


            # *********************************************************************************
            # [자료 전처리] 건축 인허가
            # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '건축 인허가', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #
            #     posData = pd.read_csv(fileList[0])
            #     posData['addrDtlInfo'] = posData['platPlc']
            #
            #     addrDtlList = set(posData['addrDtlInfo'])
            #
            #     posDataL1 = pd.DataFrame()
            #     for i, addrDtlInfo in enumerate(addrDtlList):
            #         log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
            #
            #         posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
            #         posDataL1.loc[i, 'lat'] = None
            #         posDataL1.loc[i, 'lon'] = None
            #
            #         try:
            #             rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
            #             if (len(rtnGeo) < 1): continue
            #
            #             # 위/경도 반환
            #             posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
            #             posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
            #
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            #     posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     posDataL2.to_csv(saveFile, index_label=False, index=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            #
            # # *********************************************************************************
            # # [자료 가공] 건축 인허가 및 아파트 실거래 간의 공간 일치
            # # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     houseConsData = pd.read_csv(fileList[0])
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 실거래_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     aptPriceData = pd.read_csv(fileList[0])
            #
            #     # *********************************************************************************
            #     # 최근접 좌표
            #     # *********************************************************************************
            #     # BallTree를 위한 초기 데이터
            #     houseConsRef = houseConsData[['addrDtlInfo', 'lat', 'lon']].dropna().reset_index(drop=True)
            #     baTree = BallTree(np.deg2rad(houseConsRef[['lat', 'lon']].values), metric='haversine')
            #
            #     aptPriceDataL1 = aptPriceData
            #     for i, posInfo in aptPriceDataL1.iterrows():
            #         if (pd.isna(posInfo['lat']) or pd.isna(posInfo['lon'])): continue
            #
            #         closest = baTree.query(np.deg2rad(np.c_[posInfo['lat'], posInfo['lon']]), k=1)
            #         cloDist = closest[0][0][0] * 1000.0
            #         cloIdx = closest[1][0][0]
            #
            #         aptPriceDataL1.loc[i, '인허가addrDtlInfo'] = houseConsRef.loc[cloIdx, 'addrDtlInfo']
            #         aptPriceDataL1.loc[i, '인허가lat'] = houseConsRef.loc[cloIdx, 'lat']
            #         aptPriceDataL1.loc[i, '인허가lon'] = houseConsRef.loc[cloIdx, 'lon']
            #         aptPriceDataL1.loc[i, '인허가distKm'] = cloDist
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 실거래', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     aptPriceDataL1.to_csv(saveFile, index_label=False, index=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # *********************************************************************************
            # # [자료 가공] 건축 인허가 및 아파트 전월세 간의 공간 일치
            # # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     houseConsData = pd.read_csv(fileList[0])
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 전월세_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     aptPriceData = pd.read_csv(fileList[0])
            #
            #     # *********************************************************************************
            #     # 최근접 좌표
            #     # *********************************************************************************
            #     # BallTree를 위한 초기 데이터
            #     houseConsRef = houseConsData[['addrDtlInfo', 'lat', 'lon']].dropna().reset_index(drop=True)
            #     baTree = BallTree(np.deg2rad(houseConsRef[['lat', 'lon']].values), metric='haversine')
            #
            #     aptPriceDataL1 = aptPriceData
            #     for i, posInfo in aptPriceDataL1.iterrows():
            #         if (pd.isna(posInfo['lat']) or pd.isna(posInfo['lon'])): continue
            #
            #         closest = baTree.query(np.deg2rad(np.c_[posInfo['lat'], posInfo['lon']]), k=1)
            #         cloDist = closest[0][0][0] * 1000.0
            #         cloIdx = closest[1][0][0]
            #
            #         aptPriceDataL1.loc[i, '인허가addrDtlInfo'] = houseConsRef.loc[cloIdx, 'addrDtlInfo']
            #         aptPriceDataL1.loc[i, '인허가lat'] = houseConsRef.loc[cloIdx, 'lat']
            #         aptPriceDataL1.loc[i, '인허가lon'] = houseConsRef.loc[cloIdx, 'lon']
            #         aptPriceDataL1.loc[i, '인허가distKm'] = cloDist
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 전월세', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     aptPriceDataL1.to_csv(saveFile, index_label=False, index=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))