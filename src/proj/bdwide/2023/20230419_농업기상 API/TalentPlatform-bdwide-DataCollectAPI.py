import argparse
import datetime
import glob
import logging
import logging.handlers
import logging.handlers
import logging.handlers
import os
import platform
import sys
import traceback
import urllib
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import xmltodict
from pandas.tseries.offsets import Hour

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
# font_manager._rebuild()

#plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# # 타임존 설정
# tzKst = pytz.timezone('Asia/Seoul')
# tzUtc = pytz.timezone('UTC')
# dtKst = timedelta(hours=9)

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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        # fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        # plt.rc('font', family=fontName)

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

        log.info("[CHECK] {} / {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PWRER-IOT'

    prjName = 'bdwide'
    serviceName = 'PRJ2023'

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
                    # 시작/종료 시간
                    # 'srtDate': '2022-01-01'
                    # , 'endDate': '2022-12-31'

                    'srtDate': '2022-07-01'
                    , 'endDate': '2022-08-01'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 공공데이터포털 API
                    'apiKey': 'F6CN7k8VOt7oHN2xhXc8tEXhBJbH7lR1qMpFxCEfWNAA3P34sS/WiDexjmujb7s6lWnuAnq0UEktsxTGsi2FIw=='
                    # 'apiKey': 'bf9fH0KLgr65zXKT5D/dcgUBIj1znJKnUPrzDVZEe6g4gquylOjmt65R5cjivLPfOKXWcRcAWU0SN7KKXBGDKA=='

                    # 농촌진흥청_국립농업과학원_농업기상 관측지점
                    , 'apiUrl': 'http://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/InsttWeather/getWeatherTenMinList'

                    # 시작/종료 시간
                    , 'srtDate': '2018-01-01'
                    , 'endDate': '2018-01-02'
                    # , 'endDate': '2018-03-01'
                }

                # 입력/출력/그림 경로
                # globalVar['inpPath'] = '/DATA/INPUT'
                # globalVar['outPath'] = '/DATA/OUTPUT'
                # globalVar['figPath'] = '/DATA/FIG'

            log.info(f'[CHECK] sysOpt : {sysOpt}')

            # 날짜 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            # log.info(f'[CHECK] dtIncDateList : {dtIncDateList}')

            inpFile = '{}/{}'.format(globalVar['inpPath'], '농촌진흥청_국립농업과학원_농업기상 관측지점 정보_201908.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')

            posData = pd.read_csv(fileList[0], encoding='cp949')
            # posDataL1 = posData[['도구분', '지점코드', '지점명']]

            dataL2 = pd.DataFrame()
            for i, posInfo in posData.iterrows():
                # log.info(f'[CHECK] posInfo : {posInfo}')

                for j, dtIncDateInfo  in enumerate(dtIncDateList):
                    log.info(f'[CHECK] {posInfo["지점명"]} : {dtIncDateInfo}')

                    dtYmdHm = dtIncDateInfo.strftime('%Y%m%d%H%M')
                    dtYmd = dtIncDateInfo.strftime('%Y-%m-%d')
                    dtHm = dtIncDateInfo.strftime('%H%M')

                    saveFile = '{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '농업기상', '농업기상', posInfo['지점코드'], dtYmdHm)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)

                    # 파일 중복 검사
                    fileChkList = glob.glob(saveFile)
                    if (len(fileChkList) > 0): continue

                    dataL1 = pd.DataFrame()
                    try:
                        apiUrl = sysOpt['apiUrl']

                        inParams = {'serviceKey': sysOpt["apiKey"], 'Page_No': '1', 'Page_Size': '100', 'date': dtYmd, 'time': dtHm, 'obsr_Spot_Nm': posInfo['지점명'], 'obsr_Spot_Code': posInfo['지점코드']}
                        apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')

                        res = requests.get(apiUrl, params=apiParams)

                        resCode = res.status_code
                        if resCode != 200: continue

                        # json 읽기
                        # resData = json.loads(res.read().decode('utf-8'))

                        # xml to json 읽기
                        resData = xmltodict.parse(res.content.decode('utf-8'))

                        # if resData['OpenAPI_ServiceResponse']['cmmMsgHeader']['returnReasonCode'] == '22': continue
                        if (resData.get('OpenAPI_ServiceResponse') is not None) or (resData.get('response') is None): continue

                        resultCode = resData['response']['header']['result_Code']
                        if (resultCode != '200'): continue

                        resBody = resData['response']['body']
                        totalCnt = int(resBody['total_Count'])
                        if (totalCnt < 1): break

                        # itemList = resBody['items']['item']
                        itemList = resBody['items']
                        if (len(itemList) < 1): break

                        data = pd.DataFrame.from_dict(itemList).transpose().reset_index(drop=True)
                        data['지점명'] = posInfo['지점명']
                        data['지점코드'] = posInfo['지점코드']
                        data['dtYmdHm'] = dtYmdHm

                        dataL1 = data
                        dataL2 = pd.concat([dataL2, dataL1], ignore_index=False)

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                    # 자료 저장
                    if (len(dataL1) > 0):
                        dataL1.to_csv(saveFile, index=False)
                        log.info(f'[CHECK] saveFile : {saveFile}')

                if (len(dataL2) > 0):
                    minDate = dtIncDateList.min().strftime("%Y%m%d")
                    maxDate = dtIncDateList.max().strftime("%Y%m%d")

                    saveFile = '{}/{}/{}/{}_{}_{}-{}.csv'.format(globalVar['outPath'], serviceName, '농업기상', '농업기상', posInfo['지점코드'], minDate, maxDate)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dataL2.to_csv(saveFile, index=False)
                    log.info(f'[CHECK] saveFile : {saveFile}')

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
