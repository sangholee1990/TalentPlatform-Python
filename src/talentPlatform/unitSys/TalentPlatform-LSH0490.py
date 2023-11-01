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

import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import pandas as pd
import re
import numpy as np
import concurrent.futures

from matplotlib.pyplot import axis

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

def makeCsvProc(fileInfo):

    print(f'[START] makeCsvProc')

    result = None

    try:
        filePath = os.path.dirname(fileInfo)
        fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
        data = pd.read_csv(fileInfo, encoding='EUC-KR')

        # '고유번호', '토지소재', '지번', '지목', '면적', '변동일자', '성명', '주소', '등록번호', '공시지가', '소유권이전 변동일자', '토지가격'
        # data.columns

        data['토지가격'] = pd.to_numeric(data['토지가격'], errors='coerce')
        data['면적'] = pd.to_numeric(data['면적'], errors='coerce')
        data['isMount'] = data['지번'].str.contains('산', na=False)

        # 토지가격, 면적 순으로 내림차순 정렬
        dataL1 = data.sort_values(by=['토지가격', '면적'], ascending=[False, False], na_position='last')

        # 토지가격을 기준으로 구간 설정
        conditions = [
            dataL1['토지가격'] >= 5e8,
            (dataL1['토지가격'] < 5e8) & (dataL1['토지가격'] >= 3e8),
            (dataL1['토지가격'] < 3e8) & (dataL1['토지가격'] >= 1e8),
            dataL1['토지가격'] < 1e8
        ]

        choices = ['1그룹', '2그룹', '3그룹', '4그룹']
        dataL1['구간'] = pd.np.select(conditions, choices, default='가격없음')

        # 우선 순위 선정
        rankData = dataL1.groupby(['isMount', '구간', '토지소재'])['토지가격'].sum().reset_index().rename({'토지가격': '총계'}, axis=1)
        rankData['순위'] = rankData['총계'].rank(method="min", ascending=False)

        # 병합
        dataL2 = dataL1.merge(rankData, left_on=['isMount', '구간', '토지소재'], right_on=['isMount', '구간', '토지소재'], how='left')

        # 재 정렬
        fnlData = dataL2.sort_values(by=['순위', '토지가격', '면적', 'isMount', '토지소재', '지번'])

        saveFile = '{}/{}-{}.csv'.format(filePath, fileNameNoExt, 'fnlData')
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        fnlData.to_csv(saveFile, index=False, encoding='CP949')
        print(f'[CHECK] saveFile : {saveFile}')

    except Exception as e:
        print(f'Exception : {e}')
        return result

    finally:
        print(f'[END] makeCsvProc')

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 특정 조건 (성명, 주소, 주민번호)에 따른 자료 전처리2

    # 함수 1. csv파일을 불러옵니다. 첫 줄은 헤더에요. 헤더에 '토지소재', '지번', '토지가격', '면적'이 있습니다. 파일마다 토지소재와 지번의 위치가 다를 수 있으니 정해진 열이 아니라 헤더에 따라서 토지소재와 지번을 구간별로 나누는 함수를 만들어 주세요. 구간을 나누는 규칙은 다음과 같습니다.
    # 1. '토지가격'으로 내림차순한다. 다시 말 해 비싼것부터 싼거 순으로 정렬합니다. '토지가격'이 없는 경우 '면적'이 큰 것 부터 작은 것으로 정렬합니다. '면적'도 없는 경우 에러 메세지 출력하고 종료.
    # 2. '토지가격'을 기준으로 구간을 나눕니다. 1그룹은 5억 이상, 2그룹은 5억 미만 3억 이상, 3그룹은 3억미만 1억이상, 4그룹은 1억 미만, 5그룹은 '가격없음'입니다. 그룹의 수, 그룹을 나누는 가격 기준은 제가 임의로 나중에 숫자만 바꿔서 쓸 수 있게끔 부탁 드립니다.
    # 3. 여기까지 하면 일단 가격 순으로 정렬되어 있기 때문에 토지소재를 기준으로는 뒤죽박죽일 겁니다. 같은 구간내에서 토지소재를 기준으로 오름차순 정렬하고, 토지소재가 같은 경우 지번을 기준으로 오름차순 정렬합니다.
    # 4. 그룹내에서 '지번'에 문자 '산'이 있는 것은 밑으로, '산'이 없는 것은 위로 몰아 놓습니다.
    # 예시로 쓰실 만한 파일 있다가 한 두개 첨부해 드리겠습니다.

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
    serviceName = 'LSH0491'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 수행 목록
                'nameList': ['csv']

                # 모델 정보 : 파일 경로, 파일명
                , 'nameInfo': {
                    'csv': {
                        'filePath': '/DATA/INPUT/LSH0490'
                        , 'fileName': '대전_프린트.csv'
                        # , 'fileName': '06경기용인_프린트.csv'
                    }
                }
            }


            for i, nameType in enumerate(sysOpt['nameList']):
                log.info(f'[CHECK] nameType : {nameType}')

                nameInfo = sysOpt['nameInfo'].get(nameType)
                if nameInfo is None: continue

                inpFile = '{}/{}'.format(nameInfo['filePath'], nameInfo['fileName'])
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    continue

                for j, fileInfo in enumerate(fileList):
                    log.info(f'[CHECK] fileInfo : {fileInfo}')

                    makeCsvProc(fileInfo)

        except Exception as e:
            log.error(f'Exception : {e}')

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
