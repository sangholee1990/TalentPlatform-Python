# ================================================
# 요구사항
# ================================================

# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0627-DaemonFramework-model.py
# 0 0 * * * cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys && /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py

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
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime, timedelta
import pytz
import os
import sys
import os
import sys
import json
from darts import TimeSeries
from darts.models import Prophet
from darts.models import DLinearModel

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

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')


# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
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
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

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
        ,
        'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
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

def extYear(title):
    match = re.search(r'\(?(\d{4})\)?', str(title))
    if match:
        year = int(match.group(1))
        current_year = datetime.now().year

        if 2000 <= year <= current_year + 1:
            return str(year)
    return None

def clsBrand(title):
    title = str(title)
    if '알톤' in title:
        return '알톤 자전거'
    elif '삼천리' in title:
        return '삼천리 자전거'
    elif '스마트' in title:
        return '스마트 자전거'
    else:
        return '기타'

def clsType(title):
    title = str(title)
    if '전기' in title:
        return '전기자전거'
    elif '하이브리드' in title:
        return '하이브리드'
    elif 'MTB' in title:
        return 'MTB'
    elif '사이클' in title or '로드' in title:
        return '사이클'
    elif '미니벨로' in title:
        return '미니벨로'
    else:
        return '일반자전거'

def extKeyword(title, extKeywordList):
    title = str(title)
    incKeyword = [keyword for keyword in extKeywordList if keyword in title]
    if incKeyword:
        return ",".join(incKeyword)
    return None

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

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0612'

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
                # 입력 데이터
                'inpFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_자전거.csv',
                'model': {
                    # 시계열 딥러닝
                    'dl': {
                        'input_chunk_length': 2,
                        'output_chunk_length': 7,
                        'n_epochs': 50,
                    },
                    # 예측
                    'prdCnt': 7,
                },
                # 예측 데이터
                'saveFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_prd.csv',
                'preDt': datetime.now(),
                'extKeywordList': ['호환', '할인', '부품', '액세서리', '안장', '헬멧', '거치대', '펌프', '공구', '수리', '세팅', '대여', '렌탈', '전시'],
            }

            # =================================================================
            # 모델링
            # =================================================================
            mlModel = Prophet()
            dlModel = DLinearModel(
                input_chunk_length=sysOpt['model']['dl']['input_chunk_length'],
                output_chunk_length=sysOpt['model']['dl']['output_chunk_length'],
                n_epochs=sysOpt['model']['dl']['n_epochs'],
                random_state=None,
            )

            orgData = pd.read_csv(sysOpt['inpFile'])
            data = orgData.sort_values(by='lprice', ascending=True).drop_duplicates(subset=['title', 'date'], keep='first')
            data['dtDate'] = pd.to_datetime(data['date'], format='%Y%m%d')

            # 자전거 상품 별로 최저가 2개 이상인 경우
            # modTitleList = sorted(data.groupby("title").filter(lambda x: x['lprice'].nunique() > 1)['title'].unique())
            modTitleList = sorted(data['title'].unique())

            # modTitleInfo = modTitleList[0]
            # modTitleInfo = '하운드 2025 <b>삼천리자전거</b> 시애틀F 21단 26 접이식 자전거'
            # modTitleInfo = '하운드 2022 <b>삼천리자전거</b> 하운드 주니어 <b>자전거</b> 시애틀MT 20인치'
            mlPrdDataL1 = pd.DataFrame()
            dlPrdDataL1 = pd.DataFrame()
            for i, modTitleInfo in enumerate(modTitleList):
                per = round(i / len(modTitleList) * 100, 1)
                log.info(f'i : {i}, per : {per}%')

                selData = data[(data['title'] == modTitleInfo)]
                if len(selData) < 1: continue

                # selDataL1 = TimeSeries.from_dataframe(selData, time_col='dtDate', value_cols='lprice', fill_missing_dates=True, freq='D')
                selDataL1 = TimeSeries.from_dataframe(selData, time_col='dtDate', value_cols='lprice', freq='D')

                try:
                    mlModel.fit(selDataL1)
                    mlPrd = mlModel.predict(n=sysOpt['model']['prdCnt'])
                    mlPrdData = pd.DataFrame({
                        'title': modTitleInfo,
                        'dtDate': mlPrd.time_index,
                        'mlPrd': mlPrd.values().flatten()
                    })

                    if len(mlPrdData) > 0:
                        mlPrdDataL1 = pd.concat([mlPrdDataL1, mlPrdData], ignore_index=True)
                except Exception as e:
                    log.error(f'Exception : {e}')

                try:
                    dlModel.fit(selDataL1)
                    dlPrd = dlModel.predict(n=sysOpt['model']['prdCnt'])
                    dlPrdData = pd.DataFrame({
                        'title': modTitleInfo,
                        'dtDate': dlPrd.time_index,
                        'dlPrd': dlPrd.values().flatten()
                    })

                    if len(dlPrdData) > 0:
                        dlPrdDataL1 = pd.concat([dlPrdDataL1, dlPrdData], ignore_index=True)
                except Exception as e:
                    log.error(f'Exception : {e}')

            dataL1 = data
            if (len(mlPrdDataL1) > 0) & (len(dlPrdDataL1) > 0):
                prdData = pd.merge(mlPrdDataL1, dlPrdDataL1, on=['title', 'dtDate'], how='inner').drop_duplicates(subset=['title', 'dtDate'], keep='first')
                dataL1 = pd.merge(dataL1, prdData, on=['title', 'dtDate'], how='outer')

            dataL2 = dataL1.sort_values(['title', 'date'], ascending=False).reset_index(drop=True)
            # dataL2[(dataL2['title'] == modTitleInfo)]

            dataL2['yearByTitle'] = dataL2['title'].apply(extYear)
            dataL2['brandByTitle'] = dataL2['title'].apply(clsBrand)
            dataL2['typeByTitle'] = dataL2['title'].apply(clsType)
            dataL2['keywordByTitle'] = dataL2['title'].apply(lambda x: extKeyword(x, sysOpt['extKeywordList']))
            dataL3 = dataL2[dataL2['keywordByTitle'].isna() & dataL2['yearByTitle'].notna()].sort_values(['title', 'date'], ascending=False).reset_index(drop=True)
            if len(dataL3) > 0:
                saveFile = sysOpt['preDt'].strftime(sysOpt['saveFile'])
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL3.to_csv(saveFile, index=False)
                log.info(f"[CHECK] saveFile : {saveFile}")
                
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
