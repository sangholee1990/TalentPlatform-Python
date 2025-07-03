# ================================================
# 요구사항
# ================================================
# Python을 이용한 청소년 인터넷 게임 중독 관련 소셜데이터 수집과 분석을 위한 한국형 온톨로지 개발 및 평가

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0612-DaemonFramework-analy-allData.py

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
# from matplotlib import font_manager, rc
# from dbfread import DBF, FieldParser
import csv
import os
import pandas as pd
import re

import json
from datetime import datetime, timedelta
from konlpy.tag import Okt
from collections import Counter
import pytz
from wordcloud import WordCloud
import io
import re
import os

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

    log.propagate = False

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
                'preDt': datetime.now(),
                'fontInfo': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/fontInfo/malgun.ttf',

                # 수행 목록
                'modelList': ['googleNews', 'naverNews', 'naverBlog', 'naverCafe', 'kci'],
                # 'modelList': ['naverNews', 'naverBlog', 'naverCafe', 'kci'],
                # 'modelList': ['googleNews'],
                # 'modelList': ['naverNews'],
                # 'modelList': ['naverBlog'],
                # 'modelList': ['naverCafe'],

                # 세부 정보
                'googleNews': {
                    'inpFile': '/DATA/OUTPUT/LSH0612/gnews_*.csv',
                    'itemList': {'title': '제목', 'text': '본문', 'summary': '요약'},
                },
                'naverNews': {
                    'inpFile': '/DATA/OUTPUT/LSH0612/naverNewsL1_*.csv',
                    'itemList': {'title': '제목', 'description': '본문', 'summary': '요약'},
                },
                'naverBlog': {
                    'inpFile': '/DATA/OUTPUT/LSH0612/naverBlog_*.csv',
                    'itemList': {'title': '제목', 'description': '본문'},
                },
                'naverCafe': {
                    'inpFile': '/DATA/OUTPUT/LSH0612/naverCafe_*.csv',
                    'itemList': {'title': '제목', 'description': '본문'},
                },
                'kci': {
                    'inpFile': '/DATA/OUTPUT/LSH0612/kci_*.csv',
                    'itemList': {'title': '제목', 'text': '본문'},
                },

                'saveFile': '/DATA/OUTPUT/LSH0612/%Y%m%d_{key}_빈도분포_{type}.xlsx',
                'saveImg': '/DATA/OUTPUT/LSH0612/%Y%m%d_{key}_단어구름_{type}.png',
            }

            # =================================================================
            # 공통 설정
            # =================================================================
            okt = Okt()

            # =================================================================
            # 제목/본문/요약 명사 추출, 단어구름 시각화, 빈도분포 저장
            # =================================================================
            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                fileList = sorted(glob.glob(modelInfo['inpFile']), reverse=True)
                data = pd.read_csv(fileList[0])
    
                keywordDataL2 = pd.DataFrame()
                for key, name in modelInfo['itemList'].items():
                    log.info(f'[CHECK] {key} : {name}')

                    textList = data[key].astype(str).tolist()
                    log.info(f'[CHECK] textList : {len(textList)}')

                    # text = '\n'.join(textList)
                    # if text is None or len(text) < 1: continue
                    # posTagList = okt.pos(text, stem=True)

                    posTagList = []
                    for textLine in textList:
                        posTagList.extend(okt.pos(textLine, stem=True))

                    keywordList = [word for word, pos in posTagList if pos in ['Noun']]
    
                    keywordOrder = 100
                    keywordCnt = Counter(keywordList)
                    keywordData = pd.DataFrame(keywordCnt.items(), columns=['keyword', 'cnt']).sort_values(by='cnt', ascending=False)
                    keywordDataL1 = keywordData[keywordData['keyword'].str.len() >= 2].reset_index(drop=True).head(keywordOrder)
                    keywordDataL1['rat'] = keywordDataL1['cnt'] / keywordDataL1['cnt'].sum() * 100
                    keywordDataL1['type'] = name
    
                    keywordDataL2 = pd.concat([keywordDataL2, keywordDataL1], axis=0)

                    # 단어구름 시각화
                    saveImg = sysOpt['preDt'].strftime(sysOpt['saveImg']).format(key= modelType,type=name)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    
                    wordcloud = WordCloud(
                        width=1500,
                        height=1500,
                        background_color=None,
                        mode='RGBA',
                        font_path=sysOpt['fontInfo'],
                        max_words=keywordOrder
                    ).generate_from_frequencies(keywordDataL1.set_index('keyword')['cnt'].to_dict())
    
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    # plt.show()
                    plt.close()
                    log.info(f'[CHECK] saveImg : {saveImg}')
    
                # 빈도분포 저장
                saveFile = sysOpt['preDt'].strftime(sysOpt['saveFile']).format(key= modelType, type='전체')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                keywordDataL2.to_excel(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')
    
                # # =================================================================
                # # [구글뉴스 수집 및 chatgpt 온톨로지 구축] 빈도분포 저장
                # # =================================================================
                # fileList = sorted(glob.glob(sysOpt['chatgptFile']))
                # data = pd.read_excel(fileList[0])
                # 
                # grpData = data.groupby(['대분류', '중분류', '소분류']).size()
                # grpDataL1 = grpData.reset_index(name='cnt')
                # grpDataL1['rat'] = (grpDataL1['cnt'] / grpDataL1['cnt'].sum()) * 100
                # grpDataL2 = grpDataL1.sort_values(by='cnt', ascending=False)
                # 
                # # 빈도분포 저장
                # saveFile = sysOpt['saveFile'].format(key='구글뉴스 수집 및 chatgpt 온톨로지 구축', type='전체')
                # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # grpDataL2.to_excel(saveFile, index=False)
                # log.info(f'[CHECK] saveFile : {saveFile}')
                # 
                # # =================================================================
                # # [구글뉴스 수집 및 gemini 온톨로지 구축] 빈도분포 저장
                # # =================================================================
                # fileList = sorted(glob.glob(sysOpt['geminiFile']))
                # data = pd.read_excel(fileList[0])
                # 
                # grpData = data.groupby(['대분류', '중분류', '소분류']).size()
                # grpDataL1 = grpData.reset_index(name='cnt')
                # grpDataL1['rat'] = (grpDataL1['cnt'] / grpDataL1['cnt'].sum()) * 100
                # grpDataL2 = grpDataL1.sort_values(by='cnt', ascending=False)
                # 
                # # 빈도분포 저장
                # saveFile = sysOpt['saveFile'].format(key='구글뉴스 수집 및 gemini 온톨로지 구축', type='전체')
                # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # grpDataL2.to_excel(saveFile, index=False)
                # log.info(f'[CHECK] saveFile : {saveFile}')

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
