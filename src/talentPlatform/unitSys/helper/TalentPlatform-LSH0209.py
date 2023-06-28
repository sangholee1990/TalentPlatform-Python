# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import sys
import logging
import platform
import sys
import traceback
import urllib
from datetime import datetime
from urllib import parse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import dfply
from plotnine.data import *
from plotnine import *
from sspipe import p, px

import urllib.request
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
import requests
from lxml import html
import urllib
import math
import glob
import warnings

# =================================================
# 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


# =================================================
# 함수 정의
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

    globalVar = {
        "prjName": prjName
        , "sysOs": platform.system()
        , "contextPath": contextPath
        , "resPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , "cfgPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , "inpPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , "sysPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace("\\", "/")

    return globalVar

#  초기 전달인자 설정
def initArgument(globalVar, inParams):

    for i, key in enumerate(inParams):
        if globalVar['sysOs'] in 'Linux':
            if i >= len(sys.argv[1:]): continue
            if inParams[key] == None: continue
            val = inParams[key] if sys.argv[i + 1] == None else sys.argv[i + 1]

        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] == None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

def makeDataProp(inpFile, saveXlsxFile, saveCsvFile):

    fileInfo = glob.glob(inpFile)
    if (len(fileInfo) < 1):
        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        # return

    data = pd.read_excel(fileInfo[0], skiprows=1)

    # 파일 전처리
    dataL1 = data[['시간', '절연유 상부온도', '절연유온도A', '절연유온도B', '절연유온도C', '권선온도A', '권선온도B',
                   '권선온도C']]

    sDateTime = dataL1['시간'].dt.strftime("%Y-%m-%d %H:%M")
    dataL1['시간'] = pd.to_datetime(sDateTime, format='%Y-%m-%d %H:%M')

    # 절연유 평균온도 = (절연유온도A+절연유온도B+절연유온도C)/3
    dataL1['val1'] = dataL1[['절연유온도A', '절연유온도B', '절연유온도C']].mean(axis=1)

    # 권선 평균온도 = (권선온도A+권선온도B+권선온도C)/3
    dataL1['val2'] = dataL1[['권선온도A', '권선온도B', '권선온도C']].mean(axis=1)

    # 권선 최고온도 = 권선 평균온도 +(1.1 x (절연유 상부온도 - 절연유 평균온도))
    dataL1['val3'] = dataL1['val2'] + (1.1 * (dataL1['절연유 상부온도'] - dataL1['val1']))

    # Aging Factor = exp[(15000/383) - (15000/(권선 최고온도+273))]
    dataL1['val4'] = np.exp((15000 / 383) - (15000 / (dataL1['val3'] + 273)))

    # 수명손실 = Aging Factor X 1/6[h]
    dataL1['val5'] = dataL1['val4'] * (1 / 6)

    # 누적수명손실
    dataL1['val6'] = np.cumsum(dataL1['val5'])

    # 누적수명손실율 = (누적수명손실/180000)X100
    dataL1['val7'] = (dataL1['val6'] / 180000) * 100

    # 이름 변경
    dataL1 = dataL1.rename(
        columns={
            'val1': '절연유 평균온도'
            , 'val2': '권선 평균온도'
            , 'val3': '권선 최고온도'
            , 'val4': 'Aging Factor'
            , 'val5': '수명손실'
            , 'val6': '누적수명손실'
            , 'val7': '누적수명손실율'
        }
    )

    # XLXS 파일 저장
    dataL1.to_excel(saveXlsxFile, index=False)

    # CSV 파일 저장
    dataL1.to_csv(saveCsvFile, index=False)

def searchInfo(inpFile, searchDateTime):

    # breakpoint()

    fileInfo = glob.glob(inpFile)
    if (len(fileInfo) < 1):
        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        # return

    data = pd.read_csv(fileInfo[0])
    data['시간'] = pd.to_datetime(data['시간'], format='%Y-%m-%d %H:%M')

    # searchDateTime = "2022-09-16 22:00"
    dtSearchDateTime = pd.to_datetime(searchDateTime, format='%Y-%m-%d %H:%M')

    dataL1 = data.loc[
        data['시간'] == dtSearchDateTime
        ]

    if (len(dataL1) < 1):
        log.error("[ERROR] searchDateTime : {} / {}".format(searchDateTime, '검색 날짜를 확인해주세요.'))
        return
        # raise Exception("[ERROR] searchDateTime : {} / {}".format(searchDateTime, '검색 날짜를 확인해주세요.'))

    # 1. 몇월몇일에서의 계산값은 ___입니다.
    # 2. 하루동안 가속열화시, 0.003671687%의 수명이 감소했습니다.
    # 3. 매일 이와 같이 운전시, 변압기 사용시간은 (100/0.003671687)/365 년 운전가능합니다.
    # 4. X축 시간, Y축 범례(센서값~계산값) 각각 그래프

    log.info('[CHECK] 몇월몇일에서의 계산값은 {}입니다.'.format(searchDateTime))
    log.info('[CHECK] 하루동안 가속열화시, {}%의 수명이 감소했습니다.'.format(dataL1['누적수명손실율'].values[0]))
    log.info('[CHECK] 매일 이와 같이 운전시, 변압기 사용시간은 {} 년 운전가능합니다.'.format((100 / dataL1['누적수명손실율'].values) / 365))
    # print(' X축 시간, Y축 범례(센서값~계산값) 각각 그래프'.format(sDateTime))

def makePlot(inpFile, saveImg):

    fileInfo = glob.glob(inpFile)
    if (len(fileInfo) < 1):
        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        # return

    data = pd.read_csv(fileInfo[0])
    data['시간'] = pd.to_datetime(data['시간'], format='%Y-%m-%d %H:%M')

    selColNameList = data.columns.difference(['시간'])

    plotData = (
            data >>
            dfply.gather('key', 'val', [selColNameList])
    )

    plot = (ggplot(plotData, aes(x='시간', y='val', color='key'))
            + geom_line()
            + scale_x_datetime(date_labels='%Y-%m-%d %H:%M')
            + theme(axis_text_x=element_text(angle=45, hjust=1)
                    , text=element_text(family="Malgun Gothic")
                    )
            + labs(
                x='Date [Year-Month-Day Hour:Minute]'
                , y='센서값 및 계산값'
                , title='센서 및 계산에 따른 시계열'
                )
            )

    plot.save(saveImg, bbox_inches='tight', width=10, height=6, dpi=600)

class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # LSH0209. Python을 이용한 센서 데이터 전처리 및 시계열 시각화

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0209'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar key / val : {} / {}".format(key, val))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        try:
            log.info('[START] {}'.format("exec"))

            # breakpoint()

            # 파일 정보
            inpFile = '{}/{}_{}'.format(globalVar['inpPath'], serviceName, '파이썬값(수정).xlsx')
            saveXlsxFile = '{}/{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '센서 및 계산에 따른 데이터')
            saveCsvFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, '센서 및 계산에 따른 데이터')

            # 파일 전처리 및 저장
            makeDataProp(inpFile, saveXlsxFile, saveCsvFile)

            # 이미지 저장
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '센서 및 계산에 따른 시계열')
            makePlot(saveCsvFile, saveImg)

            # 1. 몇월몇일에서의 계산값은 ___입니다.
            # 2. 하루동안 가속열화시, 0.003671687%의 수명이 감소했습니다.
            # 3. 매일 이와 같이 운전시, 변압기 사용시간은 (100/0.003671687)/365 년 운전가능합니다.
            # 4. X축 시간, Y축 범례(센서값~계산값) 각각 그래프

            # breakpoint()

            # 검색 날짜 (성공)
            searchDateTime = "2021-09-16 22:00"
            searchInfo(saveCsvFile, searchDateTime)

            # 검색 날짜 (실패)
            # searchDateTime = "2022-09-16 21:00"
            # searchInfo(saveCsvFile, searchDateTime)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

    # 수행 프로그램 (단일 코어, 다중 코어 멀티프로세싱)
    def runPython(self):
        try:
            log.info('[START] {}'.format("runPython"))

            DtaProcess.exec(self)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("runPython"))


if __name__ == '__main__':

    try:
        print('[START] {}'.format("main"))

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        # 입력 자료 : inpPath
        # 그림 자료 : figPath
        # 출력 자료 : outPath
        # 로그 자료 : logPath
        inParams = {
            # 'inpPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'figPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'outPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'logPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
        }

        print("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
