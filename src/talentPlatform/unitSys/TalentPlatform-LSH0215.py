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
        ,
        "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
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


# 아래 퍼셉트론을 위한 함수
def checkValue(val):
    if val <= 0:
        return 0
    else:
        return 1

# AND게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    val = np.sum(w * x) + b

    return checkValue(val)

# OR게이트
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    val = np.sum(w * x) + b

    return checkValue(val)

# NAND게이트
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    val = np.sum(w * x) + b

    return checkValue(val)

# XOR게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)

    return AND(s1, s2)

def add_five(x):
    x += 5

class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # LSH0215. Python을 이용한 퍼세트론 구현 및 소스 코드의 결과에 대한 근거 제시

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0215'

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

            # 아래 퍼셉트론을 python으로 구현
            log.info("[CHECK] AND(0, 0): {}".format(AND(0, 0)))
            log.info("[CHECK] AND(1, 0): {}".format(AND(1, 0)))
            log.info("[CHECK] AND(0, 1): {}".format(AND(0, 1)))
            log.info("[CHECK] AND(1, 1): {}".format(AND(1, 1)))

            log.info("[CHECK] OR(0, 0): {}".format(OR(0, 0)))
            log.info("[CHECK] OR(1, 0): {}".format(OR(1, 0)))
            log.info("[CHECK] OR(0, 1): {}".format(OR(0, 1)))
            log.info("[CHECK] OR(1, 1): {}".format(OR(1, 1)))

            log.info("[CHECK] NAND(0, 0): {}".format(NAND(0, 0)))
            log.info("[CHECK] NAND(1, 0): {}".format(NAND(1, 0)))
            log.info("[CHECK] NAND(0, 1): {}".format(NAND(0, 1)))
            log.info("[CHECK] NAND(1, 1): {}".format(NAND(1, 1)))

            log.info("[CHECK] XOR(0, 0): {}".format(XOR(0, 0)))
            log.info("[CHECK] XOR(1, 0): {}".format(XOR(1, 0)))
            log.info("[CHECK] XOR(0, 1): {}".format(XOR(0, 1)))
            log.info("[CHECK] XOR(1, 1): {}".format(XOR(1, 1)))

            # 그리고 첨부드린 그림의 파이썬 코딩의 결과를 캡처해주시고 그 이유에 대해 간략 기재해주시면 됩니다.
            A = np.array([1, 2, 3, 4, 5, 6])
            B = 50

            add_five(A)
            add_five(B)

            # 파이썬에서 모든 변수는 값에 대한 참조임
            # 따라서 add_five(A) 호출 시 x + 5로 신규 배열로 새로 참조되기 때문에
            # 기존의 A의 값 +5로 출력
            log.info("[CHECK] A: {}".format(A))

            # 반면에 인수는 파이썬에서 할당으로 전달됨
            # 즉 add_five(50) 호출 시 x = 55로 재할당되나 반환 결과(return)가 없기 때문에
            # 기존의 B의 값이 그대로 출력
            log.info("[CHECK] B: {}".format(B))


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
