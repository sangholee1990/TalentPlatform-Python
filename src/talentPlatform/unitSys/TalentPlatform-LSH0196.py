# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import sys
# from plotnine import *
# from plotnine.data import *
# from dfply import *
# import hydroeval
import dfply
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import glob
import pprint
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler
# from keras.layers import LSTM
# from keras.models import Sequential
#
# from keras.layers import Dense
# import keras.backend as K
# from keras.callbacks import EarlyStopping
from multiprocessing import Pool, Process
import traceback
import sys

# 초기 환경변수 정의
from src.talentPlatform.unitSysHelper.InitConfig import *


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 GPS 데이터셋 병합 및 전처리

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0196'
    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 파이썬 실행 시 전달인자 설정
            # pyhton3 *.py argv1 argv2 argv3 ...
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

                print(os.getcwd())

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

            fileInfoPattrn = '{}/{}'.format(globalVar['inpPath'], 'LSH0196_GPS.xlsx')
            fileInfo = glob.glob(fileInfoPattrn)

            if (len(fileInfo) < 1):
                raise Exception("[ERROR] fileInfo : {} : {}".format("입력 자료를 확인해주세요.", fileInfoPattrn))

            #++++++++++++++++++++++++++++++++++++
            # 엑셀 파일에서 시트 A 읽기
            # ++++++++++++++++++++++++++++++++++++
            dataA = pd.read_excel(fileInfo[0], sheet_name = 'A')
            dataA.columns = ['col1', 'time', 'col2', 'y', 'x']

            dtTime = pd.to_datetime(dataA['time'], format='%H:%M:%S')

            sYear = '2021'
            sMonth = '07'
            sDay = '29'
            sHour = dtTime.dt.hour.astype('str')
            sMinute = dtTime.dt.minute.astype('str')
            sSec = (dtTime.dt.second.astype('int') + (dataA['col1'] - dataA['col1'].astype(int))).astype('str')
            sDateTime = sYear + '-' + sMonth + '-' + sDay + ' ' + sHour + ':' + sMinute + ':' + sSec

            dataA['dtDateTime'] = pd.to_datetime(sDateTime, format='%Y-%m-%d %H:%M:%S.%f')

            # ++++++++++++++++++++++++++++++++++++
            # 엑셀 파일에서 시트 B 읽기
            # ++++++++++++++++++++++++++++++++++++
            dataB = pd.read_excel(fileInfo[0], sheet_name='B')
            dataB.columns = ['col1', 'x']

            dataB['dtDateTime'] = pd.to_datetime(dataB['col1'], format='%Y-%m-%d %H:%M:%S.%f')

            for i in dataB.index:
                if i >= (len(dataB) - 1): continue

                dtUnix = dataB._get_value(i, 'col1').timestamp()
                dtNextUnix = int(dataB._get_value(i + 1, 'col1').timestamp())

                # 정수형의 경우
                if dtUnix == int(dtUnix):
                    dataB._set_value(i, 'dtDateTime', pd.to_datetime(dtNextUnix, unit='s'))

            # dtDateTime 기준으로 데이터 병합
            dataL1 = pd.merge(dataA, dataB, how="left", on="dtDateTime")
            dataL2 = dataL1[['dtDateTime', 'col1_x', 'time', 'col2', 'y', 'x_x', 'col1_y', 'x_y']]

            saveFile = '{}/{}_{}'.format(globalVar['outPath'], serviceName, 'GPS.csv')
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            dataL2.to_csv(saveFile, index=False)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

    # 수행 프로그램 (단일 코어, 다중 코어 멀티프레세싱)
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
        log.info('[START] {}'.format("main"))

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

        log.info("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        log.error(traceback.format_exc())
        sys.exit(1)

    finally:
        log.info('[END] {}'.format("main"))
