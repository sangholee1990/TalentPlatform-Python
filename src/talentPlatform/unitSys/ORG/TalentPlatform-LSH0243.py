# -*- coding: utf-8 -*-

import glob
import json
import logging
import logging.handlers
import math
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
import seaborn as sns
from gnews.utils.utils import import_or_install
from pandas import json_normalize
from plotnine import *
from scipy import spatial
# import dfply as dfply

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
            if inParams[key] is None: continue
            val = inParams[key] if sys.argv[i + 1] is None else sys.argv[i + 1]

        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] is None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

def makeGridPlot(sysOpt, inpFile):

    log.info('[START] {}'.format('makeGridPlot'))

    result = None

    try:
        fileInfo = glob.glob(inpFile)

        if fileInfo is None or len(fileInfo) < 1:
            log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

        data = pd.read_csv(fileInfo[0], encoding='CP932')

        # getMaxLon = np.max(data['x'])
        # getMinLon = np.min(data['x'])
        # getMaxLat = np.max(data['y'])
        # getMinLat = np.min(data['y'])
        #
        # maxLon = getMaxLon if getMaxLon < sysOpt['maxLon'] else sysOpt['maxLon']
        # minLon = getMinLon if getMinLon > sysOpt['minLon'] else sysOpt['minLon']
        # maxLat = getMaxLat if getMaxLat < sysOpt['maxLat'] else sysOpt['maxLat']
        # minLat = getMinLat if getMinLat > sysOpt['minLat'] else sysOpt['minLat']

        maxLon = sysOpt['maxLon']
        minLon = sysOpt['minLon']
        maxLat = sysOpt['maxLat']
        minLat = sysOpt['minLat']

        log.info('[CHECK] maxLon : {}'.format(maxLon))
        log.info('[CHECK] minLon : {}'.format(minLon))
        log.info('[CHECK] maxLat : {}'.format(maxLat))
        log.info('[CHECK] minLat : {}'.format(minLat))

        # 100,000 m = 1 도
        # 2 m = 2 / 50000 도
        gridInvDeg = sysOpt['gridInv'] / 50000
        log.info('[CHECK] gridInvDeg : {}'.format(gridInvDeg))

        lonList = np.arange(minLon, maxLon, gridInvDeg)
        latList = np.arange(minLat, maxLat, gridInvDeg)

        nxGridInv = len(lonList)
        nyGridInv = len(latList)

        # nxGridInv = int(sysOpt['nxMeter'] / sysOpt['gridInv']) + 1
        # nyGridInv = int(sysOpt['nyMeter'] / sysOpt['gridInv']) + 1

        log.info('[CHECK] nxGridInv : {}'.format(nxGridInv))
        log.info('[CHECK] nyGridInv : {}'.format(nyGridInv))

        # lonList = np.linspace(minLon, maxLon, num=nxGridInv)
        # latList = np.linspace(minLat, maxLat, num=nyGridInv)

        eleData = pd.DataFrame()
        posList = []
        idx = 0

        # kdTree를 위한 초기 데이터
        for i in range(0, len(lonList)):
            for j in range(0, len(latList)):
                dict = {
                    'i': [i]
                    , 'j': [j]
                    , 'idx': [idx]
                    , 'lon': [lonList[i]]
                    , 'lat': [latList[j]]
                }

                eleData = eleData.append(pd.DataFrame.from_dict(dict))

                coord = [latList[j], lonList[i]]
                posList.append(cartesian(*coord))
                idx = idx + 1

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, sysOpt['id'], '그리드 설정')
        eleData.to_csv(saveCsvFile, index=False)
        log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        # kdTree 학습
        tree = spatial.KDTree(posList)

        # kdTree를 통해 최근접 위/경도 인덱스 설정
        for i, item2 in data.iterrows():
            coord = cartesian(item2['y'], item2['x'])
            closest = tree.query([coord], k=1)

            data._set_value(i, 'dist', closest[0][0])
            data._set_value(i, 'idx', closest[1][0])

        dataL1 = pd.merge(data, eleData, how="left", on="idx")

        fileKeyPattern = '{}_{:.4f}-{:.4f}_{:.4f}-{:.4f}_{}-{}'.format(sysOpt['id'], minLon, maxLon, minLat, maxLat,
                                                                       nxGridInv, nyGridInv)

        # 1 단계 : 격자 내 원시 관측값
        plot = (
                ggplot(dataL1, aes(x='i', y='j', fill='logSFV'))
                + geom_tile(aes(width=1.0, height=1.0))
                + labs(fill = 'logSFV\n\n', title=fileKeyPattern)
                + scale_fill_cmap(limits=[0, 1])
        )

        saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '1단계', fileKeyPattern)
        plot.save(saveImg, width=10, height=10, dpi=600)
        log.info('[CHECK] saveImg : {}'.format(saveImg))

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '1단계', fileKeyPattern)
        dataL1.to_csv(saveCsvFile, index=False)
        log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        # 2 단계 : 단일 격자 평균값
        dataL2 = dataL1.groupby(['i', 'j'])[dataL1.columns].mean()

        plot = (
                ggplot(dataL2, aes(x='i', y='j', fill='logSFV'))
                + geom_tile(aes(width=1.0, height=1.0))
                + labs(fill='logSFV\n\n', title=fileKeyPattern)
                + scale_fill_cmap(limits=[0, 1])
        )

        saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '2단계', fileKeyPattern)
        plot.save(saveImg, width=10, height=10, dpi=600)
        log.info('[CHECK] saveImg : {}'.format(saveImg))

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '2단계', fileKeyPattern)
        dataL2.to_csv(saveCsvFile, index=False)
        log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        # 3 단계 : 단일 격자에서 NA를 3x3 격자 평균값으로 대체
        # i = 15
        # j = 6

        dataL3 = pd.DataFrame()

        for i in range(0, len(lonList)):
            for j in range(0, len(latList)):
                iSel = i
                jSel = j

                selData = dataL2.loc[
                    ((dataL2['i'] == iSel) & (dataL2['j'] == jSel))
                ]

                # if (len(selData.index) > 0): continue

                # print(iSel, jSel, selData['logSFV'].values)

                # iSel, jSel를 기준으로 평균 수행
                areaData = pd.DataFrame()

                for ii in range((iSel - 1), (iSel + 1) + 1):
                    for jj in range((jSel - 1), (jSel + 1) + 1):

                        if (ii < 0) or (jj < 0): continue
                        if (jj >= len(latList)) | (ii >= len(lonList)): continue

                        getData = dataL2.loc[
                            ((dataL2['i'] == ii) & (dataL2['j'] == jj))
                        ]

                        if (len(getData.index) < 1): continue

                        areaData = areaData.append(getData)

                if (len(areaData.index) < 1): continue

                meanDict = areaData.mean().to_dict()

                # 전처 컬럼 정보가 없을 시 NA값 채우기
                if (len(selData.index) < 1):
                    selData = selData.append(meanDict, ignore_index=True)

                # 일부 컬럼 정보가 없을 시 NA값 채우기
                selData = selData.fillna(meanDict)

                selData.insert(0, 'iSel', iSel)
                selData.insert(1, 'jSel', jSel)
                selData.insert(2, 'lonSel', lonList[iSel])
                selData.insert(3, 'latSel', latList[jSel])

                dataL3 = dataL3.append(selData)

                # dict = {
                #     'iSel': iSel
                #     , 'jSel': jSel
                #     , 'lonSel': [lonList[iSel]]
                #     , 'latSel': [latList[jSel]]
                # }
                #
                # for key, val in meanData.items():
                #     dict.update( { key : [val] } )
                #
                # dataL3 = dataL3.append(pd.DataFrame.from_dict(dict))

        dataL4 = dataL3.groupby(['iSel', 'jSel'])[dataL3.columns].mean()
        # dataL4 = dataL3.groupby(['i', 'j'])[dataL3.columns].mean()

        plot = (
                ggplot(dataL4, aes(x='iSel', y='jSel', fill='logSFV'))
                + geom_tile(aes(width=1.0, height=1.0))
                + labs(fill='logSFV\n\n', title=fileKeyPattern)
                + scale_fill_cmap(limits=[0, 1])
        )

        saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '3단계', fileKeyPattern)
        plot.save(saveImg, width=10, height=10, dpi=600)
        log.info('[CHECK] saveImg : {}'.format(saveImg))

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '3단계', fileKeyPattern)
        dataL4.to_csv(saveCsvFile, index=False)
        log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        # dataL5 = dataL4
        dataL5 = pd.DataFrame()

        for i in range(0, len(lonList)):
            for j in range(0, len(latList)):
                iSel = i
                jSel = j

                selData = dataL3.loc[
                    ((dataL3['iSel'] == iSel) & (dataL3['jSel'] == jSel))
                ]

                # if (len(selData.index) > 0): continue

                # iSel, jSel를 기준으로 평균 수행
                areaData = pd.DataFrame()

                for ii in range((iSel - 1), (iSel + 1) + 1):
                    for jj in range((jSel - 1), (jSel + 1) + 1):

                        if (ii < 0) or (jj < 0): continue
                        if (jj >= len(latList)) | (ii >= len(lonList)): continue

                        getData = dataL3.loc[
                            ((dataL3['iSel'] == ii) & (dataL3['jSel'] == jj))
                        ]

                        if (len(getData.index) < 1): continue

                        areaData = areaData.append(getData)

                if (len(areaData.index) < 1): continue

                meanDict = areaData.mean().to_dict()

                # 전처 컬럼 정보가 없을 시 NA값 채우기
                if (len(selData.index) < 1):
                    selData = selData.append(meanDict, ignore_index=True)

                # 일부 컬럼 정보가 없을 시 NA값 채우기
                selData = selData.fillna(meanDict)

                selData.insert(0, 'iSelL2', iSel)
                selData.insert(1, 'jSelL2', jSel)
                selData.insert(2, 'lonSelL2', lonList[iSel])
                selData.insert(3, 'latSelL2', latList[jSel])

                dataL5 = dataL5.append(selData)

        # dataL6 = dataL5.groupby(['iSel', 'jSel'])[dataL3.columns].mean()
        # dataL6 = dataL5.groupby(['iSelL2', 'jSelL2'])[dataL3.columns].mean()

        plot = (
                ggplot(dataL5, aes(x='iSelL2', y='jSelL2', fill='logSFV'))
                + geom_tile(aes(width=1.0, height=1.0))
                + labs(fill='logSFV\n\n', title=fileKeyPattern)
                + scale_fill_cmap(limits=[0, 1])
        )

        saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '4단계', fileKeyPattern)
        plot.save(saveImg, width=10, height=10, dpi=600)
        log.info('[CHECK] saveImg : {}'.format(saveImg))

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '4단계', fileKeyPattern)
        dataL5.to_csv(saveCsvFile, index=False)
        log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        result = { 'msg': 'succ' }
        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeGridPlot'))


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 불규칙 격자를 3단계 규칙격자 수행

    # 1. 2 m x2 m 규칙 격자 생성
    # 2. 격자 내에서 NA 결측 시 평균값으로 대체
    # 3. 시각화

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0243'

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

            # ********************************************
            # 옵션 설정
            # ********************************************
            # 초기 옵션 설정
            sysOpt = {
                # 해상도 설정 : 2 m
                'gridInv': 1

                # id 설정
                , 'id': None

                # 최대/최소 위경도
                , 'minLon': None
                , 'maxLon': None
                , 'minLat': None
                , 'maxLat': None

                # 가로/세로 크기 [m]
                # , 'nxMeter': 94.35
                # , 'nyMeter': 155.56
            }

            inpXlsxFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '샘플농지정리.xlsx')
            fileInfo = glob.glob(inpXlsxFile)

            if fileInfo is None or len(fileInfo) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpXlsxFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpXlsxFile, '입력 자료를 확인해주세요.'))

            xlsxData = pd.read_excel(fileInfo[0], sheet_name='Sheet1')

            for i, item in xlsxData.iterrows():

                sysOpt['id'] = item['id']
                sysOpt['maxLon'] = item.filter(regex='x', axis=0).max()
                sysOpt['minLon'] = item.filter(regex='x', axis=0).min()
                sysOpt['maxLat'] = item.filter(regex='y', axis=0).max()
                sysOpt['minLat'] = item.filter(regex='y', axis=0).min()

                log.info("[CHECK] id : {}".format(sysOpt['id']))

                # 파일 정보
                fileNamePattern = '2021_ohtani_田植え_{}_raw'.format(sysOpt['id'])
                inpFile = '{}/{}/{}.csv'.format(globalVar['inpPath'], serviceName, fileNamePattern)

                result = makeGridPlot(sysOpt, inpFile)
                log.info('[CHECK] result : {}'.format(result))

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
