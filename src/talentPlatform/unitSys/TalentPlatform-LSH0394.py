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

import googlemaps
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import spatial

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

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        plt.rcParams['font.family'] = fontName

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

def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 탄소모니터링 도시를 기준으로 1,2순위 온도 도시 매칭 (구글 위경도 지오코딩 활용+다양한 국가)

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0394'

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
                    # 구글 API키
                    'googleApiKey' : 'AIzaSyCkYokUFIcH5OYDaYU0IrFLX89wX1o7-qc'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 구글 API키
                    'googleApiKey': 'AIzaSyCkYokUFIcH5OYDaYU0IrFLX89wX1o7-qc'
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 구글 API 설정
            gmaps = googlemaps.Client(key=sysOpt['googleApiKey'])

            # ********************************************************************
            # 탄소 모니터링 및 온도 파일 패턴 검색
            # ********************************************************************
            # carbon 파일 : hungary 없음
            inpFile = '{}/{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'carbon', '*.csv')

            # 파일 검사
            fileList = sorted(glob.glob(inpFile))

            optData = pd.DataFrame()
            for i, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo : {fileInfo}')
                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
                key = fileNameNoExt.split('-')[3].lower()

                inpTempFile = '{}/{}/{}/daily_temperature_{}.csv'.format(globalVar['inpPath'], serviceName, 'temp', key)
                fileTempList = glob.glob(inpTempFile)

                dict = {
                    'key' : [key]
                    , 'carbonInfo' : [fileInfo]
                    , 'tempInfo' : [fileTempList[0] if len(fileTempList) > 0 else np.nan]
                }

                optData = pd.concat([optData, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)

            optDataL1 = optData.dropna().reset_index(drop=True)

            # ********************************************************************
            # 탄소모니터링 도시를 기준으로 1,2순위 온도 도시 매칭
            # ********************************************************************
            for i, optInfo in optDataL1.iterrows():
                log.info(f'[CHECK] optInfo.key : {optInfo.key}')

                # 파일 읽기
                data = pd.read_csv(optInfo.carbonInfo)

                # 2019년 1월 1일 날짜 추출
                dataL1 = data.query('date == "2019-01-01"')

                # city, sector를 기준으로 개수 계산
                # 단일 개수일 경우만 추출
                dataL2 = dataL1.groupby(['city', 'sector']).count()[['country']].query('country == 1').reset_index(drop=False)

                # 미국을 대상으로 USA 키워드 추가
                # dataL2['addr'] = dataL2['city'] + ', USA'
                # dataL2['addr'] = dataL2['city'] + ', United States of America'
                dataL2['addr'] =  dataL2['city'] + ', ' +  optInfo.key

                # ********************************************************************
                # 탄소 모니터링 파일을 기준으로 구글 위경도 환산
                # ********************************************************************
                # 중복없는 주소 목록
                addrList = sorted(set(dataL2['addr']))

                matData = pd.DataFrame()
                for i, addrInfo in enumerate(addrList):
                    log.info(f'[CHECK] addrInfo : {addrInfo}')

                    # 초기값 설정
                    matData.loc[i, 'addr'] = addrInfo
                    matData.loc[i, 'glat'] = None
                    matData.loc[i, 'glon'] = None

                    try:
                        rtnGeo = gmaps.geocode(addrInfo, language='en')
                        if (len(rtnGeo) < 1): continue

                        # 위/경도 반환
                        matData.loc[i, 'glat'] = rtnGeo[0]['geometry']['location']['lat']
                        matData.loc[i, 'glon'] = rtnGeo[0]['geometry']['location']['lng']

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                # 파일 저장
                saveFile = '{}/{}/{}-{}.csv'.format(globalVar['outPath'], serviceName, 'matData', optInfo.key)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                matData.to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

                # 미리 위경도 환산된 파일 읽기
                matData = pd.read_csv(saveFile)
                # matData['city'] = matData['addr'].str.replace(', USA', '')
                # matData['city'] = matData['addr'].str.replace(', United States of America', '')
                matData['city'] = matData['addr'].str.replace(', ' + optInfo.key, '')
                matData['cityLower'] = matData['city'].str.lower()

                # ********************************************************************
                # 온도 파일 읽기
                # ********************************************************************
                # 파일 읽기
                tasData = pd.read_csv(optInfo.tempInfo)

                # 2019년 1월 1일 날짜 추출
                tasDataL1 = tasData.query('date == 20190101')

                selColName = tasDataL1.columns[tasDataL1.columns.str.contains('NAME')][0]

                # selColName를 기준으로 개수 계산
                # tasDataL2 = tasDataL1.groupby(['NAME_2']).count()[['mean']].query('mean == 1').reset_index(drop=False)
                # tasDataL2 = tasDataL1.groupby(['NAME_3']).count()[['mean']].query('mean == 1').reset_index(drop=False)
                tasDataL2 = tasDataL1.groupby([selColName]).count()[['mean']].query('mean == 1').reset_index(drop=False)

                # selColName를을 대상으로 optInfo.key 키워드 추가
                # tasDataL2['addr'] = tasDataL2['NAME_2'] + ', USA'
                # tasDataL2['addr'] = tasDataL2['NAME_2'] + ', United States of America'
                tasDataL2['addr'] = tasDataL2[selColName] + ', ' + optInfo.key

                # ********************************************************************
                # 온도 파일을 기준으로 구글 위경도 환산
                # ********************************************************************
                # 중복없는 주소 목록
                addrList = sorted(set(tasDataL2['addr']))

                matTasData = pd.DataFrame()
                for i, addrInfo in enumerate(addrList):
                    log.info(f'[CHECK] addrInfo : {addrInfo}')

                    # 초기값 설정
                    matTasData.loc[i, 'addr'] = addrInfo
                    matTasData.loc[i, 'glat'] = None
                    matTasData.loc[i, 'glon'] = None

                    try:
                        rtnGeo = gmaps.geocode(addrInfo, language='en')
                        if (len(rtnGeo) < 1): continue

                        # 위/경도 반환
                        matTasData.loc[i, 'glat'] = rtnGeo[0]['geometry']['location']['lat']
                        matTasData.loc[i, 'glon'] = rtnGeo[0]['geometry']['location']['lng']

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                # 파일 저장
                saveFile = '{}/{}/{}-{}.csv'.format(globalVar['outPath'], serviceName, 'matTasData', optInfo.key)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                matTasData.to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

                # 미리 위경도 환산된 파일 읽기
                matTasData = pd.read_csv(saveFile)
                # matTasData['city'] = matTasData['addr'].str.replace(', USA', '')
                # matTasData['city'] = matTasData['addr'].str.replace(', United States of America', '')
                matTasData['city'] = matTasData['addr'].str.replace(', ' + optInfo.key, '')
                matTasData['cityLower'] = matTasData['city'].str.lower()

                # ****************************************************************************
                # 탄소 모니터링 기준으로 최근접 화소 찾기
                # ****************************************************************************
                # kdTree를 위한 초기 데이터 (온도 데이터)
                posList = []
                for idx in range(0, len(matTasData)):
                    coord = [matTasData.loc[idx, 'glat'], matTasData.loc[idx, 'glon']]
                    posList.append(cartesian(*coord))

                tree = spatial.KDTree(posList)

                # 탄소 모니터링 데이터에서 반복문 수행
                matDataL1 = matData
                for i, posInfo in matDataL1.iterrows():
                    # NA 검사
                    if (posInfo.isna()[['glon', 'glat']].any() == True): continue
                    coord = cartesian(posInfo['glat'], posInfo['glon'])

                    # 최근접 화소 개수 (1순위, 2순위)
                    cloCnt = 2
                    closest = tree.query([coord], k=cloCnt)

                    # 순위 별로 반복문 수행
                    for j in range(0, cloCnt):

                        # 거리 계산
                        cloDist = closest[0][0][j]

                        # 인덱스 계산
                        cloIdx = closest[1][0][j]

                        # 인덱스를 통해 데이터 추출
                        cloData = matTasData.iloc[cloIdx]

                        # 각 N순위에 따라 동적 컬럼 추가
                        matDataL1.loc[i, f'tas-dist-{j}'] = cloDist
                        matDataL1.loc[i, f'tas-addr-{j}'] = cloData['addr']
                        matDataL1.loc[i, f'tas-city-{j}'] = cloData['city']
                        matDataL1.loc[i, f'tas-glat-{j}'] = cloData['glat']
                        matDataL1.loc[i, f'tas-glon-{j}'] = cloData['glon']


                # 파일 저장
                # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'matDataL1')
                saveFile = '{}/{}/{}-{}.csv'.format(globalVar['outPath'], serviceName, 'matDataL1', optInfo.key)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                matDataL1.to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

                # ****************************************************************************
                # 공간 일치 데이터 가공
                # ****************************************************************************
                matDataL2 = matDataL1.dropna().reset_index(drop=True)
                matDataL3 = pd.DataFrame()
                for i, matInfo in matDataL2.iterrows():
                    matDataL3.loc[i, 'city'] = matInfo['city']
                    matDataL3.loc[i, 'temperature2'] = '-' if  matInfo['tas-dist-0'] == 0 else matInfo['tas-city-1']
                    matDataL3.loc[i, 'temperature'] = matInfo['tas-city-0']

                # 파일 저장
                saveFile = '{}/{}/{}-{}.csv'.format(globalVar['outPath'], serviceName, 'matDataL3', optInfo.key)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                matDataL3.to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

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
        inParams = { }
        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))