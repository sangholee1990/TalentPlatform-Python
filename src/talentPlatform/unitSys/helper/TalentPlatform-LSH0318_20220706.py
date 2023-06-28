# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import logging.handlers
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
import xarray as xr

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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 앙상블 모델 구축, 과거대비 변화율 산정, 연 강수량 및 평균온도 계산

    # 새롭게 문의드릴 내용은 다음과 같습니다.
    # 1) 가중치를 적용하여 각 모델의 어셈블 모델을 구축하려고합니다!(미래, 과거 데이터 다 확보하였고 가중치는 제가 직접계산하여 파일로 적용할 예정입니다.)
    # 2.) 과거기간대비 변화율을 산정하려고합니다. (1980년-2014년) 미래 (2031-2060)/ (2071-2100) 입니다.
    # (미래 - 현재)/현재 * 100
    # 3) 연 총 강수량과 연 평균 온도를 구하려고합니다.

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0318'

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
                    'srtDate': '2020-09-01'
                    , 'endDate': '2020-09-03'

                    # 경도 최소/최대/간격
                    , 'lonMin': 0
                    , 'lonMax': 360
                    , 'lonInv': 0.5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.5
                }
            else:
                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                }


            # ************************************************************************************************
            # 입력 자료 (가중치, M1, M2, M3) 읽기
            # ************************************************************************************************
            # weight 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Topsis Weight.csv')
            fileList = sorted(glob.glob(inpFile))

            weiData = pd.read_csv(fileList[0], index_col=False)
            # weiData.loc[0, 'M1']
            # weiData.loc[0, 'M2']
            # weiData.loc[0, 'M3']

            # M1 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ACCESS-CM2 ssp126 2015-2100_pr.nc')
            fileList = sorted(glob.glob(inpFile))
            m1Data = xr.open_dataset(fileList[0]).rename({ 'pr' : 'M1' })
            # m1DataL1 = m1Data.copy().sel(lat = 0, lon = 180)
            m1DataL1 = m1Data.copy()
            # 날짜 변환 (연-월을 기준)
            m1DataL1['time'] = pd.to_datetime(pd.to_datetime(m1DataL1['time'].values).strftime("%Y-%m"), format='%Y-%m')

            # M2 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ACCESS-ESM1-5 ssp126 2015-2100_pr.nc')
            fileList = sorted(glob.glob(inpFile))
            m2Data = xr.open_dataset(fileList[0]).rename({ 'pr' : 'M2' })
            # m2DataL1 = m2Data.copy().sel(lat=0, lon=180)
            m2DataL1 = m2Data.copy()
            # 날짜 변환 (연-월을 기준)
            m2DataL1['time'] = pd.to_datetime(pd.to_datetime(m2DataL1['time'].values).strftime("%Y-%m"), format='%Y-%m')

            # M3 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'BCC-CSM2-MR ssp126 2015-2100_pr.nc')
            fileList = sorted(glob.glob(inpFile))
            m3Data = xr.open_dataset(fileList[0]).rename({ 'pr' : 'M3' })
            # m3DataL1 = m3Data.copy().sel(lat=0, lon=180)
            m3DataL1 = m3Data.copy()
            # 날짜 변환 (연-월을 기준)
            m3DataL1['time'] = pd.to_datetime(pd.to_datetime(m3DataL1['time'].values).strftime("%Y-%m"), format='%Y-%m')

            # 2015-01-01T00:00:00.000000000
            # m1DataL1['time'].values

            # 2015-01-01T00:00:00.000000000
            # m2DataL1['time'].values

            # 2015-01-16T12:00:00.000000000
            # m3DataL1['time'].values

            # 자료 병합
            data = xr.merge([m1DataL1['M1'], m2DataL1['M2'], m3DataL1['M3']])


            # ************************************************************************************************
            # 1) 가중치를 적용하여 각 모델의 어셈블 모델을 구축하려고합니다!(미래, 과거 데이터 다 확보하였고
            # 가중치는 제가 직접계산하여 파일로 적용할 예정입니다.)
            # ************************************************************************************************
            # 앙상블 강수
            data['pr'] = (data['M1'] * weiData.loc[0, 'M1']) + (data['M2'] * weiData.loc[0, 'M2'])+ (data['M3'] * weiData.loc[0, 'M3'])


            # ************************************************************************************************
            # 2) 과거기간대비 변화율을 산정하려고합니다. (1980년-2014년) 미래 (2031-2065)/ (2066-2100) 입니다.
            # 변화율 = (미래 - 현재)/현재 * 100
            # 현재 데이터 없음
            # ************************************************************************************************
            # 현재 (1980년-2014년) : 자료 없음
            # nowData = data.sel(time = slice('1980-01', '2014-12'))

            # 임시 (2015년-2030년)
            nowData = data.sel(time = slice('2015-01', '2030-12'))

            # 위경도에 따른 연별 합계
            # nowSum2D = nowData['pr'].groupby('time.year').sum(skipna = True)

            # 위경도에 따른 평균 수행
            # nowMean2D = nowSum2D.mean(dim = ['year'], skipna = True)

            # 평균 수행
            nowMean = nowData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 현재 여름 (6, 7, 8)
            nowSumerData = nowData.sel(time = nowData.time.dt.month.isin([6, 7, 8]))
            nowSumerMean = nowSumerData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 현재 겨울 (12, 1, 2)
            nowWntrData = nowData.sel(time = nowData.time.dt.month.isin([1, 2, 12]))
            nowWntrMean = nowWntrData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 미래 (2031-2065)
            nextData = data.sel(time = slice('2031-01', '2065-12'))
            nextMean = nextData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 미래 여름 (6, 7, 8)
            nextSumerData = nextData.sel(time=nextData.time.dt.month.isin([6, 7, 8]))
            nextSumerMean = nextSumerData['pr'].groupby('time.year').sum(skipna=True).mean(skipna=True)

            # 미래 겨울 (12, 1, 2)
            nextWntrData = nextData.sel(time=nextData.time.dt.month.isin([1, 2, 12]))
            nextWntrMean = nextWntrData['pr'].groupby('time.year').sum(skipna=True).mean(skipna=True)

            # 먼 미래 (2066-2100)
            futData = data.sel(time = slice('2066-01', '2100-12'))
            futMean = futData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 먼 미래 여름 (6, 7, 8)
            futSumerData = futData.sel(time=nextData.time.dt.month.isin([6, 7, 8]))
            futSumerMean = futSumerData['pr'].groupby('time.year').sum(skipna=True).mean(skipna=True)

            # 먼 미래 겨울 (12, 1, 2)
            futWntrData = futData.sel(time=nextData.time.dt.month.isin([1, 2, 12]))
            futWntrMean = futWntrData['pr'].groupby('time.year').sum(skipna=True).mean(skipna=True)

            dict = {
                'nowMean': [nowMean.values]
                , 'nextMean': [nextMean.values]
                , 'futMean': [futMean.values]
                , 'nowSumerMean': [nowSumerMean.values]
                , 'nextSumerMean': [nextSumerMean.values]
                , 'futSumerMean': [futSumerMean.values]
                , 'nowWntrMean': [nowWntrMean.values]
                , 'nextWntrMean': [nextWntrMean.values]
                , 'futWntrMean': [futWntrMean.values]
            }

            # 현재, 미래, 먼 미래에 대한 통계 데이터
            resData = pd.DataFrame.from_dict(dict)

            # 변화율 계산
            resData['A-P1'] = ( (nextMean.values - nowMean.values) / nowMean.values ) * 100.0
            resData['S-P1'] = ( (nextSumerMean.values - nowSumerMean.values) / nowSumerMean.values ) * 100.0
            resData['W-P1'] = ( (nextWntrMean.values - nowWntrMean.values) / nowWntrMean.values ) * 100.0

            resData['A-P3'] = ( (futMean.values - nowMean.values) / nowMean.values ) * 100.0
            resData['S-P3'] = ( (futSumerMean.values - nowSumerMean.values) / nowSumerMean.values ) * 100.0
            resData['W-P3'] = ( (futWntrMean.values - nowWntrMean.values) / nowWntrMean.values ) * 100.0

            saveFile = '{}/{}_{}'.format(globalVar['outPath'], serviceName, 'Rate_Change.xlsx')
            resData.to_excel(saveFile, index=False)

            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # ************************************************************************************************
            # 3) 연 총 강수량과 연 평균 온도를 구하려고합니다.
            # ************************************************************************************************
            # 연도별 총 강수량
            sumData = data.groupby('time.year').sum()

            # NetCDF 파일 저장
            saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'sum_weg_pr')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            sumData.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 연도별 시각화
            yearList = sumData['year'].values
            # yearInfo =  yearList[0]
            for i, yearInfo in enumerate(yearList):
                log.info("[CHECK] yearInfo : {}".format(yearInfo))

                # 연도별 자료 추출
                sumDataL1 = sumData.sel(year = yearInfo)

                saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, yearInfo, 'sum_weg_pr')

                sumDataL1['pr'].plot()
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent = True)
                plt.show()
                plt.close()

                log.info('[CHECK] saveImg : {}'.format(saveImg))

            # 연도별 평균 온도
            # meanData = data.groupby('time.year').mean()

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
