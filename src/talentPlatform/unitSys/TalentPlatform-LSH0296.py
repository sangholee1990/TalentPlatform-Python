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
import HydroErr as he
import skill_metrics as sm
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import mean_gamma_deviance
# from pymcdm import methods as mcdm_methods
# from pymcdm import weights as mcdm_weights
# from pymcdm import normalizations as norm
# from pymcdm import correlations as corr
# from pymcdm.helpers import rankdata, rrankdata

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

        log.info("[CHECK] {} / val : {}".format(key, val))

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
    # R을 이용한 NetCDF 파일 비교 및 검증스코어 계산

    # 세부내용은 다음과 같습니다!

    # 1.NC파일 형태에서 OBS는 육지만 데이터를 보유하고있습니다.
    # 따라서 바다부분은 데이터가 NA 값입니다!
    # 따라서 Model NC파일과 OBS NC 파일을 비교해주시면 감사합니다 (격자별 비교입니다!).

    # 2. 평가지표는 대도록이면 많이 사용하려고합니다.
    # 약 (20개 정도) 여기 평가지표에 기후 인덱스를 몇개 추가하려고합니다.
    # (연간 총 강수량, 월 최대 강수량, 월 최소 강수량, 월 최대 온도, 월 최소 온도)를 제외하고 나머지 16개는 평가지표( 예시 RMSE)를 사용하려고합니다.
    # 여기 평가지표를 정리한 엑셀파일을 보내드리겠습니다!

    # 3. 평가지표의 모든 값을 CSV파일로 저장해주시고,
    # 이후 mcdm 패키지에 있는 TOPSIS 방법을 이용해서 계산해주시고 저장해주시면됩니다!
    # 여기서 우선순위를 선정하는 기준은 모델 격자의 평가지표 결과의 평균 입니다!

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0296'

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
                    'srtDate': '1950-01-01'
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

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_Amon_MRI-ESM2-0_ssp585_r1i1p1f1_gn_201501-210012.nc')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'OBS GPCC 1891-2016 precipMonTotal.nc')
            # E:\Global climate models\Historical
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'OBS GPCC 1891-2016 precipMonTotal.nc')
            # globalVar['inpPath'] = 'E:/Global climate models/Historical'
            # inpFile = '{}/{}'.format(globalVar['inpPath'], 'OBS GPCC 1891-2016.nc')
            fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[0]
            obsData = xr.open_mfdataset(fileList)

            # 경도 변환 (0~360 to -180~180) 확인필요
            # obsData.coords['lon'] = (obsData.coords['lon'] + 180) % 360 - 180
            # obsData = obsData.sortby(obsData.lon)

            # 경도 변환 (-180~180 to 0~360)
            # obsData.coords['lon'] = (obsData.coords['lon'] + 180) % 360 - 180
            # obsData = obsData.sortby(obsData.lon)

            obsTimeList = pd.to_datetime(obsData['time'].values).strftime('%Y%m')
            obsLonList = obsData['lon'].values
            obsLatList = obsData['lat'].values


            inpFile = '{}/ACCESS-CM2/{}'.format(globalVar['inpPath'], '*.nc')
            # inpFile = '{}/*/{}'.format(globalVar['inpPath'], '*.nc')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'INM-CM4-8 historical*.nc')
            fileList = sorted(glob.glob(inpFile))

            for j, fileInfo in enumerate(fileList):
                key = fileInfo.split('\\')[1]
                log.info('[CHECK] key : {}'.format(key))

                if (key == 'MCM-UA-1-0'): continue

                # fileInfo = fileList[0]
                modelData = xr.open_mfdataset(fileInfo)

                # modelTimeList[0]
                # 1안
                # modelTimeList = modelData['time'].values
                modelTimeList = pd.to_datetime(modelData['time'].values).strftime('%Y%m')

                # 2안
                # modelTimeList = modelData.indexes['time'].to_datetimeindex().to_numpy()
                # modelData['time'] = modelTimeList

                modelLonList = modelData['lon'].values
                modelLatList = modelData['lat'].values

                # 시공간 일치 및 중복 제거
                # 서로간의 시간대 안 맞음
                timeList = sorted(list(set(obsTimeList) & set(modelTimeList)))
                #timeList = sorted(list(set(obsTimeList) | set(modelTimeList)))
                lonList = sorted(list(set(obsLonList) & set(modelLonList)))
                latList = sorted(list(set(obsLatList) & set(modelLatList)))

                log.info('[CHECK] len(timeList) : {}'.format(len(timeList)))
                log.info('[CHECK] len(lonList) : {}'.format(len(lonList)))
                log.info('[CHECK] len(latList) : {}'.format(len(latList)))

                selTimeList = []
                for i, timeInfo in enumerate(timeList):
                    dtTimeInfo = pd.to_datetime(timeInfo, format='%Y%m')
                    if (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d') > dtTimeInfo): continue
                    selTimeList.append(dtTimeInfo)

                if (len(selTimeList) < 1): continue

                #obsDataL1 = obsData.sel(time=selTimeList, lat=latList, lon=lonList)
                obsDataL1 = obsData.interp(time=selTimeList, lat=latList, lon=lonList, method='nearest', kwargs={'fill_value': 'extrapolate'})
                # 1안
                #modelDataL1 = modelData.sel(time=selTimeList, lat=latList, lon=lonList)
                modelDataL1 = modelData.interp(time=selTimeList, lat=latList, lon=lonList, method='nearest', kwargs={'fill_value': 'extrapolate'})
                # 2안
                # modelDataL1 = modelData.sel(time=selTimeList, lat=latList, lon=lonList)

                # 단위 변환
                # obsDataL2 = obsDataL1['precip'] / 2592000.0
                obsDataL2 = obsDataL1['precip']
                modelDataL2 = modelDataL1['pr']

                # 임시파일
                # tmpData = obsData.copy().sel(time = timeList[5])
                # tmpDataL1 = tmpData
                # tmpDataL1['precip'] = tmpData['precip'] / 2592000.0
                # # 0보다 같거나 큰 값을 사용
                # # 반면에 0보다 작으면 모두 0으로 사용
                # tmpDataL2 = tmpDataL1.where(tmpDataL1.precip >= 0, 0)

                saveData = xr.Dataset().merge(obsDataL2)\
                    .merge(modelDataL2)

                uniqTimeList = pd.to_datetime(saveData['time'].values).strftime('%Y-%m')
                uniqLonList = saveData['lon'].values
                uniqLatList = saveData['lat'].values

                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], key, serviceName, 'unique_time')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                pd.DataFrame({'time': uniqTimeList}).to_csv(saveFile, index=False)

                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], key, serviceName, 'unique_lon')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                pd.DataFrame({'lon': uniqLonList}).to_csv(saveFile, index=False)

                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], key, serviceName, 'unique_lat')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                pd.DataFrame( {'lat' : uniqLatList } ).to_csv(saveFile, index=False)

                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], key, serviceName, 'save_data')
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                saveData.to_dataframe().reset_index().dropna().to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 포트란 소스코드 대체
            # data = pd.DataFrame()
            # for i, lonInfo in enumerate(lonList):
            #     for j, latInfo in enumerate(latList):
            #
            #         # 테스트
            #         # if (i > 0): continue
            #         # if (j > 2): continue
            #
            #         log.info('[CHECK] lonInfo : {}, latInfo : {}'.format(lonInfo, latInfo))
            #
            #         obsDataL3 = obsDataL2.sel(lat=latInfo, lon=lonInfo)
            #         modelDataL3 = modelDataL2.sel(lat=latInfo, lon=lonInfo)
            #
            #         # 사장님 혹시 평가지표 결과를 저위도 (0-29), 중위도 (30-59) 고위도 (60-90)로 구분해서 저장해주실수 있으실까요?
            #         type = np.nan
            #         if 0 <= latInfo < 30:
            #             type = 'low'
            #         elif 30 <= latInfo < 60:
            #             type = 'middle'
            #         elif 60 <= latInfo < 91:
            #             type = 'high'
            #         elif -30 <= latInfo < 0:
            #             type = '-low'
            #         elif -60 <= latInfo < -30:
            #             type = '-middle'
            #         elif -91 <= latInfo < -60:
            #             type = '-high'
            #
            #         dict = {
            #             'lat': [latInfo]
            #             , 'lon': [lonInfo]
            #             , 'type': [type]
            #             , 'ME': [he.me(obsDataL3, modelDataL3)]
            #             , 'MAE': [he.mae(obsDataL3, modelDataL3)]
            #             , 'MSE': [he.mse(obsDataL3, modelDataL3)]
            #             , 'MDE': [he.mde(obsDataL3, modelDataL3)]
            #             , 'MDAE': [he.mdae(obsDataL3, modelDataL3)]
            #             , 'MDSE': [he.mdse(obsDataL3, modelDataL3)]
            #             , 'ED': [he.ed(obsDataL3, modelDataL3)]
            #             , 'RMSE': [he.rmse(obsDataL3, modelDataL3)]
            #             , 'NRMSE': [he.rmsle(obsDataL3, modelDataL3)]
            #             , 'IRMSE': [he.irmse(obsDataL3, modelDataL3)]
            #             , 'MASE': [he.mase(obsDataL3, modelDataL3)]
            #             , 'R-SQUARED': [he.r_squared(obsDataL3, modelDataL3)]
            #             , 'Pearson_r': [he.pearson_r(obsDataL3, modelDataL3)]
            #             , 'ACC': [he.acc(obsDataL3, modelDataL3)]
            #             , 'MAPE': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else he.mape(obsDataL3, modelDataL3)]
            #             , 'MAPD': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else he.mapd(obsDataL3, modelDataL3)]
            #             , 'Dmod': [he.dmod(obsDataL3, modelDataL3)]
            #             , 'drel': [he.drel(obsDataL3, modelDataL3)]
            #             , 'dr': [he.dr(obsDataL3, modelDataL3)]
            #             , 'NSE': [he.nse(obsDataL3, modelDataL3)]
            #             , 'KGE_2012': [he.kge_2012(obsDataL3, modelDataL3)]
            #             , 'lm_index': [he.lm_index(obsDataL3, modelDataL3)]
            #             , 've': [he.ve(obsDataL3, modelDataL3)]
            #             , 'sa': [he.sa(obsDataL3, modelDataL3)]
            #             , 'sc': [he.sc(obsDataL3, modelDataL3)]
            #             , 'sid': [he.sid(obsDataL3, modelDataL3)]
            #             , 'skill': [sm.skill_score_murphy(obsDataL3.values, modelDataL3.values)]
            #             , 'adjust': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else adjusted_mutual_info_score(obsDataL3.values, modelDataL3.values)]
            #             , 'homogeneity': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else homogeneity_score(obsDataL3.values, modelDataL3.values)]
            #             , 'complete': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else completeness_score(obsDataL3.values, modelDataL3.values)]
            #             # , 'gamma': [np.nan if (pd.isna(he.mae(obsDataL3, modelDataL3))) else mean_gamma_deviance(obsDataL3.values, modelDataL3.values)]
            #         }
            #
            #         data = pd.concat([data, pd.DataFrame.from_dict(dict)], axis=0)
            #
            # saveFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'lon_lat_INM-CM4-8')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # data.to_csv(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # *******************************************************************************
            # 격자별 검증스코어를 통해 엔트로피 계산
            # *******************************************************************************
            # matrix = np.random.rand(5, 10) * 10
            # # 엔트로피
            # weights = mcdm_weights.entropy_weights(matrix)
            # i, j = matrix.shape
            # maxEle = i if (i > j) else j
            # types = np.repeat(3, maxEle)
            #
            # topsis = mcdm_methods.TOPSIS()
            # topRes = topsis(matrix, weights, types)
            # meanRes = np.nanmean(topRes)

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
