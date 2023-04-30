# -*- coding: utf-8 -*-

import glob
import json
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from collections import Counter
import glob
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import os
import glob
# import googlemaps
import plotly.express as px
import matplotlib as mpl
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pycparser.ply.lex import _form_master_re
from scipy.stats import linregress
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
import matplotlib.dates as mdates
from autots import AutoTS
from scipy.interpolate import interp1d

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
warnings.filterwarnings('ignore')

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font='Malgun Gothic', rc={'axes.unicode_minus': False}, style='darkgrid')

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

    saveLogFile = '{}/{}_{}_{}_{}.log'.format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.node()
        , prjName
        , datetime.now().strftime('%Y%m%d')
    )

    if not os.path.exists(os.path.dirname(saveLogFile)):
        os.makedirs(os.path.dirname(saveLogFile))

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0: return log

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

    log.info('[CHECK] inParInfo : {}'.format(inParInfo))

    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info('[CHECK] {} / val : {}'.format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


# 산점도 시각화
def makeUserScatterPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, isSame):
    # 그리드 설정
    plt.grid(True)

    plt.scatter(prdVal, refVal)

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
    Bias = np.nanmean(prdVal - refVal)
    rBias = (Bias / np.nanmean(refVal)) * 100.0
    RMSE = np.sqrt(np.nanmean((prdVal - refVal) ** 2))
    rRMSE = (RMSE / np.nanmean(refVal)) * 100.0
    MAPE = np.nanmean(np.abs((prdVal - refVal) / prdVal)) * 100.0

    # 결측값 마스킹
    mask = ~np.isnan(refVal)
    N = len(refVal[mask])

    # 선형회귀곡선에 대한 계산
    slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

    lmfit = (slope * prdVal) + intercept
    # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle='-')
    plt.plot([minVal, maxVal], [minVal, maxVal], color='black', linestyle='--', linewidth=2)
    plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle='-')
    # 라벨 추가
    plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept), xy=(minVal + xIntVal, maxVal - yIntVal),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (rVal, pVal), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')

    if (isSame == True):
        # plt.axes().set_aspect('equal')
        # plt.axes().set_aspect(1)
        # plt.gca().set_aspect('equal')
        plt.xlim(minVal, maxVal)
        plt.ylim(minVal, maxVal)

        plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(minVal + xIntVal, maxVal - yIntVal * 3),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(minVal + xIntVal, maxVal - yIntVal * 4),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(minVal + xIntVal, maxVal - yIntVal * 5),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 6), color='black',
                     xycoords='data', horizontalalignment='left', verticalalignment='center')
    else:
        plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 3), color='black',
                     xycoords='data', horizontalalignment='left', verticalalignment='center')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):
    log.info('[START] {}'.format('makeUserTimeSeriesPlot'))

    result = None

    try:
        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        mlRMSE = np.sqrt(np.nanmean((mlPrdVal - refVal) ** 2))
        mlReRMSE = (mlRMSE / np.nanmean(refVal)) * 100.0

        dlRMSE = np.sqrt(np.nanmean((dlPrdVal - refVal) ** 2))
        dlReRMSE = (dlRMSE / np.nanmean(refVal)) * 100.0

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        number = len(refVal[mask])

        # 선형회귀곡선에 대한 계산
        mlSlope, mlInter, mlR, mlPvalue, mlStdErr = linregress(mlPrdVal[mask], refVal[mask])
        dlSlope, dlInter, dlR, dlPvalue, dlStdErr = linregress(dlPrdVal[mask], refVal[mask])

        prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(mlPrdValLabel, mlR, mlReRMSE)
        prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(dlPrdValLabel, dlR, dlReRMSE)
        refValLabel_ref = '{0:s} : N = {1:d}'.format(refValLabel, number)

        plt.grid(True)

        plt.plot(dtDate, mlPrdVal, label=prdValLabel_ml, marker='o')
        plt.plot(dtDate, dlPrdVal, label=prdValLabel_dnn, marker='o')
        plt.plot(dtDate, refVal, label=refValLabel_ref, marker='o')

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        # plt.ylim(0, 1000)

        if (isFore == True):
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=45, ha='right', minor=False)

        else:
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=0, ha='center')

        plt.legend(loc='upper left')

        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserTimeSeriesPlot'))


#
# # 시계열 시각화
# def makeUserTimeSeriesPlot(dtDate, prdVal1, prdVal2, refVal, prdValLabel1, prdValLabel2, refValLabel, xlab, ylab, mainTitle, saveImg):
#     #####################################################################################
#     # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
#     RMSE_ml = np.sqrt(np.mean((prdVal1 - refVal) ** 2))
#     rRMSE_ml = (RMSE_ml / np.mean(refVal)) * 100.0
#
#     RMSE_dnn = np.sqrt(np.mean((prdVal2 - refVal) ** 2))
#     rRMSE_dnn = (RMSE_dnn / np.mean(refVal)) * 100.0
#
#     # 선형회귀곡선에 대한 계산
#     lmFit1 = linregress(prdVal1, refVal)
#     R_ml = lmFit1[2]
#
#     lmFit2 = linregress(prdVal2, refVal)
#     R_dnn = lmFit2[2]
#
#     prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel1, R_ml, rRMSE_ml)
#     prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel2, R_dnn, rRMSE_dnn)
#     #####################################################################################
#
#     plt.grid(True)
#
#     plt.plot(dtDate, prdVal1, label=prdValLabel_ml, marker='o')
#     plt.plot(dtDate, prdVal2, label=prdValLabel_dnn, marker='o')
#     plt.plot(dtDate, refVal, label=refValLabel, marker='o')
#
#     # 제목, x축, y축 설정
#     plt.title(mainTitle)
#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#
#     plt.xticks(rotation=45, ha='right')
#     plt.legend(loc='upper left')
#
#     plt.savefig(saveImg, dpi=600, bbox_inches='tight')
#     plt.show()
#     plt.close()


# 시계열 시각화2
def makeUserTimeSeriesPlot2(dtDate, prdVal1, prdVal2, refVal, dtDate2, afterVal1, afterVal2, prdValLabel1, prdValLabel2, refValLabel, xlab, ylab, mainTitle, saveImg):
    #####################################################################################
    # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
    RMSE_ml = np.sqrt(np.mean((prdVal1 - refVal) ** 2))
    rRMSE_ml = (RMSE_ml / np.mean(refVal)) * 100.0

    RMSE_dnn = np.sqrt(np.mean((prdVal2 - refVal) ** 2))
    rRMSE_dnn = (RMSE_dnn / np.mean(refVal)) * 100.0

    # 선형회귀곡선에 대한 계산
    lmFit1 = linregress(prdVal1, refVal)
    R_ml = lmFit1[2]

    lmFit2 = linregress(prdVal2, refVal)
    R_dnn = lmFit2[2]

    prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel1, R_ml, rRMSE_ml)
    prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel2, R_dnn, rRMSE_dnn)
    #####################################################################################

    plt.grid(True)

    plt.plot(dtDate2, afterVal1, label=prdValLabel_ml, marker='o')
    plt.plot(dtDate2, afterVal2, label=prdValLabel_dnn, marker='o')

    # plt.plot(dtDate2, afterVal1,marker='o')
    # plt.plot(dtDate2, afterVal2,marker='x')

    plt.plot(dtDate, refVal, label=refValLabel, marker='o')

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# 딥러닝 매매가/전세가 예측
def makeDlModel(subOpt=None, xCol=None, yCol=None, inpData=None, modelKey=None):
    log.info('[START] {}'.format('makeDlModel'))

    result = None

    try:

        saveModel = '{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'h2o', 'act', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        inpDataL1 = inpData[xyCol]

        if (not subOpt['isDlModelInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isDlModelInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(inpDataL1, test_size=0.3)
            # trainData = inpData

            # dlModel = H2OAutoML(max_models=30, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainData), validation_frame=h2o.H2OFrame(validData))
            # dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(inpDataL1))
            fnlModel = dlModel.get_best_model()

            # 학습 모델 저장
            saveModel = '{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'h2o', 'act', datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            # h2o.save_model(model=fnlModel, path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
            fnlModel.save_mojo(path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = h2o.import_mojo(saveModel)

        result = {
            'msg': 'succ'
            , 'dlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeDlModel'))


# 머신러닝 매매가/전세가 예측
def makeMlModel(subOpt=None, xCol=None, yCol=None, inpData=None, modelKey=None):
    log.info('[START] {}'.format('makeMlModel'))

    result = None

    try:

        saveModel = '{}/{}-{}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'pycaret', 'act', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        inpDataL1 = inpData[xyCol]

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(inpDataL1, test_size=0.3)
            # trainData = inpData

            pyModel = setup(
                data=trainData
                , session_id=123
                , silent=True
                , target=yCol
            )

            # 각 모형에 따른 자동 머신러닝
            modelList = compare_models(sort='RMSE', n_select=3)

            # 앙상블 모형
            blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(tuneModel)

            # 학습 모형 저장
            saveModel = '{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'pycaret', 'act', datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            save_model(fnlModel, saveModel)

        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = load_model(os.path.splitext(saveModel)[0])

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeMlModel'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 부동산 데이터 분석 및 가격 예측

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0250'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] __init__ : {}'.format('init'))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error('Exception : {}'.format(e))
            raise e
        finally:
            log.info('[END] __init__ : {}'.format('init'))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        try:
            log.info('[START] {}'.format('exec'))

            # breakpoint()

            # ********************************************
            # 옵션 설정
            # ********************************************
            sysOpt = {
                #  딥러닝
                'dlModel': {
                    # 초기화
                    # 'isDlModelInit': False
                    'isDlModelInit': True
                    # 모델 업데이트 여부
                    , 'isOverWrite': True
                    # , 'isOverWrite': False
                }

                #  머신러닝
                , 'mlModel': {
                    # 모델 업데이트 여부
                    # 'isOverWrite': True
                    'isOverWrite': False
                }

                #  시계열
                , 'tsModel': {
                    # 모델 업데이트 여부
                    'isOverWrite': True
                    # 'isOverWrite': False
                    # 아파트 설정
                    , 'aptList': ['미아동부센트레빌(숭인로7가길 37)', '송천센트레빌(숭인로 39)', '에스케이북한산시티(솔샘로 174)']
                }
            }

            # *****************************************************
            # 인허가 데이터
            # *****************************************************
            # 서울특별시 강북구 인허가.csv
            lcnsInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 인허가.csv')
            lcnsFileList = glob.glob(lcnsInpFile)
            if lcnsFileList is None or len(lcnsFileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(lcnsFileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(lcnsFileList, '입력 자료를 확인해주세요.'))

            lcnsData = pd.read_csv(lcnsFileList[0])
            lcnsData.drop(['Unnamed: 0'], axis=1, inplace=True)
            lcnsDataL1 = lcnsData.groupby(['주소'], as_index=False)['archGbCdNm'].count()

            # *****************************************************
            # 전월세 데이터
            # *****************************************************
            # prvsMntsrInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 전월세가_20111101_20201101.csv')
            prvsMntsrInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 전월세가_인허가_20111101_20201101.csv')

            prvsMntsrFileList = glob.glob(prvsMntsrInpFile)
            if prvsMntsrFileList is None or len(prvsMntsrFileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(prvsMntsrFileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(prvsMntsrFileList, '입력 자료를 확인해주세요.'))

            prvsMntsrFileInfo = prvsMntsrFileList[0]
            prvsMntsrData = pd.read_csv(prvsMntsrFileInfo)
            prvsMntsrData.drop(['Unnamed: 0.1'], axis=1, inplace=True)

            prvsMntsrData['name'] = prvsMntsrData['단지명'] + '(' + prvsMntsrData['도로명'] + ')'
            # prvsMntsrDataL1 = prvsMntsrData[['name', '전용면적(㎡)', '보증금(만원)', '층', '건축년도', 'lat', 'lon', '계약년월', '전월세구분']]
            prvsMntsrDataL2 = prvsMntsrData.loc[
                (prvsMntsrData['전월세구분'] == '전세')
                & (prvsMntsrData['층'] != 1)
                ].reset_index(drop=True)

            # prvsMntsrDataL2['계약년도'] = prvsMntsrDataL2['계약년월'].astype(str).str.slice(0, 4)
            prvsMntsrDataL2['date'] = pd.to_datetime(prvsMntsrDataL2['계약년월'], format='%Y%m')
            prvsMntsrDataL2['year'] = prvsMntsrDataL2['date'].dt.strftime("%Y").astype('int')
            prvsMntsrDataL2['month'] = prvsMntsrDataL2['date'].dt.strftime("%m").astype('int')
            prvsMntsrDataL2['보증금(만원)'] = prvsMntsrDataL2['보증금(만원)'].astype(str).str.replace(',', '').astype('float')

            # prvsMntsrDataL3 = prvsMntsrDataL2
            # prvsMntsrDataL3 = prvsMntsrDataL2.groupby(['name', '전용면적(㎡)', '건축년도', 'lat', 'lon', 'year'], as_index=False)['보증금(만원)'].mean()
            # prvsMntsrDataL3 = prvsMntsrDataL2.groupby(['name', '전용면적(㎡)', '건축년도', 'lat', 'lon', 'date', 'year', 'month'], as_index=False)['보증금(만원)'].mean()

            prvsMntsrDataL3 = pd.merge(
                left=prvsMntsrDataL2
                # , right=lcnsData[['archGbCdNm', '주소', 'lat', 'lon']]
                , right=lcnsDataL1[['archGbCdNm', '주소']]
                , left_on=['인허가addr']
                , right_on=['주소']
                , how='left'
            ).rename(
                columns={
                    'archGbCdNm': 'inhuga'
                }
            )

            prvsMntsrDataL4 = prvsMntsrDataL3.rename(
                columns={
                    '전용면적(㎡)': 'capacity'
                    , '건축년도': 'conYear'
                    , '보증금(만원)': 'real_bjprice'
                    , 'archGbCdNm': 'inhuga'
                    , '거래금액(만원)': 'real_price'
                }
            ).sort_values(by=['name', 'capacity', 'year']).reset_index(drop=True)

            # *****************************************************
            # 실거래가 데이터
            # *****************************************************
            realPriceInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 실거래가_인허가_20111101_20201101.csv')

            realPriceFileList = glob.glob(realPriceInpFile)
            if realPriceFileList is None or len(realPriceFileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(realPriceFileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(realPriceFileList, '입력 자료를 확인해주세요.'))

            realPriceFileInfo = realPriceFileList[0]
            realPriceData = pd.read_csv(realPriceFileInfo)
            realPriceData.drop(['Unnamed: 0.1'], axis=1, inplace=True)

            realPriceData['name'] = realPriceData['단지명'] + '(' + realPriceData['도로명'] + ')'
            # realPriceDataL1 = realPriceData[['name','전용면적(㎡)','거래금액(만원)','층','건축년도','lat','lon','계약년월','인허가addr']]
            realPriceDataL2 = realPriceData.loc[
                (realPriceData['층'] != 1)
            ].reset_index(drop=True)

            # pd.to_datetime(prvsMntsrDataL2['계약년월'], format='%Y%m')
            # realPriceDataL2['계약년도'] = realPriceDataL2['계약년월'].astype(str).str.slice(0, 4)
            realPriceDataL2['date'] = pd.to_datetime(realPriceDataL2['계약년월'], format='%Y%m')
            realPriceDataL2['year'] = realPriceDataL2['date'].dt.strftime("%Y").astype('int')
            realPriceDataL2['month'] = realPriceDataL2['date'].dt.strftime("%m").astype('int')
            realPriceDataL2['거래금액(만원)'] = realPriceDataL2['거래금액(만원)'].astype(str).str.replace(',', '').astype('float')

            # realPriceDataL3 = realPriceDataL2.groupby(['name', '전용면적(㎡)', '건축년도', 'lat', 'lon', 'date', 'year', 'month', '인허가addr'], as_index=False)['거래금액(만원)'].mean()
            # realPriceDataL3 = realPriceDataL2.groupby(['name', '전용면적(㎡)', '건축년도', 'lat', 'lon', 'year', '인허가addr'], as_index=False)['거래금액(만원)'].mean()
            # realPriceDataL3 = realPriceDataL2
            realPriceDataL3 = pd.merge(
                left=realPriceDataL2
                # , right=lcnsData[['archGbCdNm', '주소', 'lat', 'lon']]
                , right=lcnsDataL1[['archGbCdNm', '주소']]
                , left_on=['인허가addr']
                , right_on=['주소']
                , how='left'
            ).rename(
                columns={
                    'archGbCdNm': 'inhuga'
                }
            )

            realPriceDataL4 = realPriceDataL3.rename(
                columns={
                    '전용면적(㎡)': 'capacity'
                    , '건축년도': 'conYear'
                    , '보증금(만원)': 'real_bjprice'
                    , 'archGbCdNm': 'inhuga'
                    , '거래금액(만원)': 'real_price'
                }
            ).sort_values(by=['name', 'capacity', 'year']).reset_index(drop=True)

            # gg = realPriceDataL2.loc[
            #     (realPriceDataL2['name'] == apaInfo)
            #     & (realPriceDataL2['전용면적(㎡)'] == capInfo)
            #     ].reset_index(drop=True)
            #
            # gg2 = realPriceDataL3.loc[
            #     (realPriceDataL3['name'] == apaInfo)
            #     & (realPriceDataL3['전용면적(㎡)'] == capInfo)
            #     ].reset_index(drop=True)

            # *****************************************************
            # 데이터 통합
            # *****************************************************
            # data = pd.merge(
            #     left=prvsMntsrDataL3[['name', '전용면적(㎡)', '건축년도', '보증금(만원)', 'year', 'lat', 'lon']]
            #     , right=realPriceDataL3[['name', '전용면적(㎡)', '건축년도', '거래금액(만원)', 'year', 'lat', 'lon', '인허가addr']]
            #     , left_on=['name', '전용면적(㎡)', '건축년도', 'year', 'lat', 'lon']
            #     , right_on=['name', '전용면적(㎡)', '건축년도', 'year', 'lat', 'lon']
            #     # left=prvsMntsrDataL3[['name', '전용면적(㎡)', '보증금(만원)', 'date', 'year', 'month', 'lat', 'lon']]
            #     # , right=realPriceDataL3[['name', '전용면적(㎡)', '거래금액(만원)', 'date', 'year', 'month', 'lat', 'lon', '인허가addr']]
            #     # , left_on=['name', '전용면적(㎡)', 'date', 'year', 'month', 'lat', 'lon']
            #     # , right_on=['name', '전용면적(㎡)', 'date', 'year', 'month', 'lat', 'lon']
            #     , how='outer'
            #     )
            #
            # dataL1 = data.rename(
            #         columns={
            #             '전용면적(㎡)': 'capacity'
            #             , '건축년도': 'conYear'
            #             , '보증금(만원)': 'real_bjprice'
            #             , 'archGbCdNm': 'inhuga'
            #             , '거래금액(만원)': 'real_price'
            #         }
            #     )

            # dataL2 = pd.DataFrame()
            # nameList = dataL1['name'].unique()
            # # nameInfo = nameList[0]
            # for i, nameInfo in enumerate(nameList):
            #     selData = dataL1.loc[
            #         (dataL1['name'] == nameInfo)
            #         ].reset_index(drop=True)
            #
            #     if (len(selData) < 1): continue
            #
            #     capList = selData['capacity'].unique()
            #     # capInfo = capList[0]
            #     for j, capInfo in enumerate(capList):
            #         selDataL1 = selData.loc[
            #             (selData['capacity'] == capInfo)
            #         ].reset_index(drop=True)
            #
            #         if (len(selDataL1) < 1):continue
            #
            #         selDataL2 = selDataL1.sort_values(by='year').reset_index(drop=True)
            #         # selDataL2.index = selDataL2['date']
            #         selDataL2.index = selDataL2['year']
            #         selDataL2 = selDataL2.interpolate(method='linear')
            #         # selDataL2 = selDataL2.interpolate(method='polynomial')
            #
            #         getAddr = selDataL1['인허가addr'].dropna().unique()
            #         if (len(getAddr) > 0):
            #             selDataL2['인허가addr'] = getAddr[0]
            #
            #         dataL2 = pd.concat([dataL2, selDataL2])
            #
            # dataL3 = pd.merge(
            #     left=dataL2
            #     # , right=lcnsData[['archGbCdNm', '주소', 'lat', 'lon']]
            #     , right=lcnsDataL1[['archGbCdNm', '주소']]
            #     , left_on=['인허가addr']
            #     , right_on=['주소']
            #     , how='left'
            # ).rename(
            #     columns={
            #         'archGbCdNm': 'inhuga'
            #     }
            # )

            # tmpData = dataL1.loc[
            #     (dataL1['capacity'] == capInfo)
            #     & (dataL1['capacity'] == capInfo)
            #     ].reset_index(drop=True)
            #
            # tmpData2 = dataL2.loc[
            #     (dataL2['capacity'] == capInfo)
            #     & (dataL2['capacity'] == capInfo)
            #     ].reset_index(drop=True)
            #
            # plt.scatter(tmpData2['date'], tmpData2['real_bjprice'], c='black', label='b')
            # plt.scatter(tmpData['date'], tmpData['real_bjprice'], c='blue', label='a')
            # plt.legend()
            # plt.show()

            # 데이터 유형
            # dataL3.info()

            # sysData = realPriceDataL4[['name', 'year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga', 'real_price']]
            #
            # sysData = pd.merge(
            #     left=prvsMntsrDataL3[['name', '전용면적(㎡)', '건축년도', '보증금(만원)', 'year', 'lat', 'lon']]
            #     , right=realPriceDataL3[['name', '전용면적(㎡)', '건축년도', '거래금액(만원)', 'year', 'lat', 'lon', '인허가addr']]
            #     , left_on=['name', '전용면적(㎡)', '건축년도', 'year', 'lat', 'lon']
            #     , right_on=['name', '전용면적(㎡)', '건축년도', 'year', 'lat', 'lon']
            #     # left=prvsMntsrDataL3[['name', '전용면적(㎡)', '보증금(만원)', 'date', 'year', 'month', 'lat', 'lon']]
            #     # , right=realPriceDataL3[['name', '전용면적(㎡)', '거래금액(만원)', 'date', 'year', 'month', 'lat', 'lon', '인허가addr']]
            #     # , left_on=['name', '전용면적(㎡)', 'date', 'year', 'month', 'lat', 'lon']
            #     # , right_on=['name', '전용면적(㎡)', 'date', 'year', 'month', 'lat', 'lon']
            #     , how='outer'
            #     )

            prvsMntsrDataL5 = prvsMntsrDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['real_bjprice'].mean()
            realPriceDataL5 = realPriceDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['real_price'].mean()

            inpDataL1 = pd.merge(
                left=prvsMntsrDataL5
                , right=realPriceDataL5
                , left_on=['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga']
                , right_on=['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga']
                , how='outer'
            )

            # 데이터 보간
            # dataL1 = inpDataL1
            #
            # dataL2 = pd.DataFrame()
            # nameList = dataL1['name'].unique()
            # # nameInfo = nameList[0]
            # for i, nameInfo in enumerate(nameList):
            #     selData = dataL1.loc[
            #         (dataL1['name'] == nameInfo)
            #         ].reset_index(drop=True)
            #
            #     if (len(selData) < 1): continue
            #
            #     capList = selData['capacity'].unique()
            #     # capInfo = capList[0]
            #     for j, capInfo in enumerate(capList):
            #         selDataL1 = selData.loc[
            #             (selData['capacity'] == capInfo)
            #         ].reset_index(drop=True)
            #
            #         if (len(selDataL1) < 1):continue
            #
            #         selDataL2 = selDataL1.sort_values(by='year').reset_index(drop=True)
            #         # selDataL2.index = selDataL2['date']
            #         selDataL2.index = selDataL2['year']
            #         selDataL2 = selDataL2.interpolate(method='linear')
            #         # selDataL2 = selDataL2.interpolate(method='polynomial')
            #
            #         # getAddr = selDataL1['인허가addr'].dropna().unique()
            #         # if (len(getAddr) > 0):
            #         #     selDataL2['인허가addr'] = getAddr[0]
            #
            #         dataL2 = pd.concat([dataL2, selDataL2])
            #
            # inpDataL1 = dataL2

            # **********************************************************************************************************
            # 딥러닝 매매가
            # **********************************************************************************************************
            inpData = realPriceDataL4
            # inpData ㅊ= realPriceDataL5

            # inpData.info()

            # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'real_bjprice', 'inhuga']
            # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon']
            # xCol = ['name', 'year', 'conYear', 'capacity', 'lat', 'lon']
            # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
            xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
            yCol = 'real_price'
            modelKey = 'realPrice'

            # 딥러닝 매매가 불러오기
            result = makeDlModel(sysOpt['dlModel'], xCol, yCol, inpData, modelKey)
            log.info('[CHECK] result : {}'.format(result))

            # 딥러닝 매매가 예측
            realPriceDlModel = result['dlModel']
            inpDataL1['realPriceDL'] = realPriceDlModel.predict(h2o.H2OFrame(inpDataL1)).as_data_frame()

            mainTitle = '강북구 아파트 매매가 예측 결과 (딥러닝)'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeUserScatterPlot(inpDataL1['realPriceDL'], inpDataL1['real_price'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

            # **********************************************************************************************************
            # 딥러닝 전세가
            # **********************************************************************************************************
            inpData = prvsMntsrDataL4

            # inpData.info()

            # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'real_price', 'inhuga']
            # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
            xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            yCol = 'real_bjprice'
            modelKey = 'realBjPrice'

            # 딥러닝 전세가 불러오기
            result = makeDlModel(sysOpt['dlModel'], xCol, yCol, inpData, modelKey)
            log.info('[CHECK] result : {}'.format(result))

            # 딥러닝 전세가 예측
            realBjPriceDlModel = result['dlModel']
            inpDataL1['realBjPriceDL'] = realBjPriceDlModel.predict(h2o.H2OFrame(inpDataL1)).as_data_frame()

            mainTitle = '강북구 아파트 전세가 예측 결과 (딥러닝)'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeUserScatterPlot(inpDataL1['realBjPriceDL'], inpDataL1['real_bjprice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

            # **********************************************************************************************************
            # 머신러닝 매매가
            # **********************************************************************************************************
            inpData = realPriceDataL4

            # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'real_bjprice', 'inhuga']
            # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
            xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
            yCol = 'real_price'
            modelKey = 'realPrice'

            # 머신러닝 매매가 불러오기
            result = makeMlModel(sysOpt['mlModel'], xCol, yCol, inpData, modelKey)
            log.info('[CHECK] result : {}'.format(result))

            # 머신러닝 매매가 예측
            realPriceMlModel = result['mlModel']
            inpDataL1['realPriceML'] = predict_model(realPriceMlModel, data=inpDataL1)['Label']

            mainTitle = '강북구 아파트 매매가 예측 결과 (머신러닝)'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeUserScatterPlot(inpDataL1['realPriceML'], inpDataL1['real_price'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

            # **********************************************************************************************************
            # 머신러닝 전세가
            # **********************************************************************************************************
            inpData = prvsMntsrDataL4
            # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'real_price', 'inhuga']
            # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
            xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
            yCol = 'real_bjprice'
            modelKey = 'realBjPrice'

            # 머신러닝 전세가 불러오기
            result = makeMlModel(sysOpt['mlModel'], xCol, yCol, inpData, modelKey)
            log.info('[CHECK] result : {}'.format(result))

            # 머신러닝 전세가 예측
            realBjPriceMlModel = result['mlModel']
            inpDataL1['realBjPriceML'] = predict_model(realBjPriceMlModel, data=inpDataL1)['Label']

            mainTitle = '강북구 아파트 전세가 예측 결과 (머신러닝)'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            makeUserScatterPlot(inpDataL1['realBjPriceML'], inpDataL1['real_bjprice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

            # sys.exit()

            # **********************************************************************************************************
            # 시계열 갭투자
            # **********************************************************************************************************
            # result_data = inpDataL1

            # real_price: 거래금액(만원) = 매매가
            # real_bjprice: 보증금(만원) = 전세금
            # 갭투자 : 매매가 - 보증금
            # result_data['gap_real'] = result_data['real_price'] - result_data['real_bjprice']
            # result_data['dnn_gap'] = result_data['realPriceDL'] - result_data['realBjPriceDL']
            # result_data['ml_gap'] = result_data['realPriceML'] - result_data['realBjPriceML']

            # apaList = result_data['name'].unique()
            # capList = result_data['capacity'].unique()
            #
            # # ind = 0
            # # apaInfo = '북한산굿모닝(삼양로171길 21)'
            # apaInfo = apaList[3]
            # # capInfo = 81.98530
            # capInfo = 62.7103

            resData = inpDataL1

            nameList = resData['name'].unique()
            # nameInfo = nameList[2]
            # nameInfo = nameList[3]
            # nameInfo = nameList[36]
            for i, nameInfo in enumerate(nameList):
                selData = resData.loc[
                    (resData['name'] == nameInfo)
                ].reset_index(drop=True)

                if (len(selData) < 1): continue

                capList = selData['capacity'].unique()
                # capInfo = capList[0]
                for j, capInfo in enumerate(capList):
                    selDataL1 = selData.loc[
                        (selData['capacity'] == capInfo)
                    ].reset_index(drop=True)

                    if (len(selDataL1) < 1): continue

                    log.info('[CHECK] nameInfo : {} / capInfo : {} / cnt : {}'.format(nameInfo, capInfo, len(selDataL1)))

                    # srtDate = selDataL1['date'].min()
                    # endDate = selDataL1['date'].max()
                    srtDate = pd.to_datetime(selDataL1['year'].min(), format='%Y')
                    endDate = pd.to_datetime(selDataL1['year'].max(), format='%Y')
                    # endDate = selDataL1['date'].max() + pd.DateOffset(years=2)
                    # endDate = selDataL1['date'].max() + pd.DateOffset(years=2)
                    # dtDateList = pd.date_range(start=srtDate, end=endDate, freq=pd.DateOffset(months=1))
                    dtDateList = pd.date_range(start=srtDate, end=endDate, freq=pd.DateOffset(years=1))

                    dataL4 = pd.DataFrame()
                    # dtDateInfo = dtDateList[0]
                    for k, dtDateInfo in enumerate(dtDateList):
                        iYear = int(dtDateInfo.strftime('%Y'))
                        # iMonth = int(dtDateInfo.strftime('%m'))

                        selInfoFirst = selDataL1.loc[0]

                        selInfo = selDataL1.loc[
                            (selDataL1['year'] == iYear)
                            # & (selDataL1['month'] == iMonth)
                        ].reset_index(drop=True)

                        dictInfo = {
                            'name': [nameInfo]
                            , 'capacity': [capInfo]
                            , 'date': [dtDateInfo]
                            , 'year': [iYear]
                            # , 'month': [iMonth]
                            , 'lat': [selInfoFirst['lat']]
                            , 'lon': [selInfoFirst['lon']]
                            , 'inhuga': [selInfoFirst['inhuga']]
                            # , 'inhuga': [np.nan if (len(selInfo) < 1) else selInfo['inhuga'][0]]
                        }

                        dictDtl = {
                            'real_price': [np.nan if (len(selInfo) < 1) else selInfo['real_price'][0]]
                            , 'real_bjprice': [np.nan if (len(selInfo) < 1) else selInfo['real_bjprice'][0]]
                            , 'realPriceDL': [realPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selInfo) < 1) else selInfo['realPriceDL'][0]]
                            , 'realBjPriceDL': [realBjPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selInfo) < 1) else selInfo['realBjPriceDL'][0]]
                            , 'realPriceML': [predict_model(realPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selInfo) < 1) else selInfo['realPriceML'][0]]
                            , 'realBjPriceML': [predict_model(realBjPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selInfo) < 1) else selInfo['realBjPriceML'][0]]
                        }

                        dict = {**dictInfo, **dictDtl}

                        # dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dictInfo)], ignore_index=True)
                        dataL4 = pd.concat([dataL4, pd.DataFrame.from_dict(dict)], ignore_index=True)

                    dataL4['gap_real'] = dataL4['real_price'] - dataL4['real_bjprice']
                    dataL4['gapML'] = dataL4['realPriceML'] - dataL4['realBjPriceML']
                    dataL4['gapDL'] = dataL4['realPriceDL'] - dataL4['realBjPriceDL']

                    # 아파트 갭 투자 시계열
                    mainTitle = '[{}, {}] 아파트 갭 투자 시계열'.format(nameInfo, capInfo)
                    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    makeUserTimeSeriesPlot(dataL4['date'], dataL4['gapML'], dataL4['gapDL'], dataL4['gap_real'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)
                    makeUserTimeSeriesPlot(dataL4['date'], dataL4['realBjPriceML'], dataL4['realBjPriceDL'], dataL4['real_bjprice'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)
                    makeUserTimeSeriesPlot(dataL4['date'], dataL4['realPriceML'], dataL4['realPriceDL'], dataL4['real_price'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)

                    # 시계열 예측
                    try:
                        tsModel = AutoTS(forecast_length=2, frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')

                        tsDlModel = tsModel.fit(dataL4, date_col='date', value_col='gapDL', id_col=None)
                        tsDlFor = tsDlModel.predict().forecast
                        tsDlFor['date'] = tsDlFor.index
                        # tsDlFor.reset_index(drop=True, inplace=True)

                        tsMlModel = tsModel.fit(dataL4, date_col='date', value_col='gapML', id_col=None)
                        tsMlFor = tsMlModel.predict().forecast
                        tsMlFor['date'] = tsMlFor.index
                        # tsMlFor.reset_index(drop=True, inplace=True)

                        tsForData = tsDlFor.merge(tsMlFor, left_on=['date'], right_on=['date'], how='inner')
                        tsForData['name'] = nameInfo
                    except Exception as e:
                        log.error('Exception : {}'.format(e))

                    valid_result_FIN_L4 = dataL4.merge(
                        tsForData
                        , left_on=['name', 'date', 'gapDL', 'gapML']
                        , right_on=['name', 'date', 'gapDL', 'gapML']
                        , how='outer'
                    )

                    # valid_result_FIN_L5 = valid_resCult_FIN_L4.interpolate(method='values')

                    # full_data_gapreal = pd.concat([full_data_gapreal, valid_result_FIN_L5])

                    # 아파트 갭 투자 시계열
                    mainTitle = '[{}, {}] 아파트 갭 투자 시계열'.format(nameInfo, capInfo)
                    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    makeUserTimeSeriesPlot(valid_result_FIN_L4['date'], valid_result_FIN_L4['gapML'], valid_result_FIN_L4['gapDL'], valid_result_FIN_L4['gap_real'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)

                    result_L2 = valid_result_FIN_L4

                    result_L3 = result_L2[['gap_real', 'gapML', 'gapDL']]

                    resDiffData = result_L3.diff(periods=1).rename(
                        columns={
                            'gap_real': 'gapDiffReal'
                            , 'gapML': 'gapDiffML'
                            , 'gapDL': 'gapDiffDL'
                        }
                        , inplace=False
                    )

                    resPctData = result_L3.pct_change(periods=1).rename(
                        columns={
                            'gap_real': 'gapPctReal'
                            , 'gapML': 'gapPctML'
                            , 'gapDL': 'gapPctDL'
                        }
                        , inplace=False
                    )

                    result_L4 = pd.concat([result_L2, resDiffData, resPctData * 100], axis=1)

                    result_L5 = result_L4.rename(
                        columns={
                            'name': '아파트(도로명)'
                            , 'capacity': '면적'
                            , 'construction_year': '건축연도'
                            , 'year': '연도'
                            , 'realPrice': '매매가'
                            , 'realPriceML': '예측 머신러닝 매매가'
                            , 'realPriceDL': '예측 딥러닝 매매가'
                            , 'real_bjprice': '전세가'
                            , 'realBjpriceML': '예측 머신러닝 전세가 '
                            , 'realBjpriceDL': '예측 딥러닝 전세가'
                            , 'gap_real': '실측 갭투자'
                            , 'gapML': '예측 머신러닝 갭투자'
                            , 'gapDL': '예측 딥러닝 갭투자'
                            , 'gapDiffReal': '실측 수익금'
                            , 'gapDiffDL': '예측 딥러닝 수익금'
                            , 'gapDiffML': '예측 머신러닝 수익금'
                            , 'gapPctReal': '실측 수익률'
                            , 'gapPctDL': '예측 딥러닝 수익률'
                            , 'gapPctML': '예측 머신러닝 수익률'
                        }
                    )

                    saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '수익률 테이블', datetime.now().strftime('%Y%m%d'))
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    result_L5.to_excel(saveFile, index=False)
                    log.info('[CHECK] saveFile : {}'.format(saveFile))


                    # # for i, apaInfo in enumerate(apaList):
                    # #     for j, capInfo in enumerate(capList):
                    # # print(apaInfo)
                    #
                    # # valid_result_FIN_L1 = result_data.loc[
                    # #     (result_data['name'] == apaInfo)
                    # #     & (result_data['capacity'] == capInfo)
                    # #     ].reset_index(drop=True)
                    # #
                    # # if (len(valid_result_FIN_L1) < 6): continue
                    # #
                    # # log.info('[CHECK] apaInfo : {} / capInfo : {} / cnt : {}'.format(apaInfo, capInfo, len(valid_result_FIN_L1)))
                    #
                    # # srtYear = pd.to_datetime(valid_result_FIN_L1['times'].min(), format='%Y')
                    # # endYear = pd.to_datetime(valid_result_FIN_L1['times'].max(), format='%Y')
                    # # # dtDateList = pd.date_range(start=srtYear, end=endYear, freq=pd.DateOffset(years=1))
                    # #
                    # # srtDate = valid_result_FIN_L1['date'].min()
                    # # # endDate = valid_result_FIN_L1['date'].max()
                    # # endDate = pd.to_datetime('2024-04', format='%Y-%m')
                    # # dtDateList = pd.date_range(start=srtDate, end=endDate, freq=pd.DateOffset(months=1))
                    # #
                    # # dataL1 = pd.DataFrame()
                    # # # dtDateInfo = dtDateList[0]
                    # # for k, dtDateInfo in enumerate(dtDateList):
                    # #     # log.info('[CHECK] dtDateInfo : {}'.format(dtDateInfo))
                    # #
                    # #     # tmpData = valid_result_FIN_L1
                    # #     # predict_model(realBjPriceMlModel, data=tmpData)['Label']
                    # #
                    # #     iYear = int(dtDateInfo.strftime('%Y'))
                    # #     iMonth = int(dtDateInfo.strftime('%m'))
                    # #
                    # #     selData = valid_result_FIN_L1.loc[
                    # #         (valid_result_FIN_L1['year'] == iYear)
                    # #         & (valid_result_FIN_L1['month'] == iMonth)
                    # #     ].reset_index(drop=True)
                    # #
                    # #     tmpData = valid_result_FIN_L1.loc[0]
                    # #     # tmpData['year'] = iYear
                    # #     # tmpData['month'] = iMonth
                    # #
                    # #     # inpDataL1['realBjPriceML'] = predict_model(realBjPriceMlModel, data=inpData)['Label']
                    # #
                    # #     # valid_result_FIN_L2 = valid_result_FIN_L1.loc[
                    # #     #     (valid_result_FIN_L1['times'] == iYear)
                    # #     # ].reset_index(drop=True)
                    # #     #
                    # #     # iMonth = 1
                    # #
                    # #     dictInfo = {
                    # #         'name': [apaInfo]
                    # #         , 'capacity': [capInfo]
                    # #         , 'date': [dtDateInfo]
                    # #         , 'year': [iYear]
                    # #         , 'month': [iMonth]
                    # #         , 'lat': [tmpData['lat']]
                    # #         , 'lon': [tmpData['lon']]
                    # #         , 'inhuga': [tmpData['inhuga']]
                    # #
                    # #         # , 'realPriceDL': [np.nan if (len(tmpData) < 1) else tmpData['realPriceDL']]
                    # #         # , 'realBjPriceDL': [np.nan if (len(tmpData) < 1) else tmpData['realBjPriceDL']]
                    # #         # , 'realPriceML': [np.nan if (len(tmpData) < 1) else tmpData['realPriceML']]
                    # #         # , 'realBjPriceML': [np.nan if (len(tmpData) < 1) else tmpData['realBjPriceML']]
                    # #         # , 'gap_real': [np.nan if (len(tmpData) < 1) else tmpData['gap_real']]
                    # #         # , 'gap_machine': [np.nan if (len(tmpData) < 1) else tmpData['ml_gap']]
                    # #         # , 'gap_dnn': [np.nan if (len(tmpData) < 1) else tmpData['dnn_gap']]
                    # #     }
                    # #
                    # #     dictDtl = {
                    # #         'real_price': [np.nan if (len(selData) < 1) else selData['real_price'][0]]
                    # #         , 'real_bjprice': [np.nan if (len(selData) < 1) else selData['real_bjprice'][0]]
                    # #         , 'realPriceDL': [realPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selData) < 1) else selData['realPriceDL'][0]]
                    # #         , 'realBjPriceDL': [realBjPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selData) < 1) else selData['realBjPriceDL'][0]]
                    # #         , 'realPriceML': [predict_model(realPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selData) < 1) else selData['realPriceML'][0]]
                    # #         , 'realBjPriceML': [predict_model(realBjPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selData) < 1) else selData['realBjPriceML'][0]]
                    # #     }
                    # #
                    # #     dict = {**dictInfo, **dictDtl}
                    # #
                    # #     # dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dictInfo)], ignore_index=True)
                    # #     dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dict)], ignore_index=True)
                    # #
                    # #
                    # # dataL1['gap_real'] = dataL1['real_price'] - dataL1['real_bjprice']
                    # # dataL1['gapML'] = dataL1['realPriceML'] - dataL1['realBjPriceML']
                    # # dataL1['gapDL'] = dataL1['realPriceDL'] - dataL1['realBjPriceDL']
                    # #
                    # #
                    # # # 아파트 갭 투자 시계열
                    # # mainTitle = '[{}, {}] 아파트 갭 투자 시계열'.format(apaInfo, capInfo)
                    # # saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    # # makeUserTimeSeriesPlot(dataL1['date'], dataL1['gapML'], dataL1['gapDL'], dataL1['gap_real'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)
                    # #
                    # #     # c = pd.DataFrame.fromC_dict(dict)
                    # #     # pd.DataFrame.from_dict(dict)
                    # #     # merged_dict = dict_one | dict_two | dict_three
                    # #
                    # #     # , 'realPriceDL': [np.nan if (len(tmpData) < 1) else tmpData['realPriceDL']]
                    # #     # , 'realBjPriceDL': [np.nan if (len(tmpData) < 1) else tmpData['realBjPriceDL']]
                    # #     # , 'realPriceML': [np.nan if (len(tmpData) < 1) else tmpData['realPriceML']]
                    # #     # , 'realBjPriceML': [np.nan if (len(tmpData) < 1) else tmpData['realBjPriceML']]
                    # #     # , 'gap_real': [np.nan if (len(tmpData) < 1) else tmpData['gap_real']]
                    # #     # , 'gap_machine': [np.nan if (len(tmpData) < 1) else tmpData['ml_gap']]
                    # #     # , 'gap_dnn': [np.nan if (len(tmpData) < 1) else tmpData['dnn_gap']]
                    # #     # }
                    # #
                    # #     # b = aa.values
                    # #
                    # #     # aa = predict_model(realBjPriceMlModel, data=pd.DataFrame.from_dict(dict))['Label']
                    # #
                    # # # a = 10
                    # #
                    # # dataL1['date'] = pd.to_datetime(dataL1['times'], format='%Y')
                    # # df_intp_values = dataL1.interpolate(method='values')
                    #
                    # try:
                    #     tsModel = AutoTS(forecast_length=2, frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')
                    #
                    #     tsDlModel = tsModel.fit(df_intp_values, date_col='date', value_col='gap_dnn', id_col=None)
                    #     tsDlFor = tsDlModel.predict().forecast
                    #     tsDlFor['date'] = tsDlFor.index
                    #     # tsDlFor.reset_index(drop=True, inplace=True)
                    #
                    #     tsMlModel = tsModel.fit(df_intp_values, date_col='date', value_col='gap_machine', id_col=None)
                    #     tsMlFor = tsMlModel.predict().forecast
                    #     tsMlFor['date'] = tsMlFor.index
                    #     # tsMlFor.reset_index(drop=True, inplace=True)
                    #
                    #     tsForData = tsDlFor.merge(tsMlFor, left_on=['date'], right_on=['date'], how='inner')
                    #     tsForData['name'] = apaInfo
                    # except Exception as e:
                    #     log.error('Exception : {}'.format(e))
                    #
                    # valid_result_FIN_L4 = df_intp_values.merge(tsForData, left_on=['name', 'date', 'gap_dnn', 'gap_machine'], right_on=['name', 'date', 'gap_dnn', 'gap_machine'], how='outer')
                    # valid_result_FIN_L5 = valid_result_FIN_L4.interpolate(method='values')
                    #
                    # # full_data_gapreal = pd.concat([full_data_gapreal, valid_result_FIN_L5])
                    #
                    # # 아파트 갭 투자 시계열
                    # mainTitle = '[{}, {}] 아파트 갭 투자 시계열'.format(apaInfo, capInfo)
                    # saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    # makeUserTimeSeriesPlot(valid_result_FIN_L4['date'], valid_result_FIN_L4['gap_machine'], valid_result_FIN_L4['gap_dnn'], valid_result_FIN_L4['gap_real'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)
                    #
                    # # 수익률 테이블
                    # # valid_result_FIN_L5
                    #
                    # # result_L2 = valid_result_FIN_L5
                    # result_L2 = valid_result_FIN_L4
                    #
                    # result_L3 = result_L2[['gap_real', 'gap_machine', 'gap_dnn']]
                    # resDiff = result_L3.diff(periods=1).rename(
                    #     columns={
                    #         'gap_real': 'gap_real_diff'
                    #         , 'gap_machine': 'gap_machine_diff'
                    #         , 'gap_dnn': 'gap_dnn_diff'
                    #     }
                    #     , inplace=False
                    # )
                    #
                    # resPct = result_L3.pct_change(periods=1).rename(
                    #     columns={
                    #         'gap_real': 'gap_real_pct'
                    #         , 'gap_machine': 'gap_machine_pct'
                    #         , 'gap_dnn': 'gap_dnn_pct'
                    #     }
                    #     , inplace=False
                    # )
                    #
                    # result_L4 = pd.concat([result_L2, resDiff, resPct * 100], axis=1)
                    # result_L5 = result_L4.rename(
                    #     columns={
                    #         'date': '연도'
                    #         , 'construction_year': '건축연도'
                    #         , 'name': '아파트'
                    #         , 'capacity': '면적'
                    #         , 'real_price': '매매가'
                    #         , 'real_bjprice': '전세가'
                    #         , 'gap_real': '실측 갭투자'
                    #         , 'gap_machine': '예측 머신러닝 갭투자'
                    #         , 'gap_dnn': '예측 딥러닝 갭투자'
                    #         , 'gap_real_diff': '실측 수익금'
                    #         , 'gap_dnn_diff': '예측 딥러닝 수익금'
                    #         , 'gap_machine_diff': '예측 머신러닝 수익금'
                    #         , 'gap_real_pct': '실측 수익률'
                    #         , 'gap_dnn_pct': '예측 딥러닝 수익률'
                    #         , 'gap_machine_pct': '예측 머신러닝 수익률'
                    #     }
                    # )
                    #
                    # result_L2.to_excel('./수익률테이블_결과/result_인허가포함학습.xlsx')

        except Exception as e:
            log.error('Exception : {}'.format(e))
            raise e
        finally:
            log.info('[END] {}'.format('exec'))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format('main'))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = {}

        print('[CHECK] inParams : {}'.format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format('main'))
