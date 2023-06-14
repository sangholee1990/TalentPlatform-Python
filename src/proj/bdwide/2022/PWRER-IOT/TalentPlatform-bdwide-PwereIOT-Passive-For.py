import glob
# import seaborn as sns
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# import datetime as dt
# from datetime import datetime
import pvlib
import matplotlib.dates as mdates
import matplotlib.cm as cm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
import pygrib
# import pykrige.kriging_tools as kt
import pytz
import requests
import datetime
import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from scipy.stats import linregress
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import xarray as xr
from pvlib import location
from pvlib import irradiance333

import h2o
from h2o.automl import H2OAutoML

from pycaret.regression import *
from matplotlib import font_manager, rc

# try:
#     from pycaret.regression import *
# except Exception as e:
#     print("Exception : {}".format(e))
#
# try:
#     from pycaret.regression import *
# except Exception as e:
#     print("Exception : {}".format(e))


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
font_manager._rebuild()

#plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')
dtKst = datetime.timedelta(hours=9)

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
        , datetime.datetime.now().strftime("%Y%m%d")
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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        plt.rc('font', family=fontName)

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


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


# 산점도 시각화
def makeUserScatterPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, isSame):

    log.info('[START] {}'.format('makeUserScatterPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        N = len(refVal[mask])

        plt.scatter(prdVal, refVal)

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)


        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        Bias = np.mean(prdVal[mask] - refVal[mask])
        rBias = (Bias / np.mean(refVal[mask])) * 100.0
        RMSE = np.sqrt(np.mean((prdVal[mask] - refVal[mask]) ** 2))
        rRMSE = (RMSE / np.mean(refVal[mask])) * 100.0

        MAPE = np.mean(np.abs((prdVal[mask] - refVal[mask]) / prdVal[mask])) * 100.0

        # 선형회귀곡선에 대한 계산
        slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

        lmfit = (slope * prdVal) + intercept
        # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle="-")
        plt.plot([minVal, maxVal], [minVal, maxVal], color="black", linestyle="--", linewidth=2)
        plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle="-")
        # 라벨 추가
        plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept),
                     xy=(minVal + xIntVal, maxVal - yIntVal),
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

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserScatterPlot'))


# 빈도분포 2D 시각화
def makeUserHist2dPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, nbinVal, isSame):

    log.info('[START] {}'.format('makeUserHist2dPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        N = len(refVal[mask])

        # plt.scatter(prdVal, refVal)
        # nbins = 250
        hist2D, xEdge, yEdge = np.histogram2d(prdVal[mask], refVal[mask], bins=nbinVal)
        # hist2D, xEdge, yEdge = np.histogram2d(prdVal, refVal)

        # hist2D 전처리
        hist2D = np.rot90(hist2D)
        hist2D = np.flipud(hist2D)

        # 마스킹
        hist2DVal = np.ma.masked_where(hist2D == 0, hist2D)

        plt.pcolormesh(xEdge, yEdge, hist2DVal, cmap=cm.get_cmap('jet'), vmin=0, vmax=100)

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        Bias = np.mean(prdVal - refVal)
        rBias = (Bias / np.mean(refVal)) * 100.0
        RMSE = np.sqrt(np.mean((prdVal - refVal) ** 2))
        rRMSE = (RMSE / np.mean(refVal)) * 100.0
        MAPE = np.mean(np.abs((prdVal - refVal) / prdVal)) * 100.0

        # 선형회귀곡선에 대한 계산
        slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

        lmfit = (slope * prdVal) + intercept
        # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle="-")
        plt.plot([minVal, maxVal], [minVal, maxVal], color="black", linestyle="--", linewidth=2)
        plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle="-")

        # 컬러바
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('빈도수')

        # 라벨 추가
        plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept),
                     xy=(minVal + xIntVal, maxVal - yIntVal),
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

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserHist2dPlot'))


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):

    log.info('[START] {}'.format('makeUserTimeSeriesPlot'))

    result = None

    try:
        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        mlRMSE = np.sqrt(np.mean((mlPrdVal - refVal) ** 2))
        mlReRMSE = (mlRMSE / np.mean(refVal)) * 100.0

        dlRMSE = np.sqrt(np.mean((dlPrdVal - refVal) ** 2))
        dlReRMSE = (dlRMSE / np.mean(refVal)) * 100.0

        # 선형회귀곡선에 대한 계산
        mlFit = linregress(mlPrdVal, refVal)
        mlR = mlFit[2]

        dlFit = linregress(dlPrdVal, refVal)
        dlR = dlFit[2]

        prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(mlPrdValLabel, mlR, mlReRMSE)
        prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(dlPrdValLabel, dlR, dlReRMSE)

        plt.grid(True)

        plt.plot(dtDate, mlPrdVal, label=prdValLabel_ml, marker='o')
        plt.plot(dtDate, dlPrdVal, label=prdValLabel_dnn, marker='o')
        plt.plot(dtDate, refVal, label=refValLabel, marker='o')

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.ylim(0, 1000)

        if (isFore == True):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=45, ha='right')

        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=0, ha='right')

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
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserTimeSeriesPlot'))

# 딥러닝 예측
def makeDlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeDlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:

        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # h2o.shutdown(prompt=False)

        if (not subOpt['isInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # dlModel = H2OAutoML(max_models=30, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=99999, balance_classes=True, seed=123)

            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainData), validation_frame=h2o.H2OFrame(validData))
            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))
            
            fnlModel = dlModel.get_best_model()
            #fnlModel = dlModel.leader

            # 학습 모델 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'for', datetime.datetime.now().strftime('%Y%m%d'))
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


# 머신러닝 예측
def makeMlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeMlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:
        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model.pkl'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(data, test_size=0.3)
            # trainData = inpData

            pyModel = setup(
                data=dataL1
                , session_id=123
                , silent=True
                , target=yCol
            )

            # pyModel = setup(
            #     data=dataL1
            #     , session_id=123
            #     , silent=True
            #     , target=yCol
            #     , remove_outliers= True
            #     , remove_multicollinearity = True
            #     , ignore_low_variance = True
            #     , normalize=True
            #     , transformation= True
            #     , transform_target = True
            #     , combine_rare_levels = True
            # )

            # 각 모형에 따른 자동 머신러닝
            modelList = compare_models(sort='RMSE', n_select=3)

            # 앙상블 모형
            blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(tuneModel)

            # 학습 모형 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', datetime.datetime.now().strftime('%Y%m%d'))
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


def initCfgInfo(sysPath):
    log.info('[START] {}'.format('initCfgInfo'))
    # log.info('[CHECK] sysPath : {}'.format(sysPath))

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')
        dbUser = config.get('mariadb-dev', 'user')
        dbPwd = config.get('mariadb-dev', 'pwd')
        dbHost = config.get('mariadb-dev', 'host')
        dbPort = config.get('mariadb-dev', 'port')
        dbName = config.get('mariadb-dev', 'dbName')

        dbEngine = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        # dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessMake = sessionmaker(bind=dbEngine)
        session = sessMake()

        # API 정보
        apiUrl = config.get('pv', 'url')
        apiToken = config.get('pv', 'token')

        result = {
            'dbEngine': dbEngine
            , 'session': session
            , 'apiUrl': apiUrl
            , 'apiToken': apiToken
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))


def subAtmosData(cfgInfo, cfgPath, posDataL1, dtIncDateList):

    log.info('[START] {}'.format('subAtmosData'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] cfgPath : {}'.format(cfgPath))
    # log.info('[CHECK] posDataL1 : {}'.format(posDataL1))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))

    result = None

    try:
        lat1D = np.array(posDataL1['LAT'])
        lon1D = np.array(posDataL1['LON'])

        cfgFile = '{}/{}'.format(cfgPath, 'modelInfo/UMKR_l015_unis_H000_202110010000.grb2')
        log.info("[CHECK] cfgFile : {}".format(cfgFile))
        cfgFileInfo = pygrib.open(cfgFile).select(name='Temperature')[1]
        lat2D, lon2D = cfgFileInfo.latlons()

        posList = []
        # kdTree를 위한 초기 데이터
        for i in range(0, lon2D.shape[0]):
            for j in range(0, lon2D.shape[1]):
                coord = [lat2D[i, j], lon2D[i, j]]
                posList.append(cartesian(*coord))

        tree = spatial.KDTree(posList)

        row1D = []
        col1D = []
        for ii, posInfo in posDataL1.iterrows():
            coord = cartesian(posInfo['LAT'], posInfo['LON'])
            closest = tree.query([coord], k=1)
            cloIdx = closest[1][0]
            row = int(cloIdx / lon2D.shape[1])
            col = cloIdx % lon2D.shape[1]

            row1D.append(row)
            col1D.append(col)

        row2D, col2D = np.meshgrid(row1D, col1D)

        # dtIncDateInfo = dtIncDateList[0]
        for ii, dtIncDateInfo in enumerate(dtIncDateList):
            log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

            dtDateYm = dtIncDateInfo.strftime('%Y%m')
            dtDateDay = dtIncDateInfo.strftime('%d')
            dtDateHour = dtIncDateInfo.strftime('%H')
            dtDateYmd = dtIncDateInfo.strftime('%Y%m%d')
            dtDateHm = dtIncDateInfo.strftime('%H%M')
            dtDateYmdHm = dtIncDateInfo.strftime('%Y%m%d%H%M')

            # UMKR_l015_unis_H001_202110010000.grb2
            inpFilePattern = 'MODEL/{}/{}/{}/UMKR_l015_unis_*_{}.grb2'.format(dtDateYm, dtDateDay, dtDateHour, dtDateYmdHm)
            inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
            fileList = sorted(glob.glob(inpFile))

            if (len(fileList) < 1): continue

            # fileInfo = fileList[0]
            for jj, fileInfo in enumerate(fileList):
                log.info("[CHECK] fileInfo : {}".format(fileInfo))

                try:
                    grb = pygrib.open(fileInfo)
                    grbInfo = grb.select(name='Temperature')[1]

                    validIdx = int(re.findall('H\d{3}', fileInfo)[0].replace('H', ''))
                    dtValidDate = grbInfo.validDate
                    dtAnalDate = grbInfo.analDate

                    uVec = grb.select(name='10 metre U wind component')[0].values[row2D, col2D]
                    vVec = grb.select(name='10 metre V wind component')[0].values[row2D, col2D]
                    WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
                    WS = np.sqrt(np.square(uVec) + np.square(vVec))
                    PA = grb.select(name='Surface pressure')[0].values[row2D, col2D]
                    TA = grbInfo.values[row2D, col2D]
                    TD = grb.select(name='Dew point temperature')[0].values[row2D, col2D]
                    HM = grb.select(name='Relative humidity')[0].values[row2D, col2D]
                    lowCA = grb.select(name='Low cloud cover')[0].values[row2D, col2D]
                    medCA = grb.select(name='Medium cloud cover')[0].values[row2D, col2D]
                    higCA = grb.select(name='High cloud cover')[0].values[row2D, col2D]
                    CA_TOT = np.mean([lowCA, medCA, higCA], axis=0)
                    SS = grb.select(name='unknown')[0].values[row2D, col2D]

                    dsDataL1 = xr.Dataset(
                        {
                            'uVec': (('anaDate', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'vVec': (('anaDate', 'time', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WD': (('anaDate', 'time', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WS': (('anaDate', 'time', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'PA': (('anaDate', 'time', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TA': (('anaDate', 'time', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TD': (('anaDate', 'time', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'HM': (('anaDate', 'time', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'lowCA': (('anaDate', 'time', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'medCA': (('anaDate', 'time', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'higCA': (('anaDate', 'time', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'CA_TOT': (('anaDate', 'time', 'lat', 'lon'), (CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'SS': (('anaDate', 'time', 'lat', 'lon'), (SS).reshape(1, 1, len(lat1D), len(lon1D)))
                        }
                        , coords={
                            'anaDate': pd.date_range(dtAnalDate, periods=1)
                            , 'time': pd.date_range(dtValidDate, periods=1)
                            , 'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                except Exception as e:
                    log.error("Exception : {}".format(e))

                for kk, posInfo in posDataL1.iterrows():
                    posId = int(posInfo['ID'])
                    posLat = posInfo['LAT']
                    posLon = posInfo['LON']
                    # posSza = posInfo['STN_SZA']
                    # posAza = posInfo['STN_AZA']

                    # log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))

                    umData = dsDataL1
                    dtanaDateInfo = umData['anaDate'].values
                    # log.info("[CHECK] dtanaDateInfo : {}".format(dtanaDateInfo))

                    try:
                        umDataL2 = umData.sel(lat=posLat, lon=posLon, anaDate=dtanaDateInfo)
                        umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=True)
                        # umDataL3['dtDate'] = pd.to_datetime(dtanaDateInfo) + (umDataL3.index.values * datetime.timedelta(hours=1))
                        umDataL3['DATE_TIME'] = pd.to_datetime(dtanaDateInfo) + (validIdx * datetime.timedelta(hours=1))
                        # umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
                        umDataL3['DATE_TIME_KST'] = umDataL3['DATE_TIME'] + dtKst
                        umDataL4 = umDataL3.rename({'SS': 'SWR'}, axis='columns')
                        umDataL5 = umDataL4[['DATE_TIME_KST', 'DATE_TIME', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR']]
                        umDataL5['SRV'] = 'SRV{:05d}'.format(posId)
                        umDataL5['TA'] = umDataL5['TA'] - 273.15
                        umDataL5['TD'] = umDataL5['TD'] - 273.15
                        umDataL5['PA'] = umDataL5['PA'] / 100.0
                        umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] < 0, 0, umDataL5['CA_TOT'])
                        umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] > 1, 1, umDataL5['CA_TOT'])

                        solPosInfo = pvlib.solarposition.get_solarposition(pd.to_datetime(umDataL5['DATE_TIME'].values), posLat, posLon, pressure=umDataL5['PA'].values * 100.0, temperature=umDataL5['TA'].values, method='nrel_numpy')
                        umDataL5['SZA'] = solPosInfo['apparent_zenith'].values
                        umDataL5['AZA'] = solPosInfo['azimuth'].values
                        umDataL5['ET'] = solPosInfo['equation_of_time'].values
                        umDataL5['ANA_DATE'] = pd.to_datetime(dtanaDateInfo)

                        # pvlib.location.Location.get_clearsky()
                        site = location.Location(posLat, posLon, tz='Asia/Seoul')
                        clearInsInfo = site.get_clearsky(pd.to_datetime(umDataL5['DATE_TIME'].values))
                        umDataL5['GHI_CLR'] = clearInsInfo['ghi'].values
                        umDataL5['DNI_CLR'] = clearInsInfo['dni'].values
                        umDataL5['DHI_CLR'] = clearInsInfo['dhi'].values

                        # poaInsInfo = irradiance.get_total_irradiance(
                        #     surface_tilt=posSza,
                        #     surface_azimuth=posAza,
                        #     dni=clearInsInfo['dni'],
                        #     ghi=clearInsInfo['ghi'],
                        #     dhi=clearInsInfo['dhi'],
                        #     solar_zenith=solPosInfo['apparent_zenith'].values,
                        #     solar_azimuth=solPosInfo['azimuth'].values
                        # )

                        # umDataL5['GHI_POA'] = poaInsInfo['poa_global'].values
                        # umDataL5['DNI_POA'] = poaInsInfo['poa_direct'].values
                        # umDataL5['DHI_POA'] = poaInsInfo['poa_diffuse'].values

                        # 혼탁도
                        turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(umDataL5['DATE_TIME'].values), posLat, posLon, interp_turbidity=True)
                        umDataL5['TURB'] = turbidity.values

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                    setAtmosDataDB(cfgInfo, umDataL5)

        result = {
            'msg': 'succ'
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subAtmosData'))


def setAtmosDataDB(cfgInfo, dbData):

    # log.info('[START] {}'.format('setAtmosDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']
        iAnaYear = int(dbData['ANA_DATE'][0].strftime("%Y"))
        selDbTable = 'TB_FOR_DATA_{}'.format(iAnaYear)

        # 테이블 생성
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS `{}`
            (
                SRV           varchar(20) not null comment '관측소 정보',
                ANA_DATE      datetime    not null comment '예보일',
                DATE_TIME     datetime    not null comment '예보날짜 UTC',
                DATE_TIME_KST datetime    null comment '예보날짜 KST',
                CA_TOT        float       null comment '전운량',
                HM            float       null comment '상대습도',
                PA            float       null comment '현지기압',
                TA            float       null comment '기온',
                TD            float       null comment '이슬점온도',
                WD            float       null comment '풍향',
                WS            float       null comment '풍속',
                SZA           float       null comment '태양 천정각',
                AZA           float       null comment '태양 방위각',
                ET            float       null comment '태양 시간각',
                TURB          float       null comment '혼탁도',
                GHI_CLR       float       null comment '맑은날 전천 일사량',
                DNI_CLR       float       null comment '맑은날 직달 일사량',
                DHI_CLR       float       null comment '맑은날 산란 일사량',
                -- GHI_POA       float       null comment '보정 맑은날 전천 일사량',
                -- DNI_POA       float       null comment '보정 맑은날 직달 일사량',
                -- DHI_POA       float       null comment '보정 맑은날 산란 일사량',
                SWR           float       null comment '일사량',
                -- ML            float       null comment '머신러닝',
                -- DL            float       null comment '딥러닝',
                -- ML2            float       null comment '머신러닝',
                -- DL2            float       null comment '딥러닝',
                REG_DATE      datetime    null comment '등록일',
                MOD_DATE      datetime    null comment '수정일',
                primary key (SRV, ANA_DATE, DATE_TIME)
            )
            comment '기상 예보 테이블_{}';
            """.format(selDbTable, iAnaYear)
        )
        session.commit()

        for k, dbInfo in dbData.iterrows():
            # 테이블 중복 검사
            resChk = pd.read_sql(
                """
                SELECT COUNT(*) AS CNT FROM `{}`
                WHERE SRV = '{}' AND ANA_DATE = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'])
                , con=dbEngine
            )

            if (resChk.loc[0, 'CNT'] > 0):
                dbInfo['MOD_DATE'] = datetime.datetime.now()

                session.execute(
                    """
                    UPDATE `{}`
                    SET DATE_TIME_KST = '{}', CA_TOT = '{}', HM = '{}', PA = '{}', TA = '{}', TD = '{}', WD = '{}', WS = '{}', SZA = '{}', AZA = '{}', ET = '{}', TURB = '{}'
                    , GHI_CLR = '{}', DNI_CLR = '{}', DHI_CLR = '{}', SWR = '{}', MOD_DATE = '{}'
                    WHERE SRV = '{}' AND ANA_DATE = '{}' AND DATE_TIME = '{}';
                    """.format(selDbTable
                               , dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR']
                               , dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['SWR'], dbInfo['MOD_DATE']
                               , dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'])
                )

            else:
                dbInfo['REG_DATE'] = datetime.datetime.now()

                session.execute(
                    """
                    INSERT INTO `{}` (SRV, ANA_DATE, DATE_TIME, DATE_TIME_KST, CA_TOT, HM, PA, TA, TD, WD, WS, SZA, AZA, ET, TURB, GHI_CLR, DNI_CLR, DHI_CLR, SWR, REG_DATE)
                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
                    """.format(selDbTable
                               , dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR'], dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['SWR'], dbInfo['REG_DATE'])
                )
            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()
        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setAtmosDataDB'))


def subModelInpData(sysOpt, cfgInfo, srvId, posVol, dtIncDateList):

    log.info('[START] {}'.format('subModelInpData'))
    # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] srvId : {}'.format(srvId))
    # log.info('[CHECK] posVol : {}'.format(posVol))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))

    result = None

    try:

        dtYearList = dtIncDateList.strftime('%Y').unique().tolist()
        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']

        # *****************************************************
        # 발전량 데이터
        # *****************************************************
        pvData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            selDbTable = 'TB_PV_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT SRV, DATE_TIME, DATE_TIME_KST, PV
                FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME_KST BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            pvData = pd.concat([pvData, res], ignore_index=False)

        # *****************************************************
        # 기상정보 데이터
        # *****************************************************
        forData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            # selDbTable = 'TB_FOR_DATA_{}'.format(dtYearInfo)
            selDbTable = 'TEST_TB_FOR_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT *
                FROM `{}`
                WHERE SRV = '{}' AND ANA_DATE BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            forData = pd.concat([forData, res], ignore_index=False)

        # *****************************************************
        # 데이터 전처리 및 병합
        # *****************************************************
        pvDataL1 = pvData.loc[(pvData['PV'] > 0) & (pvData['PV'] <= posVol)]
        pvDataL1['PV'][pvDataL1['PV'] < 0] = np.nan

        forData['CA_TOT'][forData['CA_TOT'] < 0] = np.nan
        forData['WS'][forData['WS'] < 0] = np.nan
        forData['WD'][forData['WD'] < 0] = np.nan
        forData['SWR'][forData['SWR'] < 0] = np.nan

        inpData = forData.merge(pvDataL1, how='left', left_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'], right_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'])
        inpDataL1 = inpData.sort_values(by=['ANA_DATE','DATE_TIME_KST'], axis=0)
        prdData = inpDataL1.copy()

        result = {
            'msg': 'succ'
            , 'inpDataL1': inpDataL1
            , 'prdData': prdData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subModelInpData'))


def subVisPrd(subOpt, prdData):

    log.info('[START] {}'.format('subVisPrd'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))
    # log.info('[CHECK] prdData : {}'.format(prdData))

    result = None

    try:

        anaDataList = prdData['ANA_DATE'].unique()

        minAnaData = pd.to_datetime(anaDataList).min().strftime("%Y%m%d")
        maxAnaData = pd.to_datetime(anaDataList).max().strftime("%Y%m%d")
        # anaDataInfo = anaDataList[0]
        # for idx, anaDataInfo in enumerate(anaDataList):
        #
        #     prdDataL1 = prdData.loc[
        #         prdData['ANA_DATE'] == anaDataInfo
        #         ].dropna().reset_index(drop=True)
        #
        #     mainTitle = '[{}] {}'.format(pd.to_datetime(anaDataInfo).strftime("%Y%m%d"), '기상 예보 정보 (수치모델)를 활용한 48시간 예측 시계열')
        #     saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        #
        #     if (os.path.exists(saveImg)): continue
        #     rtnInfo = makeUserTimeSeriesPlot(pd.to_datetime(prdDataL1['DATE_TIME']), prdDataL1['ML2'], prdDataL1['DL2'], prdDataL1['PV'], '예측 (머신러닝)', '예측 (딥러닝)', '실측 (발전량)', '시간 (시)', '발전량', mainTitle, saveImg, True)
        #     log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 머신러닝 (48시간 예측) 2D 산점도')
        saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        rtnInfo = makeUserHist2dPlot(prdData['ML2'], prdData['PV'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 딥러닝 (48시간 예측) 2D 산점도')
        saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        rtnInfo = makeUserHist2dPlot(prdData['DL2'], prdData['PV'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        result = {
            'msg': 'succ'
            , 'prdData': prdData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subVisPrd'))


def reqPvApi(cfgInfo, dtYmd, id):

    # log.info('[START] {}'.format('subVisPrd'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dtYmd : {}'.format(dtYmd))
    # log.info('[CHECK] id : {}'.format(id))

    result = None

    try:

        apiUrl = cfgInfo['apiUrl']
        apiToken = cfgInfo['apiToken']
        stnId = id
        srvId = 'SRV{:05d}'.format(id)

        reqUrl = '{}/{}/{}'.format(apiUrl, dtYmd, id)
        reqHeader = {'Authorization': 'Bearer {}'.format(apiToken)}
        res = requests.get(reqUrl, headers=reqHeader)

        if not (res.status_code == 200): return result
        resJson = res.json()

        if not (resJson['success'] == True): return result
        resInfo = resJson['pvs']

        if (len(resInfo) < 1): return result
        resData = pd.DataFrame(resInfo)

        resData = resData.rename(
            {
                'pv': 'PV'
            }
            , axis='columns'
        )

        resData['SRV'] = srvId
        resData['DATE_TIME_KST'] = pd.to_datetime(resData['date'], format='%Y-%m-%d %H')
        resData['DATE_TIME'] = resData['DATE_TIME_KST'] - dtKst
        resData = resData.drop(['date'], axis='columns')

        result = {
            'msg': 'succ'
            , 'resData': resData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    # finally:
    #     # try, catch 구문이 종료되기 전에 무조건 실행
    #     log.info('[END] {}'.format('reqPvApi'))


def setPvDataDB(cfgInfo, dbData, dtYear):

    # log.info('[START] {}'.format('setPvDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))
    # log.info('[CHECK] dtYear : {}'.format(dtYear))

    # result = None

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']
        selDbTable = 'TB_PV_DATA_{}'.format(dtYear)

        session.execute(
            """
                CREATE TABLE IF NOT EXISTS `{}`
                (
                    SRV           varchar(10) not null comment '관측소 정보',
                    DATE_TIME     datetime    not null comment '날짜 UTC',
                    DATE_TIME_KST datetime    null comment '날짜 KST',
                    PV            float       null comment '발전량',
                    REG_DATE      datetime    null comment '등록일',
                    MOD_DATE      datetime    null comment '수정일',
                    primary key (SRV, DATE_TIME)
                )    
                    comment '발전량 테이블_{}';
            """.format(selDbTable, dtYear)
        )

        for idx, dbInfo in dbData.iterrows():

            # 중복 검사
            resChk = pd.read_sql(
                """
                SELECT COUNT(*) AS CNT FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                , dbEngine
            )

            if (resChk.loc[0, 'CNT'] > 0):
                dbInfo['MOD_DATE'] = datetime.datetime.now()
                session.execute(
                    """
                    UPDATE `{}` SET PV = '{}', MOD_DATE = '{}' WHERE SRV = '{}' AND DATE_TIME = '{}'; 
                    """.format(selDbTable, dbInfo['PV'], dbInfo['MOD_DATE'], dbInfo['SRV'], dbInfo['DATE_TIME'])
                )

            else:
                dbInfo['REG_DATE'] = datetime.datetime.now()
                session.execute(
                    """
                    INSERT INTO `{}` (SRV, DATE_TIME, DATE_TIME_KST, PV, REG_DATE, MOD_DATE) VALUES ('{}', '{}', '{}', '{}', '{}', '{}') 
                    """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['PV'], dbInfo['REG_DATE'], dbInfo['REG_DATE'])
                )

            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()

        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setPvDataDB'))


def subPvData(cfgInfo, sysOpt, posDataL1, dtIncDateList):

    log.info('[START] {}'.format('subPvData'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
    # log.info('[CHECK] posDataL1 : {}'.format(posDataL1))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))
    result = None

    try:
        # dtIncDateInfo = dtIncDateList[0]
        stnId = sysOpt.get('stnId')

        for i, dtIncDateInfo in enumerate(dtIncDateList):
            log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

            dtYmd = dtIncDateInfo.strftime('%Y/%m/%d')
            dtYear = dtIncDateInfo.strftime('%Y')

            isSearch = True if ((stnId == None) or (len(stnId) < 1)) else False
            if (isSearch):
                for j, posInfo in posDataL1.iterrows():
                    id = int(posInfo['ID'])

                    result = reqPvApi(cfgInfo, dtYmd, id)
                    # log.info("[CHECK] result : {}".format(result))

                    resData = result['resData']
                    if (len(resData) < 1): continue

                    setPvDataDB(cfgInfo, resData, dtYear)
            else:
                id = int(stnId)

                result = reqPvApi(cfgInfo, dtYmd, id)
                # log.info("[CHECK] result : {}".format(result))

                resData = result['resData']
                if (len(resData) < 1): continue

                setPvDataDB(cfgInfo, resData, dtYear)

        result = {
            'msg': 'succ'
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subPvData'))


def setPrdDataDB(cfgInfo, dbData):

    # log.info('[START] {}'.format('setPrdDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']

        for k, dbInfo in dbData.iterrows():

            iAnaYear = int(dbInfo['ANA_DATE'].strftime("%Y"))
            selDbTable = 'TEST_TB_FOR_DATA_{}'.format(iAnaYear)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            # 테이블 중복 검사
            resChk = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM `{}`
                    WHERE SRV = '{}' AND ANA_DATE = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'])
                , con=dbEngine
            )

            if (resChk.loc[0, 'CNT'] < 1): continue

            dbInfo['MOD_DATE'] = datetime.datetime.now()
            
            session.execute(
                """
                UPDATE `{}` SET  ML2 = '{}', DL2 = '{}'
                WHERE SRV = '{}' AND ANA_DATE = '{}' AND DATE_TIME = '{}';
                """.format(selDbTable
                           , dbInfo['ML2'], dbInfo['DL2']
                           , dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'])
            )

            session.commit()

#            if (resChk.loc[0, 'CNT'] > 0):
#                dbInfo['MOD_DATE'] = datetime.datetime.now()
#
#                session.execute(
#                    """
#                    UPDATE `{}`
#                    SET DATE_TIME_KST = '{}', CA_TOT = '{}', HM = '{}', PA = '{}', TA = '{}', TD = '{}', WD = '{}', WS = '{}', SZA = '{}', AZA = '{}', ET = '{}', TURB = '{}'
#                    , GHI_CLR = '{}', DNI_CLR = '{}', DHI_CLR = '{}', GHI_POA = '{}', DNI_POA = '{}', DHI_POA = '{}', SWR = '{}', MOD_DATE = '{}', ML = '{}', DL = '{}', ML2 = '{}', DL2 = '{}'
#                    WHERE SRV = '{}' AND ANA_DATE = '{}' AND DATE_TIME = '{}';
#                    """.format(selDbTable
#                               , dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
#                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR']
#                               , dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['SWR'], dbInfo['MOD_DATE']
#                               , dbInfo['ML'], dbInfo['DL'], dbInfo['ML2'], dbInfo['DL2']
#                               , dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'])
#                )
#
#            else:
#                dbInfo['REG_DATE'] = datetime.datetime.now()
#
#                session.execute(
#                    """
#                    INSERT INTO `{}` (SRV, ANA_DATE, DATE_TIME, DATE_TIME_KST, CA_TOT, HM, PA, TA, TD, WD, WS, SZA, AZA, ET, TURB, GHI_CLR, DNI_CLR, DHI_CLR, GHI_POA, DNI_POA, DHI_POA, SWR, REG_DATE, ML, DL, ML2, DL2)
#                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
#                    """.format(selDbTable
#                               , dbInfo['SRV'], dbInfo['ANA_DATE'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
#                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR'], dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['SWR'], dbInfo['REG_DATE']
#                               , dbInfo['ML'], dbInfo['DL'], dbInfo['ML2'], dbInfo['DL2']
#                               )
#                )
#            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()
        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setPrdDataDB'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
    # nohup bash RunShell-Python.sh "2020-10" &

    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
    # python3 ${contextPath}/TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "$1" --endDate "$2"
    # python3 TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "20210901" --endDate "20210902"
    # python3 /SYSTEMS/PROG/PYTHON/PV/TalentPlatform-LSH0255-RealTime-For.py --inpPath "/DATA" --outPath "/SYSTEMS/OUTPUT" --modelPath "/DATA" --srtDate "20220101" --endDate "20220102"

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV_20220523'

    prjName = 'test'
    serviceName = 'LSH0000'

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

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'
                globalVar['figPath'] = 'E:/DATA/OUTPUT'
                globalVar['modelPath'] = 'E:/DATA'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': '2021-10-01'
                    # , 'endDate': '2021-10-10'
                    'srtDate': '2019-01-01'
                    , 'endDate': '2022-05-22'
                    # , 'endDate': '2021-11-01'

                    , 'stnId': '1'

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        , 'isOverWrite': True
                        #, 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        # 'isOverWrite': True
                        'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                    , 'stnId': globalVar['stnId']

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        , 'isOverWrite': True
                        #, 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        'isOverWrite': True
                        #'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

            # DB 정보
            cfgInfo = initCfgInfo(globalVar['sysPath'])
            dbEngine = cfgInfo['dbEngine']

            # =======================================================================
            # 관측소 정보
            # =======================================================================
            posDataL1 = pd.read_sql(
                """
               SELECT *
               FROM TB_STN_INFO
               WHERE ID = '{}'
               """.format(sysOpt['stnId'])
            , con = dbEngine)

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))

            # =======================================================================
            # 기상정보 자료 수집 및 DB 삽입
            # =======================================================================
            subAtmosData(cfgInfo, globalVar['cfgPath'], posDataL1, dtIncDateList)

            # =======================================================================
            # 발전량 자료 수집 및 DB 삽입
            # =======================================================================
            #subPvData(cfgInfo, sysOpt, posDataL1, dtIncDateList)

            # =======================================================================
            # 발전량 관측소에 따른 머신러닝/딥러닝 예측
            # =======================================================================
            # for idx, posInfo in posDataL1.iterrows():
            #     posId = int(posInfo['ID'])
            #     posVol = posInfo['VOLUME']
            #
            #     srvId = 'SRV{:05d}'.format(posId)
            #     log.info("[CHECK] srvId : {}".format(srvId))
            #
            #     result = subModelInpData(sysOpt, cfgInfo, srvId, posVol, dtIncDateList)
            #     inpDataL1 = result['inpDataL1']
            #     prdData = result['prdData']
            #
            #     if (len(prdData) < 1): continue
            #
            #     log.info("[CHECK] len(prdData) : {}".format(len(prdData)))
            #
            #     # ******************************************************************************************************
            #     # 머신러닝
            #     # ******************************************************************************************************
            #     xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'SZA', 'AZA', 'ET']
            #     yCol = 'PV'
            #
            #     sysOpt['mlModel'].update(
            #         {
            #             'srvId' : srvId
            #             , 'modelKey' : 'AI-FOR-20220521'
            #         }
            #     )
            #
            #     # 머신러닝 불러오기
            #     result = makeMlModel(sysOpt['mlModel'], xCol, yCol, inpDataL1)
            #     log.info('[CHECK] result : {}'.format(result))
            #
            #     # 머신러닝 예측
            #     mlModel = result['mlModel']
            #     prdData['ML2'] = predict_model(mlModel, data=prdData)['Label']
            #
            #     # ******************************************************************************************************
            #     # 딥러닝
            #     # ******************************************************************************************************
            #     xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'SZA', 'AZA', 'ET']
            #     yCol = 'PV'
            #
            #     sysOpt['dlModel'].update(
            #         {
            #             'srvId': srvId
            #             , 'modelKey': 'AI-FOR-20220521'
            #         }
            #     )
            #
            #     result = makeDlModel(sysOpt['dlModel'], xCol, yCol, inpDataL1)
            #     log.info('[CHECK] result : {}'.format(result))
            #
            #     # 딥러닝 예측
            #     dlModel = result['dlModel']
            #     # prdData['DL2'] = dlModel.predict(h2o.H2OFrame(inpDataL1)).as_data_frame()
            #     prdData['DL2'] = dlModel.predict(h2o.H2OFrame(prdData[xCol])).as_data_frame()
            #
            #     setPrdDataDB(cfgInfo, prdData)
            #
            #     # ******************************************************************************************************
            #     # 시각화
            #     # ******************************************************************************************************
            #     subVisPrd(sysOpt, prdData)

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
