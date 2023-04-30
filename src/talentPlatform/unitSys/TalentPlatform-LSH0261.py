# -*- coding: utf-8 -*-
import glob
# import seaborn as sns
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
import statsmodels.api as sm
from scipy.stats import linregress

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

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace('\\', '/')

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    for i, key in enumerate(inParams):
        # 리눅스 환경
        if globalVar['sysOs'] in 'Linux':
            if i >= len(sys.argv[1:]): continue
            if inParams[key] is None: continue
            val = inParams[key] if sys.argv[i + 1] is None else sys.argv[i + 1]

        # 원도우 또는 맥 환경
        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] is None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, prdVal, refVal, prdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg):

    # 그리드 설정
    plt.grid(True)

    plt.plot(dtDate, prdVal, label=prdValLabel)
    plt.plot(dtDate, refVal, label=refValLabel)

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

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
    Bias = np.mean(prdVal - refVal)
    rBias = (Bias / np.mean(refVal)) * 100.0
    RMSE = np.sqrt(np.mean((prdVal - refVal) ** 2))
    rRMSE = (RMSE / np.mean(refVal)) * 100.0
    MAPE = np.mean(np.abs((prdVal - refVal) / prdVal)) * 100.0

    # 선형회귀곡선에 대한 계산
    lmFit = linregress(prdVal, refVal)
    slope = lmFit[0]
    intercept = lmFit[1]
    R = lmFit[2]
    Pvalue = lmFit[3]
    N = len(prdVal)

    lmfit = (slope * prdVal) + intercept
    plt.plot(prdVal, lmfit, color='red', linewidth=2)

    # 라벨 추가
    plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept), xy=(minVal + xIntVal, maxVal - yIntVal),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')

    if (isSame == True):
        plt.axes().set_aspect('equal')

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


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0261'

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
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar[{}] {}".format(key, val))

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

            # 코스피지수 파일 패턴
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '코스피지수+20년치.xls')

            # 코스피지수 파일 찾기
            fileInfo = glob.glob(inpFile)

            # 코스피지수 파일 없을 경우 예외 처리
            if fileInfo is None or len(fileInfo) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            # 코스피지수 파일 읽기
            tmpData = pd.read_excel(fileInfo[0], sheet_name='Sheet1')

            # 코스피지수 컬럼명 설정
            tmpData.columns = ['yyyymm', 'tmp', 'tmp2', 'kospiLogVal']

            # 수출액 파일 패턴
            inpFile2 = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '한국+월별+수출액+20년치.xls')

            # 수출액 파일 찾기
            fileInfo2 = glob.glob(inpFile2)

            # 수출액 파일 없을 경우 예외 처리
            if fileInfo2 is None or len(fileInfo2) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile2, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile2, '입력 자료를 확인해주세요.'))

            # 수출액 파일 읽기
            tmpData2 = pd.read_excel(fileInfo2[0], sheet_name='Sheet1')

            # 수출액 컬럼명 설정
            tmpData2.columns = ['yyyymm', 'exprtVal', 'tmp', 'tmp2', 'tmp3', 'tmp4']

            # 연월 컬럼을 기준으로 데이터 병합
            data = pd.merge(tmpData, tmpData2, how="left", on="yyyymm")

            # 결측값 제거
            dataL1 = data[['yyyymm', 'kospiLogVal', 'exprtVal']].dropna(axis=0)

            # 연월 컬럼에서 문자열을 날짜형으로 변환
            dataL1["dtDate"] = pd.to_datetime(dataL1["yyyymm"], format='%Y%m')

            # ************************************************************************
            # 산점도 시각화
            # ************************************************************************
            mainTitle = '한국 월별 수출액과 코스피 종간 간의 산점도'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserScatterPlot(dataL1['exprtVal'], dataL1['kospiLogVal'], '수출액', '코스피', mainTitle, saveImg, 100, 3.5, 5, 0.06, False)

            # ************************************************************************
            # 회귀분석
            # ************************************************************************
            # 회귀모형

            smfModelFor = sm.OLS.from_formula('kospiLogVal ~ exprtVal', dataL1)

            # 회귀모형 적합
            smfModel = smfModelFor.fit()

            # 회귀모형 예측 결과
            smfModelRes = smfModel.predict()
            dataL1['smfModelRes'] = smfModelRes

            # 회귀모델 결과 요약
            smfModel.summary()

            # ************************************************************************
            # 산점도 시각화
            # ************************************************************************
            mainTitle = '한국 월별 수출액을 이용한 코스피 예측 결과'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserScatterPlot(dataL1['smfModelRes'], dataL1['kospiLogVal'], '코스피 예측', '코스피 실측', mainTitle, saveImg,
                                2.8, 3.6, 0.02, 0.04, True)

            # ************************************************************************
            # 시계열 시각화
            # ************************************************************************
            mainTitle = '한국 월별 수출액을 이용한 코스피 시계열 결과'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserTimeSeriesPlot(dataL1['dtDate'], dataL1['kospiLogVal'], dataL1['smfModelRes'], '예측', '실측', '날짜',
                                   '코스피', mainTitle, saveImg)

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
