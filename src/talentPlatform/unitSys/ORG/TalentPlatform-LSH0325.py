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
import glob
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import datetime as dt
from plotnine import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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
    # Python을 이용한 최적화 알고리즘을 이용한 회귀식 추정 (GA 알고리즘, SA 알고리즘)

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0325'

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

        try:
            log.info('[START] {}'.format("exec"))

            # ********************************************
            # 옵션 설정
            # ********************************************
            sysOpt = {
            }

            # ==========================================================================
            # 회귀식 추정
            # ==========================================================================
            # 각각 하나의 독립변수와 종속변수로 이루어진 데이터 집합에 대해,
            # 두 변수의 관계를 가장 잘 설명할 수 있는 수학적 모형(1차 또는 2차 회귀식)을 가정하고
            # 에러를 최소화하는 모수 값을 최적화 알고리즘(Genetic Algorithm, Simulated Annealing)을 이용하여 추정하세요.

            # 1974년에 미국의 모터 트렌드 잡지에 실린 1973 ~ 1974년 자동차 모델의 연료 소비, 10가지 디자인 요소, 성능을 비교한 mtcars 데이터 사용
            data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv')

            # 무게 (wt)에 따른 연비 (mpg) 예측 수행
            # 독립변수 : 무게 (wt)
            # 종속변수 : 연비 (mpg)
            x = 'wt'
            y = 'mpg'

            # *******************************************************************************************************
            # Genetic Algorithm
            # *******************************************************************************************************
            log.info('[CHECK] {}'.format('Genetic Algorithm'))

            # 회귀모형 초기화
            model = smf.ols('mpg ~ wt', data=data)
            model = model.fit()

            # 요약
            # model.summary()

            # (mpg) = -5.344472 * (wt) + 37.285126
            log.info('[CHECK] params : {}'.format(model.params))

            # mpg 예측
            data['prd'] = model.predict()

            # 산점도 시각화
            mainTitle = '{}'.format('무게에 따른 연비 예측')
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, mainTitle)

            plt.plot(data['wt'], data['mpg'], 'o')
            plt.plot(data['wt'], data['prd'], 'r', linewidth=2)
            plt.annotate('%s = %.2f x (%s) + %.2f' % ('mpg', model.params[1], 'wt', model.params[0]), xy=(3.5, 34),
                         color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('R = %.2f  (p-value < %.2f)' % (np.sqrt(model.rsquared), model.f_pvalue), xy=(3.5, 32),
                         color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('RMSE = %.2f' % (np.sqrt(mean_squared_error(data['mpg'], data['prd']))), xy=(3.5, 30),
                         color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.xlabel('wt')
            plt.ylabel('mpg')
            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()

            log.info('[CHECK] saveImg : {}'.format(saveImg))

            # *******************************************************************************************************
            # Simulated Annealing
            # *******************************************************************************************************
            log.info('[CHECK] {}'.format('Simulated Annealing'))
            slopeList = np.arange(-100, 100, 1)
            interceptList = np.arange(-100, 100, 1)

            # slopeList = np.arange(-5, 5, 0.01)
            # slopeList = np.arange(-10, 10, 0.1)
            # interceptList = np.arange(30, 40, 0.1)

            ref = data['mpg']

            dataL1 = pd.DataFrame()
            for i, slope in enumerate(slopeList):
                for j, intercept in enumerate(interceptList):
                    prd = (data['wt'] * slope) + intercept
                    rmse = np.sqrt(mean_squared_error(ref, prd))

                    dict = {
                        'slope': [slope]
                        , 'intercept': [intercept]
                        , 'rmse': [rmse]
                    }

                    dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)

            idx = dataL1.idxmin()['rmse']

            # 앞선 회귀모형과 유사하게 근사값으로 출력
            # 보다 상세한 시뮬레이션을 위해서 slopeList, interceptList의 간격을 조밀하게 설정 (현재 0.1 간격)
            log.info('[CHECK] params : {}'.format(dataL1.iloc[idx,]))

            # 산점도 시각화
            mainTitle = '{}'.format('시뮬레이션에 따른 RMSE 결과')
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, mainTitle)

            plt.plot(dataL1['rmse'], 'o')
            plt.xlabel('index')
            plt.ylabel('RMSE')
            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()

            log.info('[CHECK] saveImg : {}'.format(saveImg))

            # # ==========================================================================
            # # 혹시 월 별 따릉이 이용건수 이런 것도 가능할까요??
            # # ==========================================================================
            # print('[CHECK] Unit : {}'.format('Visualization'))
            #
            # # 위 사이트에서 21년도 01월-7월 데이터로 하고싶습니다
            # visData = pd.read_csv('./서울특별시 공공자전거 일별 대여건수_21.07-21.12.csv', encoding='EUC-KR')
            #
            # visData['dtDate'] = pd.to_datetime(visData['대여일시'], format='%Y-%m-%d')
            # visData['월'] = visData['dtDate'].dt.strftime("%m")
            # visData['대여건수'] = visData['대여건수'].replace(',', '', regex=True).astype('float64')
            #
            # visDataL1 = visData.groupby(['월']).sum().reset_index()
            #
            # mainTitle = '{}'.format('월에 따른 따릉이 이용건수')
            # saveImg = './{}'.format(mainTitle)
            #
            # plot = (
            #         ggplot(data=visDataL1) +
            #         aes(x='월', y='대여건수', fill='대여건수') +
            #         theme_bw() +
            #         geom_bar(stat='identity') +
            #         labs(title=mainTitle, xlab='월', ylab='대여 건수') +
            #         theme(
            #             text=element_text(family="Malgun Gothic", size=18)
            #             # , axis_text_x=element_text(angle=45, hjust=1, size=6)
            #             # , axis_text_y=element_text(size=10)
            #         )
            # )
            #
            # fig = plot.draw()
            # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
            # fig.show()

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