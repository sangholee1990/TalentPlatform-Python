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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split

# 초기 환경변수 정의
from src.talentPlatform.unitSysHelper.InitConfig import *


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 범죄횟수을 위한 회귀모형

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0012'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    globals()['log'] = initLog(env, contextPath, prjName)

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

            # fileInfoPattrn = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'data/csv/inp_01.csv')
            # fileInfo = glob.glob(fileInfoPattrn)
            # if (len(fileInfo) < 1): raise Exception("[ERROR] fileInfo : {} : {}".format("자료를 확인해주세요.", fileInfoPattrn))

            # saveFile = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '2021_nagano_S1_01_raw.png')
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # fileInfo = fileList[0]
            # for i, fileInfo in enumerate(fileList):
            #     globalVar['inpData{0:02d}'.format(i + 1)] = pd.read_csv(fileInfo, na_filter=False)

            # breakpoint()

            # 파일 읽기
            fileInfoPattrn = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'data/csv/inp_*.csv')
            fileList = glob.glob(fileInfoPattrn)
            if (len(fileList) < 6): raise Exception("[ERROR] fileInfo : {} : {}".format("자료를 확인해주세요.", fileInfoPattrn))

            inpData01 = pd.read_csv(fileList[0], na_filter=False)
            inpData02 = pd.read_csv(fileList[1], na_filter=False)
            inpData03 = pd.read_csv(fileList[2], na_filter=False)
            inpData04 = pd.read_csv(fileList[3], na_filter=False)
            inpData05 = pd.read_csv(fileList[4], na_filter=False)
            inpData06 = pd.read_csv(fileList[5], na_filter=False)

            # 기간 및 자치구에 따른 데이터 병합
            data = (
                (
                        inpData01 >>
                        dfply.left_join(inpData02, by=('기간', '자치구')) >>
                        dfply.left_join(inpData03, by=('기간', '자치구')) >>
                        dfply.left_join(inpData04, by=('기간', '자치구')) >>
                        dfply.left_join(inpData05, by=('기간', '자치구')) >>
                        dfply.left_join(inpData06, by=('기간', '자치구')) >>
                        dfply.mask(dfply.X.자치구 != '합계') >>
                        dfply.drop(
                            ['합계_검거', '살인_발생', '살인_검거', '강도_발생', '강도_검거', '강간강제추행_발생', '강간강제추행_검거', '절도_발생', '절도_검거',
                             '폭력_발생',
                             '폭력_검거', '합계', '소계'])
                )
            )

            # 컬럼 개수 : 42개
            len(data.columns.values)

            # 컬럼 형태
            # dataStep1.dtypes

            # ======================================================
            #  범죄횟수를 기준으로 각 상관계수 행렬 시각화
            # ======================================================
            # data = pd.DataFrame(data.dropna(axis=0))

            tmpColY = data.iloc[:, 2]
            tmpColXStep1 = data.iloc[:, 3:21:1]
            tmpColXStep2 = data.iloc[:, 22:41:1]

            dataStep1 = pd.concat([tmpColY, tmpColXStep1], axis=1)
            dataStep1Corr = dataStep1.corr(method='pearson')
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '상관계수 상단 행렬.png')

            makeCorrPlot(dataStep1, saveImg)

            dataStep2 = pd.concat([tmpColY, tmpColXStep2], axis=1)
            dataStep2Corr = dataStep2.corr(method='pearson')
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '상관계수 하단 행렬.png')

            makeCorrPlot(dataStep2, saveImg)

            # ===================================================================
            #  전체 데이터셋 (기간, 자치구)을 이용한 독립변수 및 종속 변수 선정
            # ===================================================================
            dataL1 = (
                (
                        data >>
                        dfply.drop(dfply.X.기간, dfply.X.자치구)
                )
            )

            # ===================================================================
            #  [상관분석 > 유의미한 변수] 전체 데이터셋 (기간, 자치구)을 이용한 독립변수 및 종속 변수 선정
            # ===================================================================
            selCol = ['범죄횟수', '지구대파출소치안센터', '119안전센터', 'CCTV설치현황', '비거주용건물내주택', '계_사업체수', '계_종사자수']
            dataL1 = data[selCol]

            # 결측값에 대한 행 제거 (그에 따른 index 변화로 인해 pd.DataFrame 재변환)
            dataL2 = pd.DataFrame(dataL1.dropna(axis=0))
            dataL2.rename(columns={'범죄횟수': 'total'}, inplace=True)

            # 요약 통계량
            dataL2.describe()

            # 자치구 데이터셋 (기간 평균)을 이용한 독립변수 및 종속 변수 선정
            # selCol = ['기간', '자치구', '범죄횟수', '지구대파출소치안센터', 'CCTV설치현황', '전체세대', '비거주용건물내주택', '계_사업체수']
            # dataL1 = data[selCol]
            #
            # pd.plotting.scatter_matrix(dataL1)
            # plt.show()
            #
            # dataL2 = ((dataL1 >>
            #      group_by(X.자치구) >>
            #      summarize(
            #          total=X.범죄횟수.mean()
            #          , maenX1=X.지구대파출소치안센터.mean()
            #          , maenX2=X.CCTV설치현황.mean()
            #          , maenX3=X.전체세대.mean()
            #          , maenX4=X.비거주용건물내주택.mean()
            #          , maenX5=X.계_사업체수.mean()
            #          ) >>
            #         # arrange(X.number, ascending=False)
            #         drop(X.자치구)
            #      ))

            # ========================================
            #  회귀모형 수행
            # ========================================
            selVarList = list(dataL2.columns[~dataL2.columns.str.contains('total')])

            # 다중선형회귀 모형
            result = train_test_linreg(dataL2, selVarList)

            # 릿지 모형
            # result = train_test_ridge(dataL2, selVarList, 1.0)

            # =======================================
            #  시각화
            # ======================================
            # 트레이닝 데이터
            trainValY = result['Y_train'].values
            trainPredValY = result['Y_pred_train']
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '트레이닝 데이터_상관계수 행렬.png')

            makeScatterPlot(trainValY, trainPredValY, saveImg)

            # 테스트 데이터
            testValY = result['Y_test'].values
            testPredValY = result['Y_pred_test']
            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '테스트 데이터_상관계수 행렬.png')

            makeScatterPlot(testValY, testPredValY, saveImg)

            # =======================================
            #  교차검증 수행
            # ======================================
            X = dataL2[selVarList]
            Y = dataL2.total
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

            # Pre-allocate models and corresponding parameter candidates
            models = []
            params = []

            model = ('Linear', LinearRegression())
            param = {}

            models.append(model)
            params.append(param)

            log.info("[CHECK] models : {%s}", models)
            log.info("[CHECK] params : {%s}", params)

            kfold = KFold(n_splits=10, shuffle=True)

            results = []

            # [교차검증] 트레이닝 데이터
            for i in range(1):
                model = models[i]
                param = params[i]
                result = gridsearch_cv_for_regression(model=model, param=param, kfold=kfold, train_input=X_train,
                                                      train_target=Y_train)
                result.best_score_
                results.append(result)

            # [교차검증] 테스트 데이터
            for i in range(len(results)):
                testValY = Y_test.values
                testPredValY = results[i].predict(X_test)

                saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '테스트 데이터_산점도.png')
                makeScatterPlot(testValY, testPredValY, saveImg)

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