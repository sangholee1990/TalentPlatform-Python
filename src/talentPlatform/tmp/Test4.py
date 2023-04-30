import logging
import logging.handlers
import os
import sys
# from plotnine import *
# from plotnine.data import *
# from dfply import *
# import hydroeval
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

# 초기 환경변수 정의
from src.talentPlatform.unitSysHelper.InitConfig import *

class DtaProcess(object):
    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0177'

    global log, globalVar
    log = initLog(contextPath, prjName)
    globalVar = initGlobalVar(contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 파이썬 실행 시 전달인자 설정
            for i, key in enumerate(inParams):
                if i >= len(sys.argv[1:]): continue
                val = inParams[key] if sys.argv[i + 1] == None else sys.argv[i + 1]

                log.info("[CHECK] key : {} / val : {}".format(key, val))

                setattr(self, key, val)

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

            # 금리
            # 매매가격지수
            # 토지시장
            # 소비자심리지수
            # 소비자물가
            # 총통화량
            # 경제성장률(실질GDP성장률)(분기별
            # 지가변동률

            # fileList = glob.glob(os.path.join(globalVar['inpPath'], 'LSH0110_seoul2.csv'))

            # for fileInfo in fileList:
            #     data = pd.read_csv(fileInfo, encoding="UTF-8")
            #     log.info('data : {} : {}'.format(len(data), fileInfo))


                # data["dtDate"] = pd.to_datetime(data["c1"], format='%Y년 %m월') + MonthEnd(1)
                # data = data.set_index('dtDate')
                # data.drop(['c1'], axis = 1, inplace = True)
                #
                # splitDate = pd.Timestamp('01-01-2020')
                #
                # trainData = data.loc[:splitDate, ]
                # testData = data.loc[splitDate:, ]
                #
                # # dataL1[xCol]
                #
                # # trainData, testData = train_test_split(data, test_size=30, random_state=1)
                #
                # log.info('trainData : {}'.format(len(trainData)))
                # log.info('testData : {}'.format(len(testData)))

                # breakpoint()

                # sc = MinMaxScaler()
                # trainScData = sc.fit_transform(trainData)
                # testScData = sc.transform(testData)
                #
                # xTrainData = trainScData.drop('c8', axis = 1)
                # yTrainData = trainScData[['c8']]
                #
                # xTestData = testScData.drop('c8', axis = 1)
                # yTestData = testScData[['c8']]
                #
                #
                # print("변경 전 :", yTestData, yTestData.shape)
                # xTrainData = np.expand_dims(xTrainData, axis=2)
                # yTrainData = np.expand_dims(yTrainData, axis=2)
                # xTestData = np.expand_dims(xTestData, axis=2)
                # yTestData = np.expand_dims(yTestData, axis=2)
                # print("변경 후 :", yTestData, yTestData.shape)
                #
                #
                # K.clear_session()
                #
                # # Sequeatial Model
                # model = Sequential()
                #
                # # (timestep, feature)
                # model.add(LSTM(20, input_shape=(7, 1), return_sequences = True))
                #
                # # input = 1
                # model.add(Dense(1))
                # model.compile(loss='mean_squared_error', optimizer='adam')
                # model.summary()
                #
                #
                # fileName = Path(fileInfo).stem

                # 훈련 데이터셋 CSV 저장
                # saveTrainCsvFile = "{}/{}_{}_{}_{}.csv".format(
                #     globalVar.get('outPath')
                #     , fileName
                #     , "trainData"
                #     , self.trainPerRat
                #     , 100 - self.trainPerRat
                # )
                # pd.DataFrame(trainData).to_csv(saveTrainCsvFile, index=False)
                # log.info('saveTrainCsvFile : {} : {}'.format(len(glob.glob(saveTrainCsvFile)), saveTrainCsvFile))

                # 테스트 데이터셋 CSV 저장
                # saveTestCsvFile = "{}/{}_{}_{}_{}.csv".format(
                #     globalVar.get('outPath')
                #     , fileName
                #     , "testData"
                #     , self.trainPerRat
                #     , 100 - self.trainPerRat
                # )
                # pd.DataFrame(testData).to_csv(saveTestCsvFile, index=False)
                # log.info('saveTestCsvFile : {} : {}'.format(len(glob.glob(saveTestCsvFile)), saveTestCsvFile))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

if __name__ == '__main__':

    try:
        log.info('[START] {}'.format("main"))

        inParams = {
            'fileKey': 'AOD'
            , 'sDate': '20100102'
            , 'asdasd': 'asdfasdf'
            , 'asdfasf': 'asdfadf'
        }

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.exec()

    except Exception as e:
        log.error("Exception : {}".format(e))
        sys.exit(1)
    finally:
        log.info('[END] {}'.format("main"))