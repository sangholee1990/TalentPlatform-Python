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
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

import seaborn as sns
import pickle

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
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
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 성별, 연령별에 따른 흡연자 비중 및 비율 시각화

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'BDWIDE2024'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2019-01-01'
                , 'endDate': '2023-01-01'
            }

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'SERVICE1_INPUT1_DATA.csv')
            fileList = sorted(glob.glob(inpFile))
            prdData = pd.read_csv(fileList[0]).reset_index(drop=True)
            prdColList = ['YEAR', 'MONTH', 'DAY', 'STATE_ABBR', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD']
            prdDataL1 = prdData[prdColList].groupby(['YEAR', 'MONTH', 'DAY', 'STATE_ABBR']).mean().reset_index()

            prdDataL1['DATE'] = pd.to_datetime(prdDataL1[['YEAR', 'MONTH', 'DAY']])
            prdDataL2 = prdDataL1[prdDataL1['DATE'] > '2020-07-01'].reset_index(drop=True)
            # prdDataL2 = prdDataL1.loc[prdDataL1['YEAR'].isin([2020, 2021])]


            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'SERVICE1_INPUT2_DATA.csv')
            fileList = sorted(glob.glob(inpFile))
            inpData = pd.read_csv(fileList[0]).reset_index(drop=True)
            inpDataL1 = inpData[['YEAR', 'MONTH', 'DAY', 'STATE_ABBR', 'PRD_CODE', 'PRD_AMT', 'PRD_CNT']]

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'SERVICE1_PROP_DATA.csv')
            fileList = sorted(glob.glob(inpFile))
            propData = pd.read_csv(fileList[0]).reset_index(drop=True)

            # mrgColList = ['YEAR', 'MONTH', 'DAY', 'STATE_ABBR', 'PRD_AMT']
            mrgColList = ['YEAR', 'MONTH', 'DAY', 'STATE_ABBR', 'PRD_CNT']
            propDataL1 = propData.drop(['STATE_NAME', 'REG_DATE', 'MOD_DATE'], axis=1).groupby(mrgColList).mean().reset_index()

            mrgData = pd.merge(left=propDataL1, right=inpDataL1, how='left', left_on=mrgColList, right_on=mrgColList)
            # mrgDataL1 = mrgData.drop_na()

            # prdCodeList = set(mrgData['PRD_CODE'])
            # stateAbbrList = set(mrgData['STATE_ABBR'])

            # len(prdCodeList)
            # mrgDataL1 =  mrgData.loc[(mrgData['PRD_CODE'] == 'B089GNWBT4')]
            mrgDataL1 =  mrgData

            # mrgData.columns
            # mrgDataL1.columns
            # NO, DATE_DECI, RES_CNT, REG_DATE, MOD_DATE
            # mrgDataL2 = mrgDataL1[['YEAR', 'MONTH', 'DAY', 'PRD_CNT', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD', 'PRD_CODE']]
            mrgDataL2 = mrgDataL1[['YEAR', 'MONTH', 'DAY', 'PRD_CNT', 'STATE_ABBR', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD', 'PRD_CODE']]

            # mrgColList = ['YEAR', 'MONTH', 'DAY', 'PRD_CODE']
            mrgColList = ['YEAR', 'MONTH', 'DAY', 'PRD_CODE', 'STATE_ABBR']
            sumPropData = mrgDataL2.groupby(mrgColList).mean().drop(['PRD_CNT'], axis=1).reset_index()
            meanPropData = mrgDataL2.groupby(mrgColList).sum()['PRD_CNT'].reset_index()

            mrgDataL3 = pd.merge(left=sumPropData, right=meanPropData, how='left', left_on=mrgColList, right_on=mrgColList)
            mrgDataL3['PRD_CODE'] = mrgDataL3['PRD_CODE'].astype('category')
            mrgDataL3['STATE_ABBR'] = mrgDataL3['STATE_ABBR'].astype('category')

            # len(set(mrgData['STATE_ABBR']))
            # mrgDataL3.columns

            mrgDataL4 = mrgDataL3
            mrgDataL4['DATE'] = pd.to_datetime(mrgDataL4[['YEAR', 'MONTH', 'DAY']])
            import optuna.integration.lightgbm as lgb

            # xCol = ['PRD_CODE', 'PRD_AMT', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD']
            # xCol = ['PRD_CODE', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD']
            xCol = ['PRD_CODE', 'STATE_ABBR', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD']
            yCol = 'PRD_CNT'

            # sorted(set(trainDataL1['stn']))
            # sorted(set(trainDataL4['stn']))
            lgbTrainData = lgb.Dataset(mrgDataL4[xCol], mrgDataL4[yCol])
            lgbTestData = lgb.Dataset(mrgDataL4[xCol], mrgDataL4[yCol], reference=lgbTrainData)

            params = {
                'objective': 'regression'
                , 'metric': 'rmse'
                , 'verbosity': -1
                , 'n_jobs': -1
            }

            # lgbModel = lgb.train(params=params, train_set=lgbTrainData, num_boost_round=10000, valid_sets=[lgbTrainData, lgbTestData])

            saveModel = '{}/{}/{}.model'.format(globalVar['outPath'], serviceName, 'lgbModel-20240808')
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            # pickle.dump(lgbModel, open(saveModel, 'wb'))
            log.info(f'[CHECK] saveFile : {saveModel}')

            lgbModel = pickle.load(open(saveModel, 'rb'))
            mrgDataL4['RES_VAL'] = lgbModel.predict(data=mrgDataL4[xCol])
            mrgDataL4['RES_CNT'] = mrgDataL4['RES_VAL'].apply(lambda x: round(x)).astype(int)

            # 변수 중요도 저장
            try:
                mainTitle = '{}'.format('lgb-importance')
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                lgb.plot_importance(lgbModel)
                plt.title(mainTitle)
                plt.tight_layout()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                plt.show()
                plt.close()
            except Exception as e:
                log.error('Exception : {}'.format(e))

            from sklearn.metrics import mean_squared_error
            import matplotlib.dates as mdates
            # import matplotlib as mpl
            # import matplotlib.pyplot as plt

            plt.plot(mrgDataL4['DATE'], mrgDataL4[yCol], marker='o', label='실측')
            plt.plot(mrgDataL4['DATE'], mrgDataL4['RES_CNT'], label='예측 (RMSE : {:.2f}, {:.2f}%)'.format(
                mean_squared_error(mrgDataL4[yCol], mrgDataL4['RES_CNT'], squared=False)
                , (mean_squared_error(mrgDataL4[yCol], mrgDataL4['RES_CNT'], squared=False) / np.nanmean(mrgDataL4[yCol])) * 100.0)
                )
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d %H'))
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d %H'))
            # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
            # plt.gcf().autofmt_xdate()
            # plt.xticks(rotation=45, ha='right')
            plt.xticks(rotation=45)
            plt.title(mainTitle)
            plt.legend()
            plt.tight_layout()
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()



            # mrgDataL3
            prdDataL4 = pd.DataFrame()
            prdCodeList = set(mrgDataL4['PRD_CODE'])
            for prdCode in prdCodeList:
                log.info(f'[CHECK] prdCode : {prdCode}')

                mrgDataL5 =  mrgDataL4.loc[(mrgDataL4['PRD_CODE'] == prdCode)].reset_index(drop=True)
                if len(mrgDataL5) < 1: continue

                prdDataL2['PRD_CODE'] = prdCode
                prdDataL2['PRD_CODE'] = prdDataL2['PRD_CODE'].astype('category')
                prdDataL2['STATE_ABBR'] = prdDataL2['STATE_ABBR'].astype('category')
                prdDataL2['RES_VAL'] = lgbModel.predict(data=prdDataL2[xCol])
                prdDataL2['RES_CNT'] = prdDataL2['RES_VAL'].apply(lambda x: round(x)).astype(int)

                # prdDataL2 = pd.merge(left=prdDataL1, right=mrgDataL5[['YEAR', 'MONTH', 'DAY', 'PRD_CODE', 'PRD_CNT']], how='left', on=['YEAR', 'MONTH', 'DAY', 'PRD_CODE'])
                prdDataL3 = pd.merge(left=prdDataL2, right=mrgDataL5[['YEAR', 'MONTH', 'DAY', 'PRD_CODE', 'STATE_ABBR', 'PRD_CNT']], how='left', on=['YEAR', 'MONTH', 'DAY', 'PRD_CODE', 'STATE_ABBR'])
                prdDataL4 = pd.concat([prdDataL4, prdDataL3], axis=0)

            prdDataL5 = prdDataL4.reset_index(drop=True)

            # 중복 개수
            prdDataL5.duplicated(keep = False).sum()
            prdDataL6 = prdDataL5.dropna().reset_index(drop=True)

            mean_squared_error(prdDataL6[yCol], prdDataL6['RES_CNT'], squared=False)
            prdDataL5[[yCol,'RES_CNT']].corr()

            prdDataL5['NO'] = prdDataL5.index + 1
            prdDataL5['REG_DATE'] = '2024-08-08'
            prdDataL5['MOD_DATE'] = '2024-08-08'

            # prdDataL5.columns
            prdDataL6 = prdDataL5[['NO', 'YEAR', 'MONTH', 'DAY', 'PRD_CODE', 'PRD_CNT', 'STATE_ABBR', 'INF_POP_CNT', 'MAX_TEMP', 'MIN_TEMP', 'UN_IDX', 'HEAT_IDX', 'CLD_COVER', 'HUM', 'TEMP', 'VIS', 'WIN_DIR', 'WIN_SPD', 'RES_VAL', 'RES_CNT', 'REG_DATE', 'MOD_DATE']]

            saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'SERVICE1_RESULT_DATA.csv')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            prdDataL6.to_csv(saveFile, index=False)
            log.info(f'[CHECK] saveFile : {saveFile}')

        except Exception as e:
            log.error(f"Exception : {str(e)}")
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