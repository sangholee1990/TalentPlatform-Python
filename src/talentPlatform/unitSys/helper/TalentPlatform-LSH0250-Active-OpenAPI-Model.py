# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
from datetime import datetime

import h2o
# import googlemaps
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import Prophet
from h2o.automl import H2OAutoML
from matplotlib import font_manager
from pycaret.regression import *
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

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

# plt.rc('font', family='Malgun Gothic')
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

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        if (len(fontList) > 0):
            fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
            plt.rcParams['font.family'] = fontName
            plt.rc('font', family=fontName)

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

        log.info("[CHECK] {} : {}".format(key, val))

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

    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    # plt.show()
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
            # plt.xticks(rotation=0, ha='center')
            plt.xticks(rotation=45, ha='right', minor=False)

        plt.legend(loc='upper left')
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        # plt.show()
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

# 딥러닝 매매가/전세가 예측
def makeDlModel(subOpt=None, xCol=None, yCol=None, inpData=None, modelKey=None, addrCode=None):

    log.info('[START] {}'.format('makeDlModel'))

    result = None

    try:

        # saveModel = '{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'h2o', 'act', '*')
        saveModel = '{}/{}/{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, 'MODEL', addrCode, modelKey, 'final', 'h2o', 'act', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        # data = inpData[xyCol]
        data = inpData[xyCol].dropna()

        # h2o.shutdown(prompt=False)

        if (not subOpt['isInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(data, test_size=0.3)
            # trainData = inpData

            # dlModel = H2OAutoML(max_models=30, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=60, balance_classes=True, seed=123)
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainData), validation_frame=h2o.H2OFrame(validData))
            # dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(data))
            fnlModel = dlModel.get_best_model()

            # 학습 모델 저장
            # saveModel = '{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'h2o', 'act', datetime.now().strftime('%Y%m%d'))
            saveModel = '{}/{}/{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, 'MODEL', addrCode, modelKey, 'final', 'h2o', 'act', datetime.now().strftime('%Y%m%d'))
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
def makeMlModel(subOpt=None, xCol=None, yCol=None, inpData=None, modelKey=None, addrCode = None):
    log.info('[START] {}'.format('makeMlModel'))

    result = None

    try:

        # saveModel = '{}/{}/{}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'pycaret', 'act', '*')
        saveModel = '{}/{}/{}/{}/{}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], serviceName, 'MODEL', addrCode, modelKey, 'final', 'pycaret', 'act', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        # data = inpData[xyCol]
        data = inpData[xyCol].dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(data, test_size=0.3)
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
            saveModel = '{}/{}/{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, 'MODEL', addrCode, modelKey, 'final', 'pycaret', 'act', datetime.now().strftime('%Y%m%d'))
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

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

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
                    'isInit': False

                    # 모델 업데이트 여부
                    # , 'isOverWrite': True
                    , 'isOverWrite': False
                }

                #  머신러닝
                , 'mlModel': {
                    # 모델 업데이트 여부
                    # 'isOverWrite': True
                    'isOverWrite': False
                }

                #  시계열
                , 'tsModel': {
                    # 미래 예측 연도
                    'forYear': 2026
                    # 'forYear': 2026

                    # 아파트 설정
                    , 'aptList': [] # 전체 아파트 검색
                    # , 'aptList': ['미아동부센트레빌']
                    # , 'aptList': ['미아동부센트레빌', '송천센트레빌', '에스케이북한산시티']
                }

                # 검색 목록
                # , 'addrList': ['서울특별시 강북구']
                # , 'addrList': ['서울특별시 강북구', '서울특별시 송파구', '서울특별시 강남구', '서울특별시 양천구', '서울특별시 서초구']
                , 'addrList': ['서울특별시 양천구']
            }

            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUT'
            globalVar['figPath'] = '/DATA/FIG'

            # *********************************************************************************
            # 법정동 코드 읽기
            # *********************************************************************************
            inpFile = '{}/{}'.format(globalVar['mapPath'], 'admCode/법정동코드_전체자료.txt')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')

            admData = pd.read_csv(fileList[0], encoding='EUC-KR', sep='\t')
            admData[['d1', 'd2', 'd3', 'd4', 'd5']] = admData['법정동명'].str.split(expand=True)
            admData['sigunguCd'] = admData['법정동코드'].astype('str').str.slice(0, 5)
            admData['bjdongCd'] = admData['법정동코드'].astype('str').str.slice(5, 10)

            # *****************************************************
            # 건축 인허가
            # *****************************************************
            addrInfo = sysOpt['addrList'][0]
            for ii, addrInfo in enumerate(sysOpt['addrList']):

                admDataL1 = admData[admData['법정동명'].str.contains(addrInfo)]
                if (len(admDataL1) < 1): continue
                addrCode = np.unique(admDataL1['sigunguCd'])[0]

                log.info(f'[CHECK] addrInfo : {addrInfo}')

                lcnsInpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가_*_*')
                lcnsFileList = sorted(glob.glob(lcnsInpFile))
                if lcnsFileList is None or len(lcnsFileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(lcnsInpFile, '입력 자료를 확인해주세요.'))
                    raise Exception('[ERROR] inpFile : {} / {}'.format(lcnsInpFile, '입력 자료를 확인해주세요.'))

                lcnsData = pd.read_csv(lcnsFileList[0])
                # lcnsData.drop(['Unnamed: 0'], axis=1, inplace=True)
                lcnsDataL1 = lcnsData.groupby(['addrDtlInfo'], as_index=False)['archGbCdNm'].count()

                # *****************************************************
                # 아파트 전월세
                # *****************************************************
                # prvsMntsrInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 전월세가_인허가_20111101_20201101.csv')
                prvsMntsrInpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 전월세_*_*')
                prvsMntsrFileList = sorted(glob.glob(prvsMntsrInpFile))
                if prvsMntsrFileList is None or len(prvsMntsrFileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(prvsMntsrFileList, '입력 자료를 확인해주세요.'))
                    raise Exception('[ERROR] inpFile : {} / {}'.format(prvsMntsrFileList, '입력 자료를 확인해주세요.'))

                prvsMntsrData = pd.read_csv(prvsMntsrFileList[0])
                # prvsMntsrData.drop(['Unnamed: 0'], axis=1, inplace=True)

                prvsMntsrData['name'] = prvsMntsrData['아파트'] + '(' + prvsMntsrData['지번'] + ')'

                # prvsMntsrDataL2 = prvsMntsrData.loc[
                #     (prvsMntsrData['전월세구분'] == '전세')
                #     & (prvsMntsrData['층'] != 1)
                #     ].reset_index(drop=True)

                prvsMntsrDataL2 = prvsMntsrData.loc[
                    (prvsMntsrData['월세금액'] == 0)
                    & (prvsMntsrData['층'] != 1)
                    ].reset_index(drop=True)

                # prvsMntsrDataL2['계약년도'] = prvsMntsrDataL2['계약년월'].astype(str).str.slice(0, 4)
                prvsMntsrDataL2['계약년월'] = prvsMntsrDataL2['년'].astype(str) + '-' +  prvsMntsrDataL2['월'].astype(str)
                prvsMntsrDataL2['date'] = pd.to_datetime(prvsMntsrDataL2['계약년월'], format='%Y-%m')
                prvsMntsrDataL2['year'] = prvsMntsrDataL2['date'].dt.strftime("%Y").astype('int')
                prvsMntsrDataL2['month'] = prvsMntsrDataL2['date'].dt.strftime("%m").astype('int')
                # prvsMntsrDataL2['보증금(만원)'] = prvsMntsrDataL2['보증금(만원)'].astype(str).str.replace(',', '').astype('float')
                prvsMntsrDataL2['보증금액'] =  pd.to_numeric(prvsMntsrDataL2['보증금액'].astype(str).str.replace(',', ''), errors='coerce')

                prvsMntsrDataL3 = pd.merge(
                    left=prvsMntsrDataL2
                    # , right=lcnsData[['archGbCdNm', '주소', 'lat', 'lon']]
                    , right=lcnsDataL1[['archGbCdNm', 'addrDtlInfo']]
                    , left_on=['인허가addrDtlInfo']
                    , right_on=['addrDtlInfo']
                    , how='left'
                ).rename(
                    columns={
                        'archGbCdNm': 'inhuga'
                    }
                )

                prvsMntsrDataL4 = prvsMntsrDataL3.rename(
                    columns={
                        '전용면적': 'capacity'
                        , '건축년도': 'conYear'
                        , '보증금액': 'realBjprice'
                        , 'archGbCdNm': 'inhuga'
                        # , '거래금액(만원)': 'realPrice'
                    }
                ).sort_values(by=['name', 'capacity', 'year']).reset_index(drop=True)

                # *****************************************************
                # 실거래가 데이터
                # *****************************************************
                # realPriceInpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 실거래가_인허가_20111101_20201101.csv')
                realPriceInpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 실거래_*_*')
                realPriceFileList = glob.glob(realPriceInpFile)
                if realPriceFileList is None or len(realPriceFileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(realPriceInpFile, '입력 자료를 확인해주세요.'))
                    raise Exception('[ERROR] inpFile : {} / {}'.format(realPriceInpFile, '입력 자료를 확인해주세요.'))

                realPriceData = pd.read_csv(realPriceFileList[0])
                # realPriceData.drop(['Unnamed: 0'], axis=1, inplace=True)

                # realPriceData['name'] = realPriceData['단지명'] + '(' + realPriceData['도로명'] + ')'
                realPriceData['name'] = realPriceData['아파트'] + '(' + realPriceData['지번'] + ')'

                realPriceDataL2 = realPriceData.loc[
                    (realPriceData['층'] != 1)
                ].reset_index(drop=True)

                realPriceDataL2['계약년월'] = realPriceDataL2['년'].astype(str) + '-' + realPriceDataL2['월'].astype(str)
                realPriceDataL2['date'] = pd.to_datetime(realPriceDataL2['계약년월'], format='%Y-%m')
                realPriceDataL2['year'] = realPriceDataL2['date'].dt.strftime("%Y").astype('int')
                realPriceDataL2['month'] = realPriceDataL2['date'].dt.strftime("%m").astype('int')
                realPriceDataL2['거래금액'] = realPriceDataL2['거래금액'].astype(str).str.replace(',', '').astype('float')

                realPriceDataL3 = pd.merge(
                    left=realPriceDataL2
                    , right=lcnsDataL1[['archGbCdNm', 'addrDtlInfo']]
                    , left_on=['인허가addrDtlInfo']
                    , right_on=['addrDtlInfo']
                    , how='left'
                ).rename(
                    columns={
                        'archGbCdNm': 'inhuga'
                    }
                )

                realPriceDataL3.columns

                realPriceDataL4 = realPriceDataL3.rename(
                    columns={
                        '전용면적': 'capacity'
                        , '건축년도': 'conYear'
                        # , '보증금(만원)': 'realBjprice'
                        , 'archGbCdNm': 'inhuga'
                        , '거래금액': 'realPrice'
                    }
                ).sort_values(by=['name', 'capacity', 'year']).reset_index(drop=True)


                # *****************************************************
                # 데이터 통합
                # *****************************************************
                # prvsMntsrDataL5 = prvsMntsrDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['realBjprice'].mean()
                # realPriceDataL5 = realPriceDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['realPrice'].mean()
                prvsMntsrDataL5 = prvsMntsrDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['realBjprice'].agg({'realBjprice': lambda x: x.mean(skipna=False)})
                realPriceDataL5 = realPriceDataL4.groupby(['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga'], as_index=False)['realPrice'].agg({'realPrice': lambda x: x.mean(skipna=False)})

                # prvsMntsrDataL5.isna().sum()
                # realPriceDataL5.isna().sum()

                data = pd.merge(
                    left=prvsMntsrDataL5
                    , right=realPriceDataL5
                    , left_on=['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga']
                    , right_on=['name', 'conYear', 'capacity', 'lat', 'lon', 'year', 'inhuga']
                    , how='outer'
                )

                # realBjprice    3440
                # realPrice      5288
                # data.isna().sum()

                # **********************************************************************************************************
                # 딥러닝 매매가
                # **********************************************************************************************************
                inpData = realPriceDataL4
                # inpData = realPriceDataL5

                # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'realBjprice', 'inhuga']
                # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon']
                # xCol = ['name', 'year', 'conYear', 'capacity', 'lat', 'lon']
                # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
                xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
                yCol = 'realPrice'
                modelKey = 'realPrice'


                # 딥러닝 매매가 불러오기
                result = makeDlModel(sysOpt['dlModel'], xCol, yCol, inpData, modelKey, addrCode)
                log.info('[CHECK] result : {}'.format(result))

                # 딥러닝 매매가 예측
                realPriceDlModel = result['dlModel']
                data['realPriceDL'] = realPriceDlModel.predict(h2o.H2OFrame(data[xCol])).as_data_frame()

                mainTitle = '강북구 아파트 매매가 예측 결과 (딥러닝)'
                saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                # makeUserScatterPlot(data['realPriceDL'], data['realPrice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

                # **********************************************************************************************************
                # 딥러닝 전세가
                # **********************************************************************************************************
                inpData = prvsMntsrDataL4

                # inpData.info()

                # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'realPrice', 'inhuga']
                # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
                xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                yCol = 'realBjprice'
                modelKey = 'realBjPrice'

                # 딥러닝 전세가 불러오기
                result = makeDlModel(sysOpt['dlModel'], xCol, yCol, inpData, modelKey, addrCode)
                log.info('[CHECK] result : {}'.format(result))

                # 딥러닝 전세가 예측
                realBjPriceDlModel = result['dlModel']
                data['realBjPriceDL'] = realBjPriceDlModel.predict(h2o.H2OFrame(data[xCol])).as_data_frame()

                mainTitle = '강북구 아파트 전세가 예측 결과 (딥러닝)'
                # saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                # makeUserScatterPlot(data['realBjPriceDL'], data['realBjprice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

                # **********************************************************************************************************
                # 머신러닝 매매가
                # **********************************************************************************************************
                # inpData = realPricedataL2
                inpData = realPriceDataL4

                # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'realBjprice', 'inhuga']
                # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
                xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
                yCol = 'realPrice'
                modelKey = 'realPrice'

                # 머신러닝 매매가 불러오기
                result = makeMlModel(sysOpt['mlModel'], xCol, yCol, inpData, modelKey, addrCode)
                log.info('[CHECK] result : {}'.format(result))

                # 머신러닝 매매가 예측
                realPriceMlModel = result['mlModel']
                data['realPriceML'] = predict_model(realPriceMlModel, data=data)['Label']

                mainTitle = '강북구 아파트 매매가 예측 결과 (머신러닝)'
                saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                # makeUserScatterPlot(data['realPriceML'], data['realPrice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)

                # **********************************************************************************************************
                # 머신러닝 전세가
                # **********************************************************************************************************
                # inpData = prvsMntsrdataL2
                inpData = prvsMntsrDataL4

                # xCol = ['times', 'capacity', 'construction_year', 'lat', 'lon', 'realPrice', 'inhuga']
                # xCol = ['times', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                # xCol = ['year', 'month', 'capacity', 'lat', 'lon', 'inhuga']
                xCol = ['year', 'conYear', 'capacity', 'lat', 'lon', 'inhuga']
                yCol = 'realBjprice'
                modelKey = 'realBjPrice'

                # 머신러닝 전세가 불러오기
                result = makeMlModel(sysOpt['mlModel'], xCol, yCol, inpData, modelKey, addrCode)
                log.info('[CHECK] result : {}'.format(result))

                # 머신러닝 전세가 예측
                realBjPriceMlModel = result['mlModel']
                data['realBjPriceML'] = predict_model(realBjPriceMlModel, data=data)['Label']

                mainTitle = '강북구 아파트 전세가 예측 결과 (머신러닝)'
                saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                # makeUserScatterPlot(data['realBjPriceML'], data['realBjprice'], '예측', '실측', mainTitle, saveImg, 0, 140000, 2000, 10000, True)


                # **********************************************************************************************************
                # 시계열 갭투자
                # **********************************************************************************************************
                nameList = data['name'].unique()
                searchAptList = sysOpt['tsModel']['aptList']

                fnlData = pd.DataFrame()
                for i, nameInfo in enumerate(nameList):

                    # 아파트 검색 모듈
                    # sysOpt['tsModel']['aptList'] = [] : 전체 아파트에 대한 수익률 예측
                    # sysOpt['tsModel']['aptList'] = ['미아동부센트레빌'] : 미아동부센트레빌 아파트에 대한 수익률 예측
                    isSearch = True if (len(searchAptList) < 1) else False
                    for ii, aptInfo in enumerate(searchAptList):
                        if (aptInfo in nameInfo):
                            isSearch = True
                            break

                    if (isSearch == False): continue

                    log.info('[CHECK] isSearch : {} / nameInfo : {}'.format(isSearch, nameInfo))

                    selData = data.loc[
                        (data['name'] == nameInfo)
                    ].reset_index(drop=True)

                    if (len(selData) < 2): continue

                    capList = selData['capacity'].unique()
                    # capList = set(selData['capacity'])
                    # capInfo = capList[0]
                    for j, capInfo in enumerate(capList):

                        selDataL1 = selData.loc[
                            (selData['capacity'] == capInfo)
                        ].reset_index(drop=True)

                        if (len(selDataL1) < 1): continue
                        selInfoFirst = selDataL1.loc[0]

                        log.info('[CHECK] nameInfo : {} / capInfo : {} / cnt : {}'.format(nameInfo, capInfo, len(selDataL1)))

                        srtDate = pd.to_datetime(selDataL1['year'].min(), format='%Y')
                        endDate = pd.to_datetime(selDataL1['year'].max(), format='%Y')
                        dtDateList = pd.date_range(start=srtDate, end=endDate, freq=pd.DateOffset(years=1))

                        dataL2 = pd.DataFrame()
                        # dtDateInfo = dtDateList[0]
                        for k, dtDateInfo in enumerate(dtDateList):
                            iYear = int(dtDateInfo.strftime('%Y'))

                            selInfo = selDataL1.loc[
                                (selDataL1['year'] == iYear)
                                # & (selDataL1['month'] == iMonth)
                            ].reset_index(drop=True)

                            dictInfo = {
                                'name': [nameInfo]
                                , 'capacity': [capInfo]
                                , 'date': [dtDateInfo]
                                , 'year': [iYear]
                                , 'lat': [selInfoFirst['lat']]
                                , 'lon': [selInfoFirst['lon']]
                                , 'inhuga': [selInfoFirst['inhuga']]
                                , 'conYear': [selInfoFirst['conYear']]
                            }

                            dictDtl = {
                                'realPrice': [np.nan if (len(selInfo) < 1) else selInfo['realPrice'][0]]
                                , 'realBjprice': [np.nan if (len(selInfo) < 1) else selInfo['realBjprice'][0]]
                                , 'realPriceDL': [realPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selInfo) < 1) else selInfo['realPriceDL'][0]]
                                , 'realBjPriceDL': [realBjPriceDlModel.predict(h2o.H2OFrame(pd.DataFrame.from_dict(dictInfo))).as_data_frame()['predict'][0] if (len(selInfo) < 1) else selInfo['realBjPriceDL'][0]]
                                , 'realPriceML': [predict_model(realPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selInfo) < 1) else selInfo['realPriceML'][0]]
                                , 'realBjPriceML': [predict_model(realBjPriceMlModel, data=pd.DataFrame.from_dict(dictInfo))['Label'][0] if (len(selInfo) < 1) else selInfo['realBjPriceML'][0]]
                            }

                            dict = {**dictInfo, **dictDtl}

                            # dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dictInfo)], ignore_index=True)
                            dataL2 = pd.concat([dataL2, pd.DataFrame.from_dict(dict)], ignore_index=True)

                        # if (len(dataL2.dropna()) < 1): continue
                        if (len(dataL2) < 2): continue

                        dataL2['gapReal'] = dataL2['realPrice'] - dataL2['realBjprice']
                        dataL2['gapML'] = dataL2['realPriceML'] - dataL2['realBjPriceML']
                        dataL2['gapDL'] = dataL2['realPriceDL'] - dataL2['realBjPriceDL']

                        # 아파트 전세가 시계열
                        mainTitle = '[{}, {}] 아파트 전세가 시계열'.format(nameInfo, capInfo)
                        saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                        # makeUserTimeSeriesPlot(dataL2['date'], dataL2['realBjPriceML'], dataL2['realBjPriceDL'], dataL2['realBjprice'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '전세가 [만원]', mainTitle, saveImg, False)

                        # 아파트 매매가 시계열
                        mainTitle = '[{}, {}] 아파트 매매가 시계열'.format(nameInfo, capInfo)
                        saveImg = '{}/{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '예측', addrInfo, mainTitle)
                        # makeUserTimeSeriesPlot(dataL2['date'], dataL2['realPriceML'], dataL2['realPriceDL'], dataL2['realPrice'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '매매가 [만원]', mainTitle, saveImg, False)

                        try:
                            tsForPeriod = sysOpt['tsModel']['forYear'] - dataL2['year'].max()
                            log.info(f'[CHECK] tsForPeriod : {tsForPeriod}')

                            # tsModel = AutoTS(forecast_length=sysOpt['tsModel']['forYear'], min_allowed_train_percent=1.0, frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')
                            # tsModel = AutoTS(forecast_length=tsForPeriod, frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')

                            # tsDlModel = tsModel.fit(dataL2, date_col='date', value_col='gapDL', id_col=None)
                            # tsDlFor = tsDlModel.predict().forecast
                            # tsDlFor['date'] = tsDlFor.index
                            # tsDlFor.reset_index(drop=True, inplace=True)

                            tsDlModel = Prophet(n_changepoints=2)
                            tsDlModel.fit(TimeSeries.from_dataframe(dataL2, time_col='date', value_cols='gapDL'))
                            tsDlFor = tsDlModel.predict(tsForPeriod).pd_dataframe().rename(columns={'0': 'gapDL'})
                            tsDlFor['date'] = tsDlFor.index
                            tsDlFor.reset_index(drop=True, inplace=True)

                            # tsModel = AutoTS(forecast_length=sysOpt['tsModel']['forYear'], frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')
                            # tsModel = AutoTS(forecast_length=tsForPeriod, min_allowed_train_percent=0, frequency='infer', ensemble='all', model_list='superfast', transformer_list='superfast')

                            # tsMlModel = tsModel.fit(dataL2, date_col='date', value_col='gapML', id_col=None)
                            # tsMlFor = tsMlModel.predict().forecast
                            # tsMlFor['date'] = tsMlFor.index
                            # tsMlFor.reset_index(drop=True, inplace=True)

                            tsMlModel = Prophet(n_changepoints=2)
                            tsMlModel.fit(TimeSeries.from_dataframe(dataL2, time_col='date', value_cols='gapML'))
                            tsMlFor = tsMlModel.predict(tsForPeriod).pd_dataframe().rename(columns={'0': 'gapML'})
                            tsMlFor['date'] = tsMlFor.index
                            tsMlFor.reset_index(drop=True, inplace=True)

                            tsForData = tsDlFor.merge(tsMlFor, left_on=['date'], right_on=['date'], how='inner')
                            tsForData['name'] = nameInfo
                            tsForData['capacity'] = capInfo
                            tsForData['year'] = tsForData['date'].dt.strftime('%Y').astype('int')
                            tsForData['lat'] = selInfoFirst['lat']
                            tsForData['lon'] = selInfoFirst['lon']
                            tsForData['inhuga'] = selInfoFirst['inhuga']
                            tsForData['conYear'] = selInfoFirst['conYear']
                        except Exception as e:
                            log.error('Exception : {}'.format(e))

                        if (len(tsForData) < 1): continue

                        dataL3 = dataL2.merge(
                            tsForData
                            , left_on=['name', 'capacity', 'year', 'lat', 'lon', 'inhuga', 'conYear', 'date', 'gapDL', 'gapML']
                            , right_on=['name', 'capacity', 'year', 'lat', 'lon', 'inhuga', 'conYear', 'date', 'gapDL', 'gapML']
                            , how='outer'
                        )

                        # 아파트 갭투자 시계열
                        # mainTitle = '[{}, {}] 아파트 갭투자 시계열'.format(nameInfo, capInfo)
                        # saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, addrInfo, mainTitle)
                        # makeUserTimeSeriesPlot(dataL3['date'], dataL3['gapML'], dataL3['gapDL'], dataL3['gapReal'], '예측 (머신러닝)', '예측 (딥러닝)', '실측', '날짜 [연도]', '갭 투자 [만원]', mainTitle, saveImg, False)

                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # 수익률 테이블
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
                        resData = dataL3[['gapReal', 'gapML', 'gapDL']]

                        resDiffData = resData.diff(periods=1).rename(
                            columns={
                                'gapReal': 'gapDiffReal'
                                , 'gapML': 'gapDiffML'
                                , 'gapDL': 'gapDiffDL'
                            }
                            , inplace=False
                        )

                        resPctData = resData.pct_change(periods=1).rename(
                            columns={
                                'gapReal': 'gapPctReal'
                                , 'gapML': 'gapPctML'
                                , 'gapDL': 'gapPctDL'
                            }
                            , inplace=False
                        )

                        resDataL2 = pd.concat([dataL3, resDiffData, resPctData * 100], axis=1)
                        resDataL3 = resDataL2.sort_values(by=['date'], ascending=False).rename(
                            columns={
                                'name': '아파트(도로명)'
                                , 'capacity': '면적'
                                , 'construction_year': '건축연도'
                                , 'year': '연도'
                                , 'date': '날짜'
                                , 'lat': '위도'
                                , 'lon': '경도'
                                , 'inhuga': '인허가'
                                , 'conYear': '건축년도'
                                , 'realPrice': '매매가'
                                , 'realBjprice': '전세가'
                                , 'realPriceDL': '예측 딥러닝 매매가'
                                , 'realBjpriceDL': '예측 딥러닝 전세가'
                                , 'realPriceML': '예측 머신러닝 매매가'
                                , 'realBjpriceML': '예측 머신러닝 전세가'
                                , 'gapReal': '실측 갭투자'
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

                        fnlData = pd.concat([fnlData, resDataL3], ignore_index=True)

                        if (len(resDataL3) > 0):
                            saveFile = '{}/{}/{}/{}/{}_{}_{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '예측', addrInfo, '수익률 테이블', addrInfo, nameInfo, capInfo, datetime.now().strftime('%Y%m%d'))
                            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                            resDataL3.to_excel(saveFile, index=False)
                            log.info('[CHECK] saveFile : {}'.format(saveFile))

                if (len(fnlData) > 0):
                    saveFile = '{}/{}/{}/{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '예측', addrInfo, '수익률 테이블', addrInfo, datetime.now().strftime('%Y%m%d'))
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    fnlData.to_excel(saveFile, index=False)
                    log.info('[CHECK] saveFile : {}'.format(saveFile))

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