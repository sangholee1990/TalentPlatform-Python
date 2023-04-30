# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import sys
import plotnine as plotnine
import hydroeval
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import glob
import pprint
import platform
from datetime import datetime
import dfply as dfply
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import re
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from src.talentPlatform.unitSysHelper.forecasting_metrics import mase
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics
from datetime import datetime
import dfply as dfply
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus":False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# =================================================
# 함수 정의
# =================================================
# 로그 설정
def initLog(prjName):
    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
    )

    # breakpoint()

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
def initGlobalVar(contextPath, prjName):
    globalVar = {
        "prjName": prjName
        , "contextPath": contextPath
        , "srcPath": os.path.join(contextPath, 'src')
        , "resPath": os.path.join(contextPath, 'resources')
        , "cfgPath": os.path.join(contextPath, 'resources', 'config')
        , "inpPath": os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": os.path.join(contextPath, 'resources', 'mapInfo')
        , "systemPath": os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        globalVar[key] = val.replace("\\", "/")

    # 전역 변수
    log.info("[Check] globalVar : {}".format(globalVar))

    return globalVar

def stepAic(model, exog, endog, **kwargs):
    """
    This select the best exogenous variables with AIC
    Both exog and endog values can be either str or list.
    (Endog list is for the Binomial family.)

    Note: This adopt only "forward" selection

    Args:
        model: model from statsmodels.formula.api
        exog (str or list): exogenous variables
        endog (str or list): endogenous variables
        kwargs: extra keyword argments for model (e.g., data, family)

    Returns:
        model: a model that seems to have the smallest AIC
    """

    exog = np.r_[[exog]].flatten()
    endog = np.r_[[endog]].flatten()
    remaining = set(exog)
    selected = []

    formula_head = ' + '.join(endog) + ' ~ '
    formula = formula_head + '1'
    aic = model(formula=formula, **kwargs).fit().aic
    print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

    current_score, best_new_score = np.ones(2) * aic

    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula_tail = ' + '.join(selected + [candidate])
            formula = formula_head + formula_tail
            aic = model(formula=formula, **kwargs).fit().aic
            print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

            scores_with_candidates.append((aic, candidate))

        scores_with_candidates.sort()
        scores_with_candidates.reverse()
        best_new_score, best_candidate = scores_with_candidates.pop()

        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = formula_head + ' + '.join(selected)

    print('The best formula: {}'.format(formula))

    return model(formula, **kwargs).fit()

# 산점도 시각화
def makeScatterPlot(valY, PredValY, savefigName, minVal, intVal):
    X = valY
    Y = PredValY

    plt.scatter(X, Y)

    arrVal = np.array([X, Y])
    setMax = np.max(arrVal)
    setMin = minVal
    # setMax = 8000
    interval = intVal

    plt.title("")
    plt.xlabel('Val')
    plt.ylabel('Pred')
    # plt.xlim(0, setMax)
    # plt.ylim(0, setMax)
    plt.grid()

    ## Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
    Bias = np.mean(X - Y)
    rBias = (Bias / np.mean(Y)) * 100.0
    RMSE = np.sqrt(np.mean((X - Y) ** 2))
    rRMSE = (RMSE / np.mean(Y)) * 100.0
    MAPE = np.mean(np.abs((X - Y) / X)) * 100.0
    # MASE = mase(X, Y)

    lmFit = linregress(X, Y)
    slope = lmFit[0]
    intercept = lmFit[1]
    R = lmFit[2]
    Pvalue = lmFit[3]
    N = len(X)

    lmfit = (slope * X) + intercept
    plt.plot(X, lmfit, color='red', linewidth=2)
    # plt.plot([0, setMax], [0, setMax], color='black')

    plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval),
                 color='red',
                 fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2),
                 color='red',
                 fontweight='bold', xycoords='data',
                 horizontalalignment='left', verticalalignment='center')
    # plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(setMin, setMax - interval * 3),
    #              color='black', fontweight='bold',
    #              xycoords='data', horizontalalignment='left', verticalalignment='center')
    # plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(setMin, setMax - interval * 4),
    #              color='black', fontweight='bold',
    #              xycoords='data', horizontalalignment='left', verticalalignment='center')
    # plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(setMin, setMax - interval * 5),
    #              color='black', fontweight='bold',
    #              xycoords='data', horizontalalignment='left', verticalalignment='center')
    # plt.annotate('MASE = %.2f' % (MASE), xy=(setMin, setMax - interval * 6),
    #              color='black', fontweight='bold',
    #              xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 3), color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left',
                 verticalalignment='center')
    plt.show()
    plt.savefig(savefigName, dpi=600, bbox_inches='tight')

# =================================================
# Set Env
# =================================================
# 작업환경 경로 설정
contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
prjName = 'test'
serviceName = 'LSH0167'
log = initLog(prjName)
globalVar = initGlobalVar(contextPath, prjName)

# =================================================
# Main
# =================================================
try:
    log.info('[START] {}'.format('Main'))

    # ***********************************************
    # 통합 데이터 전처리
    # ***********************************************
    fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0167_dataL2.csv'))
    dataL2 = pd.read_csv(fileInfo1[0], na_filter=False)

    # breakpoint()

    # ***********************************************
    # 데이터 요약 (요약통계량)
    # ***********************************************
    # 연소득당 거래금액 따른 기초 통계량
    dataL2.describe()

    # 법정동에 따른 연소득당 거래금액 따른 기초 통계량
    dataL3 = ((
            dataL2 >>
            dfply.group_by(dfply.X.d2) >>
            dfply.summarize(
                meanVal=dfply.mean(dfply.X.val)
            )
    ))

    # *******************************************************
    # 데이터 요약 (표/그래프 활용)
    # *******************************************************
    # 연소득당 거래금액 따른 히스토그램
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 따른 히스토그램')

    sns.distplot(dataL2['val'], kde=True, rug=False)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 법정동에 따른 연소득당 거래금액 히스토그램
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '법정동에 따른 연소득당 거래금액 히스토그램')

    sns.barplot(x='d2', y='meanVal', data=dataL3)
    plt.xticks(rotation = 45)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 연소득당 거래금액 따른 상자 그림
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 따른 상자 그림')

    sns.boxplot(y="val", data=dataL2)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 법정동에 따른 연소득당 거래금액 상자 그림
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '법정동에 따른 연소득당 거래금액 상자 그림')

    sns.boxplot(x = "d2", y="val", data=dataL2)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 연소득당 거래금액 산점도
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 산점도')

    makeScatterPlot(dataL2['meanCost'], dataL2['거래금액'], saveImg, 3500, 100000)

    # *******************************************************
    # 데이터 분석 (데이터 분석 기법 활용)
    # *******************************************************
    # 주택 가격 결정 요인을 위한 회귀분석
    dataL4 = ((
            dataL2 >>
            dfply.select(dfply.X.건축년도, dfply.X.전용면적, dfply.X.층, dfply.X.val2, dfply.X.d2, dfply.X.val) >>
            dfply.rename(
                면적당거래금액 = dfply.X.val2
                , 연소득당거래금액 = dfply.X.val
            )
    ))

    # 주택 가격 결정 요인을 위한 관계성
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '주택 가격 결정 요인을 위한 관계성')

    sns.pairplot(dataL4)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # +++++++++++++++++++++++++++++++++++++++++++++++
    # 전체 아파트
    dataL5 = dataL4
    # +++++++++++++++++++++++++++++++++++++++++++++++
    # 모든 변수에 대한 다중선형회귀모형
    model = sm.OLS.from_formula('연소득당거래금액 ~ 건축년도 + 전용면적 + 층 + 면적당거래금액 + d2', dataL5)
    result = model.fit()
    result.summary()

    # 단계별 다중선형회귀모형
    # 그 결과 앞서 모든 변수 다중선형회귀모형과 동일한 결과를 보임
    bestModel = stepAic(smf.ols, ['건축년도', '전용면적', '층', '면적당거래금액', 'd2'], ['연소득당거래금액'], data=dataL5)
    bestModel.summary()

    # # 전체 아파트에 대한 주택가격 결정요인 (연소득당 거래금액) 예측 산점도
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '전체 아파트에 대한 주택가격 결정요인 예측 산점도')

    makeScatterPlot(bestModel.predict(), dataL5['연소득당거래금액'], saveImg, 0, 15)

except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] {}'.format('Main'))
