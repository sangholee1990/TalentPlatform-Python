# -*- coding: utf-8 -*-

import json
import logging
import logging.handlers
import os
import platform
import shutil
import warnings
from configparser import ConfigParser
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from sklearn.model_selection import GridSearchCV

import logging
import logging.handlers
import os
import sys
# import plotnine as plotnine
# import hydroeval
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
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics
from datetime import datetime
import dfply as dfply
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import sys

# =================================================
# Set Opt
# =================================================
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
def initLog(env = None, contextPath = None, prjName = None):

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
def initGlobalVar(env = None, contextPath = None, prjName = None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    globalVar = {
        "prjName": prjName
        , "sysOs": platform.system()
        , "contextPath": contextPath
        , "resPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , "cfgPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , "inpPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , "sysPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace("\\", "/")

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
def makeTimePlot(valY, PredValY, savefigName, minVal, intVal):
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

# 산점도 시각화
def makeScatterPlot(PredValY, valY, savefigName):

    X = PredValY
    Y = valY

    plt.scatter(X, Y)

    arrVal = np.array([X, Y])
    setMin = np.min(arrVal)
    setMax = np.max(arrVal)
    interval = (setMax - setMin) / 20

    # setMin = 200
    # setMax = 20000
    # interval = 1000

    plt.xlabel('Val')
    plt.ylabel('Pred')
    plt.xlim(setMin, setMax)
    plt.ylim(setMin, setMax)
    plt.grid(True)

    # Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
    Bias = np.mean(X - Y)
    rBias = (Bias / np.mean(Y)) * 100.0
    RMSE = np.sqrt(np.mean((X - Y) ** 2))
    rRMSE = (RMSE / np.mean(Y)) * 100.0
    MAPE = np.mean(np.abs((X - Y) / X)) * 100.0

    slope, intercept, R, Pvalue, std_err = stats.linregress(X, Y)
    N = len(X)

    lmfit = (slope * X) + intercept
    plt.plot(X, lmfit, color='red', linewidth=2)
    plt.plot([setMin, setMax], [setMin, setMax], color='black')

    plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval), color='red',
                 fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R-square = %.2f  (p-value < %.2f)' % (R ** 2, Pvalue), xy=(setMin, setMax - interval * 2),
                 color='red',
                 fontweight='bold', xycoords='data',
                 horizontalalignment='left', verticalalignment='center')
    plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(setMin, setMax - interval * 3),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(setMin, setMax - interval * 4),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(setMin, setMax - interval * 5),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 6), color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left',
                 verticalalignment='center')
    plt.savefig(savefigName, dpi=600, bbox_inches='tight')
    plt.show()

# 산점도 시각화
def makeUserScatterPlot(PredValY, valY, mainTitle, saveImg, minVal, intVal):
    X = PredValY
    Y = valY

    plt.scatter(X, Y)

    arrVal = np.array([X, Y])
    setMax = np.max(arrVal)
    setMin = minVal
    # setMax = 8000
    interval = intVal

    plt.title(mainTitle)
    plt.xlabel('예측')
    plt.ylabel('실측')
    # plt.xlim(0, setMax)
    # plt.ylim(0, setMax)
    plt.grid(True)

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
                 color='red',  xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2),
                 color='red', xycoords='data',
                 horizontalalignment='left', verticalalignment='center')
    plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(setMin, setMax - interval * 3),
                 color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(setMin, setMax - interval * 4),
                 color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(setMin, setMax - interval * 5),
                 color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
    # plt.annotate('MASE = %.2f' % (MASE), xy=(setMin, setMax - interval * 6),
    #              color='black', fontweight='bold',
    #              xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 6), color='black',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

# 교차검증을 통해 하이퍼 파라미터 찾기
def gridsearch_cv_for_regression(model, param, kfold, train_input, train_target,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1, tracking=True):
    '''
    [Parameters]
    - model: A tuple like ('name', MODEL)
    - param
    - scoring: neg_mean_absolute_error, neg_mean_squared_error, neg_median_absolute_error, r2
               (http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
    - n_jobs: default as -1 (if it is -1, all CPU cores are used to train and validate models)
    - tracking: whether trained model's name and duration time are printed
    '''

    name = model[0]
    estimator = model[1]

    if tracking:
        start_time = datetime.now()
        print("[%s] Start parameter search for model '%s'" % (start_time, name))
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=kfold, scoring=scoring,
                                  n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target)
        end_time = datetime.now()
        duration_time = (end_time - start_time).seconds
        print(
            "[%s] Finish parameter search for model '%s' (time: %d seconds)" % (end_time, name, duration_time))
        print()
    else:
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=kfold, scoring=scoring,
                                  n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target)

    return gridsearch


"""
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        디렉토리 경로 이동 처리. 대상 경로가 있을 경우 삭제한 다음 이동 처리 수행
"""
def relocPath(src, dst):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)

"""
    parameter:
	dst - 대상 경로
	uid - OS 유저 ID
	gid - OS 그룹 ID
    return:         
        none
    description: 
        대상 경로 하위의 모든 파일에 대하여 일괄 소유자 변경 처리 수행
"""
def changeOwner(dst, uid, gid):
    os.chown(dst, uid, gid)

    for fn in os.listdir(dst):
        path = dst + "/" + fn
        os.chown(path, uid, gid)

"""
    parameter:
	path - 생성 경로
	isdir - 디렉토리 여부
	mod - 권한 정보
    return:         
        none
    description: 
        대상 경로 생성 처리 수행. 대상 경로가 존재할 경우 삭제 후 생성.
"""
def makeDirs(path, isdir=True, mod=None):
    dir = path
    if not isdir:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        if mod:
            os.chmod(dir, mod)

"""
    parameter:
	strs - 경로 문자 배열
    return:         
        string - 전체 경로 문자열
    description: 
        해당 문자열에 해당하는 디렉토리 경로를 문자열을 생성
"""
def concatPath(strs):
    fullPath = ""

    for str in strs:
        if not str: continue
        if str == '': continue
        fullPath = fullPath + "/" + str
    return fullPath

"""
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        원본 경로의 디렉토리를 대상 경로로 복사. 이미 대상 경로가 있을 경우 삭제 후 처리
"""
def copyDir(src, dst):
    makeDirs(dst)

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

"""
    module:
        commonUtil
    function:
     	copyFile
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        원본 경로의 파일을 대상 경로에 복사. 이미 대상 파일이 있을 경우 삭제 후 처리
"""
def copyFile(src, dst):
    dstDir = os.path.dirname(dst)
    makeDirs(dstDir)

    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)

"""
    module:
        commonUtil
    function:
     	loadModule
    parameter:
	modname - 모듈명
	fromlist - 패키지명
    return:         
        none
    description: 
        python 모듈 동적 로딩 처리 수행
"""
def loadModule(modname,fromlist):
    mod = None
    try:
        mod = __import__(modname, fromlist=fromlist)
    except:
        pass
    return mod

"""
    module:
        commonUtil
    function:
     	ftpIsDirectory
    parameter:
	ftp - ftp 객체
	dir - ftp 대상 경로
    return:         
        bool - 디렉토리 여부 값
    description: 
        ftp 의 해당 경로가 존재하는지 여부 확인
"""
def ftpIsDirectory(ftp, dir):
    try:
        ftp.cwd(dir)
    except:
        return False
    return True

"""
    module:
        commonUtil
    function:
     	sftpIsDirectory
    parameter:
	sftp - sftp 객체
	dir - sftp 대상 경로
    return:         
        bool - 디렉토리 여부 값
    description: 
        sftp 의 해당 경로가 존재하는지 여부 확인
"""
def sftpIsDirectory(sftp, dir):
    try:
        sftp.stat(dir)
    except:
        return False
    return True

"""
    module:
        commonUtil
    function:
     	ftpMakeDirs
    parameter:
	ftp - ftp 객체
	dir - ftp 대상 경로
    return:         
        none
    description: 
        ftp 의 대상 경로를 생성
"""
def ftpMakeDirs(ftp, dir):
    dirs = dir.split("/")

    for i in range( 1, len(dirs) + 1 ):
        d = "/".join(dirs[0:i])
        if not ftpIsDirectory(ftp, d):
            ftp.mkd(d)

"""
    module:
        commonUtil
    function:
     	getDirFileSize
    parameter:
	rootDir - 대상 경로
    return:         
        int - 파일 사이즈 총합 (바이트 단위)
    description: 
        대상 경로 하위의 파일 사이즈 계산
"""
def getDirFileSize(rootDir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(rootDir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

"""
    module:
        commonUtil
    function:
     	getDirFileCount
    parameter:
	rootDir - 대상 경로
    return:         
        int - 파일 수량 총합
    description: 
        대상 경로 하위의 파일 수량 계산
"""
def getDirFileCount(rootDir):
    total_cnt = 0
    for dirpath, dirnames, filenames in os.walk(rootDir):
        for f in filenames:
            total_cnt += 1
    return total_cnt

"""
    module:
        commonUtil
    function:
     	getMatchFilename
    parameter:
	rootDir - 대상 경로
	exts - 확장자 배열
    return:         
        string - 파일명
    description: 
        대상 경로 하위에 해당하는 확장자가 매칭되는 파일 검색
"""
def getMatchFilename(rootDir, exts):
    for filenm in os.listdir(rootDir):
        fn, ext = os.path.splitext(filenm)
        if ext in exts:
            return filenm

"""
    module:
        commonUtil
    function:
     	isValidate
    parameter:
	dst - 대상 경로
    return:         
    	bool - 유효성 여부
    description: 
        대상 경로 하위의 파일 사이즈로부터 유효성 검사 수행
"""
def isValidate(dst):
    if os.path.isdir(dst):
        for fn in os.listdir(dst):
            path = dst + "/" + fn
            if os.stat(path).st_size <= 1:
                return False
    else:
        if os.stat(dst).st_size <= 1:
            return False

    return True

def loadConfig(self, confFile):
    try:
        if not os.path.exists(confFile):
            raise Exception("%s file not found..\n" %confFile)
        self.config = ConfigParser.ConfigParser()
        self.config.read(confFile)
    except(Exception, OSError):
        raise("load Configuration file error")

def getValue(self, section, key):
    value = self.config.get(section, key)
    return value

def getJson(self, section, key):
    value = self.config.get(section, key)
    j = json.loads(value)
    return j

# 상관계수 행렬 시각화
def makeCorrPlot(data, savefigName):

    corr = data.corr(method='pearson')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, square=True, annot=False, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.savefig(savefigName, dpi=600, bbox_inches='tight')
    plt.show()

# 산점도 시각화
# def makeScatterPlot(valY, PredValY, savefigName):
#
#     X = valY
#     Y = PredValY
#
#     plt.scatter(X, Y)
#
#     # arrVal = np.array([X, Y])
#     # setMax = np.max(arrVal)
#     setMin = 200
#     setMax = 8000
#     interval = 500
#
#     plt.title("")
#     plt.xlabel('Val')
#     plt.ylabel('Pred')
#     plt.xlim(0, setMax)
#     plt.ylim(0, setMax)
#     plt.grid()
#
#     ## Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
#     Bias = np.mean(X - Y)
#     rBias = (Bias / np.mean(Y)) * 100.0
#     RMSE = np.sqrt(np.mean((X - Y) ** 2))
#     rRMSE = (RMSE / np.mean(Y)) * 100.0
#     MAPE = np.mean(np.abs((X - Y) / X)) * 100.0
#     MASE = mase(X, Y)
#
#     slope = linregress(X, Y)[0]
#     intercept = linregress(X, Y)[1]
#     R = linregress(X, Y)[2]
#     Pvalue = linregress(X, Y)[3]
#     N = len(X)
#
#     lmfit = (slope * X) + intercept
#     plt.plot(X, lmfit, color='red', linewidth=2)
#     plt.plot([0, setMax], [0, setMax], color='black')
#
#     plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval), color='red',
#                     fontweight='bold',
#                     xycoords='data', horizontalalignment='left', verticalalignment='center')
#     plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2), color='red',
#                     fontweight='bold', xycoords='data',
#                     horizontalalignment='left', verticalalignment='center')
#     plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(setMin, setMax - interval * 3),
#                     color='black', fontweight='bold',
#                     xycoords='data', horizontalalignment='left', verticalalignment='center')
#     plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(setMin, setMax - interval * 4),
#                     color='black', fontweight='bold',
#                     xycoords='data', horizontalalignment='left', verticalalignment='center')
#     plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(setMin, setMax - interval * 5),
#                     color='black', fontweight='bold',
#                     xycoords='data', horizontalalignment='left', verticalalignment='center')
#     plt.annotate('MASE = %.2f' % (MASE), xy=(setMin, setMax - interval * 6),
#                     color='black', fontweight='bold',
#                     xycoords='data', horizontalalignment='left', verticalalignment='center')
#     plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 7), color='black', fontweight='bold',
#                     xycoords='data', horizontalalignment='left',
#                     verticalalignment='center')
#     plt.savefig(savefigName, dpi=600, bbox_inches='tight')
#     plt.show()

# 선형회귀모형 트레이닝 모형
def train_test_linreg(data, feature_cols):
    X = data[feature_cols]
    Y = data.total
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Make series using selected features and corresponding coefficients
    formula = pd.Series(model.coef_, index=feature_cols)

    # Save intercept
    intercept = model.intercept_

    # Calculate training RMSE and testing RMSE
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
    rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))

    # Calculate training R-square and testing R-square
    rsquared_train = model.score(X_train, Y_train)
    rsquared_test = model.score(X_test, Y_test)

    # Make result dictionary
    result = {'formula': formula, 'intercept': intercept, 'rmse_train': rmse_train, 'rmse_test': rmse_test,
                'rsquared_train': rsquared_train, 'rsquared_test': rsquared_test,
                'Y_train': Y_train, 'Y_pred_train': Y_pred_train, 'Y_test': Y_test, 'Y_pred_test': Y_pred_test,
                'model': model
                }

    return result

# 릿지모형 트레이닝 모형
def train_test_ridge(data, feature_cols, alpha_value):
    X = data[feature_cols]
    Y = data.total
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    model = Ridge(alpha=alpha_value)
    model.fit(X_train, Y_train)

    # Make series using selected features and corresponding coefficients
    formula = pd.Series(model.coef_, index=list(X.columns.values))

    # Save intercept
    intercept = model.intercept_

    # Calculate training RMSE and testing RMSE
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
    rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))

    # Calculate training R-square and testing R-square
    rsquared_train = model.score(X_train, Y_train)
    rsquared_test = model.score(X_test, Y_test)

    # Make result dictionary
    result = {'formula': formula, 'intercept': intercept, 'rmse_train': rmse_train, 'rmse_test': rmse_test,
                'rsquared_train': rsquared_train, 'rsquared_test': rsquared_test}

    return result
