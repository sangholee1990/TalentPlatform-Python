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
from urllib.parse import quote
# import the method to use to create the animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# set the dimensions of the visualization
# import libraries needed
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen

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

# plt.rc('font', family='Malgun Gothic')
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


def load_data(url):
    try:
        # download most recent HadCrut dataset
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        content = urlopen(req).read()
        file = open("HadCrut.txt", "wb")
        file.write(content)
        file.close()

        # read the data into a Pandas dataframe
        hadcrut = pd.read_csv(file.name, delim_whitespace=True, usecols=[0, 1], header=None)
        # split the first column into month and year columns
        hadcrut["month"] = hadcrut[0].str.split("/").str[1].astype(int)
        hadcrut["year"] = hadcrut[0].str.split("/").str[0].astype(int)
        # rename the 1 column to value
        hadcrut.rename(columns={1: "value"}, inplace=True)
        # select and save all but the first column (0)
        hadcrut = hadcrut[["value", "month", "year"]].copy()

        # print("before",hadcrut["year"].value_counts(ascending=True).head())
        # check if the most recent year has complete data recordings
        recent = hadcrut["year"].value_counts(ascending=True).head(1)
        if recent.values < [12]:
            # if not complete remove the most recent year
            hadcrut = hadcrut.drop(hadcrut[hadcrut["year"] == (recent.index.values[0])].index)
        # print("after",hadcrut["year"].value_counts(ascending=True).head())

        # create a multiindex using the year and month columns
        hadcrut = hadcrut.set_index(["year", "month"])
        # compute the mean of the global temperatures from 1850 to 1900 and subtract that value from the entire dataset
        hadcrut -= hadcrut.loc[1850:1900].mean()

        # return the column names
        hadcrut = hadcrut.reset_index()
    except Exception as e:
        print("Problem in loading the data, reason", e)
    else:
        return hadcrut


def load_viz(df):
    try:
        # fig = plt.figure(figsize=(8, 8))
        fig = plt.figure(figsize=(8, 8))

        # set the projection to polar (not cartesian) system
        ax1 = plt.subplot(111, projection='polar')

        # plot the temperature rings at 0,1.5 and 2
        full_circle_thetas = np.linspace(0, 2 * np.pi, 1000)
        blue_one_radii = [0.0 + 1.0] * 1000
        red_one_radii = [1.5 + 1.0] * 1000
        red_two_radii = [2.0 + 1.0] * 1000
        ax1.plot(full_circle_thetas, blue_one_radii, c='blue')
        ax1.plot(full_circle_thetas, red_one_radii, c='red')
        ax1.plot(full_circle_thetas, red_two_radii, c='red')

        # remove the ticks for both axes
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        # set the limit for r axis
        ax1.set_ylim(0, 3.25)

        # set the color for the foreground and background
        fig.set_facecolor("#323331")
        ax1.set_facecolor("#000100")

        # add the plot title
        # ax1.set_title("Global Temperature Change (1850-{})".format(df["year"].max()), color="white", fontsize=20)
        ax1.set_title("Global Temperature Change (1850-{})".format(df["year"].max()), color="white", fontsize=20, y=1.08)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # add the temperatures for the temperature rings
        # ax1.text(np.pi/2, 0.90, "0.0 C", color="blue", ha='center',fontsize= 15)
        ax1.text(np.pi / 2, 2.40, "1.5 C", color="red", ha='center', fontsize=15,
                 bbox=dict(facecolor='#000100', edgecolor='#000100'))
        ax1.text(np.pi / 2, 2.90, "2.0 C", color="red", ha='center', fontsize=15,
                 bbox=dict(facecolor='#000100', edgecolor='#000100'))

        # add the months outer rings
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_angles = np.linspace((np.pi / 2) + (2 * np.pi), np.pi / 2, 13)
        for i, month in enumerate(months):
            ax1.text(months_angles[i], 3.4, month, color="white", fontsize=15, ha="center")

        # add the source
        fig.text(0.78, 0.01, "HadCRUT 4.6", color="white", fontsize=15)
        fig.text(0.05, 0.03, "Anis Ismail", color="white", fontsize=15)
        fig.text(0.05, 0.01, "Based on Ed Hawkins's 2017 Visualization", color="white", fontsize=10)

        # prepare the update in each frame function to be used by Funcanimation method
        def update(i):
            # Specify how we want the plot to change in each frame
            # Remove the previous year text at the center of the plot
            for txt in ax1.texts:
                if (txt.get_position() == (0, 0)):
                    txt.set_visible(False)
                    # We need to unravel the for loop we had earlier.
            year = years[i]
            r = df[df['year'] == year]['value'] + 1
            theta = np.linspace(0, 2 * np.pi, 12)
            ax1.plot(theta, r, c=plt.cm.viridis(i * 2))
            ax1.text(0, 0, year, fontsize=20, color="white", ha="center")
            return ax1

        # call the function that will create the animation
        years = df["year"].unique()
        anim = FuncAnimation(fig, update, frames=len(years), interval=10)

        saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'climate_spiral.mp4')
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        anim.save(saveFile, writer=FFMpegWriter(), savefig_kwargs={'facecolor': '#323331'})
        log.info(f"[CHECK] saveFile : {saveFile}")

    except Exception as e:
        # in case of any exception, inform the user
        return "climate_spiral.mp4 was not created successfully,\n \
        Reason:{}".format(e)
    else:
        # else inform the user that the procedure was successful
        return "climate_spiral.mp4 was created successfully"


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
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0365'

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

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'HN_M.csv')
            # fileList = sorted(glob.glob(inpFile))

            # url = input()
            url = 'https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.monthly_ns_avg.txt'
            # encode the url
            urlenc = quote(url)
            # load the data into a Pandas dataframe and return a
            # cleaned version
            hadcrut = load_data(url)

            # create the visualization
            result = load_viz(hadcrut)
            print(result)


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
