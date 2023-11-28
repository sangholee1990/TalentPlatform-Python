# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

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

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

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
    # Python을 이용한 유럽 회원국 무역의존도 최근 10년 데이터 분석

    # 데이터 분석 주제: 유럽 회원국 무역의존도 최근 10년 데이터 분석
    # 세부목표1. 유럽 국가별 무역의존도 비교 분석
    # 세부목표2. 유럽 시기별(수출)무역의존도 추이 분석
    # 세부목표3. 유럽 시기별(수입)무역의존도 추이 분석
    #
    # 요청 분석 내용
    # - 데이터 전처리 과정
    # - 데이터 분석 (pandas, matplotlib, seaborn, autopct, scatter, strip plot, dis plot 함수 포함 사용 요청)
    # 1) 기초 데이터 분석
    #   주요 칼럼 별 분포 분석 (범주형인 경우 counplot 막대 차트 분포 분석, 수치형인 경우 histogram 으로 분포 분석)
    # 2) 세부목표1,2,3 각각에 대한 분석
    #
    # 요청 파일
    # - ipynb 파일 (분석결과가 표시되는 Colab의 ipynb 파일)
    # - pdf파일 (ipynb 파일을 pdf 파일로 출력한 내용)
    # - 데이터 분석 각 구간(함수)에 대한 해석 필요
    # - (Colab에 업로드하기 위해 전처리된 데이터 파일)

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
    serviceName = 'LSH0509'

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # ********************************************************************
            # 파일 읽기
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '무역의존도_수출입의_대_GDP_비율__OECD회원국.xlsx')
            fileList = sorted(glob.glob(inpFile))
            data = pd.read_excel(fileList[0], engine='openpyxl', skiprows=2)


            dataL1 = data

            # 컬럼 재 정의
            colList = ['국가']
            for year in range(2012, 2022):
                for type in ['수출', '수입']:
                    for key in ['원데이터', '전년대비증감', '증감률']:
                        colList.append(f'{year}-{type}-{key}')

            dataL1.columns = colList

            # 국가 컬럼에서 공백 존재 (\u3000\u3000\u3000)
            dataL1['국가'] = dataL1['국가'].str.strip()

            dataL2 = dataL1.melt(id_vars=['국가'], var_name='key', value_name='val')
            dataL2[['연도', '종류', '특성']] = dataL2['key'].str.split('-', expand=True)

            # 유럽 국가 목록
            eurCotList = [
                "오스트리아", "벨기에", "체코", "덴마크", "에스토니아", "핀란드", "프랑스", "독일", "그리스", "헝가리", "아이슬란드", "아일랜드", "이탈리아", "라트비아"
                , "리투아니아", "룩셈부르크", "네덜란드", "노르웨이", "폴란드", "포르투갈", "슬로바키아", "슬로베니아", "스웨덴", "스위스", "영국"
            ]

            # 유럽 국가 데이터 필터링
            dataL3 = dataL2[dataL2['국가'].isin(eurCotList)]

            # 결측값 제거
            dataL4 = dataL3.dropna()


            # 유럽 국가별 무역의존도 분석

            # 기초 데이터 분석:
            sns.barplot(x='국가', y='val', data=dataL4)
            plt.xticks(rotation=45)
            plt.show()

               # 데이터 분석 주제: 유럽 회원국 무역의존도 최근 10년 데이터 분석
               #  세부목표1. 유럽 국가별 무역의존도 비교 분석
               #  세부목표2. 유럽 시기별(수출)무역의존도 추이 분석
               #  세부목표3. 유럽 시기별(수입)무역의존도 추이 분석
               #
               #  요청 분석 내용
               #  - 데이터 전처리 과정
               #  - 데이터 분석 (pandas, matplotlib, seaborn, autopct, scatter, strip plot, dis plot 함수 포함 사용 요청)
               #  1) 기초 데이터 분석-주요 칼럼 별 분포 분석 (범주형인 경우 counplot 막대 차트 분포 분석, 수치형인 경우 histogram 으로 분포 분석)
               #  2) 세부목표1,2,3 각각에 대한 분석
                #
                # 요청 파일
                # - ipynb 파일 (분석결과가 표시되는 Colab의 ipynb 파일)
                # - pdf파일 (ipynb 파일을 pdf 파일로 출력한 내용)
                # - 데이터 분석 각 구간(함수)에 대한 해석 필요
                # - (Colab에 업로드하기 위해 전처리된 데이터 파일)

            # 국가 컬럼

            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            # fig, axs = plt.subplots(2, 2)
            plt.subplots_adjust(hspace=0.3)
            axs = axs.flatten()

            # dataL2 = xr.Dataset()
            typeList = ['coal', 'gas', 'oil', 'mean']
            for i, type in enumerate(typeList):
                log.info(f"[CHECK] type : {type}")

                inpFileNamePattern = f'leak_basin_GFEI_grid_y19-21_*{type}*.nc'
                inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpFileNamePattern)
                fileList = sorted(glob.glob(inpFile))

                dataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    fileNameNoExt = os.path.basename(fileInfo).split('.nc')[0]

                    if re.search('mean', type, re.IGNORECASE):
                        # label = type + '_' + fileNameNoExt.split('_')[5].upper()
                        label = fileNameNoExt.split('_')[5].upper()
                    else:
                        # label = type + '_' + fileNameNoExt.split('_')[6]
                        label = fileNameNoExt.split('_')[6]

                    data = xr.open_dataset(fileInfo)

                    varList = list(data.variables.keys())
                    dataL1 = data.rename({varList[2] : label})
                    dataL1[label] = np.log(dataL1[label])

                    dataL2 = xr.merge([dataL2, dataL1])

                dataL3 = dataL2.to_dataframe().reset_index(drop=False).drop(columns=['lat', 'lon'])
                dataL4 = dataL3.to_dict(orient='list')

                # ******************************************************************************
                # 다중 그림
                # ******************************************************************************
                # x축 눈금의 범위 설정
                bin_edges = np.arange(-24, 28, 4)
                valList = list(dataL4.values())
                keyList = list(dataL4.keys())

                ax = axs[i]
                n, bins, patches = ax.hist(valList, bins=bin_edges, alpha=1.0, label=keyList, zorder=3)

                colors = plt.cm.coolwarm(np.linspace(0, 1, len(dataL4)))
                for patch, color in zip(patches, colors):
                    for rect in patch:
                        rect.set_facecolor(color)

                # x축 눈금 레이블 설정
                bin_labels = [f"{int(bins[j])}~{int(bins[j + 1])}" for j in range(len(bins) - 1)]
                ax.set_xticks(bins[:-1])
                ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=10)

                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

                # 그래프 제목과 축 레이블 추가
                mainTitle = f'{type.upper()}'
                ax.set_title(mainTitle, loc='right')
                # ax.set_xlabel('Ln Emission [kg/C/year]')
                # ax.set_ylabel('Number of Grids')
                ax.legend(loc='upper left')
                ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)

            # plt.xlabel('Ln Emission [kg/C/year]')
            # plt.ylabel('Number of Grids')

            fig.text(0.5, 0.02, 'Ln Emission [kg/C/year]', ha='center', va='center', fontsize=12)
            fig.text(0.08, 0.5, 'Number of Grids', ha='center', va='center', rotation='vertical', fontsize=12)

            # mainTitle = f'Ln Emission'
            # plt.suptitle(mainTitle)
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'leak_basin_GFEI_grid_y19-21')
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
            plt.tight_layout()
            plt.show()
            plt.close()
            log.info(f'[CHECK] saveImg : {saveImg}')

            # ******************************************************************************
            # 단일 그림
            # ******************************************************************************
            # # x축 눈금의 범위를 설정
            # bin_edges = np.arange(-24, 28, 4)  # -24부터 20까지 4 단위로 설정 (마지막 bin을 포함하기 위해 24까지)
            #
            # # 히스토그램 생성
            # n, bins, patches = plt.hist(list(dataL4.values()), bins=bin_edges, alpha=1.0, label=list(dataL4.keys()), zorder=3)
            #
            # # 색상 팔레트 설정 (각 데이터 세트마다 다른 색상)
            # colors = plt.cm.coolwarm(np.linspace(0, 1, len(dataL4)))
            # for patch, color in zip(patches, colors):
            #     for rect in patch:
            #         rect.set_facecolor(color)
            #
            # # x축 눈금 레이블 설정 (요청하신 형태로)
            # bin_labels = [f"{int(bins[i])}~{int(bins[i + 1])}" for i in range(len(bins) - 1)]
            # plt.xticks(bins[:-1], labels=bin_labels, rotation=45, ha='right')
            #
            # # 그래프 제목과 축 레이블 추가
            # plt.xlabel('Ln Emission [kg/C/year]')
            # plt.ylabel('Number of Grids')
            #
            # plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            # # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            #
            # # 범례 추가
            # plt.legend(loc='upper left')
            # # plt.grid(True)
            # plt.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)
            #
            # mainTitle = f'{type.upper()} Ln Emission'
            # plt.title(mainTitle)
            # saveImg = '{}/{}/{}_leak_basin_GFEI_grid_y19-21.png'.format(globalVar['figPath'], serviceName, type.upper())
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
            # plt.tight_layout()
            # # plt.show()
            # plt.close()
            # log.info(f'[CHECK] saveImg : {saveImg}')

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
