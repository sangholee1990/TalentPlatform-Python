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

            # ********************************************************************
            # 데이터 전처리 과정
            # ********************************************************************
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

            # ********************************************************************
            # 기초 데이터 분석
            # ********************************************************************
            # 유럽 국가별 수입/수출 빈도 분포 시각화
            # 유럽 국가마다 종류 (수입, 수출)에 관계 없이 10년간 (2012~2021년) 동안 균일하게 분포됨 (데이터 개수 = 10개)
            dataL5 = dataL4.loc[(dataL4['특성'] == '원데이터')]

            sns.countplot(y="국가", hue="종류", data=dataL5)
            plt.title('유럽 국가별 수입/수출 빈도 분포')
            plt.xlabel('개수')
            plt.show()

            # 무역의존도 수입/수출 히스토그램 시각화
            # 무역의존도 수입은 17.03 ~ 93.22 (평균 45.29)으로 나타내는 반면 수출에서는 보다 넓게 14.58 ~ 92.88 (평균 45.42)로 분포함
            # 특히 수입 및 수출은 무역의존도 25 부근에 높은 빈도를 보임
            sns.histplot(x='val', hue='종류', data=dataL5, multiple='dodge')
            plt.title('무역의존도 수입/수출 히스토그램')
            plt.xlabel('무역의존도')
            plt.ylabel('개수')
            plt.show()

            # 수입/수출에 따른 요약 통계량
            # dataL5.loc[(dataL5['종류'] == '수입')].describe()
            # dataL5.loc[(dataL5['종류'] == '수출')].describe()

            # ********************************************************************
            # 세부 목표1. 유럽 국가별 무역의존도 비교 분석
            # ********************************************************************
            # 유럽 국가별 무역의존도 비교 분석 시각화
            # 특정 국가 (벨기에, 슬로바키아)의 경우 수출 및 수입의 무역의존도가 둘다 높은 반면 일부 국가 (체코, 라트비아 등)은 하나만 의존 (수출 또는 수입) 경향을 보임
            # 이는 벨기에 및 슬로바키아는 유럽연합 회원국으로서 EU 시장과 긴밀한 관계를 유지하고 특히 벨기는 유럽 내에서 중요한 국제 물류 및 운송의 중심지 역할을 수행함
            dataL5 =  dataL4.loc[(dataL4['특성'] == '원데이터')]

            statDataL1 = dataL5.groupby(['국가', '종류'])['val'].mean().reset_index(drop=False)
            statDataL2 = statDataL1.sort_values('val', ascending=False)

            sns.barplot(x='val', y='국가', hue='종류', data=statDataL2)
            plt.title('유럽 국가별 무역의존도 비교 분석')
            plt.xlabel('평균 무역의존도')
            # plt.ylabel('국가')
            plt.show()

            # ********************************************************************
            # 세부 목표2. 유럽 시기별(수출) 무역의존도 추이 분석
            # ********************************************************************
            # 유럽 시기별(수출) 무역의존도 추이 분석 시각화
            # 10년간 (2019~2020년) 무역의존도는 다양한 변화 패턴을 보임
            # 즉 2012~2015년은 일정한 감소 경향을 보이다가 2016년 다소 증가함
            # 또한 2017~2019년까지 점차 감소 경향을 보이며 2020년 급격히 감소함
            # 그러나 2021년에는 폭발적으로 증가하여 2012년과 유사한 분포를 보임
            #  특히 2020년의 급격한 변화는 COVID-19 팬데믹과 관련이 있을 수 있으며, 이는 많은 국가들의 수출입 활동에 큰 변화를 가져왔습니다. 팬데믹으로 인한 글로벌 무역의 중단이나 제약이 무역의존도 감소에 기여했을 수 있으며, 이후의 회복 과정에서 무역 활동이 다시 증가하여 2021년에 무역의존도가 상승한 것으로 추측

            # 특히
            dataL5 = dataL4.loc[(dataL4['특성'] == '원데이터') & (dataL4['종류'] == '수출')]

            statDataL1 = dataL5.groupby(['연도', '종류'])['val'].mean().reset_index(drop=False)
            statDataL2 = statDataL1.sort_values('연도', ascending=True)

            sns.lineplot(x='연도', y='val', hue='종류', style="종류", markers=['o'], data=statDataL2)
            plt.title('유럽 시기별 수출 무역의존도 추이 분석')
            plt.xlabel('연도')
            plt.ylabel('평균 무역의존도')
            plt.show()

            # ********************************************************************
            # 세부 목표3. 유럽 시기별(수입) 무역의존도 추이 분석
            # ********************************************************************
            dataL5 = dataL4.loc[(dataL4['특성'] == '원데이터') & (dataL4['종류'] == '수입')]

            statDataL1 = dataL5.groupby(['연도', '종류'])['val'].mean().reset_index(drop=False)
            sstatDataL2 = statDataL1.sort_values('연도', ascending=True)

            sns.lineplot(x='연도', y='val', hue='종류', style="종류", markers=['o'], data=statDataL2)
            plt.title('유럽 시기별 수입 무역의존도 추이 분석')
            plt.xlabel('연도')
            plt.ylabel('평균 무역의존도')
            plt.show()

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
