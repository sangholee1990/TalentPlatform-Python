# ================================================
# 요구사항
# ================================================
# 엑셀을 이용한 구글폼 및 박람회폼 통합화 및 통계 분석

# 약 4만건의 2025서울국제불교박람회 사전등록자의 데이터를 한개의 엑셀로 정리한뒤 몇가지 분석과 통계를 진행하고 싶습니다!
# 데이터 정리>>>하단 데이터 일괄 취합
# 1. 구글폼 2만여명의 데이터
# 2. 1,000개 씩 나뉘어진 24개의 엑셀 데이터

# 4만건의 데이터 분석 내용은
# 1. 사전등록자 연령대 분석 10대~70대까지의 수치와 퍼센트
# 2. 사전등록자 남녀 성비(연령 성비 분류 포함)하여 수치와 퍼센트
# 3. 사전등록자 종교 수치와 퍼센트
# 4. 사전등록자 관심경로 수치와 퍼센트
# 5. 사전등록자 관람일시/시간 분석 수치와 퍼센트
# 6. 사전등록자 일반인과 스님(연령 성비 분류 포함) 수치와 퍼센트

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
import re

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
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

def getPhoneNumber(phone_number):
    if pd.isna(phone_number) or phone_number == '':
        return ""
    phone_number = re.sub(r'\D', '', str(phone_number))
    if len(phone_number) == 11:
        return re.sub(r'(\d{3})(\d{4})(\d{4})', r'\1-\2-\3', phone_number)
    elif len(phone_number) == 10:
        return re.sub(r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', phone_number)
    else:
        return phone_number

def getEmail(email):
    if pd.isna(email) or email.strip() == '':
        return ""
    email = str(email).strip()
    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        return email
    else:
        return ""

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

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
    serviceName = 'LSH0607'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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
                'saveFile': '/DATA/OUTPUT/LSH0607/%Y%m%d_total.xlsx',
            }


            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.csv')
            fileList = sorted(glob.glob(inpFile))
            csvData = pd.read_csv(fileList[0])
            df1 = csvData

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'app_2025_250306*.xlsx')
            fileList = sorted(glob.glob(inpFile))

            xlsxData = pd.DataFrame()
            for fileInfo in fileList:
                data = pd.read_excel(fileInfo, engine='openpyxl')
                xlsxData = pd.concat([xlsxData, data])
            xlsxDataL1 = xlsxData.reset_index(drop=False)
            df2 = xlsxDataL1

            # 컬럼명 통일 및 정리
            df1 = df1.rename(columns={
                '1-1. 구분(Division)': '구분',
                '1-2. 성별(Sex)': '성별',
                '1-3. 성명 또는 법명(Name)': '성명',
                '1-4. 휴대전화(Mobile)': '휴대폰',
                '1-6. 연령(Age)': '연령',
                '1-7. 종교(Religion)': '종교',
                '1-8. 거주지역(Residence)': '거주지역',
                '1-9. 관심경로(Funnels)': '관심경로',
                '1-10. 관심분야(Interest)': '관심분야',
                '1-11. 관람목적(Purpose)': '관람목적',
                '1-5. 관람 시간(Time)': '관람시간',
                '2-1.   개인정보 취급방침 및 초상권 활용동의 안내': '개인정보 및 초상권 동의'
            })
            # df1.columns
            # df2.columns

            # "여성"을 "여자"로 통일
            df1['성별'] = df1['성별'].replace('여자', '여성')
            df1['성별'] = df1['성별'].replace('남자', '남성')
            df2['성별'] = df2['성별'].replace('여자', '여성')
            df2['성별'] = df2['성별'].replace('남자', '남성')

            # 개인정보 및 초상권 동의여부를 "동의"로 변경
            df1['개인정보 및 초상권 동의'] = df1['개인정보 및 초상권 동의'].apply(lambda x: "동의" if "동의" in str(x) else "")

            # 주소 결합
            df2['주소'] = df2['주소1'].fillna("").astype(str) + ' ' +  df2['주소2'].fillna("").astype(str)

            # 휴대전화 번호 형식 통일 적용
            df1['휴대폰'] = df1['휴대폰'].apply(getPhoneNumber)
            df2['휴대폰'] = df2['휴대폰'].apply(getPhoneNumber)

            # 이메일 형식 검사 적용
            df2['이메일'] = df2['이메일'].apply(getEmail)

            # 컬럼 설정 & 병합
            df1ColList = ['거주지역', '개인정보 및 초상권 동의']
            df2ColList = ['이메일', '우편번호', '주소', '날짜', '아이피', '작성일']
            comColList = ['구분', '성명', '성별', '휴대폰', '연령', '종교', '관심분야', '관람목적', '관람시간', '관심경로']

            mrgData = pd.merge(df1[comColList + df1ColList], df2[comColList + df2ColList], on=comColList, how='outer')
            # mrgData.columns

            # 필요한 컬럼만 선택 (순서도 조정)
            fnlData = mrgData[['구분', '성명', '성별', '휴대폰', '연령', '종교', '거주지역', '관심분야', '관람목적', '관람시간', '관심경로', '이메일', '우편번호', '주소', '개인정보 및 초상권 동의', '날짜', '아이피', '작성일']]
            fnlDataL1 = fnlData.fillna("")
            # fnlDataL1.columns

            if len(fnlData) > 0:
                saveFile = datetime.now().strftime(sysOpt['saveFile'])
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                fnlDataL1.to_excel(saveFile, index=False)
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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))