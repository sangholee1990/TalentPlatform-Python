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
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import pandas as pd
import re
import numpy as np
import concurrent.futures

from matplotlib.pyplot import axis

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
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

def makeCsvProc(fileInfo):

    print(f'[START] makeCsvProc')

    result = None

    try:

        filePath = os.path.dirname(fileInfo)
        fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
        data = pd.read_csv(fileInfo, encoding='EUC-KR')

        dataL1 = data.copy()

        # 가공 변수
        dataL1['성명2'] = dataL1['성명'].str[:3]
        dataL1['주소'] = dataL1['주소'].str.replace(" ", "")
        dataL1['등록번호2'] = dataL1['등록번호'].str[:6]

        dataL1['성명'] = dataL1['성명'].where(~ pd.isna(dataL1['성명']), other="")
        dataL1['주소'] = dataL1['주소'].where(~ pd.isna(dataL1['주소']), other="")
        dataL1['등록번호'] = dataL1['등록번호'].where(~ pd.isna(dataL1['등록번호']), other="")

        dataL1['성명2'] = dataL1['성명2'].where(~ pd.isna(dataL1['성명2']), other="")
        dataL1['등록번호2'] = dataL1['등록번호2'].where(~ pd.isna(dataL1['등록번호2']), other="")

        groupData = pd.DataFrame()
        matchIdxList = set()
        matchIdx = 1
        for i, row in dataL1.iterrows():

            print(f'[CHECK] i : {i} / matchIdxList : {len(matchIdxList)}')

            # 이미 포함된 경우 제외
            if i in matchIdxList: continue

            # 처음/중간/끝 4글자
            isRegPattern = [row['등록번호2'][:4], row['등록번호2'][1:5], row['등록번호2'][2:]]

            len_diff = abs(dataL1['주소'].str.len() - len(row['주소']))
            max_len = dataL1['주소'].str.len().combine(len(row['주소']), max)

            # 성명, 주소, 등록번호 일치 검사
            # isName = (len(row['성명']) > 0) & (row['성명'] == dataL1['성명'])
            isName = (len(row['성명2']) > 0) & (dataL1['성명'].str.contains(r'^' + re.escape(row['성명2']), regex=True))
            isAddr = (len(row['주소']) > 0) & (row['주소'] == dataL1['주소']) | ((5 <= max_len) & (max_len <= 9)  & (len_diff <= 1)) | ((max_len >= 10) & (len_diff <= 2))

            # 처음 6글자 만족
            isReg = (len(row['등록번호2']) > 0) & (row['등록번호2'] == dataL1['등록번호2'])
            # 처음/중간/끝 4글자 모두 만족
            # isReg = (len(row['등록번호2']) > 0) & pd.concat([dataL1['등록번호'].str.contains(pat) for pat in isRegPattern], axis=1).all(axis=1)

            # 일치 검사에 대한 개수
            isFlagData = pd.concat([isName, isAddr, isReg], keys=['name', 'addr', 'reg'], ignore_index=False, axis=1)
            isFlagData['cnt'] = isFlagData.sum(axis=1)

            # 2개 이상 매칭
            filterData = isFlagData[isFlagData['cnt'] >= 2]
            filterDataL1 = filterData[~ filterData.index.isin(matchIdxList)]
            matchIdxList.update(filterData.index)
            if (len(filterData) < 2): continue

            data['matchIdx'] = matchIdx
            matchIdx += 1
            groupData = pd.concat([groupData, data.loc[filterDataL1.index].reset_index(drop=False) ], ignore_index=True)

        # 면적과 공시지가 있을 경우 가격 계산, 그 외 None
        groupData['가격'] = np.where(pd.notna(groupData['면적']) & pd.notna(groupData['공시지가']), groupData['면적'] * groupData['공시지가'], np.nan)

        rankData = groupData.groupby('matchIdx')['가격'].sum().reset_index().rename({'가격': '총계'}, axis=1)
        rankData['순위'] = rankData['총계'].rank(method="min", ascending=False)

        dataL3 = groupData.merge(rankData, left_on=['matchIdx'], right_on=['matchIdx'], how='left')

        fnlData = dataL3.groupby('순위').apply(lambda x: x.sort_values(['총계', '가격'], ascending=False, na_position='last'))

        saveFile = '{}/{}-{}.csv'.format(filePath, fileNameNoExt, 'fnlData')
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        fnlData.to_csv(saveFile, index=False, encoding='CP949')
        print(f'[CHECK] saveFile : {saveFile}')

    except Exception as e:
        print(f'Exception : {e}')
        return result

    finally:
        print(f'[END] makeCsvProc')

# def grpProc(start_index, end_index, orgData):
#
#     groups = []
#
#     for i in range(start_index, end_index):
#         current_land = orgData.loc[i]
#         matched_groups = []
#
#         for group in groups:
#             match_count = 0
#
#             # 성명 비교
#             for land in group:
#                 if pd.isna(current_land['성명']) or pd.isna(land['성명']): continue
#                 if land['성명'] == current_land['성명']:
#                     match_count += 1
#                     break  # 그룹 내 다른 토지와 비교할 필요 없음
#
#             # 주소 비교
#             for land in group:
#                 if pd.isna(current_land['주소']) or pd.isna(land['주소']): continue
#                 current_address = current_land['주소'].replace(" ", "")
#                 land_address = land['주소'].replace(" ", "")
#
#                 len_diff = abs(len(current_address) - len(land_address))
#                 max_len = max(len(current_address), len(land_address))
#
#                 if (current_address == land_address) or (5 <= max_len <= 9 and len_diff <= 1) or (max_len >= 10 and len_diff <= 2):
#                     match_count += 1
#                     break
#
#             # 등록번호 비교 (앞 6글자)
#             for land in group:
#                 if pd.isna(current_land['등록번호']) or pd.isna(land['등록번호']): continue
#                 if land['등록번호'][:6] == current_land['등록번호'][:6]:
#                     match_count += 1
#                     break
#
#             # 일치 조건 확인
#             if match_count >= 2:
#                 matched_groups.append(group)
#
#         # 매칭된 그룹 처리
#         if len(matched_groups) == 0:
#             groups.append([current_land])
#         else:
#             # 매칭된 모든 그룹에 추가
#             for group in matched_groups:
#                 group.append(current_land)
#
#     return groups
#
#
# def makeCsvMultiProc(fileInfo, cpuCnt = 4):
#
#     print(f'[START] makeCsvMultiProc')
#
#     result = None
#
#     try:
#
#         filePath = os.path.dirname(fileInfo)
#         fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
#         data = pd.read_csv(fileInfo, encoding='EUC-KR')
#
#         data_length = len(data)
#         split_size = data_length // cpuCnt
#
#         futures = []
#         groups = []
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # 병렬 프로세스 수행
#             for i in range(cpuCnt):
#                 start_index = i * split_size
#                 end_index = (i + 1) * split_size if i != cpuCnt - 1 else data_length
#                 futures.append(executor.submit(grpProc, start_index, end_index, data))
#
#             # 병렬 프로세스 완료 및 결과
#             for future in concurrent.futures.as_completed(futures):
#                 groups.extend(future.result())
#
#         dataL2 = pd.DataFrame()
#         for i, groupInfo in enumerate(groups):
#             if len(groups[i]) < 2: continue
#             dataL1 = pd.DataFrame(groups[i])
#
#             # 면적과 공시지가 있을 경우 가격 계산, 그 외 None
#             dataL1['가격'] = np.where(pd.notna(dataL1['면적']) & pd.notna(dataL1['공시지가']), dataL1['면적'] * dataL1['공시지가'], np.nan)
#             dataL1['i'] = i
#             dataL1['j'] = dataL1.index
#             dataL1['cnt'] = len(dataL1)
#
#             dataL2 = pd.concat([dataL2, dataL1], ignore_index=True)
#
#         rankData = dataL2.groupby('i')['가격'].sum().reset_index().rename({'가격': '총계'}, axis=1)
#         rankData['순위'] = rankData['총계'].rank(method="min", ascending=False)
#
#         dataL3 = dataL2.merge(rankData, left_on=['i'], right_on=['i'], how='left')
#
#         fnlData = dataL3.groupby('순위').apply(lambda x: x.sort_values(['총계', '가격'], ascending=False, na_position='last'))
#
#         saveFile = '{}/{}-{}.csv'.format(filePath, fileNameNoExt, 'fnlData')
#         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
#         fnlData.to_csv(saveFile, index=False)
#         print(f'[CHECK] saveFile : {saveFile}')
#
#     except Exception as e:
#         print(f'Exception : {e}')
#         return result
#
#     finally:
#         print(f'[END] makeCsvMultiProc')


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 특정 조건 (성명, 주소, 주민번호)에 따른 자료 전처리

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
    serviceName = 'LSH0484'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 수행 목록
                'nameList': ['csv']

                # 모델 정보 : 파일 경로, 파일명
                , 'nameInfo': {
                    'csv': {
                        'filePath': '/DATA/INPUT/LSH0484'
                        , 'fileName': '04경기용인_data.csv'
                    }
                }
            }


            for i, nameType in enumerate(sysOpt['nameList']):
                log.info(f'[CHECK] nameType : {nameType}')

                nameInfo = sysOpt['nameInfo'].get(nameType)
                if nameInfo is None: continue

                inpFile = '{}/{}'.format(nameInfo['filePath'], nameInfo['fileName'])
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    continue

                for j, fileInfo in enumerate(fileList):
                    log.info(f'[CHECK] fileInfo : {fileInfo}')

                    makeCsvProc(fileInfo)
                    # makeCsvMultiProc(fileInfo, 8)

        except Exception as e:
            log.error(f'Exception : {e}')

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
