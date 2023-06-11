# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import urllib
import warnings
from datetime import datetime

import googlemaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict
from sklearn.neighbors import BallTree

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
    # Python을 이용한 공공데이터포털 실거래가 API (아파트, 단독다가구, 연립다세대, 오피스텔) 수집 및 구글 스튜디오 시각화

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
    serviceName = 'LSH0431'

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

                # 옵션 설정
                sysOpt = {
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2000-01-01'
                    , 'endDate': datetime.now().strftime("%Y-%m-%d")

                    # 공공데이터포털 API
                    , 'apiKey': 'bf9fH0KLgr65zXKT5D/dcgUBIj1znJKnUPrzDVZEe6g4gquylOjmt65R5cjivLPfOKXWcRcAWU0SN7KKXBGDKA=='

                    # 건축 인허가
                    , 'apiUrl': 'https://apis.data.go.kr/1613000/ArchPmsService_v2/getApBasisOulnInfo'
                    # 아파트 실거래
                    , 'apiUrl2': 'http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev'
                    #  아파트 전월세
                    , 'apiUrl3': 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent'

                    # 검색 목록
                    , 'addrList': ['서울특별시 강북구']
                    # , 'addrList': ['서울특별시']

                    # 네이버 API 정보
                    , 'naverId': 'q6rz1sjycu'
                    , 'naverPw': 'KYof7RAjdDffoyEyQPFzb0mUiiPc3DKy2qxvaaZf'
                    , 'naverApi': 'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode'

                    # 구글 API 정보
                    , 'googleApiKey': 'AIzaSyCkYokUFIcH5OYDaYU0IrFLX89wX1o7-qc'
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # 구글 API 설정
            gmap = googlemaps.Client(key=sysOpt['googleApiKey'])

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

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용하여 아파트매매 실거래
            # # *********************************************************************************
            # 요청 API
            apiUrl = sysOpt['apiUrl2']

            # addrInfo = sysOpt['addrList'][0]
            for ii, addrInfo in enumerate(sysOpt['addrList']):
                log.info(f'[CHECK] addrInfo : {addrInfo}')

                # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
                admDataL1 = admData[admData['법정동명'].str.contains(addrInfo)]
                if (len(admDataL1) < 1): continue

                # 시군구 목록
                sigunguCdList = set(admDataL1['sigunguCd'])

                # 페이지 설정
                pageList = np.arange(0, 9999, 1)

                for i, sigunguCdInfo in enumerate(sigunguCdList):
                    log.info(f'[CHECK] sigunguCdInfo : {sigunguCdInfo}')

                    for j, dtMonthInfo in enumerate(dtMonthList):
                        # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')

                        dtYearMonth = dtMonthInfo.strftime('%Y%m')

                        saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '아파트 실거래', addrInfo, '아파트 실거래', addrInfo, dtYearMonth)
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)

                        dataL1 = pd.DataFrame()

                        fileChkList = glob.glob(saveFile)
                        # if (len(fileChkList) > 0): continue

                        for k, pageInfo in enumerate(pageList):

                            try:
                                # reqUrl = f'{apiUrl}?serviceKey={apiKey}&pageNo=1&numOfRows=100&LAWD_CD={sigunguCdInfo}&DEAL_YMD={dtYearMonth}'
                                #
                                # res = urllib.request.urlopen(reqUrl)
                                # resCode = res.getcode()

                                inParams = {'serviceKey': sysOpt['apiKey'], 'pageNo': 1, 'numOfRows': 100, 'LAWD_CD': sigunguCdInfo, 'DEAL_YMD': dtYearMonth}
                                apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')

                                res = requests.get(apiUrl, params=apiParams)

                                resCode = res.status_code
                                if resCode != 200: continue

                                # json 읽기
                                # resData = json.loads(res.read().decode('utf-8'))

                                # xml to json 읽기
                                resData = xmltodict.parse(res.read().decode('utf-8'))
                                resultCode = resData['response']['header']['resultCode']
                                if (resultCode != '00'): continue

                                resBody = resData['response']['body']
                                totalCnt = int(resBody['totalCount'])
                                if (totalCnt < 1): break

                                itemList = resBody['items']['item']
                                if (len(itemList) < 1): break

                                data = pd.DataFrame.from_dict(itemList)
                                data['addrInfo'] = addrInfo
                                data['sigunguCdInfo'] = sigunguCdInfo
                                data['dtYearMonth'] = dtYearMonth
                                data['pageInfo'] = pageInfo

                                dataL1 = pd.concat([dataL1, data], ignore_index=False)

                                if (totalCnt < (pageInfo * 100)): break

                            except Exception as e:
                                log.error("Exception : {}".format(e))

                        # 자료 저장
                        if (len(dataL1) > 0):
                            data.to_csv(saveFile, index=False)
                            log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 수집] 오픈API를 이용하여 아파트 전월세
            # *********************************************************************************
            # 요청 API
            # apiKey = sysOpt['apiKey']
            # apiUrl = sysOpt['apiUrl3']
            #
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # admDataL1 = admData[admData['법정동명'].str.contains('서울')]
            #     admDataL1 = admData[admData['법정동명'].str.contains(addrInfo)]
            #     if (len(admDataL1) < 1): continue
            #
            #     # 시군구 목록
            #     sigunguCdList = set(admDataL1['sigunguCd'])
            #
            #     # 페이지 설정
            #     pageList = np.arange(0, 9999, 1)
            #
            #     for i, sigunguCdInfo in enumerate(sigunguCdList):
            #         log.info(f'[CHECK] sigunguCdInfo : {sigunguCdInfo}')
            #
            #         for j, dtMonthInfo in enumerate(dtMonthList):
            #             # log.info(f'[CHECK] dtMonthInfo : {dtMonthInfo}')
            #
            #             dtYearMonth = dtMonthInfo.strftime('%Y%m')
            #
            #             saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '아파트 전월세', addrInfo, '아파트 전월세', addrInfo, dtYearMonth)
            #             os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #
            #             fileChkList = glob.glob(saveFile)
            #             if (len(fileChkList) > 0): continue
            #
            #             dataL1 = pd.DataFrame()
            #
            #             for k, pageInfo in enumerate(pageList):
            #
            #                 try:
            #                     reqUrl = f'{apiUrl}?serviceKey={apiKey}&pageNo=1&numOfRows=100&LAWD_CD={sigunguCdInfo}&DEAL_YMD={dtYearMonth}'
            #                     res = urllib.request.urlopen(reqUrl)
            #                     resCode = res.getcode()
            #                     if resCode != 200: continue
            #
            #                     # json 읽기
            #                     # resData = json.loads(res.read().decode('utf-8'))
            #
            #                     # xml to json 읽기
            #                     resData = xmltodict.parse(res.read().decode('utf-8'))
            #                     resultCode = resData['response']['header']['resultCode']
            #                     if (resultCode != '00'): continue
            #
            #                     resBody = resData['response']['body']
            #                     totalCnt = int(resBody['totalCount'])
            #                     if (totalCnt < 1): break
            #
            #                     itemList = resBody['items']['item']
            #                     if (len(itemList) < 1): break
            #
            #                     data = pd.DataFrame.from_dict(itemList)
            #                     data['addrInfo'] = addrInfo
            #                     data['sigunguCdInfo'] = sigunguCdInfo
            #                     data['dtYearMonth'] = dtYearMonth
            #                     data['pageInfo'] = pageInfo
            #
            #                     dataL1 = pd.concat([dataL1, data], ignore_index=False)
            #
            #                     if (totalCnt < (pageInfo * 100)): break
            #
            #                 except Exception as e:
            #                     log.error("Exception : {}".format(e))
            #
            #             # 자료 저장
            #             if (len(dataL1) > 0):
            #                 data.to_csv(saveFile, index=False)
            #                 log.info(f'[CHECK] saveFile : {saveFile}')

            # *********************************************************************************
            # [자료 전처리] 아파트 실거래
            # *********************************************************************************
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '????1101*.xlsx')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '????1101*.xlsx')
            addrInfo = sysOpt['addrList'][4]
            for ii, addrInfo in enumerate(sysOpt['addrList']):
                log.info(f'[CHECK] addrInfo : {addrInfo}')

                inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '아파트 실거래', addrInfo, '아파트 실거래_*_*')
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                    raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                posData = pd.DataFrame()
                for fileInfo in fileList:
                    # tmpData = pd.read_excel(fileInfo, skiprows=16)
                    # posData = posData.append(tmpData)
                    tmpData = pd.read_csv(fileInfo)
                    posData = pd.concat([posData, tmpData], ignore_index=True)

                # posData = posData.reset_index()
                # posData['주소'] = addrInfo + ' ' + posData['도로명'] + ' ' + posData['단지명']
                posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'] + ' ' + posData['지번']

                # addrList = posData['주소'].unique()
                addDtlrList = set(posData['addrDtlInfo'])

                posDataL1 = pd.DataFrame()
                for i, addrDtlInfo in enumerate(addDtlrList):
                    log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')

                    posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
                    posDataL1.loc[i, 'lat'] = None
                    posDataL1.loc[i, 'lon'] = None

                    try:
                        rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
                        if (len(rtnGeo) < 1): continue

                        # 위/경도 반환
                        posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
                        posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                # for i, addrInfo in enumerate(addrList):
                #     log.info(f'[CHECK] addrInfo : {addrInfo}')
                #
                #     addUrlenc = parse.quote(addrInfo)
                #     url = f'{sysOpt["naverApi"]}?query={addUrlenc}'
                #     request = Request(url)
                #     request.add_header('X-NCP-APIGW-API-KEY-ID', sysOpt['naverId'])
                #     request.add_header('X-NCP-APIGW-API-KEY', sysOpt['naverPw'])
                #
                #     # posDataL1.loc[i, '주소'] = addrInfo
                #     # posDataL1.loc[i, 'lat'] = None
                #     # posDataL1.loc[i, 'lon'] = None
                #
                #     try:
                #         response = urlopen(request)
                #         resCode = response.getcode()
                #
                #         if (resCode != 200): continue
                #
                #         responseBody = response.read().decode('utf-8')
                #         responseBody = json.loads(responseBody)
                #
                #         resCnt = responseBody['meta']['count']
                #         if (resCnt < 1): continue
                #
                #         log.info("[CHECK] [{}] {}".format(i, addrInfo))
                #
                #         posDataL1.loc[i, 'lat'] = responseBody['addresses'][0]['y']
                #         posDataL1.loc[i, 'lon'] = responseBody['addresses'][0]['x']
                #
                #     except HTTPError as e:
                #         log.error("Exception : {}".format(e))

                posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
                # posDataL2.drop(['index'], axis=1, inplace=True)

                saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 실거래', addrInfo, datetime.now().strftime('%Y%m%d'))
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                posDataL2.to_csv(saveFile, index_label=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

            # # *********************************************************************************
            # # [자료 전처리] 아파트 전월세
            # # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '아파트(전월세)_????.xlsx')
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '아파트 전월세', addrInfo, '아파트 전월세_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     posData = pd.DataFrame()
            #     for fileInfo in fileList:
            #         # tmpData = pd.read_excel(fileInfo, skiprows=16)
            #         # posData = posData.append(tmpData)
            #         tmpData = pd.read_csv(fileInfo)
            #         posData = pd.concat([posData, tmpData], ignore_index=True)
            #
            #     # posData = posData.reset_index()
            #     # posData['주소'] = '서울특별시 강북구' + ' ' + posData['도로명'] + ' ' + posData['단지명']
            #     posData['addrDtlInfo'] = posData['addrInfo'] + ' ' + posData['법정동'] + ' ' + posData['아파트'] + ' ' + posData['지번']
            #
            #     # addrList = posData['주소'].unique()
            #     addrDtlList = set(posData['addrDtlInfo'])
            #
            #     posDataL1 = pd.DataFrame()
            #     for i, addrDtlInfo in enumerate(addrDtlList):
            #         log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
            #
            #         posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
            #         posDataL1.loc[i, 'lat'] = None
            #         posDataL1.loc[i, 'lon'] = None
            #
            #         try:
            #             rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
            #             if (len(rtnGeo) < 1): continue
            #
            #             # 위/경도 반환
            #             posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
            #             posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
            #
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            #     # for i, addrInfo in enumerate(addrList):
            #     #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #     #
            #     #     addUrlenc = parse.quote(addrInfo)
            #     #     url = f'{sysOpt["naverApi"]}?query={addUrlenc}'
            #     #     request = Request(url)
            #     #     request.add_header('X-NCP-APIGW-API-KEY-ID', sysOpt['naverId'])
            #     #     request.add_header('X-NCP-APIGW-API-KEY', sysOpt['naverPw'])
            #     #
            #     #     # posDataL1.loc[i, '주소'] = addrInfo
            #     #     # posDataL1.loc[i, 'lat'] = None
            #     #     # posDataL1.loc[i, 'lon'] = None
            #     #
            #     #     try:
            #     #         response = urlopen(request)
            #     #         resCode = response.getcode()
            #     #
            #     #         if (resCode != 200): continue
            #     #
            #     #         responseBody = response.read().decode('utf-8')
            #     #         responseBody = json.loads(responseBody)
            #     #
            #     #         resCnt = responseBody['meta']['count']
            #     #         if (resCnt < 1): continue
            #     #
            #     #         log.info("[CHECK] [{}] {}".format(i, addrInfo))
            #     #
            #     #         posDataL1.loc[i, 'lat'] = responseBody['addresses'][0]['y']
            #     #         posDataL1.loc[i, 'lon'] = responseBody['addresses'][0]['x']
            #     #
            #     #     except HTTPError as e:
            #     #         log.error("Exception : {}".format(e))
            #
            #     posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
            #     # posDataL2.drop(['index'], axis=1, inplace=True)
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 전월세', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     posDataL2.to_csv(saveFile, index_label=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # *********************************************************************************
            # # [자료 전처리] 건축 인허가
            # # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '건축 인허가', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     posData = pd.read_csv(fileList[0])
            #     posData['addrDtlInfo'] = posData['platPlc']
            #
            #     addrDtlList = set(posData['addrDtlInfo'])
            #
            #     posDataL1 = pd.DataFrame()
            #     for i, addrDtlInfo in enumerate(addrDtlList):
            #         log.info(f'[CHECK] addrDtlInfo : {addrDtlInfo}')
            #
            #         posDataL1.loc[i, 'addrDtlInfo'] = addrDtlInfo
            #         posDataL1.loc[i, 'lat'] = None
            #         posDataL1.loc[i, 'lon'] = None
            #
            #         try:
            #             rtnGeo = gmap.geocode(addrDtlInfo, language='ko')
            #             if (len(rtnGeo) < 1): continue
            #
            #             # 위/경도 반환
            #             posDataL1.loc[i, 'lat'] = rtnGeo[0]['geometry']['location']['lat']
            #             posDataL1.loc[i, 'lon'] = rtnGeo[0]['geometry']['location']['lng']
            #
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            #     posDataL2 = pd.merge(left=posData, right=posDataL1, how='left', left_on='addrDtlInfo', right_on='addrDtlInfo')
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     posDataL2.to_csv(saveFile, index_label=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # *********************************************************************************
            # # [자료 가공] 건축 인허가 및 아파트 실거래 간의 공간 일치
            # # *********************************************************************************
            # # addrInfo = sysOpt['addrList'][0]
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     houseConsData = pd.read_csv(fileList[0])
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 실거래_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     aptPriceData = pd.read_csv(fileList[0])
            #
            #     # *********************************************************************************
            #     # 최근접 좌표
            #     # *********************************************************************************
            #     # BallTree를 위한 초기 데이터
            #     houseConsRef = houseConsData[['addrDtlInfo', 'lat', 'lon']].dropna().reset_index(drop=True)
            #     baTree = BallTree(np.deg2rad(houseConsRef[['lat', 'lon']].values), metric='haversine')
            #
            #     aptPriceDataL1 = aptPriceData
            #     for i, posInfo in aptPriceDataL1.iterrows():
            #         if (pd.isna(posInfo['lat']) or pd.isna(posInfo['lon'])): continue
            #
            #         closest = baTree.query(np.deg2rad(np.c_[posInfo['lat'], posInfo['lon']]), k=1)
            #         cloDist = closest[0][0][0] * 1000.0
            #         cloIdx = closest[1][0][0]
            #
            #         aptPriceDataL1.loc[i, '인허가addrDtlInfo'] = houseConsRef.loc[cloIdx, 'addrDtlInfo']
            #         aptPriceDataL1.loc[i, '인허가lat'] = houseConsRef.loc[cloIdx, 'lat']
            #         aptPriceDataL1.loc[i, '인허가lon'] = houseConsRef.loc[cloIdx, 'lon']
            #         aptPriceDataL1.loc[i, '인허가distKm'] = cloDist
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 실거래', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     aptPriceDataL1.to_csv(saveFile, index_label=False)
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # *********************************************************************************
            # # [자료 가공] 건축 인허가 및 아파트 전월세 간의 공간 일치
            # # *********************************************************************************
            # addrInfo = sysOpt['addrList'][0]
            # for ii, addrInfo in enumerate(sysOpt['addrList']):
            #     log.info(f'[CHECK] addrInfo : {addrInfo}')
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     houseConsData = pd.read_csv(fileList[0])
            #
            #     inpFile = '{}/{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '아파트 전월세_*_*')
            #     fileList = sorted(glob.glob(inpFile))
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     aptPriceData = pd.read_csv(fileList[0])
            #
            #     # *********************************************************************************
            #     # 최근접 좌표
            #     # *********************************************************************************
            #     # BallTree를 위한 초기 데이터
            #     houseConsRef = houseConsData[['addrDtlInfo', 'lat', 'lon']].dropna().reset_index(drop=True)
            #     baTree = BallTree(np.deg2rad(houseConsRef[['lat', 'lon']].values), metric='haversine')
            #
            #     aptPriceDataL1 = aptPriceData
            #     for i, posInfo in aptPriceDataL1.iterrows():
            #         if (pd.isna(posInfo['lat']) or pd.isna(posInfo['lon'])): continue
            #
            #         closest = baTree.query(np.deg2rad(np.c_[posInfo['lat'], posInfo['lon']]), k=1)
            #         cloDist = closest[0][0][0] * 1000.0
            #         cloIdx = closest[1][0][0]
            #
            #         aptPriceDataL1.loc[i, '인허가addrDtlInfo'] = houseConsRef.loc[cloIdx, 'addrDtlInfo']
            #         aptPriceDataL1.loc[i, '인허가lat'] = houseConsRef.loc[cloIdx, 'lat']
            #         aptPriceDataL1.loc[i, '인허가lon'] = houseConsRef.loc[cloIdx, 'lon']
            #         aptPriceDataL1.loc[i, '인허가distKm'] = cloDist
            #
            #     saveFile = '{}/{}/{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '전처리', addrInfo, '건축 인허가-아파트 전월세', addrInfo, datetime.now().strftime('%Y%m%d'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     aptPriceDataL1.to_csv(saveFile, index_label=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

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
