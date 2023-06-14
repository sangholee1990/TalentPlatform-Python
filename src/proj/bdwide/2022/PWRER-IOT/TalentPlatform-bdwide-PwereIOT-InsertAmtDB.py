import argparse
import configparser
import datetime
import glob
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import traceback
import warnings

import dask.dataframe as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pymysql

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
import pytz
from matplotlib import font_manager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from datetime import timedelta

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
# font_manager._rebuild()

#plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
# tzKst = pytz.timezone('Asia/Seoul')
# tzUtc = pytz.timezone('UTC')
# dtKst = datetime.timedelta(hours=9)

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.node()
        , prjName
        , datetime.datetime.now().strftime("%Y%m%d")
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        # fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        # plt.rc('font', family=fontName)

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

        log.info("[CHECK] {} / val : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

def initCfgInfo(sysPath):
    log.info('[START] {}'.format('initCfgInfo'))
    # log.info('[CHECK] sysPath : {}'.format(sysPath))

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')
        dbUser = config.get('mariadb-dms03', 'user')
        dbPwd = config.get('mariadb-dms03', 'pwd')
        dbHost = config.get('mariadb-dms03', 'host')
        dbPort = config.get('mariadb-dms03', 'port')
        dbName = config.get('mariadb-dms03', 'dbName')

        dbEngine = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        # dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessMake = sessionmaker(bind=dbEngine)
        session = sessMake()

        # API 정보
        apiUrl = config.get('pv', 'url')
        apiToken = config.get('pv', 'token')

        result = {
            'dbEngine': dbEngine
            , 'session': session
            , 'apiUrl': apiUrl
            , 'apiToken': apiToken
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))


def setPwrerDataDB(cfgInfo, sYm, data):

    # log.info('[START] {}'.format('setAtmosDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']
        dbData = data
        selDbTable = 'TB_PWRER_DATA_{}'.format(sYm)

        # 테이블 생성
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS `{}`
            (
                GOOD_CODE        varchar(10) not null comment '상품코드',
                CNSMR_NO         varchar(15) not null comment '수용가번호',
                LEGAL_DONG_CD    varchar(10) null comment '법정동코드',
                MESURE_DATE_TM   datetime    not null comment '측정일시',
                SG_PWRER_USE_AM  float       null comment '전력사용량',
                ALL_PWRER_USE_AM float       null comment '전체전력사용량',
                primary key (GOOD_CODE, CNSMR_NO, MESURE_DATE_TM)
            )
            comment '전력 테이블_{}';
            """.format(selDbTable, sYm)
        )
        session.commit()

        for k, dbInfo in dbData.iterrows():

            # 테이블 중복 검사
            resChk = pd.read_sql(
                """
                SELECT COUNT(*) AS CNT FROM `{}`
                WHERE GOOD_CODE = '{}' AND CNSMR_NO = '{}' AND MESURE_DATE_TM = '{}'
                """.format(selDbTable, dbInfo['GOOD_CODE'], dbInfo['CNSMR_NO'], dbInfo['MESURE_DATE_TM'])
                , con=dbEngine
            )

            log.info("[CHECK] MESURE_DATE_TM : {}".format(dbInfo['MESURE_DATE_TM']))

            if (resChk.loc[0, 'CNT'] > 0):
                session.execute(
                    """
                    UPDATE `{}`
                    SET GOOD_CODE = '{}', CNSMR_NO = '{}', LEGAL_DONG_CD = '{}', MESURE_DATE_TM = '{}', SG_PWRER_USE_AM = '{}', ALL_PWRER_USE_AM = '{}'
                    WHERE GOOD_CODE = '{}' AND CNSMR_NO = '{}' AND MESURE_DATE_TM = '{}';
                    """.format(selDbTable
                               , dbInfo['GOOD_CODE'], dbInfo['CNSMR_NO'], dbInfo['LEGAL_DONG_CD'], dbInfo['MESURE_DATE_TM'], dbInfo['SG_PWRER_USE_AM'], dbInfo['ALL_PWRER_USE_AM']
                               , dbInfo['GOOD_CODE'], dbInfo['CNSMR_NO'], dbInfo['MESURE_DATE_TM'])
                )

            else:
                session.execute(
                    """
                    INSERT INTO `{}` (GOOD_CODE, CNSMR_NO, LEGAL_DONG_CD, MESURE_DATE_TM, SG_PWRER_USE_AM, ALL_PWRER_USE_AM)
                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}')
                    """.format(selDbTable
                               , dbInfo['GOOD_CODE'], dbInfo['CNSMR_NO'], dbInfo['LEGAL_DONG_CD'], dbInfo['MESURE_DATE_TM'], dbInfo['SG_PWRER_USE_AM'], dbInfo['ALL_PWRER_USE_AM'])
                )
            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()
        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setAtmosDataDB'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
    # nohup bash RunShell-Python.sh "2020-10" &

    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
    # python3 ${contextPath}/TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "$1" --endDate "$2"
    # python3 TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "20210901" --endDate "20210902"
    # python3 /SYSTEMS/PROG/PYTHON/PV/TalentPlatform-LSH0255-RealTime-For.py --inpPath "/DATA" --outPath "/SYSTEMS/OUTPUT" --modelPath "/DATA" --srtDate "20220101" --endDate "20220102"

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PWRER-IOT'

    prjName = 'test'
    serviceName = 'bdwide'

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

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'
                globalVar['figPath'] = 'E:/DATA/OUTPUT'
                globalVar['modelPath'] = 'E:/DATA'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2018-01-01'
                    , 'endDate': '2020-07-01'
                }

                globalVar['inpPath'] = 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/proj/bdwide/2022/PWRER-IOT'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                }

            # DB 정보
            cfgInfo = initCfgInfo(globalVar['sysPath'])
            dbEngine = cfgInfo['dbEngine']

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='M')

            # =======================================================================
            # 기상정보 자료 수집 및 DB 삽입
            # =======================================================================
            # dtIncDateInfo = dtIncDateList[0]
            for i, dtIncDateInfo in enumerate(dtIncDateList):
                log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
                sYm = dtIncDateInfo.strftime('%Y%m')

                inpFilePattern = '{}.csv'.format(sYm)
                inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'CSV', inpFilePattern)
                fileList = sorted(glob.glob(inpFile))

                if (len(fileList) < 1): continue

                # 파일 읽기
                # fileInfo = fileList[0]
                # data = pd.read_csv(fileInfo)
                data = ds.read_csv(fileList[0])
                data.columns = ['GOOD_CODE', 'CNSMR_NO', 'LEGAL_DONG_CD', 'MESURE_DATE_TM', 'SG_PWRER_USE_AM', 'ALL_PWRER_USE_AM']

                setPwrerDataDB(cfgInfo, sYm, data)


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
