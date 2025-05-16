# ================================================
# 요구사항
# ================================================
# Python을 이용한 청소년 인터넷 게임 중독 관련 소셜데이터 수집과 분석을 위한 한국형 온톨로지 개발 및 평가
# lsof -i :9998 | awk '{print $2}' | xargs kill -9
# lsof -i :9999 | awk '{print $2}' | xargs kill -9

# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/TCPIP
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/TCPIP/TalentPlatform-bdwide-DaemonFramework-tcpipRec.py &
# tail -f nohup.out

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
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import pytz
import struct
from twisted.internet import reactor, protocol, endpoints
from twisted.python import log
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from urllib.parse import quote_plus
from twisted.internet.error import CannotListenError
from twisted.protocols.policies import TimeoutMixin

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

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

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

    log.propagate = False

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
    log.info(f"inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

# 서버측 프로토콜
class ReceivingProtocol(protocol.Protocol, TimeoutMixin):
    def __init__(self, sysOptForProtocol):
        self._buffer = b''
        self.sysOpt = sysOptForProtocol

    def connectionMade(self):
        peer = self.transport.getPeer()
        self.sysOpt['tcpip']['clientHost'] = peer.host
        self.sysOpt['tcpip']['clientPort'] = peer.port
        self.setTimeout(self.sysOpt['tcpip']['timeout'])
        log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] 클라이언트 연결")

    def dataReceived(self, data):
        self.resetTimeout()
        # self._buffer += data
        self._buffer = data
        headerSize = 4

        while len(self._buffer) >= headerSize:
            try:
                sof = self._buffer[0]
                if sof != 0xFF:
                    log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] sof 수신 실패 : {sof:#02x} 연결 종료")
                    self.transport.loseConnection()
                    return

                msgIdH = self._buffer[1]
                msgIdL = self._buffer[2]
                payloadSize = self._buffer[3]
                msgId = (msgIdH << 8) | msgIdL

                msgSize = headerSize + payloadSize
                # if not len(self._buffer) >= msgSize:
                #     log.info(f"[Server] 메시지 일부 수신. Payload 길이({payloadSize}), 현재 버퍼({len(self._buffer)}). 대기 중...")
                #     break

                msgData = self._buffer[:msgSize]
                payload = msgData[headerSize:]
                self._buffer = self._buffer[msgSize:]

                log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] SOF : {sof:#02x}")
                log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] Msg ID : {msgId:#04x} ({msgId})")
                log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] Payload Length : {payloadSize}")
                log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] Payload : {payload!r}")

                self.handleMsg(msgId, payload)

            except Exception as e:
                log.error(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] 메시지 처리 오류: {e}")
                self._buffer = b''
                self.transport.loseConnection()
                return

    def handleMsg(self, msgId, payload):

        # 가공 포맷
        resPayload = b'404'
        nowUtc = datetime.now(pytz.utc)
        nowKst = nowUtc.astimezone(tzKst)

        dbData = {'device': f"{self.sysOpt['tcpip']['clientHost']}:{self.sysOpt['tcpip']['clientPort']}:{msgId}", 'eventTime': nowKst, 'eventType': '연결', 'address': f"{self.sysOpt['tcpip']['clientHost']}"}
        dbMergeData(self.sysOpt['mysql']['session'], self.sysOpt['mysql']['table'][f"tbConnLog"], dbData, pkList=['id'], excList=[])

        try:
            if msgId == 0x0000:
                resPayload = payload
            elif msgId == 0x0003:
                resPayload = struct.pack('>HBBBBB', nowKst.year, nowKst.month, nowKst.day, nowKst.hour, nowKst.minute, nowKst.second)
            elif msgId == 0x0014:
                payloadOpt = [
                    ('YEAR', 'H', 2),
                    ('CUSTOMER_LINK_NUMBER', 'I', 4),
                    ('DATE_TIME', 'ascii', 19),
                ]
                dbData = payloadProc(payload, payloadOpt)
                tbOutputData = self.sysOpt['mysql']['table'][f"tbOutputData{dbData['YEAR']}"]
                listDbProc = self.sysOpt['mysql']['session'].query(tbOutputData).filter(tbOutputData.c.CUSTOMER_LINK_NUMBER == dbData['CUSTOMER_LINK_NUMBER'], tbOutputData.c.DATE_TIME == dbData['DATE_TIME']).first()
                resPayload = ",".join(str(item) for item in listDbProc).encode('utf-8')
            elif msgId == 0x0015:
                payloadOpt = [
                    ('YEAR', 'H', 2),
                    ('CUSTOMER_LINK_NUMBER', 'I', 4),
                    ('DATE_TIME', 'ascii', 19),
                ]
                dbData = payloadProc(payload, payloadOpt)
                tbOutputStatData = self.sysOpt['mysql']['table'][f"tbOutputStatData{dbData['YEAR']}"]
                listDbProc = self.sysOpt['mysql']['session'].query(tbOutputStatData).filter(tbOutputStatData.c.CUSTOMER_LINK_NUMBER == dbData['CUSTOMER_LINK_NUMBER'], tbOutputStatData.c.DATE_TIME == dbData['DATE_TIME']).first()
                resPayload = ",".join(str(item) for item in listDbProc).encode('utf-8')
            elif msgId == 0x0030:
                payloadOpt = [
                    ('YEAR', 'H', 2),
                    ('PRODUCT_SERIAL_NUMBER', 'ascii', 49),
                    ('DATE_TIME', 'ascii', 19),
                    ('TEMP', 'f', 4),
                    ('HMDTY', 'f', 4),
                    ('PM25', 'f', 4),
                    ('PM10', 'f', 4),
                    ('MVMNT', 'ascii', 20),
                    ('TVOC', 'f', 4),
                    ('HCHO', 'f', 4),
                    ('CO2', 'f', 4),
                    ('CO', 'f', 4),
                    ('BENZO', 'f', 4),
                    ('RADON', 'f', 4),
                ]
                dbData = payloadProc(payload, payloadOpt)
                dbData['MOD_DATE'] = nowKst
                isDbProc = dbMergeData(self.sysOpt['mysql']['session'], self.sysOpt['mysql']['table'][f"tbInputData{dbData['YEAR']}"], dbData, pkList=['PRODUCT_SERIAL_NUMBER', 'DATE_TIME'], excList=['YEAR'])
                log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] isDbProc : {isDbProc} / dbData : {dbData}")
                resPayload = b'200' if dbData and isDbProc else b'400'

        except Exception as e:
            log.error(f"Exception : {e}")

        # 반환 포맷
        log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] msgId : {msgId} / resPayload : {resPayload!r}")
        if msgId in [0x0000, 0x0003]:
            resHeader = self.createHeader(msgId, len(resPayload))
            resMsg = resHeader + resPayload
        else:
            resMsg = resPayload
        self.transport.write(resMsg)

        dbData = {'device': f"{self.sysOpt['tcpip']['clientHost']}:{self.sysOpt['tcpip']['clientPort']}:{msgId}", 'eventTime': nowKst, 'eventType': '해제', 'address': f"{self.sysOpt['tcpip']['clientHost']}"}
        dbMergeData(self.sysOpt['mysql']['session'], self.sysOpt['mysql']['table'][f"tbConnLog"], dbData, pkList=['id'], excList=[])

    def createHeader(self, msg_id, payloadSize):
        sof = 0xFF
        msgIdH = (msg_id >> 8) & 0xFF
        msgIdL = msg_id & 0xFF
        if payloadSize > 255:
             log.warn(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] 페이로드 길이가 255를 초과({payloadSize})하지만 헤더는 1바이트 길이만 지원합니다.")
             payloadSize = 255

        header = struct.pack('>BBBB', sof, msgIdH, msgIdL, payloadSize)
        return header

    def connectionLost(self, reason):
        log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] 클라이언트 해제 : {reason.getErrorMessage()}")
        self._buffer = b''

    # def timeoutConnection(self):
    #     log.info(f"[{self.sysOpt['tcpip']['clientHost']}][{self.sysOpt['tcpip']['clientPort']}] 클라이언트 타임아웃")
    #     self.transport.loseConnection()

# 서버 측 팩토리
class ReceivingFactory(protocol.Factory):

    protocol = ReceivingProtocol

    def __init__(self, sysOptForProtocol):
        self.sysOpt = sysOptForProtocol

    # 신규 연결 시 프로토콜 인스턴스 생성 메서드
    def buildProtocol(self, addr):
        log.info(f"프로토콜 인스턴스 생성 : {addr}")
        p = self.protocol(self.sysOpt)
        p.factory = self
        return p

def payloadProc(payload, payloadOpt):
    data = {}
    offset = 0
    try:
        for field, fmt, size in payloadOpt:
            fieldByte = payload[offset:offset + size]

            value = None
            if fmt == 'ascii':
                value = fieldByte.decode('ascii').rstrip('\x00')
            elif fmt:
                tmp = struct.unpack('>' + fmt, fieldByte)
                value = round(tmp[0], 1) if fmt == 'f' else tmp[0]
            else:
                log.warn(f"{field} : {value} : 필드 변환 실패")
                value = fieldByte

            data[field] = value
            offset += size

        return data
    except Exception as e:
         log.error(f"Exception : {e} : {field} / {offset}")
         return None

def dbMergeData(session, table, dataList, pkList, excList):
    try:
        # excList 컬럼 제외
        dataList = {key: value for key, value in dataList.items() if key not in excList}

        # 배열 선언
        if isinstance(dataList, dict): dataList = [dataList]

        # PK에 따른 수정/등록 처리
        stmt = mysql_insert(table)
        setData = {
            key: stmt.inserted[key] for key in dataList[0].keys() if key not in pkList
        }
        onConflictStmt = stmt.on_duplicate_key_update(**setData)
        session.execute(onConflictStmt, dataList)
        session.commit()
        return True

    except Exception as e:
        session.rollback()
        log.error(f'Exception : {e}')
        return False
    finally:
        session.close()

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'bdwide'

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
                'tcpip': {
                    'serverPort': 9999,
                    'timeout': 30,
                    'clientHost': None,
                    'clientPort': None,
                },
                'mysql': {
                    # 설정
                    'host': 'localhost',
                    'user': 'dms01user01',
                    'password': 'Bdwide365!@',
                    'port': '3306',
                    'schema': 'DMS02',

                    # 세션
                    'session': None,
                    'table': {
                        'tbConnLog': None
                    },
                },
            }

            try:
                # dbUrl = f"mysql+mysqlclient://{sysOpt['mysql']['user']}:{sysOpt['mysql']['password']}@{sysOpt['mysql']['host']}:{sysOpt['mysql']['port']}/{sysOpt['mysql']['schema']}"
                dbUrl = f"mysql+pymysql://{sysOpt['mysql']['user']}:{quote_plus(sysOpt['mysql']['password'])}@{sysOpt['mysql']['host']}:{sysOpt['mysql']['port']}/{sysOpt['mysql']['schema']}"
                dbEngine = create_engine(
                    dbUrl,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    pool_timeout=30,
                    echo=False,
                )

                sessionMake = sessionmaker(autocommit=False, autoflush=False, bind=dbEngine)
                sysOpt['mysql']['session'] = sessionMake()

                # 테이블 정보
                metaData = MetaData()
                for year in range(2022, 2027):
                    try:
                        sysOpt['mysql']['table'][f"tbInputData{year}"] = Table(f"TB_INPUT_DATA_{year}", metaData, autoload_with=dbEngine, schema=sysOpt['mysql']['schema'])
                        sysOpt['mysql']['table'][f"tbOutputData{year}"] = Table(f"TB_OUTPUT_DATA_{year}", metaData, autoload_with=dbEngine, schema=sysOpt['mysql']['schema'])
                        sysOpt['mysql']['table'][f"tbOutputStatData{year}"] = Table(f"TB_OUTPUT_STAT_DATA_{year}", metaData, autoload_with=dbEngine, schema=sysOpt['mysql']['schema'])
                    except Exception as e:
                        pass

                sysOpt['mysql']['table']['tbConnLog'] = Table(f"TB_CONN_LOG", metaData, autoload_with=dbEngine, schema=sysOpt['mysql']['schema'])
                sysOpt['mysql']['table']['tbConnLog'] = Table(f"TB_CONN_LOG", metaData, autoload_with=dbEngine, schema=sysOpt['mysql']['schema'])

            except Exception as e:
                raise Exception(f"DB 연결 실패 : {e}")

            # TCP 서버 엔드포인트 설정
            endpoint = endpoints.TCP4ServerEndpoint(reactor, sysOpt['tcpip']['serverPort'])
            log.info(f"TCP 서버 시작 : {sysOpt['tcpip']['serverPort']}")

            # 엔드포인트 리스닝 시작 (팩토리 사용)
            factory = ReceivingFactory(sysOpt)
            endpoint.listen(factory)

            # 리액터 시작 (프로그램 종료 시까지 실행)
            reactor.run()

        except Exception as e:
            log.error(f"Exception : {e}")
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