# ================================================
# 요구사항
# ================================================
# Python을 이용한 청소년 인터넷 게임 중독 관련 소셜데이터 수집과 분석을 위한 한국형 온톨로지 개발 및 평가

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
from datetime import datetime
from twisted.internet import reactor, protocol, endpoints
from twisted.python import log
from twisted.internet.defer import Deferred
import struct
from twisted.internet import reactor, protocol, endpoints
from twisted.python import log
from twisted.internet.defer import Deferred # 비동기 작업 완료 추적
from twisted.internet.error import ConnectionLost, ConnectionDone

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
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

class SendingProtocol(protocol.Protocol):
    """
    서버에 데이터를 전송하고 응답을 수신하는 프로토콜
    """
    def __init__(self, factory, message_to_send):
        self.factory = factory
        self._message_to_send = message_to_send
        self._buffer = b''

    def connectionMade(self):
        peer = self.transport.getPeer()
        log.info(f"[Client] 서버 연결 성공: {peer.host}:{peer.port}")
        log.info(f"[Client] 데이터 전송 시도: {self._message_to_send!r}")
        self.transport.write(self._message_to_send) # 연결 성공 시 메시지 전송
        log.info("[Client] 데이터 전송 완료.")

    def dataReceived(self, data):
        """서버로부터 응답 데이터를 수신합니다."""
        self._buffer += data
        log.info(f"[Client] 서버 응답 수신 (raw): {data!r}")

        # 간단히 버퍼 전체를 응답으로 간주 (실제로는 서버 응답 형식에 맞게 파싱 필요)
        response_data = self._buffer
        self._buffer = b'' # 버퍼 비우기

        log.info(f"[Client] 최종 서버 응답: {response_data!r}")

        # Factory에 응답 전달
        if hasattr(self.factory, 'handleResponse'):
             self.factory.handleResponse(response_data)

        # 응답 수신 후 연결 종료 요청 (정상 종료 유도)
        self.transport.loseConnection()


    def connectionLost(self, reason):
        log.info(f"[Client] 서버 연결 끊김: {reason.getErrorMessage()}")
        # Factory에 연결 종료 알림 (이유 전달)
        # Deferred가 이미 처리되지 않았을 경우에만 errback 호출
        if hasattr(self.factory, 'notifyConnectionLost'):
            self.factory.notifyConnectionLost(reason)


# --- 클라이언트 측 팩토리 ---
class SendingFactory(protocol.ClientFactory):
    """
    SendingProtocol 인스턴스를 생성하고 연결 상태를 관리합니다.
    """
    # protocol = SendingProtocol # buildProtocol 에서 직접 생성

    def __init__(self, message_to_send, deferred):
        self._message_to_send = message_to_send
        self.deferred = deferred # 연결 및 응답 처리를 위한 Deferred 객체

    def buildProtocol(self, addr):
        log.info(f"[Client] 프로토콜 인스턴스 생성: {addr}")
        # 프로토콜 생성 시 보낼 메시지 전달
        p = SendingProtocol(self, self._message_to_send)
        p.factory = self # 프로토콜이 팩토리를 참조할 수 있도록 설정
        return p

    def clientConnectionFailed(self, connector, reason):
        log.error(f"[Client] 서버 연결 실패: {reason.getErrorMessage()}")
        if self.deferred and not self.deferred.called:
            self.deferred.errback(reason) # 실패 콜백 호출
        # *** 여기서 reactor.stop() 제거 ***

    def notifyConnectionLost(self, reason):
        """프로토콜에서 연결 종료를 알릴 때 호출됩니다."""
        log.info(f"[Client] Factory가 연결 종료 알림 받음: {reason.getErrorMessage()}")
        # 응답을 받아서 deferred가 이미 완료된 경우는 제외하고,
        # 응답 없이 연결이 끊긴 경우에만 errback 호출
        if self.deferred and not self.deferred.called:
            # reason이 ConnectionDone (정상종료) 인지 확인하여 에러 처리 구분 가능
            if reason.check(ConnectionDone):
                log.info("[Client] 응답 없이 연결이 정상 종료되었습니다.")
                # 정상 종료지만 기대한 응답을 못 받았으므로 에러 처리 혹은 별도 로직 가능
                # 여기서는 간단히 에러로 처리
                self.deferred.errback(reason)
            elif reason.check(ConnectionLost):
                 log.info("[Client] 연결이 비정상적으로 손실되었습니다.")
                 self.deferred.errback(reason) # 실패 콜백 호출
            else:
                 log.info("[Client] 알 수 없는 이유로 연결이 종료되었습니다.")
                 self.deferred.errback(reason)
        # *** 여기서 reactor.stop() 제거 ***


    def handleResponse(self, response_data):
         """프로토콜로부터 받은 응답을 처리합니다."""
         if self.deferred and not self.deferred.called:
              self.deferred.callback(response_data) # 성공 콜백 호출


def build_ctrl_create_input_message(data):
    """
    #12 CTRL CREATE INPUT DATA 메시지를 생성합니다. (API 가이드 기반)
    (이 함수는 이전과 동일)
    """
    msg_id = 0x0030 # 48

    # 페이로드 구성 (Big Endian)
    payload = b''
    try:
        # Year (2 bytes, unsigned short)
        payload += struct.pack('>H', data['year'])
        # Product Serial Number (21 bytes, ASCII) - 길이 맞춰서 패딩
        payload += data['serial'].encode('ascii').ljust(49, b'\x00')
        # Datetime (19 bytes, ASCII: YYYY-MM-DD HH:MM:SS) - 길이 맞춰서 패딩
        payload += data['datetime'].encode('ascii').ljust(19, b'\x00')
        # Float 값들 (각 4 bytes)
        payload += struct.pack('>f', data['temp'])
        payload += struct.pack('>f', data['hmdty'])
        payload += struct.pack('>f', data['pm25'])
        payload += struct.pack('>f', data['pm10'])
        # MVMNT (20 bytes, ASCII) - 예제값 사용 및 패딩
        payload += data['mvmnt'].encode('ascii').ljust(20, b'\x00')
        # 나머지 Float 값들
        payload += struct.pack('>f', data['tvoc'])
        payload += struct.pack('>f', data['hcho'])
        payload += struct.pack('>f', data['co2'])
        payload += struct.pack('>f', data['co'])
        payload += struct.pack('>f', data['benzo'])
        payload += struct.pack('>f', data['radon'])

        # 페이로드 길이 계산 (실제 페이로드 길이, 예: 102 바이트)
        payload_length = len(payload)

        # 헤더 생성 (SOF, MsgID_H, MsgID_L, PayloadLength)
        sof = 0xFF
        msg_id_h = (msg_id >> 8) & 0xFF
        msg_id_l = msg_id & 0xFF

        # Payload 길이는 1바이트로 가정 (API 문서 헤더 그림 기준)
        if payload_length > 255:
             log.warn(f"페이로드 길이가 255를 초과({payload_length})하지만 헤더는 1바이트 길이만 지원합니다.")
             payload_length_byte = 255 # 최대값으로 제한
        else:
             payload_length_byte = payload_length # 실제 길이 사용 (0~255)

        # Big Endian: SOF(B), MsgID_H(B), MsgID_L(B), Length(B)
        # 예: payload_length가 102(0x66)이면, header = b'\xff\x00\x30\x66'
        header = struct.pack('>BBBB', sof, msg_id_h, msg_id_l, payload_length_byte)

        full_message = header + payload
        # 생성된 메시지 길이 로그 수정 (payload_length_byte 대신 실제 payload_length 사용)
        log.info(f"생성된 메시지 길이: {len(full_message)} (Header: {len(header)}, Payload: {payload_length})")

        log.info(f"[CHECK] payload : {payload}")

        # 생성된 메시지 반환
        return full_message

    except struct.error as e:
        log.error(f"페이로드 패킹 오류: {e}")
        return None
    except KeyError as e:
        log.error(f"데이터 딕셔너리에 필요한 키 없음: {e}")
        return None
    except Exception as e:
        log.error(f"메시지 생성 중 오류: {e}")
        return None

# *** Deferred 객체에 콜백/에러백 추가 및 reactor.stop() 호출 ***
def handleSucc(response):
    log.info(f"[Client] 최종 처리 성공: 응답={response!r}")

    ascii_ignored = response.decode('ascii', errors='ignore')
    print(f"오류 무시: '{ascii_ignored}'")

    if reactor.running:  # Reactor가 실행 중일 때만 stop 호출
        reactor.stop()

def handleFail(failure):
    # 실패 이유 로깅 (ConnectionDone 등 정상 종료도 포함될 수 있음)
    log.error(f"[Client] 최종 처리 실패: 이유={failure.getErrorMessage()}")
    # 에러 종류에 따라 다른 처리 가능
    # failure.printTraceback() # 상세 트레이스백 출력
    if reactor.running:  # Reactor가 실행 중일 때만 stop 호출
        reactor.stop()

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
                # 시작/종료 시간
                'serverHost': 'localhost',
                'serverPort': 9998,
            }

            serverHost = sysOpt['serverHost']
            serverPort = sysOpt['serverPort']

            # 보낼 데이터 준비 (#12 CTRL CREATE INPUT DATA 형식) - 예제 값 사용
            msg = {
                'year': 2025,
                'serial': 'BDWIDE-0033f-05a3776796-89ff44-b7b3ec0-d30403e426',
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'temp': 24.2,
                'hmdty': 88.2,
                'pm25': 10.6,
                'pm10': 2.4,
                'mvmnt': 'movement',
                'tvoc': 4.5,
                'hcho': 46.2,
                'co2': 2.9,
                'co': 72.1,
                'benzo': 22.3,
                'radon': 56.1
            }

            # 보낼 메시지 생성
            msgInfo = build_ctrl_create_input_message(msg)
            if not msgInfo:
                raise Exception("[Client] 메시지 생성 실패")

            log.info("[Client] 서버에 연결 시도 중...")

            log.info(f"[CHECK] msg : {msg}")
            log.info(f"[CHECK] msgInfo : {msgInfo}")
            # 비동기 작업 완료를 추적할 Deferred 객체 생성
            conDef = Deferred()
            conDef.addCallbacks(handleSucc, handleFail)

            # 클라이언트 팩토리 생성 및 메시지 전달
            factory = SendingFactory(msgInfo, conDef)

            # TCP 연결 시작
            reactor.connectTCP(serverHost, serverPort, factory)

            # 리액터 시작
            reactor.run()

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