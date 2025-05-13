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
import struct
from twisted.internet import reactor, protocol, endpoints
from twisted.python import log

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

# 서버측 프로토콜
class ReceivingProtocol(protocol.Protocol):
    def __init__(self):
        self._buffer = b''

    def connectionMade(self):
        peer = self.transport.getPeer()
        log.info(f"[CHECK] 클라이언트 연결 : {peer.host}:{peer.port}")

    def dataReceived(self, data):
        self._buffer += data
        log.info(f"[CHECK] 데이터 수신 : {len(self._buffer)} : {data!r}")

        headerSize = 4
        while len(self._buffer) >= headerSize:
            try:
                sof = self._buffer[0]
                msgIdH = self._buffer[1]
                msgIdL = self._buffer[2]
                payloadSize = self._buffer[3]
                msgId = (msgIdH << 8) | msgIdL

                if sof != 0xFF:
                    log.error(f"[CHECK] 잘못된 SOF 수신: {sof:#02x}. 연결 종료.")
                    self.transport.loseConnection()
                    return

                msgSize = headerSize + payloadSize
                if not len(self._buffer) >= msgSize:
                    log.info(f"[Server] 메시지 일부 수신. Payload 길이({payloadSize}), 현재 버퍼({len(self._buffer)}). 대기 중...")
                    break

                msgData = self._buffer[:msgSize]
                payload = msgData[headerSize:]
                self._buffer = self._buffer[msgSize:]

                log.info(f"[CHECK] SOF: {sof:#02x}")
                log.info(f"[CHECK] Msg ID: {msgId:#04x} ({msgId})")
                log.info(f"[CHECK] Payload Length: {payloadSize}")
                log.info(f"[CHECK] Payload: {payload!r}")

                self.handleMsg(msgId, payload)
            except Exception as e:
                log.error(f"메시지 처리 오류: {e}")
                self._buffer = b'' # 오류 발생 시 버퍼 비우기 (선택적)
                self.transport.loseConnection()
                return

    def handleMsg(self, msgId, payload):

        resPayload = b'404'

        try:
            if msgId == 0x0000:
                resPayload = payload
            elif msgId == 0x0003:
                now = datetime.now()
                resPayload = struct.pack('>HBBBBB', now.year, now.month, now.day, now.hour, now.minute, now.second)
            elif msgId == 0x0030:
                parsed_result = parse_input_data_payload(payload)
                resPayload = b'200' if parsed_result else b'400'
        except Exception as e:
            log.error(f"Exception : {e}")

        log.info(f"[CHECK] msgId : {msgId} / resPayload : {resPayload!r}")

        # 반환 포맷
        if msgId == 0x0000 or msgId == 0x0003:
            resHeader = self.createHeader(msgId, len(resPayload))
            resMsg = resHeader + resPayload
        else:
            resMsg = resPayload

        self.transport.write(resMsg)

    def createHeader(self, msg_id, payloadSize):
        sof = 0xFF
        msg_id_h = (msg_id >> 8) & 0xFF
        msg_id_l = msg_id & 0xFF
        if payloadSize > 255:
             log.warn(f"페이로드 길이가 255를 초과({payloadSize})하지만 헤더는 1바이트 길이만 지원합니다.")
             payloadSize = 255

        header = struct.pack('>BBBB', sof, msg_id_h, msg_id_l, payloadSize)
        return header

    def connectionLost(self, reason):
        log.info(f"[CHECK] 클라이언트 해제  : {reason.getErrorMessage()}")
        self._buffer = b'' # 버퍼 초기화

# 서버 측 팩토리
class ReceivingFactory(protocol.Factory):
    # 프로토콜 클래스 지정
    protocol = ReceivingProtocol

    # 신규 연결 시 프로토콜 인스턴스 생성 메서드
    def buildProtocol(self, addr):
        log.info(f"[CHECK] 프로토콜 인스턴스 생성 : {addr}")
        p = self.protocol()
        p.factory = self
        return p

def parse_input_data_payload(payload_bytes):
    """
    #12 CTRL CREATE INPUT DATA 페이로드 바이트를 파싱하여
    딕셔너리 형태로 반환합니다. (총 102 바이트 기준)
    """
    # expected_length = 102
    if len(payload_bytes) < 1:
        print(f"오류: 페이로드 길이가 충분하지 않습니다. (실제: {len(payload_bytes)})")
        return None

    parsed_data = {}
    offset = 0

    try:
        parsed_data['year'], = struct.unpack('>H', payload_bytes[offset:offset+2])
        offset += 2

        parsed_data['serial'] = payload_bytes[offset:offset+49].decode('ascii').rstrip('\x00')
        offset += 49

        parsed_data['datetime'] = payload_bytes[offset:offset+19].decode('ascii').rstrip('\x00')
        offset += 19

        parsed_data['temp'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['hmdty'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['pm25'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['pm10'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['mvmnt'] = payload_bytes[offset:offset+20].decode('ascii').rstrip('\x00')
        offset += 20

        parsed_data['tvoc'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['hcho'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['co2'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['co'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['benzo'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        parsed_data['radon'], = struct.unpack('>f', payload_bytes[offset:offset+4])
        offset += 4

        return parsed_data

    except struct.error as e:
        log.error(f"구조체 언패킹 오류: {e} (offset: {offset})")
        return None
    except UnicodeDecodeError as e:
        log.error(f"문자열 디코딩 오류: {e} (offset: {offset})")
        return None
    except IndexError:
        # 이 오류는 보통 expected_length 체크에서 걸리지만, 만약을 위해 남겨둠
        log.error(f"인덱스 오류: 페이로드 데이터가 예상보다 짧습니다. (offset: {offset})")
        return None
    except Exception as e:
         # 예상치 못한 다른 오류 처리
         log.error(f"페이로드 파싱 중 예상치 못한 오류: {e} (offset: {offset})")
         return None


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
                'srtDate': '2025-01-01',
                'endDate': '2025-01-05',
                'invDate': '1d',
                'searchMaxPage': 99,

                # 언어 설정
                # , 'language' : 'en'
                'language': 'ko',

                # 국가 설정
                # , 'country' : 'US'
                'country': 'KR',

                # 키워드 설정
                'keywordList': ['청소년 게임 중독'],

                # 저장 경로
                'saveCsvFile': '/DATA/OUTPUT/LSH0612/gnews_%Y%m%d.csv',
                'saveXlsxFile': '/DATA/OUTPUT/LSH0612/gnews_%Y%m%d.xlsx',
            }

            listen_port = 9998

            # TCP 서버 엔드포인트 설정
            endpoint = endpoints.TCP4ServerEndpoint(reactor, listen_port)
            log.info(f"[Server] TCP 서버 시작 중 (포트: {listen_port})...")

            # 엔드포인트 리스닝 시작 (팩토리 사용)
            endpoint.listen(ReceivingFactory())

            # 리액터 시작 (프로그램 종료 시까지 실행)
            reactor.run()

            # =================================================================
            # from gnews import GNews
            # from newspaper import Article
            # =================================================================
            # okt = Okt()
            #
            # unitGoogleNews = GNews(language='ko', country='KR')
            # searchList = unitGoogleNews.get_news('청소년 게임 중독')
            # log.info(f'[CHECK] searchList : {len(searchList)}')
            #
            # flatList = []
            # for data in searchList:
            #     flatData = {
            #         'title': data['title'],
            #         'description': data['description'],
            #         'publishedDate': data['published date'],
            #         'url': data['url'],
            #         'publisherTitle': data['publisher']['title'],
            #         'publisherHref': data['publisher']['href']
            #     }
            #
            #     flatList.append(flatData)
            #
            # data = pd.DataFrame.from_dict(flatList)
            # # description                               [기획] 청소년 게임중독 문제 심각  매일일보
            # # publishedDate                         Thu, 30 May 2024 07:00:00 GMT
            # # url               https://news.google.com/rss/articles/CBMiZEFVX...
            # # publisherTitle                                                 매일일보
            # # publisherHref                                    https://www.m-i.kr
            #
            # # i = 16
            # for i, row in data.iterrows():
            #
            #     per = round(i / len(data) * 100, 1)
            #     log.info(f'[CHECK] i : {i} / {per}%')
            #
            #     try:
            #         # https://www.m-i.kr/news/articleView.html?idxno=1125607
            #         # decInfo = gnewsdecoder(row['url'])
            #         decInfo = gnewsdecoder(data.loc[i, f'url'])
            #         if not (decInfo['status'] == True): continue
            #
            #         articleInfo = Article(decInfo['decoded_url'], language='ko')
            #
            #         #날짜 변환
            #         dtUtcPubDate = tzUtc.localize(datetime.strptime(data.loc[i, f'publishedDate'][:-4], '%a, %d %b %Y %H:%M:%S'))
            #         sKstPubDate = dtUtcPubDate.astimezone(tzKst).strftime('%Y-%m-%d %H:%M:%S')
            #
            #         # 뉴스 다운로드/파싱/자연어 처리
            #         articleInfo.download()
            #         articleInfo.parse()
            #         articleInfo.nlp()
            #
            #         # 명사/동사/형용사 추출
            #         text = articleInfo.text
            #         if text is None or len(text) < 1: continue
            #         posTagList = okt.pos(text, stem=True)
            #
            #         # i = 0
            #         keyData = {}
            #         keyList = ['Noun', 'Verb', 'Adjective']
            #         for keyInfo in keyList:
            #             # log.info(f'[CHECK] keyInfo : {keyInfo}')
            #
            #             keywordList = [word for word, pos in posTagList if pos in keyInfo]
            #
            #             # 불용어 제거
            #             # keywordList = [word for word in keywordList if word not in stopWordList and len(word) > 1]
            #
            #             # 빈도수 계산
            #             keywordCnt = Counter(keywordList).most_common(20)
            #             keywordData = pd.DataFrame(keywordCnt, columns=['keyword', 'cnt']).sort_values(by='cnt', ascending=False)
            #             keywordDataL1 = keywordData[keywordData['keyword'].str.len() >= 2].reset_index(drop=True)
            #             keyCnt = keywordDataL1['cnt'].astype(str) + " " + keywordDataL1['keyword']
            #             keyData.update({keyInfo : keyCnt.values.tolist()})
            #
            #         # log.info(f"[CHECK] keyData['Noun'] : {keyData['Noun']}")
            #         # log.info(f"[CHECK] keyData['Verb'] : {keyData['Verb']}")
            #         # log.info(f"[CHECK] keyData['Adjective'] : {keyData['Adjective']}")
            #
            #         data.loc[i, f'decUrl'] = None if decInfo['decoded_url'] is None or len(decInfo['decoded_url']) < 1 else str(decInfo['decoded_url'])
            #         data.loc[i, f'text'] = text
            #         data.loc[i, f'summary'] = None if articleInfo.summary is None or len(articleInfo.summary) < 1 else str(articleInfo.summary)
            #         data.loc[i, f'keywordNoun'] = None if keyData['Noun'] is None or len(keyData['Noun']) < 1 else str(keyData['Noun'])
            #         data.loc[i, f'keywordVerb'] = None if keyData['Verb'] is None or len(keyData['Verb']) < 1 else str(keyData['Verb'])
            #         data.loc[i, f'keywordAdjective'] = None if keyData['Adjective'] is None or len(keyData['Adjective']) < 1 else str(keyData['Adjective'])
            #         data.loc[i, f'authors'] = None if articleInfo.authors is None or len(articleInfo.authors) < 1 else str(articleInfo.authors)
            #         data.loc[i, f'publishedKstDate'] = None if sKstPubDate is None or len(sKstPubDate) < 1 else str(sKstPubDate)
            #         data.loc[i, f'top_image'] = None if articleInfo.top_image is None or len(articleInfo.top_image) < 1 else str(articleInfo.top_image)
            #         data.loc[i, f'images'] = None if articleInfo.images is None or len(articleInfo.images) < 1 else str(articleInfo.images)
            #     except Exception as e:
            #         log.error(f"Exception : {str(e)}")
            #
            # if len(data) > 0:
            #     saveCsvFile = datetime.now().strftime(sysOpt['saveCsvFile'])
            #     os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
            #     data.to_csv(saveCsvFile, index=False)
            #     log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')
            #
            #     saveXlsxFile = datetime.now().strftime(sysOpt['saveXlsxFile'])
            #     os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
            #     data.to_excel(saveXlsxFile, index=False)
            #     log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

            # =================================================================
            # from GoogleNews import GoogleNews
            # =================================================================
            # from GoogleNews import GoogleNews
            # 시작일/종료일 설정
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            # for dtDateInfo in dtDateList:
            #     log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #
            #     unitGoogleNews = GoogleNews(
            #         lang=sysOpt['language'],
            #         region=sysOpt['country'],
            #         start=dtDateInfo.strftime('%m/%d/%Y'),
            #         end=(dtDateInfo + timedelta(days=1)).strftime('%m/%d/%Y'),
            #         encode='UTF-8'
            #     )
            #
            #     searchGoogleNews(unitGoogleNews, sysOpt, dtDateInfo)

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