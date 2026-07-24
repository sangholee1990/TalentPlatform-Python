# ================================================
# 요구사항
# ================================================
# Python을 이용한 공공기관 메뉴 URL 및 디바이스별 캡처

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-INDI2026-DaemonFramework-captWebPage.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-INDI2026-DaemonFramework-captWebPage.py &
# tail -f nohup.out

import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import pytz
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
import json
from datetime import datetime, timedelta
import os
import sys
import os
import sys
import json
import os
import glob
# from langchain_community.document_loaders import PyPDFLoader, CSVLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
import base64
import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

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
        contextPath = os.getcwd() if env in 'local' else 'C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'chatTrouShoot'
    serviceName = 'INDI2026'

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
                'docPath': 'C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/bdwide/2026/trouShoot/doc',
                'chatModel': 'D:/ollama/gemma-4-E2B-it-Q8_0.gguf',
                'visModel': 'D:/ollama/mmproj-F16.gguf',
                'embModel': 'D:/ollama/multilingual-e5-small',
                'imgFile': '"D:/ollama/20260722_143203.png',
                'vecDbPath': 'D:/ollama/trouShoot_chromadb',
            }

            # ==========================================================================================================
            # 테스트
            # ==========================================================================================================
            MODEL_PATH = sysOpt['chatModel']
            PROJ_PATH = sysOpt['visModel']
            IMAGE_PATH = sysOpt['imgFile']
            CHROMA_DB_DIR = sysOpt['vecDbPath']

            def encode_image_to_base64(image_path: str) -> str:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            # ==========================================
            # 2. 모델 및 데이터베이스 로드
            # ==========================================
            print("로딩 중: 임베딩 모델 및 ChromaDB 불러오기...")
            # 한국어 임베딩 모델 로드 (가벼운 CPU 환경 설정)
            embeddings = HuggingFaceEmbeddings(
                # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_name=sysOpt['embModel'],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # 이전에 생성해 둔 로컬 벡터 DB 연결
            if not os.path.exists(CHROMA_DB_DIR):
                print("경고: ChromaDB 폴더가 없습니다. PDF 임베딩 코드를 먼저 실행해 주세요.")
                exit()

            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )

            print("로딩 중: Gemma 4 LLM 및 Vision 프로젝터 불러오기...")
            chat_handler = Llava15ChatHandler(clip_model_path=PROJ_PATH)
            llm = Llama(
                model_path=MODEL_PATH,
                chat_handler=chat_handler,
                n_ctx=2048,  # 컨텍스트 윈도우 (검색된 문서 길이에 유의)
                n_gpu_layers=0,  # RAM 환경 고려 CPU 전용
                verbose=False
            )

            # base64_image = encode_image_to_base64(IMAGE_PATH)
            print("\n준비 완료! 챗봇을 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
            print("-" * 50)

            # ==========================================
            # 3. 챗봇 대화 루프
            # ==========================================
            while True:
                user_query = input("\n사용자: ")
                if user_query.lower() in ['exit', 'quit', '종료']:
                    print("챗봇을 종료합니다.")
                    break

                if not user_query.strip():
                    continue

                # 1단계: 질문과 유사한 PDF 내용 검색 (Context Retrieval)
                # k=2: 컨텍스트 길이 제한(2048)을 넘지 않도록 가장 연관성 높은 2개의 청크만 가져옵니다.
                docs = vectorstore.similarity_search(user_query, k=10)

                # 검색된 문서 내용 병합
                retrieved_context = "\n\n".join([doc.page_content for doc in docs])

                # 2단계: RAG 기반 프롬프트 구성 (Prompt Engineering)
                # 이미지 + PDF 검색 결과 + 사용자의 실제 질문을 하나로 묶습니다.
                prompt_text = f"""
                다음 제공된 [참고 문서]를 바탕으로 원인, 우선확인사항, 조치사항, 재발 방지 등에 답변해 줘
    
                [참고 문서]
                {retrieved_context}
    
                [질문]
                {user_query}
                """

                messages = [
                    {
                        "role": "system",
                        "content": "당신은 제공된 문서를 바탕으로 분석하는 친절하고 전문적인 AI 어시스턴트입니다."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                        ]
                    }
                ]

                # {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}

                # 3단계: LLM 추론 및 답변 스트리밍 (Inference)
                print("\nAI 챗봇: ", end="", flush=True)

                response_stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=None,
                    temperature=0.3,
                    stream=True
                )

                for chunk in response_stream:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
                print()  # 줄바꿈

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