import os

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Any, Dict
import configparser
import pymysql
from fastapi.staticfiles import StaticFiles
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
import os
import shutil
from pydantic import BaseModel, Field, constr
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
import argparse
import socketio
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from socketio import AsyncClient
import asyncio
from fastapi import FastAPI
import socket
import json
from sqlalchemy import Float, Integer
import subprocess
from threading import Thread
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import psutil
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import io
import os
from enum import Enum

# =================================================
# 도움말
# =================================================
# cd /SYSTEMS/PROG/PYTHON/FAST-API
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/fast-api
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
# conda activate py38

# 테스트
# ps -ef | grep uvicorn | awk '{print $2}' | xargs kill -9
# uvicorn TalentPlatform-LSH0413-FastAPI:app --reload --host=0.0.0.0 --port=9000
# uvicorn TalentPlatform-LSH0413-FastAPI:app --reload --host=0.0.0.0 --port=9001

# http://223.130.134.136:9000/docs
# gunicorn TalentPlatform-LSH0413-FastAPI:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --daemon --access-logfile ./main.log --bind 0.0.0.0:8000 --reload

# 현업
# ps -ef | grep gunicorn | awk '{print $2}' | xargs kill -9
# gunicorn TalentPlatform-LSH0413-FastAPI:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --daemon --bind 0.0.0.0:9000 --reload


# {
#   "type": "esp-32-fota-https",
#   "version": "2",
#   "host": "192.168.x.xxx",
#   "port": 80,
#   "bin": "/test/http_test.bin"
# }

# /SYSTEMS/PROG/NODE/NetIO.js
# node NetIO.js

# =================================================
# 유틸리티 함수
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
        , datetime.datetime.now().strftime("%Y%m%d")
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

    globalVar = {
        "prjName": prjName
        , "sysOs": platform.system()
        , "contextPath": contextPath
        , "resPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , "cfgPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , "inpPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , "sysPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace("\\", "/")

    return globalVar


def getPubliIp():
    response = requests.get('https://api.ipify.org')
    return response.text


def getDb():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 인증키를 확인하는 미들웨어 함수
# async def checkApiKey(api_key: str = Header(...)):
#     if api_key != "my_secret_key":
#         raise HTTPException(status_code=401, detail="Invalid API Key")
async def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    # if api_key != "123":
    if api_key != "api-20230604":
        raise HTTPException(status_code=401, detail="Invalid API Key")

def resRespone(status: str, code: int, message: str, cnt: int = 0, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }


def firmwareToDict(firmware):
    return {
        "ID": firmware.ID,
        "TYPE": firmware.TYPE,
        "VER": firmware.VER,
        "HOST": firmware.HOST,
        "PORT": firmware.PORT,
        "BIN": firmware.BIN,
        "REG_DATE": firmware.REG_DATE.strftime("%Y-%m-%d %H:%M:%S"),  # Assuming REG_DATE is a datetime
        "DOWN_LINK": f"{getPubliIp()}:9998/firm/down/?file={firmware.BIN}"
    }


async def run_script(cmd):
    loop = asyncio.get_event_loop()
    process = await loop.run_in_executor(None, lambda: subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'))
    # process = await loop.run_in_executor(None, lambda: subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'))
    stdout, stderr = await loop.run_in_executor(None, process.communicate)

    if process.returncode != 0:
        print(f'[ERROR] cmd : {cmd}')
        print(f'[ERROR] stderr : {stderr}')

    # print(f'[CHECK] stdout : {stdout}')


def findProceByCmdline(regex):
    pattern = re.compile(regex)
    matches = []
    for proc in psutil.process_iter(['cmdline']):
        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
        if pattern.search(cmdline):
            matches.append(proc)
    return matches


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
serviceName = 'LSH0413'

# 환경 변수 설정 (로그 설정)
log = initLog(env, contextPath, prjName)

# 환경 변수 설정 (초기 변수)
globalVar = initGlobalVar(env, contextPath, prjName)

# DB 설정
config = configparser.ConfigParser()
config.read(f"{globalVar['cfgPath']}/system.cfg", encoding='utf-8')
configKey = 'mysql-clova-dms02user01'
dbUser = config.get(configKey, 'user')
dbPwd = quote_plus(config.get(configKey, 'pwd'))
dbHost = config.get(configKey, 'host')
dbHost = 'localhost' if dbHost == getPubliIp() else dbHost
dbPort = config.get(configKey, 'port')
dbName = config.get(configKey, 'dbName')

# DB 세션
SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# SessionLocal().execute('SELECT * FROM TB_FILE_INFO_DTL').fetchall()

# 전역 설정
plt.rcParams['font.family'] = 'NanumGothic'

# 옵션 설정
sysOpt = {
    # 상위 비율
    'topPerInfo': 20
}

# 공유 폴더
UPLOAD_PATH = "/DATA/UPLOAD"
CSV_PATH = "/DATA/CSV"

app = FastAPI()
app.mount('/VIDEO', StaticFiles(directory=UPLOAD_PATH), name='/DATA/VIDEO')
app.mount('/CSV', StaticFiles(directory=CSV_PATH), name='/DATA/CSV')

origins = [
    "http://localhost:8080"
    , "http://localhost:9000"
    , "http://localhost:9100"
    , "http://riakorea.co.kr"
    , "http://riakorea.co.kr"
]

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set the appropriate origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 미들웨어 함수를 적용하여 헤더 인증키 확인
# app.middleware("http")(check_api_key)

Base = declarative_base()


# 테이블 정의
class FileInfo(Base):
    # __tablename__ = "TB_FILE_INFO"
    __tablename__ = "TB_VIDEO_INFO"

    ID = Column(Integer, primary_key=True, index=True, comment="고유번호")
    VIDEO_PATH = Column(String(500), index=True, comment="비디오 파일경로")
    VIDEO_NAME = Column(String(500), index=True, comment="비디오 파일명")
    REG_DATE = Column(DateTime, default=datetime.datetime.now(pytz.timezone('Asia/Seoul')), comment="등록일")


Base.metadata.create_all(bind=engine)


class FirmwareBase(BaseModel):
    """
    펌웨어 기본 정보

    펌웨어의 기본 정보를 나타내는 데이터 모델입니다.

    :param TYPE: 펌웨어 유형
    :param VER: 펌웨어 버전
    :param HOST: 호스트
    :param PORT: 포트
    """
    TYPE: str = Field(..., description="펌웨어 유형")
    VER: str = Field(..., description="펌웨어 버전")
    HOST: str = Field(..., description="호스트")
    PORT: str = Field(..., description="포트")


class DownloadResponse(BaseModel):
    filename: str

class Encoding(str, Enum):
    UTF_8 = "UTF-8"
    EUC_KR = "EUC-KR"
    CP949 = "CP949"

@app.post("/video/upload", dependencies=[Depends(chkApiKey)])
async def viedo_upload(
        file: UploadFile = File(...),
        db: Session = Depends(getDb)
):
    """
    기능 : 비디오 영상 파일 업로드 (mp4, MP4) \n
    파라미터 : API키 없음, file 비디오 영상 파일 \n
    """
    try:
        if re.search(r'\.(?!(mp4|MP4)$)[^.]*$', file.filename) is not None:
            raise Exception("비디오 영상 파일 (mp4, MP4)을 확인해주세요.")

        proc = findProceByCmdline('TalentPlatform-LSH0413-detect_and_track.py')
        proc2 = findProceByCmdline('TalentPlatform-LSH0413-deep_sort_tracking_id.py')
        maxProcCnt = 0
        # maxProcCnt = 1

        if len(proc) > maxProcCnt or len(proc2) > maxProcCnt:
            raise Exception("현재 프로세스 수행 중이오니 1시간 이후로 다시 실행 부탁드립니다.")

        dtDateTime = datetime.datetime.now(pytz.timezone('Asia/Seoul'))

        # Save the uploaded file
        updFileInfo = f"{UPLOAD_PATH}/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{file.filename}"
        os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)
        with open(updFileInfo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        fileInfo = FileInfo(VIDEO_PATH=dtDateTime.strftime('%Y%m/%d/%H%M'), VIDEO_NAME=file.filename, REG_DATE=dtDateTime)
        db.add(fileInfo)
        db.commit()
        db.refresh(fileInfo)

        cmd = '{}/{} "{}" "{}"'.format(os.getcwd(), 'RunShell-LSH0413-PROC.sh', dtDateTime.strftime('%Y%m/%d/%H%M'), file.filename)
        os.chmod(f'{os.getcwd()}/RunShell-LSH0413-PROC.sh', 0o755)
        asyncio.create_task(run_script(cmd))

        return resRespone("succ", 200, "처리 완료", 0, f"{dtDateTime.strftime('%Y%m/%d/%H%M')}/{file.filename}")

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", len(str(e)), str(e)))


@app.get("/video/down", dependencies=[Depends(chkApiKey)])
async def video_down(file: str):
    """
    기능 : 비디오 영상 파일 다운로드\n
    파라미터 : API키 없음, file 비디오 영상 파일 \n
    파일 저장소 : /DATA/UPLOAD/%Y%m/%d/%H/파일명.zip \n
    """
    try:
        fileInfo = os.path.join(UPLOAD_PATH, file)

        if not os.path.exists(fileInfo):
            raise Exception("다운로드 파일이 없습니다.")

        return FileResponse(fileInfo, media_type="application/octet-stream", filename=file)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post("/file/upload")
async def file_upload(
        file: UploadFile = File(...)
        , column: str = Form(...)
        , encoding: Encoding = Form(Encoding.UTF_8)
):
    """
    기능 : 단어 구름 및 업종 빈도 시각화 \n
    파라미터 : API키 없음, file 파일 업로드 (csv, xlsx), column 컬럼, encoding CSV 인코딩 (UTF-8, EUC-KR, CP949) \n
    """
    try:
        if re.search(r'\.(?!(csv|xlsx)$)[^.]*$', file.filename, re.IGNORECASE) is not None:
            raise Exception("csv 또는 xlsx 파일을 확인해주세요.")

        try:
            contents = await file.read()
            basename, extension = os.path.splitext(file.filename)

            # log.info(f'[CHECK] encoding : {encoding.value}')
            if extension.lower() == '.csv': data = pd.read_csv(io.StringIO(contents.decode(encoding.value)))[column]
            if extension.lower() == '.xlsx': data =  pd.read_excel(io.BytesIO(contents))[column]
        except Exception as e:
            log.error(f'Exception : {e}')
            raise Exception("파일 읽기를 실패했습니다 (EUC-KR 인코딩 필요 또는 컬럼명 불일치).")

        dtDateTime = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        fileNameNoExt = os.path.basename(file.filename).split('.')[0]

        nounList = data.tolist()
        countList = Counter(nounList)

        dictData = {}
        for none, cnt in countList.most_common():
            if (cnt < 2): continue
            if (len(none) < 2): continue

            dictData[none] = cnt

        # 빈도분포
        saveData = pd.DataFrame.from_dict(dictData.items()).rename(
            {
                0: 'none'
                , 1: 'cnt'
            }
            , axis=1
        )
        saveData['cum'] = saveData['cnt'].cumsum() / saveData['cnt'].sum() * 100
        maxCnt = (saveData['cum'] > 20).idxmax()

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # 단어 구름
        wordcloud = WordCloud(
            width=1500
            , height=1500
            , background_color=None
            , mode='RGBA'
            , font_path="NanumGothic.ttf"
        ).generate_from_frequencies(dictData)

        ax1 = axs[0]
        ax1.imshow(wordcloud, interpolation="bilinear")
        ax1.axis("off")

        # 빈도 분포
        ax2 = axs[1]
        bar = sns.barplot(x='none', y='cnt', data=saveData, ax=ax2, linewidth=0)
        ax2.set_title('업종 빈도 분포도')
        ax2.set_xlabel(None)
        ax2.set_ylabel('빈도 개수')
        ax2.set_xlim([-1.0, len(countList)])
        # ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=45, horizontalalignment='right')
        line = ax2.twinx()
        line.plot(saveData.index, saveData['cum'], color='black', marker='o', linewidth=1)
        line.set_ylabel('누적 비율', color='black')
        line.set_ylim(0, 101)

        # 20% 누적비율에 해당하는 가로줄 추가
        line.axhline(y=20, color='r', linestyle='-')

        # 7번째 막대에 대한 세로줄 추가
        ax2.axvline(x=maxCnt, color='r', linestyle='-')

        saveImg = '{}/{}/{}.png'.format(UPLOAD_PATH, dtDateTime.strftime('%Y%m/%d/%H%M'), fileNameNoExt)
        downImg =  f"http://{getPubliIp()}:9000/VIDEO/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{fileNameNoExt}.png"
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.tight_layout()
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()
        log.info(f'[CHECK] downImg : {downImg}')

        return resRespone("succ", 200, "처리 완료", 0, f"{downImg}")

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", len(str(e)), str(e)))


# @app.post("/firm/file_info", dependencies=[Depends(chkApiKey)])
# @app.post("/firm/file_info")
# def file_info(id: int = None, db: Session = Depends(getDb)):
#     """
#     기능 : 최근 파일 정보 가져오기\n
#     파라미터 : API키, id 인덱스 (없을 시 최신 목록) \n
#     """
#     try:
#         if id:
#             selData = db.query(FileInfo).filter(FileInfo.ID == id)
#             file_info = selData.first()
#             cnt = selData.count()
#
#         else:
#             # file_info = db.query(Firmware).order_by(Firmware.ID.desc()).first()
#             selData = db.query(FileInfo).order_by(FileInfo.ID.desc())
#             file_info = selData.first()
#             cnt = selData.count()
#
#         if file_info is None:
#             raise Exception("파일 정보가 없습니다.")
#
#         return resRespone("succ", 200, "처리 완료", cnt, file_info)
#
#         # return file_info
#
#     except Exception as e:
#         log.error(f'Exception : {e}')
#         raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))


# @app.post("/firm/file_list", dependencies=[Depends(chkApiKey)])
# @app.post("/firm/file_list")
# def file_list(page: int = 1, per_page: int = 10, db: Session = Depends(getDb)):
#     """
#     기능 : 모든 파일 목록 가져오기 \n
#     파라미터 : API키, page 페이지 번호, per_page : 페이지당 개수 \n
#     """
#     try:
#         offset = (page - 1) * per_page
#         selData = db.query(FileInfo).offset(offset).limit(per_page)
#         file_list = selData.all()
#         cnt = selData.count()
#
#         if file_list is None:
#             raise Exception("파일 목록이 없습니다.")
#
#         return resRespone("succ", 200, "처리 완료", cnt, file_list)
#
#     except Exception as e:
#         log.error(f'Exception : {e}')
#         raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

# @app.post("/firm/send_data", dependencies=[Depends(chkApiKey)])
# def send_data(id: int = None, db: Session = Depends(getDb)):
#     #     """
#     #     기능 : 최근 파일 정보 가져오기\n
#     #     파라미터 : API키, id 인덱스 (없을 시 최신 목록) \n
#     #     """
#     try:
#         if id:
#             selData = db.query(Firmware).filter(Firmware.ID == id)
#             file_info = selData.first()
#             cnt = selData.count()
#
#         else:
#             # file_info = db.query(Firmware).order_by(Firmware.ID.desc()).first()
#             selData = db.query(Firmware).order_by(Firmware.ID.desc())
#             file_info = selData.first()
#             cnt = selData.count()
#
#         if file_info is None:
#             raise HTTPException(status_code=404, detail=resRespone("fail", 400, "처리 실패"))
#
#         # TCP 소켓 생성
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         server_address = ('localhost', 9997)
#
#         # 서버에 연결
#         sock.connect(server_address)
#
#         # 데이터 전송
#         # message = b'Hello, TCP Server!'
#         # message = json.dumps([firmwareToDict(firmware) for firmware in file_info]).encode()
#         # message = json.dumps([firmwareToDict(file_info)]).encode()
#         message = json.dumps([resRespone("succ", 200, "처리 완료", cnt, firmwareToDict(file_info))]).encode()
#
#         sock.sendall(message)
#
#         # 데이터 수신
#         data = sock.recv(1024)
#         response = data.decode('utf-8')
#
#         return response
#
#     except Exception as e:
#         log.error(f'Exception : {e}')
#         raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))
#
#     finally:
#         # 소켓 닫기
#         sock.close()
