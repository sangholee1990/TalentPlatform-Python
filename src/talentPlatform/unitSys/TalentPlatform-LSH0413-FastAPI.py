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
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Any
from starlette.responses import FileResponse
from starlette.requests import Request
# from fastapi.responses import FileResponse, MultipartForm
import multipart
from fastapi.responses import HTMLResponse
import zipfile
import urllib.parse

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

# nohup uvicorn TalentPlatform-LSH0413-FastAPI:app --reload --host=0.0.0.0 --port=9000 &

# http://223.130.134.136:9000/docs
# http://223.130.134.136:9001/docs
# gunicorn TalentPlatform-LSH0413-FastAPI:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --daemon --access-logfile ./main.log --bind 0.0.0.0:8000 --reload

# 현업
# ps -ef | grep gunicorn | awk '{print $2}' | xargs kill -9
# netstat -ntlp | grep 9000

# gunicorn TalentPlatform-LSH0413-FastAPI:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --daemon --bind 0.0.0.0:9000 --reload


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


# 맵 시각화
def makeProc(data, saveFile, saveImg):
    log.info(f'[START] makeProc')
    result = None

    try:
        # 명사만 추출
        nounList = data.tolist()

        # 빈도 계산
        countList = Counter(nounList)

        dictData = {}
        for none, cnt in countList.most_common():
            # 빈도수 2 이상
            if (cnt < 2): continue
            # 명사  2 글자 이상
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

        # 자료 저장
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        saveData.to_csv(saveFile, index=False)

        # 시각화
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
        # ax2.set_xlim([-1.0, len(countList)])
        ax2.set_xlim([-1.0, len(dictData)])
        # ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=45, horizontalalignment='right')
        ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=6, rotation=90)
        line = ax2.twinx()
        line.plot(saveData.index, saveData['cum'], color='black', marker='o', linewidth=1)
        line.set_ylabel('누적 비율', color='black')
        line.set_ylim(0, 101)

        # 20% 누적비율에 해당하는 가로줄 추가
        line.axhline(y=20, color='r', linestyle='-')

        # 7번째 막대에 대한 세로줄 추가
        ax2.axvline(x=maxCnt, color='r', linestyle='-')

        # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.tight_layout()
        # plt.subplots_adjust(hspace=1)
        # plt.subplots_adjust(hspace=0, left=0, right=1)
        # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
        # plt.show()
        plt.close()
        # log.info(f'[CHECK] saveImg : {saveImg}')

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isFileExist': os.path.exists(saveFile)
            , 'saveImg': saveImg
            , 'isImgExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        print("Exception : {}".format(e))
        return result
    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info(f'[END] makeProc')


async def makeZip(fileList):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for fileInfo in fileList:
            with open(fileInfo, "rb") as file:
                file_contents = file.read()
                zipf.writestr(os.path.basename(fileInfo), file_contents)

    buffer.seek(0)
    yield buffer.read()

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
VIDEO_PATH = "/DATA/VIDEO"
CSV_PATH = "/DATA/CSV"
UPLOAD_PATH = "/DATA/UPLOAD"

app = FastAPI()
app.mount('/VIDEO', StaticFiles(directory=VIDEO_PATH), name=VIDEO_PATH)
app.mount('/CSV', StaticFiles(directory=CSV_PATH), name=CSV_PATH)
app.mount('/UPLOAD', StaticFiles(directory=UPLOAD_PATH), name=UPLOAD_PATH)

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
        updFileInfo = f"{VIDEO_PATH}/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{file.filename}"
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
    파일 저장소 : /DATA/VIDEO/%Y%m/%d/%H/파일명.zip \n
    """
    try:
        fileInfo = os.path.join(VIDEO_PATH, file)

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

        saveFile = '{}/{}/{}.csv'.format(UPLOAD_PATH, dtDateTime.strftime('%Y%m/%d/%H%M'), fileNameNoExt)
        saveImg = '{}/{}/{}.png'.format(UPLOAD_PATH, dtDateTime.strftime('%Y%m/%d/%H%M'), fileNameNoExt)
        result = makeProc(data=data, saveFile=saveFile, saveImg=saveImg)
        log.info(f'[CHECK] result : {result}')

        if not (result['isImgExist'] and result['isFileExist']):
            raise Exception("이미지 또는 파일 저장을 실패")

        resData = {
            'downFile' : f"http://{getPubliIp()}:9000/UPLOAD/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{fileNameNoExt}.csv"
            , 'downImg' : f"http://{getPubliIp()}:9000/UPLOAD/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{fileNameNoExt}.png"
        }

        return resRespone("succ", 200, "처리 완료", 0, f"{resData}")

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", len(str(e)), str(e)))

@app.post("/file/down")
async def file_down(
        file: UploadFile = File(...)
        , column: str = Form(...)
        , encoding: Encoding = Form(Encoding.UTF_8)
):
    """
    기능 : 단어 구름 및 업종 빈도 다운로드 \n
    파라미터 : API키 없음, file 파일 업로드 (csv, xlsx), column 컬럼, encoding CSV 인코딩 (UTF-8, EUC-KR, CP949) \n
    """
    try:
        if re.search(r'\.(?!(csv|xlsx)$)[^.]*$', file.filename, re.IGNORECASE) is not None:
            raise Exception("csv 또는 xlsx 파일을 확인해주세요.")

        try:
            contents = await file.read()
            basename, extension = os.path.splitext(file.filename)

            if extension.lower() == '.csv': data = pd.read_csv(io.StringIO(contents.decode(encoding.value)))[column]
            if extension.lower() == '.xlsx': data =  pd.read_excel(io.BytesIO(contents))[column]
        except Exception as e:
            log.error(f'Exception : {e}')
            raise Exception("파일 읽기를 실패했습니다 (EUC-KR 인코딩 필요 또는 컬럼명 불일치).")

        dtDateTime = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        fileNameNoExt = os.path.basename(file.filename).split('.')[0]

        saveFile = '{}/{}/{}.csv'.format(UPLOAD_PATH, dtDateTime.strftime('%Y%m/%d/%H%M'), fileNameNoExt)
        saveImg = '{}/{}/{}.png'.format(UPLOAD_PATH, dtDateTime.strftime('%Y%m/%d/%H%M'), fileNameNoExt)
        result = makeProc(data=data, saveFile=saveFile, saveImg=saveImg)
        log.info(f'[CHECK] result : {result}')

        if not (result['isImgExist'] and result['isFileExist']):
            raise Exception("이미지 또는 파일 저장을 실패")

        # 파일 경로 및 파일명 리스트
        fileList = [saveFile, saveImg]

        zipFileNmae = urllib.parse.quote(fileNameNoExt, safe='')

        headers = {
            "Content-Disposition": f"attachment; filename={zipFileNmae}.zip"
        }

        return StreamingResponse(makeZip(fileList), media_type="application/zip", headers=headers)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", len(str(e)), str(e)))

# resData = {
#     'downFile' : f"http://{getPubliIp()}:9001/UPLOAD/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{fileNameNoExt}.csv"
#     , 'downImg' : f"http://{getPubliIp()}:9001/UPLOAD/{dtDateTime.strftime('%Y%m/%d/%H%M')}/{fileNameNoExt}.png"
# }

# csv_data = open(saveFile, "r")
# csv_response = FileResponse(saveFile, media_type="text/csv")
#
# image_data = open(saveImg, "rb")
# image_response = FileResponse(saveImg, media_type="image/png")
#
# async def zip_generator():
#     buffer = io.BytesIO()
#     with zipfile.ZipFile(buffer, "w") as zipf:
#         zipf.writestr(f"{fileNameNoExt}.csv", csv_response)
#         zipf.writestr(f"{fileNameNoExt}.png", image_response)
#
#     buffer.seek(0)
#     yield buffer.read()
#     buffer.close()
#
# headers = {
#     "Content-Disposition": f"attachment; filename={fileNameNoExt}.zip"
# }
#
# return StreamingResponse(zip_generator(), headers=headers, media_type="application/zip")

# html_content = f"""
# <html>
#     <head>
#         <title>Streaming HTML</title>
#     </head>
#     <body>
#         <h1>Streaming Example</h1>
#         <img src="{resData['downImg']}" alt="{fileNameNoExt}.png">
#         <a href="{resData['downFile']}">{fileNameNoExt}.csv</a>
#     </body>
# </html>
# """
#
# async def stream_generator():
#     yield html_content.encode()
#
# return StreamingResponse(stream_generator(), media_type="text/html")


# csv_data = open(saveFile, "r")
# csv_response = FileResponse(saveFile, media_type="text/csv")
#
# image_data = open(saveImg, "rb")
# image_response = FileResponse(saveImg, media_type="image/png")
#
# multipart_response = multipart(
#     parts=[
#         ("image", image_response),
#         ("csv", csv_response),
#     ]
# )

# multipart_response = Request.multipart()
# multipart_response.append(image_response)
# multipart_response.append(csv_response)

# return multipart_response


# async def image_stream():
#     # with open(resData['downImg'], mode='rb') as file:
#     with open(saveImg, mode='rb') as file:
#         yield await file.read()
#
# async def csv_stream():
#     # with open(resData['downFile'], mode='rb') as file:
#     with open(saveImg, mode='rb') as file:
#         yield await file.read()
#
# html_content = f"""
# <html>
#     <head>
#         <title>Some HTML in here</title>
#     </head>
#     <body>
#         <h1>Look ma! HTML!</h1>
#         <img src="/image" alt="{fileNameNoExt}.png">
#         <a href="/csv">{fileNameNoExt}.csv</a>
#     </body>
# </html>
# """
#
# return StreamingResponse(content=html_content, media_type='text/html')

#
# html_content = f"""
# <html>
#     <head>
#         <title>Some HTML in here</title>
#     </head>
#     <body>
#         <h1>Look ma! HTML!</h1>
#         <img src="{resData['downImg']}" alt="{fileNameNoExt}.png">
#         <a href="{resData['downFile']}">{fileNameNoExt}.csv</a>
#     </body>
# </html>
# """
#
# return HTMLResponse(content=html_content, status_code=200, media_type='text/html')

# html_content = f"""
# <html>
#     <head>
#         <title>Some HTML in here</title>
#     </head>
#     <body>
#         <h1>Look ma! HTML!</h1>
#         <img src={resData}img><
#     </body>
# </html>
# """
# return HTMLResponse(content=html_content, status_code=200)

# # Stream CSV data
# csv_data = open(saveFile, "r")
# csv_response = StreamingResponse(iter(csv_data), media_type="text/csv")
#
# # Stream image data
# image_data = open(saveImg, "rb")  # Replace "image.png" with your own image file path
# image_response = StreamingResponse(image_data, media_type="image/png")
#
# # Create multipart response
# boundary = "boundary"
# multipart_data = (
#     f"--{boundary}\n"
#     f"Content-Disposition: form-data; name=csv\n"
#     f"Content-Type: text/csv\n\n"
# )
# multipart_data += csv_data.decode('UTF-8') + "\n"
# # multipart_data += csv_data + "\n"
#
# multipart_data += (
#     f"--{boundary}\n"
#     f"Content-Disposition: form-data; name=image; filename=image.png\n"
#     f"Content-Type: image/png\n\n"
# )
#
# multipart_response = StreamingResponse(
#     iter([multipart_data.encode(), image_data.read(), f"\n--{boundary}--\n".encode()]),
#     media_type="multipart/form-data; boundary=boundary"
# )
#
# return multipart_response

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
