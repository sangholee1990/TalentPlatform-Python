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
# from datetime import datetime
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

# =================================================
# 도움말
# =================================================
# cd /SYSTEMS/PROG/PYTHON/FAST-API
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/fast-api
# conda activate py38
# uvicorn TalentPlatform-bdwide-firmware:app --reload --host=0.0.0.0 --port=9998
# http://223.130.134.136:9998/docs

# eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2FjY291bnQiOjI0fQ.pXluG0rOyeoO8xSvAFYCOrkIaYofUkUR3dIijJOT6xg

# {
#   "type": "esp-32-fota-https",
#   "version": "2",
#   "host": "192.168.x.xxx",
#   "port": 80,
#   "bin": "/test/http_test.bin"
# }

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
    if api_key != "123":
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


# ================================================================================================
# 환경변수 설정
# ================================================================================================
# 로그 설정
log = initLog()

# DB 설정
config = configparser.ConfigParser()
config.read('config/system.cfg', encoding='utf-8')
dbUser = config.get('mysql-clova-dms02', 'user')
dbPwd = quote_plus(config.get('mysql-clova-dms02', 'pwd'))
dbHost = config.get('mysql-clova-dms02', 'host')
dbHost = 'localhost' if dbHost == getPubliIp() else dbHost
dbPort = config.get('mysql-clova-dms02', 'port')
dbName = config.get('mysql-clova-dms02', 'dbName')

# DB 세션
SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 공유 폴더
UPLOAD_PATH = "/DATA/UPLOAD"

app = FastAPI()

# Enable CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Set the appropriate origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 미들웨어 함수를 적용하여 헤더 인증키 확인
# app.middleware("http")(check_api_key)

Base = declarative_base()


# 테이블 정의
class Firmware(Base):
    __tablename__ = "TB_IOT_FIRM"

    ID = Column(Integer, primary_key=True, index=True, comment="고유번호")
    TYPE = Column(String(200), index=True, comment="타입")
    VER = Column(String(200), index=True, comment="버전")
    HOST = Column(String(200), index=True, comment="호스트")
    PORT = Column(String(200), index=True, comment="포트")
    BIN = Column(String(200), index=True, comment="바이너리")
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

@app.post("/firm/upload", dependencies=[Depends(chkApiKey)])
def file_upload(
        file: UploadFile = File(...),
        type: str = Form(...),
        ver: str = Form(...),
        host: str = Form(...),
        port: str = Form(...),
        db: Session = Depends(getDb),
):
    """
    기능 : 펌웨어 업로드\n
    파라미터 : API키, file 파일 (bin 컬럼), type 펌웨어 유형, ver 펌웨어 버전, host 호스트, port 포트  \n
    """
    try:
        dtDateTime = datetime.datetime.now(pytz.timezone('Asia/Seoul'))

        # Save the uploaded file
        binFileInfo = f"{dtDateTime.strftime('%Y%m/%d/%H')}/{file.filename}"
        updFileInfo = f"{UPLOAD_PATH}/{binFileInfo}"
        os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)
        with open(updFileInfo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create the firmware object in the database
        firmware = Firmware(TYPE=type, VER=ver, HOST=host, PORT=port, REG_DATE=dtDateTime, BIN=binFileInfo)
        db.add(firmware)
        db.commit()
        db.refresh(firmware)

        # return db
        # return {"status": "success", "code": 200, "message": "Firmware uploaded successfully", "data": db}
        return resRespone("succ", 200, "처리 완료", len(db), db)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))


# @app.post("/firm/down/", dependencies=[Depends(chkApiKey)])
@app.get("/firm/down/")
async def file_down(file: str):
    """
    기능 : 펌웨어 다운로드\n
    파라미터 : API키, file 파일 (file_info 또는 file_list에서 bin 컬럼 정보) \n
    파일 저장소 : /DATA/UPLOAD/%Y%m/%d/%H/DrivePro_Toolbox_Setup_v4.0 (64bit).exe \n
    """
    try:
        fileInfo = os.path.join(UPLOAD_PATH, file)
        print(fileInfo)

        if not os.path.exists(fileInfo):
            raise HTTPException(status_code=404, detail=resRespone("fail", 400, "처리 실패"))

        return FileResponse(fileInfo, media_type="application/octet-stream", filename=file)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))


@app.post("/firm/file_info", dependencies=[Depends(chkApiKey)])
def file_info(id: int = None, db: Session = Depends(getDb)):
    """
    기능 : 최근 파일 정보 가져오기\n
    파라미터 : API키, id 인덱스 (없을 시 최신 목록) \n
    """
    try:
        if id:
            selData = db.query(Firmware).filter(Firmware.ID == id)
            file_info = selData.first()
            cnt = selData.count()

        else:
            # file_info = db.query(Firmware).order_by(Firmware.ID.desc()).first()
            selData = db.query(Firmware).order_by(Firmware.ID.desc())
            file_info = selData.first()
            cnt = selData.count()

        if file_info is None:
            raise HTTPException(status_code=404, detail=resRespone("fail", 400, "처리 실패"))

        return resRespone("succ", 200, "처리 완료", cnt, file_info)

        # return file_info

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))


@app.post("/firm/file_list", dependencies=[Depends(chkApiKey)])
def file_list(page: int = 1, per_page: int = 10, db: Session = Depends(getDb)):
    """
    기능 : 모든 파일 목록 가져오기 \n
    파라미터 : API키, page 페이지 번호, per_page : 페이지당 개수 \n
    """
    try:
        offset = (page - 1) * per_page
        selData = db.query(Firmware).offset(offset).limit(per_page)
        file_list = selData.all()
        cnt = selData.count()

        if file_list is None:
            raise HTTPException(status_code=404, detail=resRespone("fail", 400, "처리 실패"))

        # resDict = [{
        #     "ID": file.ID,
        #     "TYPE": file.TYPE,
        #     "VER": file.VER,
        #     "HOST": file.HOST,
        #     "POST": file.PORT,
        #     "BIN": file.BIN,
        #     "REG_DATE": file.REG_DATE,
        #     "DOWN_LINK": f"{getPubliIp()}:9998/firm/down/?file={file.BIN}"
        # } for file in file_list]

        return resRespone("succ", 200, "처리 완료", cnt, file_list)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post("/firm/send_data", dependencies=[Depends(chkApiKey)])
def send_data(id: int = None, db: Session = Depends(getDb)):
    #     """
    #     기능 : 최근 파일 정보 가져오기\n
    #     파라미터 : API키, id 인덱스 (없을 시 최신 목록) \n
    #     """
    try:
        if id:
            selData = db.query(Firmware).filter(Firmware.ID == id)
            file_info = selData.first()
            cnt = selData.count()

        else:
            # file_info = db.query(Firmware).order_by(Firmware.ID.desc()).first()
            selData = db.query(Firmware).order_by(Firmware.ID.desc())
            file_info = selData.first()
            cnt = selData.count()

        if file_info is None:
            raise HTTPException(status_code=404, detail=resRespone("fail", 400, "처리 실패"))

        # TCP 소켓 생성
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', 9997)

        # 서버에 연결
        sock.connect(server_address)

        # 데이터 전송
        # message = b'Hello, TCP Server!'
        # message = json.dumps([firmwareToDict(firmware) for firmware in file_info]).encode()
        # message = json.dumps([firmwareToDict(file_info)]).encode()
        message = json.dumps([resRespone("succ", 200, "처리 완료", cnt, firmwareToDict(file_info))]).encode()

        sock.sendall(message)

        # 데이터 수신
        data = sock.recv(1024)
        response = data.decode('utf-8')

        return response

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

    finally:
        # 소켓 닫기
        sock.close()