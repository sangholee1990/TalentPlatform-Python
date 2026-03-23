# ============================================
# 요구사항
# ============================================
# 모자이크앱  데이터 적재
# 명세1 http://49.247.41.71:9910/docs
# 인증키 20260221-bdwide

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026
# conda activate py39

# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-mosaic:app --reload --host=0.0.0.0 --port=9910 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-mosaic:app --reload --host=0.0.0.0 --port=9910

# 프로그램 종료
# pkill -f TalentPlatform-BDWIDE2026-DaemonApi-mosaic
# ps -ef | grep "TalentPlatform-BDWIDE2026-DaemonApi-mosaic" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9910
# lsof -i :9910 | awk '{print $2}' | xargs kill -9

# ============================================
# 라이브러리
# ============================================
import glob
import os
import platform
import warnings
from datetime import timedelta
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Any, Dict
import configparser
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
import os
import shutil
# from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi import FastAPI, UploadFile, File, Form
import argparse
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
import asyncio
from fastapi import FastAPI
import socket
import json
import requests
# from google.cloud import bigquery
# from google.oauth2 import service_account
# import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import urllib
import os
import shutil
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import os
from enum import Enum
from pydantic import BaseModel, Field, constr, validator
from konlpy.tag import Okt
from collections import Counter
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from urllib import parse
import time
from urllib.parse import quote_plus, urlencode
import pytz
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
from pytrends.request import TrendReq
from fastapi.responses import StreamingResponse
from io import BytesIO
from google import genai
import configparser
import httpx
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
import threading
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymysql
import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
import warnings
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional
from fastapi import File, UploadFile, Form
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import Form, HTTPException, Depends
from sqlalchemy import text
import re
from email.utils import formataddr
warnings.filterwarnings('ignore')

# ============================================
# 유틸리티 함수
# ============================================
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

# 인증키 검사
def chkKey(key: str = Depends(APIKeyHeader(name="key"))):
    if key != '20260221-bdwide':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

def initCfgInfo(config, key):

    result = None

    try:
        log.info(f'[CHECK] key : {key}')

        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        engine = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessionMake = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        # session = sessionMake()

        base = automap_base()
        base.prepare(autoload_with=engine)
        tableList = base.classes.keys()

        result = {
            'engine': engine
            , 'sessionMake': sessionMake
            , 'tableList': tableList
            , 'tableCls': base.classes
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'BDWIDE2026'
prjName = 'mosaic'

# ctxPath = os.getcwd()
ctxPath = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026'

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # CORS 설정
    'oriList': ['*'],

    # 설정 정보
    'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'cfgDbKey': 'mysql-iwin-bdwide-DMS04',
    'cfgDb': None,

    # 텔레그램 정보
    'cfgTgKey': 'telegram-bdwideMosaic',
    'cfgTg': None,

    # 메일 정보
    'cfgMailKey': 'mail-bdwideMosaic',
    'cfgMail': None,
}

app = FastAPI(
    title="영상 모자이크 API",
    description="",
    version="1.0.0"
    ,openapi_url='/api'
    , docs_url='/docs'
    , redoc_url='/redoc'
)

# 공유 설정
# app.mount('/UPLOAD', StaticFiles(directory='/DATA/UPLOAD'), name='/DATA/UPLOAD')

app.add_middleware(
    CORSMiddleware
    , allow_origins=sysOpt['oriList']
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

clientAsync = httpx.AsyncClient()

# ============================================
# 비즈니스 로직
# ============================================
# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')

sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
sysOpt['cfgMail'] = {
    'email': config.get(sysOpt['cfgMailKey'], 'email'),
    'appPwd': config.get(sysOpt['cfgMailKey'], 'appPwd'),
}
sysOpt['cfgTg'] = {
    'botToken': config.get(sysOpt['cfgTgKey'], 'botToken'),
    'chatId': config.get(sysOpt['cfgTgKey'], 'chatId'),
}

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sendEmail", dependencies=[Depends(chkKey)])
@app.post(f"/api/sendEmail")
async def sendEmail(
    recvEmail: str = Form(..., description='받는 사람 이메일', examples=['backjoi@naver.com']),
    subject: str = Form(..., description='이메일 제목', examples=['테스트 이메일입니다.']),
    content: str = Form(..., description='이메일 내용', examples=['안녕하세요. 테스트 이메일 내용입니다.']),
    file: Optional[UploadFile] = File(..., description='첨부파일')
):
    """
    기능\n
        이메일 발송 API\n
    파라미터\n
        recvEmail: 이메일 받는 사람\n
        subject: 이메일 제목\n
        content: 이메일 내용\n
        file: 첨부파일\n
    """
    try:
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', recvEmail):
            return resResponse("fail", 400, f"이메일 발송 실패, 수신 이메일 주소를 확인")

        sendEmail = sysOpt['cfgMail']['email']
        sendAppPwd = sysOpt['cfgMail']['appPwd']

        msg = MIMEMultipart()
        # msg['From'] = sendEmail
        msg['From'] = formataddr(("비디와이드 고객지원", sendEmail))

        msg['To'] = recvEmail
        msg['Subject'] = subject
        msg.attach(MIMEText(content, 'plain'))

        # 첨부파일 처리
        if not file:
            return resResponse("fail", 400, f"이메일 발송 실패, 첨부파일 없음")

        fileContent = await file.read()
        
        maintype = "application"
        subtype = "octet-stream"
        if file.content_type and "/" in file.content_type:
            maintype, subtype = file.content_type.split("/", 1)
            
        from email.mime.base import MIMEBase
        from email import encoders
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(fileContent)
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', 'attachment', filename=file.filename)
        msg.attach(attachment)

        # SMTP 서버 연결 및 발송
        if sendEmail.endswith("@gmail.com"):
            server = 'smtp.gmail.com'
        elif sendEmail.endswith("@naver.com"):
            server = 'smtp.naver.com'
        else:
            return resResponse("fail", 400, f"이메일 발송 실패, 지원하지 않는 이메일")

        with smtplib.SMTP(server, 587) as server:
            server.starttls()
            server.login(sendEmail, sendAppPwd)
            result = server.send_message(msg)
            if result:
                return resResponse("fail", 400, f"이메일 발송 실패, {result}")

        return resResponse("succ", 200, f"이메일 발송 완료")
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"이메일 발송 실패, {str(e)}")


# @app.post(f"/api/sendTelegram", dependencies=[Depends(chkKey)])
@app.post(f"/api/sendTelegram")
async def sendTelegram(
    msg: str = Form(..., description='메시지 내용', examples=['메시지']),
):
    """
    기능\n
        텔레그램 메시지 발송 API\n
    파라미터\n
        msg: 전송할 메시지 내용\n
    """
    try:
        botToken =  sysOpt['cfgTg']['botToken']
        chatId = sysOpt['cfgTg']['chatId']
        url = f"https://api.telegram.org/bot{botToken}/sendMessage"
        payload = {
            'chat_id': chatId,
            'text': msg
        }
        
        response = requests.post(url, json=payload)
        resData = response.json()

        if response.status_code == 200 and resData.get('ok'):
            return resResponse("succ", 200, "텔레그램 메시지 발송 완료")
        else:
            return resResponse("fail", 400, f"텔레그램 메시지 발송 실패 : {resData}")
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"텔레그램 메시지 발송 실패 : {str(e)}")


@app.post(f"/api/insConsult")
async def insConsult(
        name: str = Form(..., description='기관/성함', examples=['홍길동(모자이크랩)']),
        contact: str = Form(..., description='연락처', examples=['010-0000-0000']),
        email: str = Form(..., description="이메일", examples=['test@asdasd.com']),
        msg: str = Form(..., description="요청사항", examples=['모자이크 앱 도입 문의드립니다.']),
        status: str = Form(..., description="상태 (대기, 완료 등)", examples=['대기'])
):
    """
    기능\n
        긴급상담 요청\n
    """
    try:
        if not name: return resResponse("fail", 400, f"기관/성함 없음")
        if not contact: return resResponse("fail", 400, f"연락처 없음")
        if not email: return resResponse("fail", 400, f"이메일 없음")
        if not msg: return resResponse("fail", 400, f"요청사항 없음")
        if not status: return resResponse("fail", 400, f"상태 없음")

        params = {
            "name": name,
            "contact": contact,
            "email": email,
            "msg": msg,
            "status": status,
        }
        log.info(f"params : {params}")

        with sysOpt['cfgDb']['sessionMake']() as session:
            with session.begin():
                try:
                    query = text("""
                                 SELECT 1
                                 FROM TB_CONSULT
                                 WHERE CONTACT = :contact AND EMAIL = :email AND MSG = :msg
                                 LIMIT 1
                                 """)
                    isExist = session.execute(query, params).fetchone()
                    if isExist:
                        return resResponse("fail", 400, "이미 동일한 요청사항 (연락처, 이메일, 요청 내용)이 있습니다.", 0, None)

                    query = text("""
                                 INSERT INTO TB_CONSULT (NAME, CONTACT, EMAIL, MSG, STATUS, REG_DATE)
                                 VALUES (:name, :contact, :email, :msg, :status, NOW())
                                 """)
                    result = session.execute(query, params)
                    log.info(f"result : {result.rowcount}")
                    return resResponse("succ", 200, "처리 완료", result.rowcount, None)
                except Exception as e:
                    log.error(f'Exception : {str(e)}')
                    raise e
    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400)


@app.post(f"/api/selConsult")
async def selConsult(
):
    """
    기능\n
        긴급상담 조회\n
    """
    try:
        with sysOpt['cfgDb']['sessionMake']() as session:
            with session.begin():
                query = text("""
                             WITH RankedConsults AS (SELECT ID,
                                                            NAME,
                                                            CONTACT,
                                                            EMAIL,
                                                            MSG,
                                                            STATUS,
                                                            REG_DATE,
                                                            ROW_NUMBER() OVER(PARTITION BY CONTACT, EMAIL, MSG ORDER BY REG_DATE DESC) AS rn
                                                     FROM TB_CONSULT)
                             SELECT C.ID,
                                    C.NAME,
                                    C.CONTACT,
                                    C.EMAIL,
                                    C.MSG,
                                    C.STATUS,
                                    DATE_FORMAT(C.REG_DATE, '%Y-%m-%d %H:%i:%s') AS REG_DATE,
                                    H.ID AS HIST_ID,
                                    H.DOC_TYPE,
                                    H.MEMO,
                                    DATE_FORMAT(H.REG_DATE, '%Y-%m-%d %H:%i:%s') AS HIST_REG_DATE
                             FROM RankedConsults C
                             LEFT JOIN TB_CONSULT_HIST H ON C.ID = H.CONSULT_ID
                             WHERE C.rn = 1
                             ORDER BY C.REG_DATE DESC, H.ID DESC
                             """)

                result = session.execute(query)
                rows = result.mappings().all()
                
                consultDict = {}
                for row in rows:
                    cid = row['ID']
                    if cid not in consultDict:
                        consultDict[cid] = {
                            'ID': row['ID'],
                            'NAME': row['NAME'],
                            'CONTACT': row['CONTACT'],
                            'EMAIL': row['EMAIL'],
                            'MSG': row['MSG'],
                            'STATUS': row['STATUS'],
                            'REG_DATE': row['REG_DATE'],
                            'hst': []
                        }
                    
                    if row['HIST_ID']:
                        consultDict[cid]['hst'].append({
                            'ID': row['HIST_ID'],
                            'CONSULT_ID': row['ID'],
                            'DOC_TYPE': row['DOC_TYPE'],
                            'MEMO': row['MEMO'],
                            'REG_DATE': row['HIST_REG_DATE']
                        })

                consultList = list(consultDict.values())
                return resResponse("succ", 200, "처리 완료", len(consultList), consultList)
    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400)

@app.post(f"/api/insConsultHist")
async def insConsultHist(
        consultId: int = Form(..., description='상담 ID', examples=[1]),
        docType: str = Form(..., description='문서 종류 (개인정보, 보안서약, 파기확인, 현금영수증)', examples=['보안서약서']),
        memo: str = Form(..., description='비고', examples=['발송 완료'])
):
    """
    기능\n
        긴급상담 이력 추가\n
    """
    try:
        if not consultId: return resResponse("fail", 400, f"상담 ID 없음")
        if not docType: return resResponse("fail", 400, f"문서 종류 없음")
        
        params = {"consultId": consultId, "docType": docType, "memo": memo}
        with sysOpt['cfgDb']['sessionMake']() as session:
            with session.begin():
                try:
                    query = text("""
                                 INSERT INTO TB_CONSULT_HIST (CONSULT_ID, DOC_TYPE, MEMO, REG_DATE)
                                 VALUES (:consultId, :docType, :memo, NOW())
                                 """)
                    result = session.execute(query, params)
                    return resResponse("succ", 200, "처리 완료", result.rowcount, None)
                except Exception as e:
                    log.error(f'Exception : {str(e)}')
                    raise e
    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400)

@app.post(f"/api/updConsultStatus")
async def updConsultStatus(
        consultId: int = Form(..., description='상담 ID', examples=[1]),
        status: str = Form(..., description='상태 (대기, 완료 등)', examples=['완료'])
):
    """
    기능\n
        긴급상담 상태 업데이트 (STATUS 필드 변경)\n
    """
    try:
        if not consultId: return resResponse("fail", 400, "상담 ID 없음")
        if not status: return resResponse("fail", 400, "상태 데이터 없음")

        params = {"consultId": consultId, "status": status}
        with sysOpt['cfgDb']['sessionMake']() as session:
            with session.begin():
                try:
                    query = text("""
                                 UPDATE TB_CONSULT
                                 SET STATUS = :status
                                 WHERE ID = :consultId
                                 """)
                    result = session.execute(query, params)
                    return resResponse("succ", 200, "상태 변경 완료", result.rowcount, None)
                except Exception as e:
                    log.error(f'Exception : {str(e)}')
                    raise e
    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400)