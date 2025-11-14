# ============================================
# 요구사항
# ============================================
# 트라이포드랩 API - 데이터 적재
# 명세1 http://49.247.41.71:9900/docs
# 인증키 20251114-bdwide

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/TRIPOD
# conda activate py39

# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-bdwide-DaemonApi-tripod:app --host=0.0.0.0 --port=9900 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-bdwide-DaemonApi-tripod:app --reload --host=0.0.0.0 --port=9900

# 프로그램 종료
# ps -ef | grep "TalentPlatform-bdwide-DaemonApi-tripod" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9900
# lsof -i :9900 | awk '{print $2}' | xargs kill -9

# 명세1 http://49.247.41.71:9900/docs
# 명세2 http://49.247.41.71:9900/redoc
# 인증키 20251114-bdwide

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
import re
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

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    # logger instance에 format 설정
    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    # logger instance에 handler 설정
    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    # logger instance로 log 기록
    log.setLevel(level=logging.INFO)

    return log

# 인증키 검사
def chkKey(key: str = Depends(APIKeyHeader(name="key"))):
    if key != '20251114-bdwide':
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
        # dbHost = 'localhost'
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
            # , 'session': session
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
env = 'local'
serviceName = 'BDWIDE2025'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api'

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
    'cfgDbKey': 'mysql-iwin-dms01user01-DMS03',
    'cfgDb': None,
}

app = FastAPI(
    title="트라이포드랩 API",
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

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post(f"/api/insTripodData", dependencies=[Depends(chkKey)])
# @app.post(f"/api/insTripodData")
async def selBrandModel(
    sn: str = Form(..., description='시리얼 번호', examples=['sn']),
    sid: int = Form(..., description='디바이스', examples=[1]),
    weg: float = Form(..., description="무게", examples=[99.5]),
    bat: int = Form(..., description="배터리 잔량", examples=[100]),
    inv: int = Form(..., description="전송 간격", examples=[1]),
):
    """
    기능\n
        트라이포드랩 API - 데이터 적재\n
    파라미터\n
        sn: 시리얼 번호\n
        sid: 디바이스 ID\n
        weg: 무게\n
        bat: 배터리 잔량\n
        inv: 전송 간격\n
    """
    try:
        if sn is None: return resResponse("fail", 400, f"시리얼 번호 없음, sn : {sn}")
        if sid is None: return resResponse("fail", 400, f"디바이스 없음, sid : {sid}")
        if weg is None: return resResponse("fail", 400, f"무게 없음, weg : {weg}")
        if bat is None: return resResponse("fail", 400, f"배터리 잔량 없음, bat : {bat}")
        if inv is None: return resResponse("fail", 400, f"전송 간격 없음, inv : {inv}")

        params = {
            "sn": sn,
            "sid": sid,
            "weg": weg,
            "bat": bat,
            "inv": inv
        }
        log.info(f"params : {params}")

        with sysOpt['cfgDb']['sessionMake']() as session:
            with session.begin():
                query = text(f"""
                        INSERT INTO TB_TRIPOD_DATA (
                            TM, SN, SID, WEG, BAT, INV, REG_DATE
                        )
                        VALUES ( 
                            NOW(), :sn, :sid, :weg, :bat, :inv, NOW()
                        )
                        ON DUPLICATE KEY UPDATE
                            WEG = VALUES(WEG),
                            BAT = VALUES(BAT),
                            INV = VALUES(INV),
                            MOD_DATE = NOW();
                    """)
                result = session.execute(query, params)
                log.info(f"result : {result}")
                return resResponse("succ", 200, "처리 완료", result.rowcount, result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400)