# ============================================
# 요구사항
# ============================================
# 모자이크앱  데이터 적재
# 명세1 http://49.247.41.71:9920/docs
# 인증키 20260221-bdwide

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026
# conda activate py39

# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-teethSeg:app --reload --host=0.0.0.0 --port=9920 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-teethSeg:app --reload --host=0.0.0.0 --port=9920

# 프로그램 종료
# pkill -f TalentPlatform-BDWIDE2026-DaemonApi-teethSeg
# ps -ef | grep "TalentPlatform-BDWIDE2026-DaemonApi-teethSeg" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9920
# lsof -i :9920 | awk '{print $2}' | xargs kill -9

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
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
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

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'BDWIDE2026'
prjName = 'teethSeg'

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

    # 모델 정보
    'modelInfo': '/HDD/SYSTEMS/models/best_float32.tflite',
}

app = FastAPI(
    title="구강 검진 API",
    description="",
    version="1.0",
    openapi_url='/api',
    docs_url='/docs',
    redoc_url='/redoc',
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


try:
    modelInfo = YOLO(sysOpt['modelInfo'], task='segment')
    log.info(f"[CHECK] YOLO Model successfully loaded from {modelInfo}")
except Exception as e:
    log.error(f"Exception during model load : {e}")
    model = None

# sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
# sysOpt['cfgMail'] = {
#     'email': config.get(sysOpt['cfgMailKey'], 'email'),
#     'appPwd': config.get(sysOpt['cfgMailKey'], 'appPwd'),
# }
# sysOpt['cfgTg'] = {
#     'botToken': config.get(sysOpt['cfgTgKey'], 'botToken'),
#     'chatId': config.get(sysOpt['cfgTgKey'], 'chatId'),
# }

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/detectTeeth", dependencies=[Depends(chkKey)])
@app.post(f"/api/detectTeeth")
async def detectTeeth(
        file: UploadFile = File(..., description='치아 이미지 첨부파일')
):
    """
    기능\n
        YOLOv8 기반 치아 객체 탐지 및 세그멘테이션 API\n
    파라미터\n
        file: 이미지 첨부파일 (jpg, png 등)\n
    """
    try:
        if not file:
            return resResponse("fail", 400, "치아 탐지 실패, 이미지 첨부파일 없음")

        if model is None:
            return resResponse("fail", 500, "치아 탐지 실패, 학습 모델 없음")

        fileContent = await file.read()
        nparr = np.frombuffer(fileContent, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return resResponse("fail", 400, "치아 탐지 실패, 이미지 첨부파일 이상")

        results = model(img)[0]

        polygons = []
        if results.masks is not None:
            for poly in results.masks.xy:
                polygon = [[float(x), float(y)] for x, y in poly]
                polygons.append(polygon)
        return resResponse("succ", 200, "처리 완료", len(polygons), {"polygons": polygons})
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"치아 탐지 실패, {e}")