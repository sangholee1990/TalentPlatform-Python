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
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import os
from enum import Enum
from pydantic import BaseModel, Field, constr, validator
import pytz
from fastapi.responses import StreamingResponse
from io import BytesIO
import configparser
import numpy as np
from datetime import datetime, timedelta
import numpy as np
import warnings
from fastapi import Form, HTTPException, Depends
import re
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
import time
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
    'modelInfo': '/HDD/DATA/INPUT/BDWIDE2026/models/best_float32.tflite',
    'modelInfo2': '/HDD/DATA/INPUT/BDWIDE2026/models2/best.pt',
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

# ============================================
# 비즈니스 로직
# ============================================
# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')

# 모델 분류 클래스 정보
CLASS_NAMES = ['cavity', 'normal']  # index 0 = cavity, index 1 = normal
CONFIDENCE_THRESHOLD = 0.25

try:
    # model = YOLO(sysOpt['modelInfo'], task='segment')
    model = YOLO(sysOpt['modelInfo2'])
except Exception as e:
    log.error(f"Exception during model load : {e}")
    sys.exit(1)


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
        file: UploadFile = File(..., description='치아 이미지 첨부파일'),
        confidence: float = Form(CONFIDENCE_THRESHOLD, description='신뢰도 임계값 (예: 0.25)')
):
    """
    기능\n
        YOLOv8 기반 치아 객체 탐지 및 세그멘테이션 API\n
    파라미터\n
        file: 이미지 첨부파일 (jpg, png 등)\n
    """
    try:
        start_time = time.time()
        if not file:
            return resResponse("fail", 400, "치아 탐지 실패, 이미지 첨부파일 없음")

        if model is None:
            return resResponse("fail", 500, "치아 탐지 실패, 학습 모델 없음")

        fileContent = await file.read()
        nparr = np.frombuffer(fileContent, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return resResponse("fail", 400, "치아 탐지 실패, 이미지 첨부파일 이상")

        # results = model(img)[0]
        #
        # polygons = []
        # if results.masks is not None:
        #     for poly in results.masks.xy:
        #         polygon = [[float(x), float(y)] for x, y in poly]
        #         polygons.append(polygon)
        # return resResponse("succ", 200, "처리 완료", len(polygons), {"polygons": polygons})

        results = model.predict(source=img, conf=confidence, verbose=False)[0]

        inference_time = (time.time() - start_time) * 1000

        annotated_img = img.copy()
        polygons = []
        detections = []

        # 1. Polygon 데이터 추출 (기존 로직 유지)
        if results.masks is not None:
            for poly in results.masks.xy:
                polygon = [[float(x), float(y)] for x, y in poly]
                polygons.append(polygon)

        # 2. Bounding Box 시각화 및 Detection 정보 추출 (신규 로직 추가)
        if results.boxes is not None:
            for idx, box in enumerate(results.boxes, start=1):
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

                # Cavity = Red, Normal = Green
                color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)

                # Bounding Box 및 Object Number 그리기
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_img, str(idx), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 확률 배열: [cavity_prob, normal_prob]
                probs = [0.0, 0.0]
                if cls_id == 0:  # cavity
                    probs[0] = conf
                elif cls_id == 1:  # normal
                    probs[1] = conf

                detections.append({
                    'object_id': idx,
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': round(conf, 4),
                    'bbox': [x1, y1, x2, y2],
                    'probabilities': probs
                })

        # 3. 처리된 이미지를 Base64 형태로 변환
        # _, buffer = cv2.imencode('.jpg', annotated_img)
        # img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 반환할 데이터 조합
        responseData = {
            "inference_time_ms": round(inference_time, 2),
            "confidence_threshold": confidence,
            "num_detections": len(detections),
            "polygons": polygons,
            "detections": detections,
            # "annotated_image": img_base64
        }

        return resResponse("succ", 200, "처리 완료", len(detections), responseData)

    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"치아 탐지 실패, {e}")