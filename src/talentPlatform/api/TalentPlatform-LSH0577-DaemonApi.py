# ============================================
# 요구사항
# ============================================
# [완료] LSH0577. Python을 이용한 빅쿼리 기반 API

# =================================================
# 도움말
# =================================================
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py38
# uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000
# nohup uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000 &

# http://49.247.41.71:9000/docs
# http://49.247.41.71:9000/redoc

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
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Any, Dict
import configparser
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

from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends

# ============================================
# 유틸리티 함수
# ============================================
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

# def getPubliIp():
#     response = requests.get('https://api.ipify.org')
#     return response.text

# 인증키 검사
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    if api_key != '20240922-topbds':
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "인증 실패"))

def resRespone(status: str, code: int, message: str, cnt: int = 0, data: Any = None) -> dict:
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
env = 'local'
serviceName = 'LSH0577'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

globalVar = {
    'ctxPath': f"{ctxPath}"
    , 'inpPath': f"/DATA/INPUT/{serviceName}"
    , 'outPath': f"/DATA/OUTPUT/{serviceName}"
    , 'figPath': f"/DATA/FIG/{serviceName}"
    , 'cfgPath': f"/SYSTEMS/PROG/PYTHON/IDE/resources/config"
}

for key, val in globalVar.items():
    if key.__contains__('Path'):
        os.makedirs(val, exist_ok=True)
        print(f"[CHECK] {key} : {val}")

# 작업 경로 설정
# os.chdir(f"{globalVar['ctxPath']}")
# print(f"[CHECK] getcwd : {os.getcwd()}")

log = initLog(env, ctxPath, prjName)

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2018-01-01'
    , 'endDate': '2018-12-31'
}


# 공유 폴더
# UPLOAD_PATH = "/DATA/UPLOAD"
# Base = declarative_base()
app = FastAPI(
    openapi_url='/api'
    , docs_url='/docs'
    , redoc_url='/redoc'
)

jsonFile = '{}/{}'.format(globalVar['cfgPath'], 'iconic-ruler-239806-7f6de5759012.json')
jsonList = sorted(glob.glob(jsonFile))
jsonInfo = jsonList[0]

credentials = service_account.Credentials.from_service_account_file(jsonInfo)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# ============================================
# 비즈니스 로직
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post(f"/api/sel-real", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-real")
def selReal(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="아파트")
        , area: str = Query(None, description="평수")
        , year: int = Query(None, description="연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_REAL 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        아파트: 시영\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if apt: condList.append(f"apt LIKE '%{apt}%'")
        if area: condList.append(f"area LIKE '%{area}%'")
        if year: condList.append(f"year = {year}")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-real", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-prd")
def selReal(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="아파트")
        , area: str = Query(None, description="평수")
        , year: int = Query(None, description="연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_PRD 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        아파트: 시영\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_PRD`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if apt: condList.append(f"apt LIKE '%{apt}%'")
        if area: condList.append(f"area LIKE '%{area}%'")
        if year: condList.append(f"year = {year}")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-infra", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-infra")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_INFRA 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_INFRA`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-keyword", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-keyword")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_KEYWORD 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_KEYWORD`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-yearPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-yearPopTrend")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_YEAR-POP-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_YEAR-POP-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-upComSupply", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-upComSupply")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_UP-COM-SUPPLY 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_UP-COM-SUPPLY`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-monthPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-monthPopTrend")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_UP-COM-SUPPLY 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_MONTH-POP-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-unSoldTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-unSoldTrend")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_UN-SOLD-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_UN-SOLD-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))

@app.post(f"/api/sel-largeComRank", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-largeComRank")
def selReal(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_LARGE-COM-RANK 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_LARGE-COM-RANK`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        query_job = client.query(sql)
        results = query_job.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            raise Exception("검색 결과 없음")

        return resRespone("succ", 200, "처리 완료", len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패", str(e)))
