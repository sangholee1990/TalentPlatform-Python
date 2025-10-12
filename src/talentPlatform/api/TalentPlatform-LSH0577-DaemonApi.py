# ============================================
# 요구사항
# ============================================
# LSH0577. Python을 이용한 빅쿼리 기반으로 API 자료서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39
# uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api/TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000
# nohup uvicorn TalentPlatform-LSH0577-DaemonApi:app --host=0.0.0.0 --port=9000 &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0577-DaemonApi" | awk '{print $2}' | xargs kill -9

# 테스트 서버
# uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9400
# nohup uvicorn TalentPlatform-LSH0597-DaemonApi:app --reload --host=0.0.0.0 --port=9400 &
# lsof -i :9400 | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9000
# lsof -i :9000 | awk '{print $2}' | xargs kill -9

# "[TOP BDS] [통합] 아파트 보고서 (데이터 분석, 가격 예측)" 및 빅쿼리 기반으로 API 배포체계를 전달하오니 확인 부탁드립니다.
# 명세1) http://49.247.41.71:9000/docs
# 명세2) http://49.247.41.71:9000/redoc

# 2024.11.08 아실 플랫폼과 같은 시군구/읍면동/계약년월을 기준으로 최고상승/최고하락 아파트 목록
# https://docs.google.com/document/d/1jUxkICwo2WqHACLc_dEziGl7_Y4f5MMClHW7yyKn_XQ/edit?tab=t.0#heading=h.ebgsmeszn577
# 실거래가 기준 전월 대비 상승 비율이 높은 아파트 리스트 구성
# 금액의 비율이 가장 높은 순으로만 아파트 구성
# 전월의 평균금액과 이번월의 평균금액의 비율 변화
# 전월에 거래가 없으면 해당 아파트는 미표시
# 읍면동 단위로 구분하여 아파트 랭킹 (상위 10개, 더보기 통해 상위 30개)
# 초기 화면은 시 단위, 메뉴바 통해 시군구까지 조회 가능
# 아파트의 평형별로 구분하여 아파트 랭킹 구성
# 레퍼런스 : 아실 데이터 https://asil.kr/asil/index.jsp

# 1. 현재 실거래 리스트에서 연도는 선택만으로 가능한 상황인데 여러 연도로 해야할까요? 그렇게 하려면 API에서 여러 연도를 추가해주시면 (시작연도, 끝연도) 작업 가능합니다.
# 2. 아파트 상세 내용에서 전체 세대수 API가 대부분의 아파트에서 정보가 없습니다. 혹시 세대수를 따로 받을 수 있는 API가 있으면 알려주세요.
# 3. 시군구데이터에서 현재 최대거래량을 전체 데이트를 가져와서 하나하나 세야 하는 방식입니다. 그래서 시군구별 한개만 선택해서 데이터 산출이 가능한데요. 여러개 선택할 경우 서버 응답 시간이 초과되는 상황입니다. 혹시 API에서 시군구별로 거래량을 산출한 후에 보내주면 여러 시군구별이나 전체 데이터로 가능할 것으로 보이는데 어떠신지요?

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
# from pydantic import BaseModel
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
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import requests
import time
from concurrent.futures import ProcessPoolExecutor
import configparser

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

# 인증키 검사
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    if api_key != '20240922-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, allCnt: int = 0, rowCnt: int = 0, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "allCnt": allCnt
        , "rowCnt": rowCnt
        , "data": data
    }

def fetchApi(apiUrl, payload, apiType, recAptData):

    result = pd.DataFrame(columns=['idx', 'score'])

    try:
        response = requests.post(apiUrl, data=payload, verify=False, timeout=30)
        response.raise_for_status()
        resJson = response.json().get('recommends')
        resData = pd.DataFrame(resJson[apiType], columns=['idx', 'score'])
        result = pd.merge(resData, recAptData, how='left', left_on=['idx'], right_on=['idx'])
    except Exception as e:
        log.error(f"Exception : {e}")

    return result

# ============================================
# 주요 설정
# ============================================
env = 'local'
serviceName = 'LSH0577'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # 빅쿼리 설정 정보
    'jsonFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/iconic-ruler-239806-7f6de5759012.json',

    # 설정 정보
    'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'cfgKey': 'gemini-api-key',
    'cfgVal': 'oper',
    # 'cfgVal': 'local',

    # CORS 설정
    'oriList': ['*'],

    'rcmd': {
        'apiCfUrl': 'http://125.251.52.42:9010/recommends_cf',
        'apiSimUrl': 'http://125.251.52.42:9010/recommends_simil',
        'propAptFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/xlsx/20250526_tbl_apts.xlsx',
        'propUserFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/xlsx/20250526_tbl_users.xlsx',
    },
}

app = FastAPI(
    openapi_url='/api'
    , docs_url='/docs'
    , redoc_url='/redoc'
)

# 공유 설정
# app.mount('/UPLOAD', StaticFiles(directory='/DATA/UPLOAD'), name='/DATA/UPLOAD')

app.add_middleware(
    CORSMiddleware
    # , allow_origins=["*"]
    , allow_origins=sysOpt['oriList']
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# 빅쿼리 설정 정보
jsonFile = sysOpt['jsonFile']
jsonList = sorted(glob.glob(jsonFile))
if jsonList is None or len(jsonList) < 1:
    log.error(f'jsonFile : {jsonFile} / 설정 파일 검색 실패')
    exit(1)

jsonInfo = jsonList[0]

try:
    credentials = service_account.Credentials.from_service_account_file(jsonInfo)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
except Exception as e:
    log.error(f'Exception : {e} / 빅쿼리 연결 실패')
    exit(1)

# 사용자 설정 정보
inpFile = sysOpt['rcmd']['propUserFile']
fileList = sorted(glob.glob(inpFile), reverse=True)
if fileList is None or len(fileList) < 1:
    log.error(f'파일 없음 : {inpFile}')
    sys.exit(1)

recUserData = pd.read_excel(fileList[0])

# 아파트 설정 정보
inpFile = sysOpt['rcmd']['propAptFile']
fileList = sorted(glob.glob(inpFile), reverse=True)
if fileList is None or len(fileList) < 1:
    log.error(f'파일 없음 : {inpFile}')
    sys.exit(1)

recAptData = pd.read_excel(fileList[0])

# ============================================
# 비즈니스 로직
# ============================================
class cfgRcmd(BaseModel):
    gender: str = Field(..., description='성별 (1 남성, 2 여성)', examples=['1'])
    age: str = Field(..., description='나이 (최소-최대, 20-39)', examples=['20-39'])
    price: str = Field(..., description='가격 억원 (최소-최대, 3-6)', examples=['3-6'])
    area: str = Field(..., description='면적 m² (최소-최대, 58-100)', examples=['58-100'])
    debtRat: str = Field(..., description='부채 비율', examples=['0.25'])
    apt: str = Field(..., description='아파트 도로명주소, 두산(가산로 99)', examples=['두산(가산로 99)'])
    cnt: str = Field(..., alias='cnt', description='추천 개수', examples=['10'])

@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post(f"/api/sel-rcmd", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-rcmd")
def selRcmd(request: cfgRcmd = Form(...)):
    """
    기능\n
        CF/유사도 기반 아파트 추천 서비스 API\n
        검색조건 사용자 설정 (gender, age, price, area, debtRat), 아파트 설정 (apt, cnt)\n
    """

    try:
        gender = request.gender
        age = request.age
        price = request.price
        area = request.area
        debtRat = request.debtRat
        apt = request.apt
        cnt = request.cnt

        minAge, maxAge = age.split('-')
        if age is None or len(age) < 1 or minAge is None or maxAge is None:
            return resResponse("fail", 400, f"나이를 확인해주세요 ({age}).", None)

        minPrice, maxPrice = price.split('-')
        if price is None or len(price) < 1 or minPrice is None or maxPrice is None:
            return resResponse("fail", 400, f"가격을 확인해주세요 ({price}).", None)

        minArea, maxArea = area.split('-')
        if area is None or len(area) < 1 or minArea is None or maxArea is None:
            return resResponse("fail", 400, f"면적을 확인해주세요 ({area}).", None)

        recUserDataL1 = recUserData.loc[
            (recUserData['gender'] == int(gender))
            & (recUserData['age'] >= float(minAge)) & (recUserData['age'] <= float(maxAge))
            & (recUserData['price_from'] >= float(minPrice)) & (recUserData['price_to'] <= float(maxPrice))
            & (recUserData['area_from'] >= float(minArea)) & (recUserData['area_to'] <= float(maxArea))
            & (recUserData['debt_ratio'] >= float(debtRat))
            # & (recUserData['prefer'] == prefer)
            ]

        if len(recUserDataL1) < 1:
            return resResponse("fail", 400, f"사용자 설정 정보를 확인해주세요.", None)

        recAptDataL1 = recAptData.loc[
            (recAptData['apt'] == apt)
            & (recAptData['area'] >= float(minArea)) & (recAptData['area'] <= float(maxArea))
            & (recAptData['price'] >= float(minPrice)) & (recAptData['price'] <= float(maxPrice))
            ]

        if len(recAptDataL1) < 1:
            return resResponse("fail", 400, f"아파트 설정 정보를 확인해주세요.", None)

        payload = {
            'user_id': recUserDataL1.iloc[0]['idx'],
            'apt_idx': recAptDataL1.iloc[0]['idx'],
            'rcmd_count': cnt,
        }

        # CF기반 및 유사도 기반 아파트 추천
        with ProcessPoolExecutor(max_workers=2) as executor:
            futureCf = executor.submit(fetchApi, sysOpt['rcmd']['apiCfUrl'], payload, 'cf', recAptData)
            futureSim = executor.submit(fetchApi, sysOpt['rcmd']['apiSimUrl'], payload, 'simil', recAptData)

            cfData = futureCf.result()
            simData = futureSim.result()

        result = {
            'user': recUserDataL1.iloc[0].to_dict(),
            'apt': recAptDataL1.iloc[0].to_dict(),
            'cf': cfData.to_dict(orient='records'),
            'sim': simData.to_dict(orient='records'),
        }

        return resResponse("succ", 200, "처리 완료", len(result), len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


@app.post(f"/api/sel-statRealMeanBySggDong", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMeanBySggDong")
def selStatRealSggApt(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 시군구 법정동 별로 평균가 API 구성\n
        검색조건 시군구, 지역, 거래년도, 평수\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,mean_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT sgg, dong, COUNT(*) AS cnt, AVG(amount) AS mean_amount FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg, dong"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchBySgg", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchBySgg")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 지역 시도와 시군구로 구분한 API 구성\n
        맞집 좌측 검색조건 지역 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT sgg, COUNT(*) AS cnt FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")


        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByYear", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByYear")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 건축년도 표기 API 구성\n
        맞집 좌측 검색조건 거래년도 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT year, COUNT(*) AS cnt FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY year"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByArea", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByArea")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 평수 리스트 API 구성\n
        맞집 좌측 검색조건 평수 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT area, COUNT(*) AS cnt FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY area"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByApt", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByApt")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트명 리스트 API 구성\n
        맞집 좌측 검색조건 아파트명 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT apt, COUNT(*) AS cnt FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")


        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY apt"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealMaxBySgg", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMaxBySgg")
def selStatRealMaxBySgg(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 시군구 별로 최대값 API 구성\n
        검색조건 시군구, 지역, 거래년도, 평수\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT sgg, COUNT(*) AS cnt, MAX(amount) AS max_amount FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealMaxBySggApt", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMaxBySggApt")
def selStatRealMaxBySggApt(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 시군구 아파트명 별로 최대값 API 구성\n
        검색조건 시군구, 아파트명, 지역, 거래년도, 평수\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT sgg, apt, COUNT(*) AS cnt, MAX(amount) AS max_amount FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg, apt"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statReal", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statReal")
def selStatReal(
        sgg: str = Query(None, description="시군구")
        , dong: str = Query(None, description="법정동")
        , area: str = Query(None, description="평수")
        , yyyymm: str = Query(None, description="연월")
    ):
    """
    기능\n
        TB_REAL 통계 목록 조회\n
    테스트\n
        시군구: 서울특별시 금천구\n
        법정동: 독산동\n
        평수: 5평\n
        연월: 202201\n

        또는 \n

        시군구: \n
        법정동: \n
        평수: \n
        연월: 202201\n
    """

    try:
        # 기본 SQL
        baseSql = """
            WITH MONTHLYAVGPRICES AS (
                SELECT
                    APT,
                    SGG,
                    DONG,
                    AREA,
                    DATE_TRUNC(DATE, MONTH) AS MONTH,
                    AVG(AMOUNT) AS AVG_AMOUNT
                FROM
                    `DMS01.TB_REAL`
                WHERE AMOUNT > 0
                GROUP BY 1, 2, 3, 4, 5
            ),

            LAGGEDPRICES AS (
                SELECT
                    APT,
                    SGG,
                    DONG,
                    AREA,
                    MONTH,
                    AVG_AMOUNT,
                    LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH) AS PREV_AVG_AMOUNT,
                    SAFE_DIVIDE(AVG_AMOUNT - LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH), LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH)) * 100 AS RATE
                FROM MONTHLYAVGPRICES
                WHERE AVG_AMOUNT > 0
            )

            SELECT * FROM (
                (SELECT
                    '최고하락' AS GRP,
                    SGG,
                    DONG,
                    DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH,
                    MONTH AS CURR_MONTH,
                    APT,
                    AREA,
                    PREV_AVG_AMOUNT,
                    AVG_AMOUNT AS CURR_AVG_AMOUNT,
                    RATE
                FROM LAGGEDPRICES
                WHERE
                    (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_AMOUNT > 0
                ORDER BY RATE ASC
                LIMIT 30)

                UNION ALL

                (SELECT
                    '최고상승' AS GRP,
                    SGG,
                    DONG,
                    DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH,
                    MONTH AS CURR_MONTH,
                    APT,
                    AREA,
                    PREV_AVG_AMOUNT,
                    AVG_AMOUNT AS CURR_AVG_AMOUNT,
                    RATE
                FROM LAGGEDPRICES
                WHERE
                    (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_AMOUNT > 0
                ORDER BY RATE DESC
                LIMIT 30)
            );
        """

        # 동적 파라미터
        queryParam = [
            bigquery.ScalarQueryParameter("sgg", "STRING", sgg if sgg else None)
            , bigquery.ScalarQueryParameter("dong", "STRING", dong if dong else None)
            , bigquery.ScalarQueryParameter("area", "STRING", area if area else None)
            , bigquery.ScalarQueryParameter("yyyymm", "STRING", yyyymm if yyyymm else None)
        ]
        log.info(f"[CHECK] queryParam : {queryParam}")

        # 쿼리 실행
        queryJobCfg = bigquery.QueryJobConfig(query_parameters=queryParam)
        queryJob = client.query(baseSql, job_config=queryJobCfg)
        results = queryJob.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-real", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-real")
def selReal(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        # , year: int = Query(None, description="연도")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 목록 조회\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): year|asc,area|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_REAL`"
        cntSql = f"SELECT COUNT(*) AS CNT FROM `iconic-ruler-239806.DMS01.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            # condList.append(f"sgg LIKE '%{sgg}%'")
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            # condList.append(f"area LIKE '%{area}%'")
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        # if year:
        #     condList.append(f"year = {year}")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql
            cntSql += condSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cnt = next(cntRes)['CNT']

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-prd", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-prd")
def selPrd(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        # , year: int = Query(None, description="연도")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_PRD 목록 조회\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): year|asc,area|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT * FROM `iconic-ruler-239806.DMS01.TB_PRD`"
        cntSql = f"SELECT COUNT(*) AS CNT FROM `iconic-ruler-239806.DMS01.TB_PRD`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            # condList.append(f"sgg LIKE '%{sgg}%'")
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            # condList.append(f"area LIKE '%{area}%'")
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        # if year:
        #     condList.append(f"year = {year}")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql
            cntSql += condSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = f" ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cnt = next(cntRes)['CNT']

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


@app.post(f"/api/sel-infra", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-infra")
def selInfra(
        lon: float = Query(127.10, description="특정 경도")
        , lat: float = Query(37.46, description="특정 위도")
        , distKm: float = Query(1, description="특정 경위도를 기준으로 거리 km")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_INFRA 목록 조회\n
        즉 특정 경도/위도 (아파트 위치 또는 관심지역 위치 등)를 기준으로 반경 km 주변 인프라 (편의시설, 교육, 주거환경, 교통) 조회
    테스트\n
        특정 경도: 127.10\n
        특정 위도: 37.46\n
        특정 경위도를 기준으로 거리 km: 1 km = 1000 m\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): distKm|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"""
            SELECT
                *,
                ST_DISTANCE(ST_GEOGPOINT(p_x, p_y), ST_GEOGPOINT({lon}, {lat})) / 1000 AS distKm
            FROM
                `iconic-ruler-239806.DMS01.TB_INFRA`
        """

        # 동적 SQL 파라미터
        condList = []
        # if sgg:
        #     sggList = [s.strip() for s in sgg.split(',')]
        #     sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
        #     condList.append(f"({' OR '.join(sggCond)})")

        if lon and lat and distKm:
            condList.append(f"ST_DISTANCE(ST_GEOGPOINT(p_x, p_y), ST_GEOGPOINT({lon}, {lat})) / 1000 <= {distKm}")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        # grpList = []
        # grpSql = " GROUP BY sgg"
        # baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")

        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql
        cntSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        data = pd.DataFrame(fileList)
        fileDict = {}
        cntDict = {}
        for cate in ['편의시설', '교육', '주거환경', '교통']:
            dataL1 = data.loc[data['category'] == cate]
            fileDict[cate] = dataL1.to_dict(orient='records')
            cntDict[cate] = len(fileDict[cate])

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = cntList[0]['cnt']

        return resResponse("succ", 200, "처리 완료", cnt, cntDict, fileDict)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-keyword", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-keyword")
def selKeyword(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-yearPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-yearPopTrend")
def selYearPopTrend(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-upComSupply", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-upComSupply")
def selUpComSupply(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-monthPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-monthPopTrend")
def selMonthPopTrend(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-unSoldTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-unSoldTrend")
def selUnSoldTrend(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-largeComRank", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-largeComRank")
def selLargeComRank(
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
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))