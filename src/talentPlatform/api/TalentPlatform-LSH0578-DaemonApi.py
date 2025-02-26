# ============================================
# 요구사항
# ============================================
# LSH0578. Python을 이용한 생성형 AI 기반 블로그 포스팅 대필 및 API 연계 서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39

# 운영 서버
# uvicorn TalentPlatform-LSH0578-DaemonApi:app --reload --host=0.0.0.0 --port=9200
# nohup uvicorn TalentPlatform-LSH0578-DaemonApi:app --host=0.0.0.0 --port=9200 &
# tail -f nohup.out

# 테스트 서버
# uvicorn TalentPlatform-LSH0578-DaemonApi:app --reload --host=0.0.0.0 --port=9400

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0578-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9200
# lsof -i :9200 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9200/docs
# 명세2) http://49.247.41.71:9200/redoc

# 20241202 기술협의
# 금칙어 제공 서비스

# 프롬프트 선택지 추천 기능
# 후기성, 정보성
# 읽는사람이 어떤 목적
# 글쓴이의 업종 등등
# -> 정확한 프롬프트를 위한 서포트장치 필요

# 기획안 예시
# 특정 글감에 대해서 필요한 주제를 나열해 달라는 요구가 있을 수 있음
# 이에 다양한 주제를 받을 수 있는 기능이 필요함

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

# from google.cloud import bigquery
# from google.oauth2 import service_account
# import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import urllib

import google.generativeai as genai
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
    if api_key != '20241014-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

async def streamExcel(dataframes: dict[str, pd.DataFrame]):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    while True:
        chunk = output.read(4096)
        if not chunk:
            break
        yield chunk
        await asyncio.sleep(0)


# ============================================
# 주요 설정
# ============================================
env = 'local'
serviceName = 'LSH0578'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # 설정 파일
    'csvFile': '/DATA/INPUT/LSH0578/20241103_13개 분야 별로 대표 템플릿 생성형 AI 4종 결과 - 최종.csv',

    # CORS 설정
    'oriList': [
        'http://localhost:9200',
        'http://49.247.41.71:9200',
        'http://localhost:9400',
        'http://49.247.41.71:9400',
    ],

    # 입력 자료
    'inpPath': '/DATA/INPUT/LSH0578',

    # 수집 설정
    'colct': {
        'naver': {
            'baseUrl': "https://datalab.naver.com/shoppingInsight/getKeywordRank.naver",
            'cateList': [
                {"name": "패션의류", "param": ["50000000"]},
                {"name": "패션잡화", "param": ["50000001"]},
                {"name": "화장품/미용", "param": ["50000002"]},
                {"name": "디지털/가전", "param": ["50000003"]},
                {"name": "가구/인테리어", "param": ["50000004"]},
                {"name": "출산/육아", "param": ["50000005"]},
                {"name": "식품", "param": ["50000006"]},
                {"name": "스포츠/레저", "param": ["50000007"]},
                {"name": "생활/건강", "param": ["50000008"]},
                {"name": "여가/생활편의", "param": ["50000009"]},
                {"name": "도서", "param": ["50005542"]},
            ],
            'headers': {
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://datalab.naver.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        },
        'whereispost': {
            'baseUrl': "https://whereispost.com/hot",
            'headers': {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            },
        },
        'ezme': {
            'baseUrl': "https://rank.ezme.net",
            'headers': {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            },
        },
    },

    # 가공 설정
    'filter': {
        'stopWordFileInfo': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/word/stopwords-ko.txt',
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

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

# Gemini Advanced
# genai.configure(api_key=None)
# genai.configure(api_key='AIzaSyCcWX2naC_JeugXS8zt4AsFeAUIFKAMaYQ')
genai.configure(api_key='AIzaSyCf8gpB1bI0sqdUhQ_TNtQnUUEHEroahoA')
model = genai.GenerativeModel('gemini-1.5-pro')

# 설정 파일
try:
    csvFile = sysOpt['csvFile']
    csvList = sorted(glob.glob(csvFile))
    if csvList is None or len(csvList) < 1:
        log.error(f'csvFile : {csvFile} / 설정 파일 검색 실패')
        exit(1)

    csvInfo = csvList[0]
    csvData = pd.read_csv(csvInfo)
except Exception as e:
    log.error(f'csvData 실패 : {csvFile} : {e}')
    exit(1)

# 불용어 목록
try:
    csvFile = sysOpt['filter']['stopWordFileInfo']
    csvList = sorted(glob.glob(csvFile))
    if csvList is None or len(csvList) < 1:
        log.error(f'csvFile : {csvFile} / 설정 파일 검색 실패')
        exit(1)

    csvInfo = csvList[0]
    stopWordData = pd.read_csv(csvInfo)
    stopWordList = stopWordData['word'].tolist()
except Exception as e:
    log.error(f'stopWordData 실패 : {csvFile} : {e}')
    exit(1)

# ============================================
# 비즈니스 로직
# ============================================
# class pdfToTxtData(BaseModel):
#     cont: str = Field(..., example='텍스트 추출', description='요청사항')
#     file: UploadFile = File(None, example=None, description='PDF 파일')

class blogTypePostData(BaseModel):
    type: str = Query(default=..., description='분야', example='뷰티화장품', enum=[
        "법률", "뷰티화장품", "병원", "부동산",
        "맛집", "반려동물", "자동차", "운동",
        "여행", "휴대폰성지", "설비", "청소", "인테리어"
    ])
    cont: str = Field(default=..., example='대표 블로그를 이용하여 필수 키워드 (알잘딱깔센)를 포함하여 블로그 포스팅을 작성해줘', description='요청사항')

class blogPostData(BaseModel):
    cont: str = Field(default=..., example='대표 블로그를 이용하여 필수 키워드 (알잘딱깔센)를 포함하여 블로그 포스팅을 작성해줘', description='요청사항')

class blogPostChkData(BaseModel):
    forbidWord: str = Field(default=..., example='만병통치약|기적의 치료|스폰서|후원|쿠팡파트너스|제휴 마케팅|도배성 홍보 문구|클릭하세요|구독해주세요|불법사이트|도박||폭력|해킹|불법|토렌트|인종|짱깨|쪽바리|깜둥이|성별|지역 비하|김치녀|한남충|맘충|틀딱|비하|정신병자|장애인|욕설|비하|시신|거지|야사|자지|보지|아다|씹고|음탕|후장|고자|충동|후회|고비|인내|참아|자살|음부|고환|후다|니미|애널|에널|해적|몰래|재생|유발|만족|무시|지저분함|등빨|탈출|살인|혐오|발언|폭력|성범죄|잠지|좆|씹|보짓물|질|음경|클리|유두|젖꼭지|가슴|엉덩이|사타구니|불알|정액|난자|월경|생리|섹스|정사|교미|성교|윤간|강간|성폭행|성추행|자위|ㄸㄸㅇ|ㅍㅍㅅㅅ|ㅇㅆ|ㅅㅅ|ㅈㅈ|ㅂㅈ|성관계|유사성행위|오랄|펠라|쿤니|애무|전희|후희|삽입|사정|절정|오르가즘|썅년|창녀|걸레|창놈|호빠|보빨|자빨|좆물|씹물|개씹|개좆|씹창|좆같다|쎅쓰|떡치다|따먹다|박다|쑤시다|조건|만남|출장|안마|오피|키스방|풀싸롱|하드코어|포르노|야동|야설|폰섹|영섹|딸감|품번|2차|노콘|노팬티|살해|폭행|구타|린치|칼빵|조폭|일진|학폭|왕따|도둑|절도|강도|사기|협박|납치|감금|살인미수|공갈|씨발|좆까|씹새끼|개새끼|미친놈|미친년|병신|찐따|retard|카지노|바카라|블랙잭|룰렛|슬롯머신|경마|경륜|경정|토토|프로토|사다리|파워볼|배팅|잭팟|올인|탕진|빚|도박중독|에이즈|임질|매독|헤르페스|불임|난임|기적|완치|즉효|특효|부작용|100%|효과|단기간|최저가|최고|비법|비밀|보장|대마초|필로폰|코카인|헤로인|LSD|엑스터시|마리화나|뽕|히로뽕|떨|은어|복수|대박|초대박|한정|긴급|서두르세요|로또|지랄|꺼져|닥쳐|엿 먹어|썅|개소리|또라이|성행위|구강성교|항문성교|음란물|나체|누드|성기|노골적|변태|AV|에이브이|AIDS|낙태|임신중절|자해|발기부전|조루증|정력제|흥분제|불치병|난치병|비아그라', description='금지어 키워드')
    cont: str = Field(default=..., example="""요즘 몸도 힘들고 마음도 힘들고 이래저래 기운없는 나날들을 보내고 있어요. 몸이 피곤하니 마음도 기분도 울적한가 봐요. 뒤늦게 가을을 타는 걸까요?
하고 싶은 제 머리는 아니지만 오늘을 딸의 생애 첫 커트에 대한 간단한 일기를 포스팅하겠습니다.

지난주 금요일 오후,
함께 누워있던 딸아이가
갑자기 '엄마, 나 머리카락이 너무 귀찮아요...
나 이제 머리카락 자르고 싶어요.'라고 해서
(무려 4년 만에... 우리 딸 4살!!)

그 말과 동시에
주섬주섬 옷을 입고
미용실로 직행!! 했습니다.

무려 4년 만에 자르는 것이라
작년부터 부쩍 길어진 머리에
아침마다 빗질도, 묶는 것도 일이기에
조금만 자르자고 꼬셔도
절대 안 자르겠다고,
아빠랑 오빠 미용실에 따라가서도
절대 안 자른다고 해왔어서
머리를 자른다는 말이
너무너무 반가웠어요. :)

미리 예약을 하고 간 것이
아니라 정말 급하게 왔더니
역시나 대기가 있었습니다.

아들은 아빠랑 아무데나 가서
커트를 하지만,
딸은 그래도 여자아이고
첫 커트니 만큼 예쁘게 잘라주고 싶은
마음에 제가 이용하는 미용실로 데리고 갔어요.
ㅎㅎㅎ""", description='블로그 포스팅 내용')

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-pdfToTxt", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-pdfToTxt")
# async def selPdfToTxt(request: pdfToTxtData = Form(...)):
async def selPdfToTxt(
        cont: str = Form('텍스트 추출', example='텍스트 추출', description='요청사항')
        ,  file: UploadFile = File(None, example=None, description='PDF 파일')
    ):
    """
    기능\n
        PDF 인쇄 파일로부터 텍스트 추출\n
    테스트\n
        cont: 요청사항\n
        file: PDF 인쇄 파일\n
    """

    tmpFileInfo = None

    try:
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        if file is None:
            return resResponse("fail", 400, f"PDF 파일이 없습니다 : {file}")

        if file.content_type != 'application/pdf':
            return resResponse("fail", 400, "PDF 파일 없음")

        log.info(f"[CHECK] cont : {cont}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=sysOpt['inpPath']) as tmpFile:
            tmpFile.write(file.file.read())
            tmpFileInfo = tmpFile.name
        log.info(f"[CHECK] tmpFileInfo : {tmpFile.name}")

        pdfFile = genai.upload_file(mime_type=file.content_type, path=tmpFileInfo, display_name=tmpFileInfo)
        res = model.generate_content([cont, pdfFile])
        result = res.candidates[0].content.parts[0].text
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmpFileInfo and os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

# @app.post(f"/api/sel-blogTypePost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-blogTypePost")
async def selBlogTypePost(request: blogTypePostData = Form(...)):
    """
    기능\n
        분야별 템플릿 및 요청사항을 기반으로 블로그 포스팅 대필\n
    테스트\n
        type: 분야\n
            - 20241103_13개 분야 별로 대표 템플릿 생성형 AI 4종 결과

            - https://docs.google.com/spreadsheets/d/1KMVTWiPQ6AZA1F3CqBLv2tw3TvG8kOLcOL3ISnFBQLk/edit?gid=472867053#gid=472867053
        cont: 요청사항\n
    """
    try:
        type = request.type
        if type is None or len(type) < 1:
            return resResponse("fail", 400, f"분야가 없습니다 : {type}")

        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        csvDataL1 = csvData.loc[csvData['분야'] == type]
        if csvDataL1.empty:
            return resResponse("fail", 400, "템플릿 파일 없음")

        log.info(f"[CHECK] csvDataL1 : {csvDataL1}")

        contTemplate = csvDataL1['텍스트 추출'].iloc[0]

        contL1 = f"{contTemplate} \n {cont}"
        log.info(f"[CHECK] contL1 : {contL1}")

        res = model.generate_content(contL1)
        result = res.candidates[0].content.parts[0].text
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-blogPost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-blogPost")
async def selBlogPost(request: blogPostData = Form(...)):
    """
    기능\n
        요청사항을 기반으로 블로그 포스팅 대필\n
    테스트\n
        cont: 요청사항\n
    """
    try:
        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        res = model.generate_content(cont)
        result = res.candidates[0].content.parts[0].text
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


# @app.post(f"/api/sel-blogPostChk", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-blogPostChk")
async def blogPostChk(request: blogPostChkData = Form(...)):
    """
    기능\n
        금지어 키워드 목록을 기반으로 블로그 포스팅 금지어 검사\n
    요청 파라미터\n
        forbidWord: 금지어 키워드 목록
        cont: 블로그 포스팅 내용
    응답 결과\n
        샘플 결과
            엑셀 파일 %Y%m%d%H%M_blogPostChk.xlsx
            세부 시트
                - forbid 금지어
                - normal 일반어
    참조\n
        소스코드 https://github.com/keunyop/BadWordCheck
    """
    try:
        # 금지어 목록
        forbidWord = request.forbidWord
        forbidWordList = list(filter(None, forbidWord.split("|")))
        if forbidWord is None or len(forbidWord) < 1 or forbidWordList is None or len(forbidWordList) < 1:
            return resResponse("fail", 400, f"금지어 키워드를 확인해주세요 : {forbidWord}", None)
        # log.info(f"[CHECK] forbidWordList : {forbidWordList}")

        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"블로그 포스팅 내용을 확인해주세요 : {cont}", None)

        okt = Okt()
        posTagList = okt.pos(cont, stem=True)

        # 명사 추출
        keywordList = [word for word, pos in posTagList if pos in ['Noun']]

        # 불용어 제거
        keywordList = [word for word in keywordList if word not in stopWordList and len(word) > 1]
        # log.info(f"[CHECK] keywordList : {keywordList}")

        # 빈도수 계산
        keywordCnt = Counter(keywordList)
        data = pd.DataFrame(keywordCnt.items(), columns=['keyword', 'cnt'])

        pattern = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, forbidWordList))))
        # log.info(f"[CHECK] forbidWordList : {len(forbidWordList)} : {forbidWordList}")

        # data['type'] = data['keyword'].apply(lambda x: '금지어' if pattern.search(str(x)) else '일반어')
        data['type'] = '일반어'
        data.loc[data['keyword'].str.contains(pattern, na=False, regex=True), 'type'] = '금지어'

        dataL1 = data.sort_values(by=['type', 'cnt'], ascending=[False, False])

        forbidData = dataL1[dataL1['type'] == '금지어']
        normalData = dataL1[dataL1['type'] == '일반어']
        # forbidList = forbidData['keyword'].tolist()
        # normalList = normalData['keyword'].tolist()

        # log.info(f"[CHECK] 금지어 목록 : {len(forbidList)} : {forbidList}")
        # log.info(f"[CHECK] 일반어 목록 : {len(normalList)} : {normalList}")

        # result = {
        #     'forbid' : {
        #         'cnt': len(forbidList),
        #         # 'list': forbidList,
        #         'data': forbidData.to_dict(orient='records'),
        #     },
        #     'normal' : {
        #         'cnt': len(normalList),
        #         # 'list': normalList,
        #         'data': normalData.to_dict(orient='records'),
        #     },
        # }

        result = {
            'forbid' : forbidData,
            'normal' : normalData,
        }
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        # return resResponse("succ", 200, "처리 완료", len(result), result)

        return StreamingResponse(
            streamExcel(result),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={datetime.now(tz=tzKst).strftime('%Y%m%d%H%M')}_blogPostChk.xlsx"}
        )

    except Exception as e:
        log.error(f'blogPostChk 실패 : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-searchKeyword", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-searchKeyword")
async def searchKeyword():
    """
    기능\n
        네이버/구글/웨어이즈포스트/이지미넷 사이트로부터 실시간 검색어\n
    요청 파라미터\n
        해당 없음
    응답 결과\n
        엑셀 파일 %Y%m%d%H%M_searchKeyword.xlsx
        세부 시트
            - naver 네이버
            - google 구글 (2025.02.26 비공식 API 장애 발생 중)
            - whereispost 웨어이즈포스트
            - ezme 이지미넷

    참조\n
        네이버 쇼핑인사이트 검색어 https://datalab.naver.com
        네이버 통합검색어 API https://openapi.naver.com/v1/datalab/search
        네이버 쇼핑인사이트 API https://openapi.naver.com/v1/datalab/shopping/categories
        구글 트렌드 검색어 https://trends.google.co.kr/trending?geo=KR&hl=ko
        웨어이즈포스트 검색어 https://whereispost.com/hot
        이지미넷 검색어 https://rank.ezme.net
    """
    try:
        dataL1 = pd.DataFrame()

        # ==========================================================================================================
        # 네이버 쇼핑인사이트 검색어
        # ==========================================================================================================
        try:
            for idx, cateInfo in enumerate(sysOpt['colct']['naver']['cateList']):
                params = {
                    "timeUnit": "date",
                    "cid": cateInfo['param'][0],
                }

                queryStr = urlencode(params)
                url = f"{sysOpt['colct']['naver']['baseUrl']}?{queryStr}"

                response = requests.post(url, headers=sysOpt['colct']['naver']['headers'])
                if not (response.status_code == 200): continue

                resData = response.json()
                resDataL1 = resData[-1]

                orgData = pd.DataFrame(resDataL1['ranks']).rename(
                    columns={
                        'rank': 'no'
                    }
                )

                orgData['type'] = 'naver'
                orgData['cate'] = cateInfo['name']
                orgData['dateTime'] = pd.to_datetime(resDataL1['date']).tz_localize('Asia/Seoul').strftime('%Y-%m-%d %H:%M')
                data = orgData[['type', 'cate', 'dateTime', 'no', 'keyword']]

                if len(data) > 0:
                    dataL1 = pd.concat([dataL1, data])
        except Exception as e:
            log.error(f"네이버 검색어 수집 실패 : {e}")

        # ==========================================================================================================
        # 구글
        # ==========================================================================================================
        try:
            pytrends = TrendReq(geo='ko-KR', tz=540)
            orgData = pytrends.trending_searches(pn='south_korea')

            orgDataL1 = orgData.rename(columns={0: 'keyword'})
            orgDataL1['no'] = orgDataL1.index + 1
            orgDataL1['dateTime'] = datetime.now(tz=tzKst).strftime('%Y-%m-%d %H:%M')
            orgDataL1['type'] = 'google'
            orgDataL1['cate'] = '전체'

            data = orgDataL1[['type', 'cate', 'dateTime', 'no', 'keyword']]
            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])
        except Exception as e:
            log.error(f"구글 검색어 수집 실패 : {e}")

        # ==========================================================================================================
        # 웨어이즈포스트 검색어
        # ==========================================================================================================
        try:
            response = requests.get(sysOpt['colct']['whereispost']['baseUrl'], headers=sysOpt['colct']['whereispost']['headers'])
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            lxml = etree.HTML(str(soup))

            try:
                tag = lxml.xpath('/html/body/content/div/div/div/div[1]/text()')[0]
                match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', tag.strip())
                sDateTime = None if match is None else match.group(0)
                dtDateTime = pd.to_datetime(sDateTime).tz_localize('Asia/Seoul')
            except Exception:
                dtDateTime = None
            # log.info(f'[CHECK] dtDateTime : {dtDateTime}')

            noList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'rank daum_color'})
            keywordList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'keyword'})

            data = pd.DataFrame()
            for noInfo, keywordInfo in zip(noList, keywordList):
                try:
                    no = None if noInfo is None or len(noInfo) < 1 else noInfo.text.strip()
                    keyword = None if keywordInfo is None or len(keywordInfo) < 1 else keywordInfo.text.strip()

                    dict = {
                        'type': ['whereispost'],
                        'cate': '전체',
                        'dateTime': [dtDateTime.strftime('%Y-%m-%d %H:%M')],
                        'no': [no],
                        'keyword': [keyword],
                    }

                    data = pd.concat([data, pd.DataFrame.from_dict(dict)])

                except Exception:
                    pass

            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])

        except Exception as e:
            log.error(f"웨어이즈포스트 검색어 수집 실패 : {e}")

        # ==========================================================================================================
        # 이지미넷 검색어
        # ==========================================================================================================
        try:
            response = requests.get(sysOpt['colct']['ezme']['baseUrl'], headers=sysOpt['colct']['ezme']['headers'])
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            try:
                tag = soup.find('div', {'id': 'content'}).find('small')
                sDateTime = None if tag is None or len(tag) < 1 else tag.text.strip()
                dtDateTime = pd.to_datetime(sDateTime).tz_localize('Asia/Seoul')
            except Exception:
                dtDateTime = None
            # log.info(f'[CHECK] dtDateTime : {dtDateTime}')

            noList = soup.find('div', {'id': 'content'}).find_all("span", {'class': 'rank_no'})
            keywordList = soup.find('div', {'id': 'content'}).find_all("span", {'class': 'rank_word'})

            data = pd.DataFrame()
            for noInfo, keywordInfo in zip(noList, keywordList):
                try:
                    no = None if noInfo is None or len(noInfo) < 1 else noInfo.text.strip(".").strip()
                    keyword = None if keywordInfo is None or len(keywordInfo) < 1 else keywordInfo.find('a').text.strip()

                    dict = {
                        'type': ['ezme'],
                        'cate': '전체',
                        'dateTime': [dtDateTime.strftime('%Y-%m-%d %H:%M')],
                        'no': [no],
                        'keyword': [keyword],
                    }

                    data = pd.concat([data, pd.DataFrame.from_dict(dict)])
                except Exception:
                    pass

            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])
        except Exception as e:
            log.error(f"이지미넷 검색어 수집 실패 : {e}")

        # ==========================================================================================================
        # 통합 데이터
        # ==========================================================================================================
        dataL2 = dataL1.reset_index(drop=True)
        if dataL2 is None or len(dataL2) < 1:
            return resResponse("fail", 400, "처리 실패")

        # result = {
        #     'cnt': len(dataL2),
        #     'data': dataL2.to_dict(orient='records'),
        # }

        result = {
            'naver': dataL2[dataL2['type'].isin(['naver'])],
            'google': dataL2[dataL2['type'].isin(['google'])],
            'whereispost': dataL2[dataL2['type'].isin(['whereispost'])],
            'ezme': dataL2[dataL2['type'].isin(['ezme'])],
        }
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        # return resResponse("succ", 200, "처리 완료", len(result), result)

        return StreamingResponse(
            streamExcel(result),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={datetime.now(tz=tzKst).strftime('%Y%m%d%H%M')}_searchKeyword.xlsx"}
        )

    except Exception as e:
        log.error(f'blogPostChk 실패 : {e}')
        raise HTTPException(status_code=400, detail=str(e))