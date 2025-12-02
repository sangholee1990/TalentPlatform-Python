# ============================================
# 요구사항
# ============================================
# LSH0627. Python을 이용한 알톤 바이크매트릭스AI 데이터 API

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39

# 운영 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-LSH0627-DaemonApi:app --host=0.0.0.0 --port=9030
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-LSH0627-DaemonApi:app --host=0.0.0.0 --port=9030 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-LSH0627-DaemonApi:app --reload --host=0.0.0.0 --port=9030

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0627-DaemonApi" | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-LSH0627-DaemonApi

# 포트 종료
# yum install lsof -y
# lsof -i :9030
# lsof -i :9030 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9030/docs
# 명세2) http://49.247.41.71:9030/redoc

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import scipy.sparse
from playwright.async_api import async_playwright
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
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    if api_key != '20251012-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

def getPageDict(data, page=1, limit=10):
    result = {}
    start_index = (page - 1) * limit
    end_index = start_index + limit

    if data.empty:
        return result

    for (brand, type), list in zip(data.index, data.values):
        cnt = len(list)
        if cnt > 0:
            key = f"{brand}-{type}"
            result[key] = {
                'cnt': cnt,
                'item': list[start_index:end_index]
            }

    return result


def load_and_standardize_columns(filepath):
    """
    엑셀 파일을 로드하고, 복잡한 한글 컬럼명을 정리하여 표준 영문명으로 변환하는 함수.
    """
    try:
        df = pd.read_excel(filepath, dtype=str)
        log.info(f"데이터 로드 성공: '{filepath}' ({df.shape})")
    except FileNotFoundError:
        log.info(f"파일을 찾을 수 없습니다: '{filepath}'")
        return pd.DataFrame()

    # --- 1. 유사/중복된 '한글' 컬럼을 대표 '한글' 컬럼으로 통합 ---
    # col_map = { '대표 한글 컬럼명': ['통합될 유사/중복 한글 컬럼명들'] }
    col_map = {
        '판매가': ['가격'],
        '권장신장(cm)': ['권장신장(최적설계)'],
        '뒷 변속기': ['뒷변속기'],
        '등판(각도)능력': ['등판능력', '등펀(각도)능력'],
        '림': ['림 / 허브', '림/허브'],
        '무게': ['중량(kg)'],
        '변속 레버': ['변속레버'],
        '변속기': ['변속시스템'],
        '시트 포스트': ['시트포스트', '싵포스트'],
        '주행거리': ['주행거리(km)', '최대주행거리'],  # 대표 컬럼을 '주행거리'로 통일
        '최대적재중량': ['최대하중(kg)'],
        '크랭크': ['크랭크 세트', '크랭크세트', '크렝크 세트'],
        '타이어': ['타어이/튜브', '타이어/튜브'],
        '핸들그립': ['바 테잎', '핸드그립'],
        '핸들바': ['헨들바'],
        '핸들스템': ['핸들바/스템'],
        '카세트': ['프리휠'],
        '모터': ['리어허브 모터', '센터 모터', '프론트허브 모터']
    }

    for representative, synonyms in col_map.items():
        # 대표 컬럼이 원본에 없으면 건너뜀
        if representative not in df.columns: continue

        for syn in synonyms:
            if syn in df.columns:
                # 대표 컬럼의 비어있는 값을 동의어 컬럼의 값으로 채움
                df[representative].fillna(df[syn], inplace=True)
                # 사용된 동의어 컬럼은 삭제
                df.drop(columns=syn, inplace=True)

    # --- 2. 정리된 한글 컬럼명을 최종 영문명으로 일괄 변환 ---
    final_rename_map = {
        '연식': 'year', '카테고리': 'category', '제품명': 'model_name', '판매가': 'price_sale',
        '소비자가': 'price_consumer', '계기판': 'display', '권장신장(cm)': 'recommended_height',
        '뒷 변속기': 'derailleur_rear', '뒷 브레이크': 'brake_rear', '등판(각도)능력': 'climbing_ability',
        '디스플레이': 'display_info', '라이트 밝기': 'light_brightness', '모터': 'motor',
        '림': 'rims', '무게': 'weight', '반사등': 'reflector', '배터리': 'battery',
        '변속 레버': 'shifter', '변속기': 'gears_info', '브레이크': 'brakes', '색상': 'color',
        '스템': 'stem', '시트 포스트': 'seatpost', '시트클램프': 'seat_clamp', '안장': 'saddle',
        '앞 변속기': 'derailleur_front', '앞 브레이크': 'brake_front', '제품사이즈': 'product_size',
        '제품특징': 'features', '주행거리': 'max_distance', '주행방식': 'drive_type',
        '차체': 'frame', '프레임': 'frame', '체인': 'chain', '최고속도': 'max_speed',
        '최대적재중량': 'max_load', '카세트': 'cassette', '크랭크': 'crankset',
        '타이어': 'tires', '포크': 'fork', '핸들그립': 'grips', '핸들바': 'handlebar',
        '핸들스템': 'stem_handlebar', '허브': 'hubs', '휠셋': 'wheelset', '대표_이미지_URL': 'image_url',
        '상세_이미지_URL_리스트': 'image_urls_detail', '제품_상세_URL': 'product_url', '기타': 'other_specs'
    }

    df.rename(columns=final_rename_map, inplace=True)

    # --- 3. 최종 정리 ---
    # 중복된 영문 컬럼이 생성되었을 경우 첫 번째 것만 남김 (예: '차체'와 '프레임'이 모두 'frame'으로 변환된 경우)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 제품명이 없는 행 제거 및 중복 제거
    if 'model_name' in df.columns:
        df.dropna(subset=['model_name'], inplace=True)
        df.drop_duplicates(subset=['model_name'], keep='first', inplace=True)

    return df.reset_index(drop=True)


def preprocess_features(df):
    """각 컬럼을 머신러닝에 적합한 형태로 가공하는 함수 (모든 컬럼명 영문으로 사용)"""
    processed_df = df.copy()

    # --- 숫자형 ---
    # --- 숫자형: price 추출 강화 ---
    def extract_single_price(price_str):
        # 숫자와 소수점 외의 모든 문자 제거 후, 가장 앞쪽의 유효한 숫자만 반환
        if isinstance(price_str, str):
            # '일반셀-파스전용: 1,010,000원일반셀-스로틀겸용: 1,040,000원'과 같은 복잡한 문자열 처리
            match = re.search(r'[\d,]+', price_str.replace('원', '').split(' PAS')[0].split('스로틀')[0].split('파스')[0])
            if match:
                # 콤마 제거 후 숫자로 변환
                return int(match.group(0).replace(',', ''))
        return np.nan

    processed_df['price'] = processed_df.get('price_sale', pd.Series(dtype='str')).astype(str).apply(
        extract_single_price)

    processed_df['weight'] = pd.to_numeric(
        processed_df.get('weight', pd.Series(dtype='str')).astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    def get_height_median(s):
        nums = re.findall(r'\d+', str(s))
        if len(nums) >= 2:
            return (int(nums[0]) + int(nums[1])) / 2
        elif len(nums) == 1:
            return int(nums[0])
        return np.nan

    processed_df['height'] = processed_df.get('recommended_height', pd.Series(dtype='str')).apply(get_height_median)

    processed_df['motor_power'] = pd.to_numeric(
        processed_df.get('motor', pd.Series(dtype='str')).astype(str).str.extract(r'(\d+)W')[0], errors='coerce')
    processed_df['battery_capacity'] = pd.to_numeric(
        processed_df.get('battery', pd.Series(dtype='str')).astype(str).str.extract(r'(\d+\.?\d*)Ah')[0],
        errors='coerce')

    # --- 범주형 ---
    # get() 안에 영문 컬럼명 사용
    processed_df['gears'] = processed_df.get('gears_info', pd.Series(dtype='str')).str.extract(r'(\d+)').fillna(
        '0').astype(str) + '단'

    def get_brake_type(s):
        s = str(s).lower()
        if '유압' in s: return '유압식디스크'
        if '기계식' in s: return '기계식디스크'
        if '디스크' in s: return '디스크'
        if 'v-브레이크' in s: return 'V브레이크'
        if '캘리퍼' in s: return '캘리퍼'
        return '기타'

    processed_df['brake_type'] = processed_df.get('brakes', pd.Series(dtype='str')).apply(get_brake_type)

    def get_frame_material(s):
        s = str(s).lower()
        if '카본' in s: return '카본'
        if 'dp780' in s: return 'DP780'
        if '크로몰리' in s: return '크로몰리'
        if '알루미늄' in s: return '알루미늄'
        if '스틸' in s: return '스틸'
        return '기타'

    processed_df['frame_material'] = processed_df.get('frame', pd.Series(dtype='str')).apply(get_frame_material)

    return processed_df


# --- 랭킹용 종합 스펙 점수 함수 ---
def calculate_total_spec_score(row):
    total_score = 0;
    category = str(row.get('category', '')).lower()
    spec_text = ' '.join(str(s) for s in row.values).lower()

    # 1. 프레임 재질
    frame_scores = {'카본': 200, 'DP780': 170, '크로몰리': 150, '알루미늄': 100, '스틸': 50, '기타': 10}
    for material, score in frame_scores.items():
        if material in str(row.get('frame_material', '')).lower(): total_score += score * 1.5; break

    # 2. 브레이크 (유압식 디스크에 가중치 부여)
    brake_scores = {'유압식디스크': 100, '기계식디스크': 50, '디스크': 25, 'v브레이크': 15, '캘리퍼': 10, '기타': 5}

    brake_type = str(row.get('brake_type', '')).lower()
    for b_type, score in brake_scores.items():
        if b_type in brake_type: total_score += score * 1.0; break

    # for brake_type, score in brake_scores.items():
    #     if brake_type in str(row.get('brake_type', '')).lower(): total_score += score * 1.0; break

    # 3. 구동계 단수 (저가 모델에서는 등급보다 단수가 중요)
    gears_info = str(row.get('gears_info', ''))
    # 21단 이상의 기어는 출퇴근 시 경사로 대응에 유리하므로 보너스
    if '21단' in gears_info or '24단' in gears_info:
        total_score += 100
    elif '7단' in gears_info:
        total_score += 50

    # 4. 신장 적합성 보너스 (180cm는 700C 휠에 가장 큰 점수 부여)
    frame_info = str(row.get('frame', ''))
    if '700C' in frame_info:
        total_score += 150  # 180cm에게 가장 적합한 휠 크기
    elif '26' in frame_info or '27.5' in frame_info:
        total_score += 50

    if '전기' in category:
        motor_power = row.get('motor_power', 0);
        battery_capacity = row.get('battery_capacity', 0)
        if pd.notna(motor_power): total_score += motor_power * 0.5
        if pd.notna(battery_capacity): total_score += battery_capacity * 15
        total_score += (int(re.sub(r'\D', '', str(row.get('gears', '0')))) * 0.5)
    else:
        drivetrain_scores = {'dura-ace': 120, 'xtr': 120, 'ultegra': 110, 'xt': 110, '105': 100, 'slx': 100,
                             'tiagra': 80, 'deore': 80, 'sora': 70, 'alivio': 70, 'claris': 60, 'acera': 60,
                             'tourney': 40, 'altus': 40, '시마노': 20}
        drivetrain_max_score = 0
        for grade, score in drivetrain_scores.items():
            if grade in spec_text: drivetrain_max_score = max(drivetrain_max_score, score)
        total_score += drivetrain_max_score * 2.0
        weight = row.get('weight', 15)
        if pd.notna(weight) and weight > 0: total_score += max(0, (15 - weight) * 5)
    return round(total_score, 1)


# --- 새로워진 5각형 스탯 점수 계산 함수 (유지보수 편의성 적용) ---
def calculate_stat_scores_final(df):
    stat_df = pd.DataFrame(index=df.index)

    def rank_to_5_score(series, ascending=True):
        if series.isnull().all() or len(series.unique()) <= 1:
            return pd.Series(3, index=series.index, dtype=int)
        ranks_pct = series.rank(method='first', ascending=ascending, pct=True, na_option='bottom')
        scores = 6 - np.ceil(ranks_pct * 5)
        return scores.astype(int)

    # 1. 종합 스펙 (Overall)
    stat_df['stat_overall'] = df.groupby('category')['spec_score'].transform(rank_to_5_score, ascending=False)

    # 2. 가성비 (Value for Money)
    value_for_money = df['spec_score'] / df['price'].replace(0, np.nan)
    stat_df['stat_value'] = df.groupby('category').apply(
        lambda group: rank_to_5_score(value_for_money.loc[group.index], ascending=False)
    ).reset_index(level=0, drop=True).sort_index()

    # 3. 제동력 (Braking)
    brake_scores_map = {'유압식디스크': 5, '기계식디스크': 4, '디스크': 3, 'V브레이크': 2, '캘리퍼': 2, '기타': 1}
    stat_df['stat_braking'] = df['brake_type'].map(brake_scores_map).fillna(1).astype(int)

    # 4. 유지보수 편의성 (Maintenance)
    def get_maintenance_score(row):
        score = 3  # 기본 3점 시작
        gears_info = str(row.get('gears_info', '')).lower()
        brake_type = str(row.get('brake_type', '')).lower()
        fork_info = str(row.get('fork', '')).lower()

        # 기어 시스템
        gears_num = int(re.sub(r'\D', '', str(row.get('gears', '0'))))
        if '싱글' in gears_info or '내장' in gears_info or gears_num <= 1:
            score += 2  # 고장 요소 적음
        elif gears_num >= 20:  # 10x2, 11x2 등 고단수
            score -= 1  # 정밀 세팅 필요

        # 브레이크 시스템
        if 'v브레이크' in brake_type or '캘리퍼' in brake_type:
            score += 1  # 자가 정비 용이
        elif '유압식' in brake_type:
            score -= 1  # 전문 정비 필요

        # 서스펜션
        if '서스펜션' in fork_info:
            score -= 1  # 추가 관리 요소

        # 최종 점수는 1~5점 사이로 제한
        return np.clip(score, 1, 5)

    stat_df['stat_maintenance'] = df.apply(get_maintenance_score, axis=1)

    # 5. 특화 성능 (X-Factor)
    def get_xfactor_raw_score(row):
        category = str(row.get('category', '')).lower()
        if '전기' in category:
            return row.get('battery_capacity', 0) or 0
        else:
            frame_scores = {'카본': 100, 'DP780': 60, '크로몰리': 50, '알루미늄': 40, '스틸': 20, '기타': 10}
            return frame_scores.get(row.get('frame_material', '기타'), 10)

    xfactor_raw_scores = df.apply(get_xfactor_raw_score, axis=1)
    df['is_electric'] = df['category'].str.contains('전기', na=False)
    stat_df['stat_xfactor'] = df.groupby('is_electric').apply(
        lambda group: rank_to_5_score(xfactor_raw_scores.loc[group.index], ascending=False)
    ).reset_index(level=0, drop=True).sort_index()
    df.drop(columns=['is_electric'], inplace=True, errors='ignore')

    return stat_df


def recommend_alton_bikes(usage=None, budget_min=None, budget_max=None, height=None, base_bike_title=None, top_n=5,
                          ranking_method='spec_score', df_processed=None, indices=None):
    """
    조건 완화 전략과 모든 랭킹 기능을 포함한 최종 추천 함수.
    """
    if 'df_processed' not in globals() or df_processed.empty:
        return "추천 시스템이 준비되지 않았습니다."

    log.info("\n--- 조건 기반 후보 탐색 시작 ---")
    purpose_map = {"출퇴근": "하이브리드|폴딩|미니벨로|전기", "운동": "로드|MTB|하이브리드", "여행": "하이브리드|전기|MTB",
                   "산악": "MTB|산악", "로드": "로드"}
    if usage not in purpose_map: return f"'{usage}'는 지원하지 않는 용도입니다."
    base_df = df_processed[df_processed['category'].str.contains(purpose_map[usage], na=False)].copy()

    # 조건 완화 로직
    # 1단계: 모든 조건 만족
    candidate_groups = {}
    strict_df = base_df.copy()
    if budget_min: strict_df = strict_df[strict_df['price'] >= budget_min]
    if budget_max: strict_df = strict_df[strict_df['price'] <= budget_max]
    if height: strict_df = strict_df[(strict_df['height'] >= height - 7) & (strict_df['height'] <= height + 7)]
    candidate_groups[1] = strict_df
    log.info(f"  - 1단계 (모든 조건 만족): {len(strict_df)}개 발견")

    # 2단계: 키 조건 완화
    if len(strict_df) < top_n:
        height_relaxed_df = base_df.copy()
        if budget_min: height_relaxed_df = height_relaxed_df[height_relaxed_df['price'] >= budget_min]
        if budget_max: height_relaxed_df = height_relaxed_df[height_relaxed_df['price'] <= budget_max]
        height_relaxed_df = height_relaxed_df.drop(strict_df.index, errors='ignore')
        candidate_groups[2] = height_relaxed_df
        log.info(f"  - 2단계 (키 조건 완화): 추가 {len(height_relaxed_df)}개 발견")

    # 3단계: 예산 조건 완화
    total_found = sum(len(df) for df in candidate_groups.values())
    # 최대 예산이 있을 때만 완화
    if total_found < top_n and budget_max:
        budget_relaxed_df = base_df.copy()
        # 최대 예산을 20% 초과 허용, 최소 예산은 유지
        budget_relaxed_df = budget_relaxed_df[budget_relaxed_df['price'] <= budget_max * 1.2]
        if budget_min: budget_relaxed_df = budget_relaxed_df[budget_relaxed_df['price'] >= budget_min]
        if height: budget_relaxed_df = budget_relaxed_df[
            (budget_relaxed_df['height'] >= height - 7) & (budget_relaxed_df['height'] <= height + 7)]
        existing_indices = pd.concat(candidate_groups.values()).index
        budget_relaxed_df = budget_relaxed_df.drop(existing_indices, errors='ignore')
        candidate_groups[3] = budget_relaxed_df
        log.info(f"  - 3단계 (최대 예산 20% 초과 허용): 추가 {len(budget_relaxed_df)}개 발견")

    all_candidates = []
    match_scores = {1: 1.0, 2: 0.7, 3: 0.5}
    for level, group_df in candidate_groups.items():
        if not group_df.empty:
            group_df['match_score'] = match_scores.get(level, 0.1)
            all_candidates.append(group_df)
    if not all_candidates: return "조건에 맞는 자전거를 전혀 찾을 수 없습니다."
    final_candidates_df = pd.concat(all_candidates)

    # 랭킹 로직
    if not (base_bike_title and base_bike_title in indices):
        log.info("\n 기준 자전거 없음. '종합 점수(스펙x조건부합)' 기반 랭킹 적용")
        max_spec, min_spec = final_candidates_df['spec_score'].max(), final_candidates_df['spec_score'].min()
        if max_spec > min_spec:
            final_candidates_df['spec_score_scaled'] = (final_candidates_df['spec_score'] - min_spec) / (
                        max_spec - min_spec)
        else:
            final_candidates_df['spec_score_scaled'] = 0.5
        final_candidates_df['final_score'] = (final_candidates_df['spec_score_scaled'] * 0.6) + (
                    final_candidates_df['match_score'] * 0.4)
        recommend_df = final_candidates_df.sort_values(by='final_score', ascending=False).head(top_n).copy()
        score_type = '종합 점수'
        score_value = recommend_df['final_score'].round(3)
    else:
        log.info("\n 기준 자전거 기반 유사도 랭킹 적용")
        base_idx = indices[base_bike_title]

        # 후보군 내에서만 유사도 계산
        final_candidates_df['similarity'] = final_candidates_df.index.map(lambda x: similarity_matrix[base_idx][x])

        # 최종 점수 = 유사도(가중치 0.6) + 조건부합점수(가중치 0.4)
        final_candidates_df['final_score'] = (final_candidates_df['similarity'] * 0.6) + (
                    final_candidates_df['match_score'] * 0.4)

        recommend_df = final_candidates_df.sort_values(by='final_score', ascending=False).head(top_n).copy()

        # 핵심 수정 부분
        score_type = '종합 점수'
        score_value = recommend_df['final_score'].round(3)
        # 유사도 컬럼을 새로 만들고, final_score 계산에 사용된 similarity 값을 할당
        recommend_df['similarity'] = recommend_df['similarity'].round(3)

    if 'score_type' in recommend_df.columns: recommend_df.drop('score_type', axis=1, inplace=True)
    if 'score' in recommend_df.columns: recommend_df.drop('score', axis=1, inplace=True)
    recommend_df['score_type'] = score_type
    recommend_df['score'] = score_value

    if recommend_df.columns.duplicated().any():
        recommend_df = recommend_df.loc[:, ~recommend_df.columns.duplicated(keep='first')]

    # 결과 정리 시 'stat_lightness'를 'stat_maintenance'로 변경
    stat_cols = ['stat_value', 'stat_overall', 'stat_braking', 'stat_maintenance', 'stat_xfactor']
    display_cols = ['model_name', 'category', 'price', 'score_type', 'score', 'match_score', 'similarity'] + stat_cols

    # recommend_df.to_excel(f'./output/{usage}_{budget_max}_{height}_recommend_df.xlsx')

    return recommend_df.reindex(columns=display_cols)

# 2-1. '판매가' 및 '무게' 전처리 함수 정의
def clean_price(price_str):
    if isinstance(price_str, str):
        # [수정] '원', ',', ' ' 및 기타 문자를 모두 제거하고 숫자만
        digits_only = re.sub(r'\D', '', price_str)
        if digits_only:
            try:
                return float(digits_only)
            except ValueError:
                return np.nan
    return np.nan

def clean_weight(weight_str):
    if isinstance(weight_str, str):
        # 'kg', 'Kg', 'KG', ' ', '약' 등의 문자 제거
        # 숫자와 소수점(.)만 추출 (혹은 첫 번째 숫자 그룹만 추출)
        match = re.search(r'\d+(\.\d+)?', weight_str)
        if match:
            try:
                # 간혹 '11.5. ' 처럼 .이 여러개 있는 경우 방지
                # 숫자와 .으로 이루어진 첫번째 유효한 숫자열을 찾도록 좀 더 정교하게
                # 첫 번째 유효한 부동소수점 숫자를 찾음.
                match = re.search(r'\d+(\.\d+)?', weight_str)
                if match:
                    return float(match.group(0))
                else:
                    return np.nan
                match = re.search(r'\d+(\.\d+)?', weight_str)
            except ValueError:
                return np.nan
    return np.nan

# 2-4. '권장신장(최적설계)' 파싱 및 Imputation (결측치 채우기)
def parse_height(height_str):
    if isinstance(height_str, str):
        match = re.search(r'\D*(\d+).*', height_str) # 문자열의 첫 번째 숫자 추출
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
    return np.nan

# 5-2. [1/2번 화면]
def recommend_by_query(user_budget_min, user_budget_max, user_height, user_purpose, top_n, df, cosine_sim, purpose_map):

    # 1. 필터링
    if user_purpose not in purpose_map:
        return f"'{user_purpose}'는 유효한 용도가 아닙니다. (선택: {list(purpose_map.keys())})"

    allowed_categories = purpose_map[user_purpose].split('|')
    df_filtered = df[df['카테고리'].isin(allowed_categories)].copy()

    if df_filtered.empty: return f"'{user_purpose}' 용도에 맞는 카테고리의 자전거가 없습니다."

    df_filtered = df_filtered[
        (df_filtered['판매가_clean'] >= user_budget_min) &
        (df_filtered['판매가_clean'] <= user_budget_max)
    ]

    if df_filtered.empty: return f"'{user_purpose}' 용도 + {user_budget_min}원 ~ {user_budget_max}원 예산에 맞는 자전거가 없습니다."

    height_range = 15.0 # (최소신장 ~ 최소신장+15) 범위로 가정
    df_filtered = df_filtered[
        (df_filtered['신장_clean'] <= user_height) &
        (user_height <= (df_filtered['신장_clean'] + height_range))
    ]

    if df_filtered.empty: return f"'{user_purpose}' 용도 + 예산 + 키 {user_height}cm에 맞는 자전거가 없습니다."

    # 2. 랭킹 (AI-Based Ranking)
    candidate_indices = df_filtered.index.tolist()

    # if len(candidate_indices) == 1:
    #     return df_filtered

    sim_subset = cosine_sim[np.ix_(candidate_indices, candidate_indices)]
    avg_similarity_to_group = sim_subset.mean(axis=1)
    rank_scores = pd.Series(avg_similarity_to_group, index=candidate_indices)
    df_filtered['AI_Rank_Score'] = rank_scores

    # 3. 결과 반환
    df_ranked = df_filtered.sort_values(by='AI_Rank_Score', ascending=False)
    return df_ranked.head(top_n)

# 5-3. [3번 화면]
def get_recommendations_from_selection(selected_titles_list, top_n, df, indices, cosine_sim):
    """
    2번 화면에서 선택한 자전거 리스트(1~4개)를 기반으로
    '평균 유사도'가 가장 높은 새로운 자전거 4개를 추천합니다.
    """

    # 1. 선택한 자전거들의 인덱스 찾기
    try:
        # 리스트에 없는 자전거가 포함될 경우를 대비
        valid_titles = [title for title in selected_titles_list if title in indices]
        if not valid_titles:
            return "선택한 자전거를 찾을 수 없습니다."

        group_indices = indices[valid_titles].tolist()
        # 1개만 선택되어도 list로 만듦
        if isinstance(group_indices, (int, np.integer)):
            group_indices = [group_indices]
    except Exception as e:
        return f"인덱스 검색 오류: {e}"

    # 2. 이 그룹의 '평균 유사도' 계산
    # (예: 2개 선택시) 2개 자전거의 유사도 벡터를 가져와서 평균을 냄
    group_sim_vectors = cosine_sim[group_indices, :]
    avg_group_sim = group_sim_vectors.mean(axis=0)

    # 3. 평균 유사도 점수를 (인덱스, 점수) 리스트로 변환
    sim_scores = list(enumerate(avg_group_sim))

    # 4. 점수 기준으로 내림차순 정렬
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 5. 추천 리스트 생성 (자기 자신 제외)
    recommended_indices = []
    for idx, score in sorted_scores:
        if idx not in group_indices: # 이미 선택한 자전거는 제외
            recommended_indices.append(idx)

        if len(recommended_indices) == top_n:
            break # top_n 개수만큼 채우면 중단

    # 6. 결과 반환 (모든 열 포함)
    result_df = df.iloc[recommended_indices].copy()

    # 결과 DataFrame에 '유사도 점수(%)' 추가
    final_scores = [avg_group_sim[i] * 100 for i in recommended_indices]
    result_df['유사도(%)'] = [round(s, 2) for s in final_scores]

    return result_df


def score_braking(row):
    # Combine brake related columns text for search
    brake_text = (
                str(row.get('브레이크', '')) + " " + str(row.get('뒷 브레이크', '')) + " " + str(row.get('앞 브레이크', ''))).lower()

    if '유압' in brake_text or 'hydraulic' in brake_text or 'mt200' in brake_text:
        return 5
    elif '기계식' in brake_text or 'mechanical' in brake_text or 'disc' in brake_text or '디스크' in brake_text:
        return 4
    elif '듀얼피봇' in brake_text or 'dual pivot' in brake_text or 'v-brake' in brake_text or 'v-브레이크' in brake_text or '브이' in brake_text:
        return 3
    elif '캘리퍼' in brake_text or 'caliper' in brake_text:
        # Single pivot assumed if not specified as dual, but usually acceptable for road
        if '싱글' in brake_text: return 2
        return 3
    elif '밴드' in brake_text or 'band' in brake_text or '드럼' in brake_text:
        return 2
    else:
        # Default for unspecified or coaster
        return 2


def score_convenience(row):
    cat = str(row.get('카테고리', '')).lower()
    desc = str(row.get('기타', '')) + str(row.get('제품특징', ''))

    score = 3  # Base score

    # Electric bikes are convenient for riding effort
    if '전기' in cat or '스마트모빌리티' in cat:
        score += 1.5

    # Folding is convenient for storage
    if '폴딩' in cat or '접이식' in desc or 'folding' in desc:
        score += 1

    # Utility features
    if '바구니' in desc or '짐받이' in desc:
        score += 0.5

    # Penalty for aggressive geometry (less convenient for casuals)
    if '로드' in cat or '픽시' in cat:
        score -= 1.5

    # Cap at 5, floor at 1
    return max(1, min(5, round(score)))


def score_performance(row):
    cat = str(row.get('카테고리', '')).lower()
    frame = str(row.get('프레임', '')).lower()
    gear = (str(row.get('변속기', '')) + str(row.get('뒷 변속기', ''))).lower()
    desc = str(row.get('기타', '')) + str(row.get('제품특징', ''))

    score = 3  # Base

    # Material
    if '카본' in frame or 'carbon' in frame:
        score += 1.5
    elif '티타늄' in frame:
        score += 2
    elif '스틸' in frame and '크로몰리' not in frame:
        score -= 1

    # Components (Hierarchy check)
    if 'xtr' in gear or 'dura-ace' in gear:
        score += 2
    elif 'xt' in gear or 'ultegra' in gear or '울테그라' in gear:
        score += 1.5
    elif '105' in gear or 'slx' in gear:
        score += 1
    elif 'deore' in gear or '데오레' in gear or 'tiagra' in gear or '티아그라' in gear:
        score += 0.5
    elif 'sora' in gear or '소라' in gear:
        score += 0.5
    elif 'tourney' in gear or '투어니' in gear or '생활' in cat:
        score -= 0.5

    # Electric boost
    if '전기' in cat or '스마트모빌리티' in cat:
        if '500w' in desc:
            score += 1.5
        else:
            score += 1

    return max(1, min(5, round(score)))


def score_maintenance(row):
    # Simpler = Better maintenance score (Easy to fix)
    cat = str(row.get('카테고리', '')).lower()
    brake_text = (str(row.get('브레이크', '')) + " " + str(row.get('뒷 브레이크', ''))).lower()
    desc = str(row.get('기타', '')).lower()

    score = 3  # Base

    # Electric components are harder to maintain
    if '전기' in cat or '스마트모빌리티' in cat:
        score -= 1.5

    # Hydraulic brakes require bleeding (harder than cable)
    if '유압' in brake_text or 'hydraulic' in brake_text:
        score -= 0.5

    # Full suspension involves more pivots/shocks
    if 'fs' in str(row.get('제품명', '')) or '풀 서스펜션' in desc:
        score -= 0.5

    # Pixie/Single gear is very simple
    if '픽시' in cat:
        score += 2

    # V-brakes are easy to adjust
    if 'v-브레이크' in brake_text or 'v-brake' in brake_text:
        score += 1

    return max(1, min(5, round(score)))


def score_value(row):
    # Heuristic: Performance per Price
    # Avoid division by zero
    price = row['판매가_clean']
    if price == 0:
        return 3  # Neutral if price unknown

    perf = row['성능']
    brake = row['제동력']

    # Simple Ratio: (Performance + Brake) / Price
    # We need to normalize this.
    # Low price (e.g. 200k) with score 3+3=6 -> ratio high
    # High price (e.g. 2m) with score 5+5=10 -> ratio low

    # Let's categorize price brackets
    if price < 300000:
        price_score = 5
    elif price < 600000:
        price_score = 4
    elif price < 1200000:
        price_score = 3
    elif price < 2500000:
        price_score = 2
    else:
        price_score = 1

    # Compare specs to price bracket
    # If spec is high for the bracket, bonus.
    spec_avg = (perf + brake) / 2

    # Adjust matrix
    val_score = 3
    if price_score >= 4 and spec_avg >= 3:
        val_score = 5  # Cheap but good
    elif price_score >= 4 and spec_avg < 2:
        val_score = 3  # Cheap and bad
    elif price_score == 3 and spec_avg >= 4:
        val_score = 5  # Mid price, great specs
    elif price_score == 3 and spec_avg == 3:
        val_score = 4
    elif price_score <= 2 and spec_avg >= 4.5:
        val_score = 4  # Expensive but top tier
    elif price_score <= 2 and spec_avg < 4:
        val_score = 2  # Expensive and mid specs

    return val_score

# ============================================
# 주요 설정
# ============================================
env = 'local'
serviceName = 'LSH0627'
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

    # 입력 데이터
    'csvFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_prd.csv',
    # 'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v2.xlsx',
    # 'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v3.xlsx',
    'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v4.xlsx',

    # 설정 정보
    'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'cfgKey': 'gemini-api-key',
    'cfgVal': 'oper',
    # 'cfgVal': 'local',
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

# Gemini API키
config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')
apiKey = config.get(sysOpt['cfgKey'], sysOpt['cfgVal'])
client = genai.Client(api_key=apiKey)

# 설정 파일
try:
    csvFile = sysOpt['csvFile']
    csvList = sorted(glob.glob(csvFile))
    if csvList is None or len(csvList) < 1:
        log.error(f'설정 파일 없음, csvFile : {csvFile}')
        exit(1)

    csvInfo = csvList[0]
    csvData = pd.read_csv(csvInfo)

    # 주요 전처리
    csvData['title'] = csvData['title'].str.replace('<[^>]*>', '', regex=True).str.strip()
    csvData['isDlPrd'] = csvData['dlPrd'].notna()
    csvData['isMlPrd'] = csvData['mlPrd'].notna()
    csvData['isLprice'] = csvData['lprice'].notna()
    csvData['typeByTitle'] = csvData['typeByTitle'].replace({'전기자전거': '전기', '일반자전거': '일반'})
    csvData['brandByTitle'] = pd.Categorical(csvData['brandByTitle'], categories=['알톤 자전거', '삼천리 자전거', '스마트 자전거', '기타'], ordered=True)
    csvData['typeByTitle'] = pd.Categorical(csvData['typeByTitle'], categories=['전기', '하이브리드', 'MTB', '사이클', '일반', '미니벨로'], ordered=True)

    prdColList = ['mlPrd', 'dlPrd', 'lprice']
    meanDataByTitle = csvData.groupby('title')[prdColList].mean()
    diffMl = (meanDataByTitle['mlPrd'] - meanDataByTitle['lprice']).abs()
    diffDl = (meanDataByTitle['dlPrd'] - meanDataByTitle['lprice']).abs()
    # meanDataByTitle['maxPrdDiff'] = np.maximum(diffMl, diffDl)
    meanDataByTitle['minPrdDiff'] = np.minimum(diffMl, diffDl)

    minDataByTitle = csvData.groupby('title')[prdColList].min()
    minDataByTitle['minPrd'] = np.minimum( minDataByTitle['mlPrd'], minDataByTitle['dlPrd'])


    # tmpData = csvData.copy()
    # tmpData = pd.merge(csvData.copy(), meanDataByTitle, on='title', how='left')
    # tmpDataL1 = tmpData.sort_values(by=['maxPrdDiff', 'title'], ascending=[False, False])
    # csvDataL1 = tmpDataL1.drop_duplicates(subset=['title'], keep='first')

    tmpData = pd.merge(csvData.copy(), meanDataByTitle[['minPrdDiff']], on='title', how='left')
    tmpDataL1 = pd.merge(tmpData, minDataByTitle[['minPrd']], on='title', how='left')
    tmpDataL2 = tmpDataL1.sort_values(by=['minPrdDiff', 'title'], ascending=[True, True])
    tmpDataL3 = tmpDataL2[tmpDataL2['minPrd'] > 0].reset_index(drop=True)
    csvDataL1 = tmpDataL3.drop_duplicates(subset=['title'], keep='first')
except Exception as e:
    log.error(f'설정 파일 실패, csvFile : {csvFile} : {e}')
    exit(1)

# 입력 파일
try:
    inpFile = sysOpt['inpFile']
    inpList = sorted(glob.glob(inpFile))
    if inpList is None or len(inpList) < 1:
        log.error(f'입력 파일 없음, inpFile : {inpFile}')
        exit(1)

    df = pd.read_excel(inpList[0])
    df = df[df['연식'] >= 2023].reset_index(drop=True)

    # 2-2. 전처리 적용
    df['판매가_clean'] = df['판매가'].apply(clean_price)
    df['무게_clean'] = df['무게'].apply(clean_weight)

    # 2-3. '판매가', '무게' 결측치 채우기 (중간값)
    price_median = df['판매가_clean'].median()
    df['판매가_clean'] = df['판매가_clean'].fillna(price_median)

    weight_median = df['무게_clean'].median()
    df['무게_clean'] = df['무게_clean'].fillna(weight_median)

    df['신장_clean'] = df['권장신장(최적설계)'].apply(parse_height)
    df['신장_clean'] = df['신장_clean'].fillna(df.groupby('카테고리')['신장_clean'].transform('median'))
    total_height_median = df['신장_clean'].median()
    if pd.isna(total_height_median):
        total_height_median = 165.0  # Fallback
    df['신장_clean'] = df['신장_clean'].fillna(total_height_median)
    # print("... '신장_clean' 파싱 및 결측치 채우기 완료")

    # --- 3단계: CBF 피처 생성 (숫자 + 텍스트) ---
    # print("\n--- (3/6) CBF 피처 생성 시작 ---")

    # 3-1. 숫자 피처 스케일링
    numerical_cols = ['판매가_clean', '무게_clean', '신장_clean']
    scaler = MinMaxScaler()
    numerical_scaled = scaler.fit_transform(df[numerical_cols])
    numerical_sparse = scipy.sparse.csr_matrix(numerical_scaled)

    # 3-2. 텍스트 피처 생성 및 벡터화
    spec_cols = [
        '카테고리', '제품명', '뒷 변속기', '림', '변속 레버',
        '브레이크', '스템', '시트포스트', '안장', '크랭크 세트',
        '타이어/튜브', '포크', '핸들바', '프리휠'
    ]
    for col in spec_cols:
        df[col] = df[col].fillna('')  # NaN을 빈 문자열로
    df['features_text'] = df[spec_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    df['제동력'] = df.apply(score_braking, axis=1)
    df['편의성'] = df.apply(score_convenience, axis=1)
    df['성능'] = df.apply(score_performance, axis=1)
    df['유지보수'] = df.apply(score_maintenance, axis=1)
    df['가성비'] = df.apply(score_value, axis=1)

    log.info(df[['제동력', '편의성', '성능', '유지보수', '가성비']].describe().to_markdown())

    tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
    text_matrix = tfidf.fit_transform(df['features_text'])

    # 3-3. 피처 결합
    combined_features = hstack([numerical_sparse, text_matrix])
    # print("... 최종 피처 행렬 결합 완료 (Shape: {})".format(combined_features.shape))

    # --- 4단계: 코사인 유사도 계산 (Bike-to-Bike) ---
    # print("\n--- (4/6) 코사인 유사도 계산 시작 ---")
    cosine_sim = cosine_similarity(combined_features)
    # print("... 코사인 유사도 계산 완료 (Shape: {})".format(cosine_sim.shape))

    # purpose_map_fixed = {
    #     "출퇴근": "하이브리드|폴딩/미니벨로|전기자전거|씨티",
    #     "운동": "로드|컴포트 산악자전거|하이브리드",
    #     "여행": "하이브리드|전기자전거|컴포트 산악자전거",
    #     "산악": "컴포트 산악자전거|MTB",
    #     "로드": "로드"
    # }
    purpose_map_fixed = {
        "출퇴근": "하이브리드|폴딩/미니벨로|전기자전거|씨티|픽시|주니어",
        "운동": "로드|컴포트 산악자전거|하이브리드|픽시|주니어|키즈",
        "여행": "하이브리드|전기자전거|컴포트 산악자전거|로드",
        "산악": "컴포트 산악자전거",
        "로드": "로드|픽시"
    }

except Exception as e:
    log.error(f'입력 파일 실패, inpFile : {inpFile} : {e}')
    exit(1)

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-brandModel", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-brandModel")
async def selBrandModel(
    year: str = Form(..., description='연식 (최소-최대)', examples=['2015-2025']),
    limit: int = Form(..., description="1쪽당 개수", examples=[10]),
    page: int = Form(..., description="현재 쪽", examples=[1]),
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 시세 조회하기 - 브랜드/모델명 목록 조회\n
    테스트\n
        year: 연식 (최소-최대)\n
        limit: 1쪽당 개수\n
        page: 현재 쪽\n
    """
    try:
        # year = request.year
        minYear, maxYear  = year.split('-')
        if year is None or len(year) < 1 or minYear is None or maxYear is None:
            return resResponse("fail", 400, f"연식 없음, year : {year}", None)

        # page = request.page
        if page is None:
            return resResponse("fail", 400, f"현재 쪽 없음, page : {page}")

        # limit = request.limit
        if limit is None:
            return resResponse("fail", 400, f"1쪽당 개수 없음, limit : {limit}")

        selData = csvDataL1.loc[
            (csvDataL1['yearByTitle'] >= float(minYear)) & (csvDataL1['yearByTitle'] <= float(maxYear))
            ]

        if len(selData) < 1:
            return resResponse("fail", 400, f"브랜드/모델명 없음", None)

        selDataL2 = selData.groupby(['brandByTitle', 'typeByTitle'], observed=False)['title'].apply(list)
        result = getPageDict(selDataL2, page=page, limit=limit)
        # log.info(f"result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-prd", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-prd")
async def selPrd(
    year: str = Form(..., description='연식 (최소-최대)', examples=['2015-2025']),
    brandModel: str = Form(..., description='자전거 모델', examples=['2020 알톤 스로틀 FS 전기자전거 앞뒤 서스펜션 20인치 미니벨로']),
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 시세 조회하기 - 브랜드/모델명의 AI시세 결과 조회\n
    테스트\n
        year: 연식 (최소-최대)\n
        brandModel: 자전거 모델\n
    """
    try:
        # year = request.year
        minYear, maxYear  = year.split('-')
        if year is None or len(year) < 1 or minYear is None or maxYear is None:
            return resResponse("fail", 400, f"연식 없음, year : {year}", None)

        # brandModel = request.brandModel
        if brandModel is None or len(brandModel) < 1:
            return resResponse("fail", 400, f"자전거 모델 없음, brandModel : {brandModel}")

        selData = csvData.loc[
            (csvData['yearByTitle'] >= float(minYear)) & (csvData['yearByTitle'] <= float(maxYear))
            & (csvData['title'] == brandModel)
            ].sort_values(by='dtDate', ascending=False)

        if len(selData) < 1:
            return resResponse("fail", 400, f"데이터 없음", None)

        jsonData = selData.to_json(orient='records')
        result = json.loads(jsonData)
        # log.info(f"result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-chatModelCont", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-chatModelCont")
async def selChatModelCont(
    chatModel: str = Form(..., description='생성형 AI 종류', examples=['gemini-2.5-flash'], enum=['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']),
    cont: str = Form(..., description='비교 리포트 (종합 성능, 상세 스펙, 종합 분석)', examples=['자전거 비교 리포트 (종합 성능, 상세 스펙, 종합 분석)']),
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 맞춤 자전거 찾기 - AI 비교 리포트 헬퍼\n
    테스트\n
        chatModel: 생성형 AI 종류 (gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite)\n
        cont: 비교 리포트 (종합 성능, 상세 스펙, 종합 분석)\n
    """
    try:
        # model = request.model
        if chatModel is None or len(chatModel) < 1:
            return resResponse("fail", 400, f"생성형 AI 모델 종류 없음, chatModel : {chatModel}")

        # cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항 없음, cont : {cont}")

        contTemplate = '''
            %cont%
           '''
        contents = contTemplate.replace('%cont%', cont)
        log.info(f"contents : {contents}")

        response = client.models.generate_content(
            model=chatModel,
            contents=contents
        )
        result = response.text
        # log.info(f"result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


# # @app.post(f"/api/sel-rcmd", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-rcmd")
# async def selRcmd(
#     usage: str = Form(..., description='용도', examples=['출퇴근'], enum=['출퇴근', '운동', '여행', '산악', '로드']),
#     budget: str = Form(..., description='예산 (최소-최대)', examples=['80000-500000']),
#     height: str = Form(..., description='키', examples=['170']),
# ):
#     """
#     기능\n
#         알톤 바이크메트릭스AI - AI 맞춤 자전거 찾기\n
#     테스트\n
#         usage: 용도 (출퇴근, 운동, 여행, 산악, 로드)\n
#         budget: 예산 (최소-최대)\n
#         height: 키\n
#     """
#     try:
#         minBudget, maxBudget  = budget.split('-')
#         if budget is None or len(budget) < 1 or minBudget is None or maxBudget is None:
#             return resResponse("fail", 400, f"예산 없음, budget : {budget}", None)
#
#         if usage is None or len(usage) < 1:
#             return resResponse("fail", 400, f"용도 없음, usage : {usage}")
#
#         if height is None or len(height) < 1:
#             return resResponse("fail", 400, f"키 없음, height : {height}")
#
#         recommendations = recommend_alton_bikes(
#             usage=usage,
#             budget_min=float(minBudget),
#             budget_max=float(maxBudget),
#             height=float(height),
#             ranking_method='spec_score',
#             top_n=4,
#             df_processed=df_processed,
#             indices=indices
#         )
#
#         selData = pd.merge(recommendations, df_std, on=['model_name'], how='left')
#
#         if len(selData) < 1:
#             return resResponse("fail", 400, f"데이터 없음", None)
#
#         jsonData = selData.to_json(orient='records')
#         result = json.loads(jsonData)
#         # log.info(f"result : {result}")
#
#         return resResponse("succ", 200, "처리 완료", len(result), result)
#
#     except Exception as e:
#         log.error(f'Exception : {e}')
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-rcmd2", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-rcmd2")
async def selRcmd(
    usage: str = Form(..., description='용도', examples=['출퇴근'], enum=['출퇴근', '운동', '여행', '산악', '로드']),
    budget: str = Form(..., description='예산 (최소-최대)', examples=['80000-500000']),
    height: str = Form(..., description='키', examples=['170']),
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 맞춤 자전거 찾기 개선\n
    테스트\n
        usage: 용도 (출퇴근, 운동, 여행, 산악, 로드)\n
        budget: 예산 (최소-최대)\n
        height: 키\n
    """
    try:
        minBudget, maxBudget  = budget.split('-')
        if budget is None or len(budget) < 1 or minBudget is None or maxBudget is None:
            return resResponse("fail", 400, f"예산 없음, budget : {budget}", None)

        if usage is None or len(usage) < 1:
            return resResponse("fail", 400, f"용도 없음, usage : {usage}")

        if height is None or len(height) < 1:
            return resResponse("fail", 400, f"키 없음, height : {height}")

        selData = recommend_by_query(
            float(minBudget),
            float(maxBudget),
            float(height),
            usage,
            top_n=4,
            df=df,
            cosine_sim=cosine_sim,
            purpose_map=purpose_map_fixed
        )

        if len(selData) < 1:
            return resResponse("fail", 400, f"데이터 없음", None)

        jsonData = selData.to_json(orient='records')
        result = json.loads(jsonData)
        # log.info(f"result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-img", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-img")
async def selImg(
    backTask: BackgroundTasks,
    url: str = Form(..., description='이미지 외부 주소', examples=['https://shopping-phinf.pstatic.net/main_5466078/54660787576.jpg'])
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 시세 조회하기 - 외부 이미지 조회\n
    테스트\n
        url: 외부 이미지 주소\n
    """
    try:
        if url is None or len(url) < 1:
            return resResponse("fail", 400, f"외부 이미지 주소 없음, url : {url}")

        stream = clientAsync.stream("GET", url)
        response = await stream.__aenter__()
        response.raise_for_status()
        mediaType = response.headers.get("Content-Type")
        backTask.add_task(stream.__aexit__, None, None, None)
        return StreamingResponse(response.aiter_bytes(), media_type=mediaType)
    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-isPage", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-isPage")
async def selImg(
    backTask: BackgroundTasks,
    url: str = Form(..., description='외부 주소', examples=['https://smartstore.naver.com/main/products/12053606391'])
):
    """
    기능\n
        알톤 바이크메트릭스AI - AI 시세 조회하기 - 외부 주소 정상/이상여부\n
    테스트\n
        url: 외부 주소\n
        https://smartstore.naver.com/smartbicycle/products/8401877446222\n
        https://smartstore.naver.com/main/products/12053606391\n
    """
    try:
        if url is None or len(url) < 1:
            return resResponse("fail", 400, f"외부 주소 없음, url : {url}", [])

        keywordList = [
            "현재 서비스 접속이 불가합니다",
            "상품이 존재하지 않습니다",
            "존재하지 않는 상품입니다",
            "판매 중지",
            "삭제된 상품",
            "판매자가 판매를 중지",
            "요청하신 페이지를 찾을 수 없습니다",
        ]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            result = {}

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=2000)

                pageUrl = page.url
                content = await page.content()

                isValid = True
                reason = ""

                for keyword in keywordList:
                    if keyword in content:
                        isValid = False
                        reason = keyword
                        break

                result = {
                    "isValid": isValid,
                    "url": pageUrl,
                    "reason": reason if not isValid else ""
                }

            except Exception as e:
                log.error(f'Exception : {e}')
                return resResponse("fail", 400, f"접속 실패 : {e}")

            finally:
                await browser.close()

            return resResponse("succ", 200, "처리 완료", 1, result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))