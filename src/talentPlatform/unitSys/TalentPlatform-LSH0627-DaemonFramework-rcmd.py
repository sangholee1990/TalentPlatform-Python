# ================================================
# 요구사항
# ================================================
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0627-DaemonFramework-model.py
# 0 0 * * * cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys && /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py

import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램 :부 프로그램을 호출
# 4. 부 프로그램 : 자료 처리를 위한 클래스로서 내부 함수 (초기 변수, 비즈니스 로직, 수행 프로그램 설정)
# 4.1. 환경 변수 설정 (로그 설정) : 로그 기록을 위한 설정 정보 읽기
# 4.2. 환경 변수 설정 (초기 변수) : 입력 경로 (inpPath) 및 출력 경로 (outPath) 등을 설정
# 4.3. 초기 변수 (Argument, Option) 설정 : 파이썬 실행 시 전달인자 설정 (pyhton3 *.py argv1 argv2 argv3 ...)
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리 또는 비즈니스 로직 구현

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')


# =================================================
# 2. 유틸리티 함수
# =================================================
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

#  초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    # 환경 변수 (local, 그 외)에 따라 전역 변수 (입력 자료, 출력 자료 등)를 동적으로 설정
    # 즉 local의 경우 현재 작업 경로 (contextPath)를 기준으로 설정
    # 그 외의 경우 contextPath/resources/input/prjName와 같은 동적으로 구성
    globalVar = {
        'prjName': prjName
        , 'sysOs': platform.system()
        , 'contextPath': contextPath
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar


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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0612'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:
            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 입력 데이터
                'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v2.xlsx',
                # 'model': {
                #     # 시계열 딥러닝
                #     'dl': {
                #         'input_chunk_length': 2,
                #         'output_chunk_length': 7,
                #         'n_epochs': 50,
                #     },
                #     # 예측
                #     'prdCnt': 10,
                # },
                # 예측 데이터
                'saveFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_prd.csv',
                'preDt': datetime.now(),
            }

            # =================================================================
            # 전처리
            # =================================================================
            df_std = load_and_standardize_columns(sysOpt['inpFile'])
            log.info(f"데이터 로드 및 표준화 완료. 최종 데이터 형태: {df_std.shape}")
            log.info("\n--- 표준화된 컬럼 목록 (샘플) ---")
            log.info(df_std.columns.tolist())

            df_processed = preprocess_features(df_std)
            log.info("Feature Engineering 완료.")

            # 한글 컬럼명을 표준 영문명으로 변경함
            # 벡터화할 컬럼 정의
            numeric_features = ['price', 'weight', 'height', 'motor_power', 'battery_capacity']
            categorical_features = ['category', 'gears', 'brake_type', 'frame_material']
            text_features = 'features'

            # 전처리 과정에서 생성된 컬럼들이 df_processed에 있는지 확인하고 없는 경우를 대비
            existing_numeric = [col for col in numeric_features if col in df_processed.columns]
            existing_categorical = [col for col in categorical_features if col in df_processed.columns]
            if text_features not in df_processed.columns:
                text_features = None  # features 컬럼이 없으면 텍스트 처리를 건너뜀

            log.info(f"사용될 숫자형 특징: {existing_numeric}")
            log.info(f"사용될 범주형 특징: {existing_categorical}")
            if text_features:
                log.info(f"사용될 텍스트 특징: {text_features}")

            # 결측치 처리
            for col in existing_numeric:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            for col in existing_categorical:
                df_processed[col].fillna('Unknown', inplace=True)
            if text_features:
                df_processed[text_features].fillna('', inplace=True)

            # 파이프라인 구성
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            text_transformer = TfidfVectorizer(max_features=500)

            # ColumnTransformer의 transformer 리스트를 동적으로 구성
            transformers = []
            if existing_numeric:
                transformers.append(('num', numeric_transformer, existing_numeric))
            if existing_categorical:
                transformers.append(('cat', categorical_transformer, existing_categorical))
            if text_features:
                transformers.append(('tfidf', text_transformer, text_features))

            if not transformers:
                log.info("벡터화할 특징이 하나도 없습니다. 컬럼명을 확인해주세요.")
            else:
                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder='drop')  # 위에서 정의하지 않은 컬럼은 버림

                # 전체 모델링 파이프라인: 전처리 -> SVD
                svd_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('svd', TruncatedSVD(n_components=30, random_state=42))])

                # 데이터에 파이프라인 적용하여 잠재 벡터 생성
                X_svd = svd_pipeline.fit_transform(df_processed)

                # 최종 유사도 행렬 계산
                similarity_matrix = cosine_similarity(X_svd)

                # 모델명으로 인덱스 찾기 위한 딕셔너리 (이제 'model_name' 사용)
                indices = pd.Series(df_processed.index, index=df_processed['model_name']).drop_duplicates()

                log.info("\n 특징 벡터화 및 SVD 모델링 완료.")
                log.info(f"  - 잠재 벡터 형태: {X_svd.shape}")
                log.info(f"  - 최종 유사도 행렬 형태: {similarity_matrix.shape}")

            # 함수 실행
            df_processed['spec_score'] = df_processed.apply(calculate_total_spec_score, axis=1)
            stat_scores_df = calculate_stat_scores_final(df_processed)
            df_processed = pd.concat([df_processed, stat_scores_df], axis=1)
            # df_processed.to_excel('df_processed.xlsx')

            log.info("최종 5각형 스탯 점수 계산 완료 ('유지보수 편의성' 지표 적용).")

            # np.max(df_processed['price'])
            # np.min(df_processed['price'])

            user_usage = '출퇴근'
            user_budget_min = 80000
            user_budget_max = 500000
            user_height = 180

            recommendations = recommend_alton_bikes(
                usage=user_usage,
                budget_min=user_budget_min,
                budget_max=user_budget_max,
                height=user_height,
                ranking_method='spec_score',
                top_n=4,
                df_processed=df_processed,
                indices=indices
            )

            rcmdData = pd.merge(recommendations, df_std, on=['model_name'], how='left')
            log.info(f"rcmdData : {rcmdData}")

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e

        finally:
            log.info('[END] {}'.format("exec"))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    log.info('[START] {}'.format("main"))

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        log.info(traceback.format_exc())
        sys.exit(1)

    finally:
        log.info('[END] {}'.format("main"))
