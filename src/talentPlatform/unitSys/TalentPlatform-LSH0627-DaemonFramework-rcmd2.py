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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import scipy.sparse
import pandas as pd
import numpy as np
import re
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import scipy.sparse

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
                # 'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v3.xlsx',
                'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v4.xlsx',
            }

            # =================================================================
            # 전처리
            # =================================================================
            df = pd.read_excel(sysOpt['inpFile'])
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

            purpose_map_fixed = {
                "출퇴근": "하이브리드|폴딩/미니벨로|전기자전거|씨티|픽시|주니어",
                "운동": "로드|컴포트 산악자전거|하이브리드|픽시|주니어|키즈",
                "여행": "하이브리드|전기자전거|컴포트 산악자전거|로드",
                "산악": "컴포트 산악자전거",
                "로드": "로드|픽시"
            }

            test_budget_min_1 = 0
            test_budget_max_1 = 500000
            # test_height_1 = 170
            test_height_1 = 140
            test_purpose_1 = "운동"

            # 2번 화면에 보여줄 4개 자전거
            recommendations_1 = recommend_by_query(test_budget_min_1, test_budget_max_1, test_height_1, test_purpose_1, top_n=4, df=df, cosine_sim=cosine_sim, purpose_map=purpose_map_fixed)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e

        finally:
            log.info('[END] {}'.format("exec"))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))