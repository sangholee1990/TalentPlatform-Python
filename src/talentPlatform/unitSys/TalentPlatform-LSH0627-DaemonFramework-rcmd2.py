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
                # # 간혹 '11.5. ' 처럼 .이 여러개 있는 경우 방지
                # # 숫자와 .으로 이루어진 첫번째 유효한 숫자열을 찾도록 좀 더 정교하게
                # # 첫 번째 유효한 부동소수점 숫자를 찾음.
                # match = re.search(r'\d+(\.\d+)?', weight_str)
                # if match:
                #     return float(match.group(0))
                # else:
                #     return np.nan
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

    if len(candidate_indices) == 1:
        return df_filtered

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
                'inpFile': '/HDD/DATA/OUTPUT/LSH0627/alton_bikes_web_v3.xlsx',
            }

            # =================================================================
            # 전처리
            # =================================================================
            df = pd.read_excel(sysOpt['inpFile'])

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

            tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
            text_matrix = tfidf.fit_transform(df['features_text'])

            # 3-3. 피처 결합
            combined_features = hstack([numerical_sparse, text_matrix])
            # print("... 최종 피처 행렬 결합 완료 (Shape: {})".format(combined_features.shape))

            # --- 4단계: 코사인 유사도 계산 (Bike-to-Bike) ---
            # print("\n--- (4/6) 코사인 유사도 계산 시작 ---")
            cosine_sim = cosine_similarity(combined_features)
            # print("... 코사인 유사도 계산 완료 (Shape: {})".format(cosine_sim.shape))

            # (제품명 <-> 인덱스) 매핑 생성
            indices_map = pd.Series(df.index, index=df['제품명']).drop_duplicates()

            # --- 5단계: 추천 함수 정의 ---
            # print("\n--- (5/6) 추천 함수 정의 ---")

            # 5-1. '용도' -> '카테고리' 매핑
            purpose_map_fixed = {
                "출퇴근": "하이브리드|폴딩/미니벨로|전기자전거|씨티",
                "운동": "로드|컴포트 산악자전거|하이브리드",
                "여행": "하이브리드|전기자전거|컴포트 산악자전거",
                "산악": "컴포트 산악자전거|MTB",
                "로드": "로드"
            }
            # print("... '용도' 맵핑 정의 완료")

            # pandas 설정으로 모든 컬럼이 보이도록 보장합니다.
            # pd.set_option('display.max_columns', None)
            # pd.set_option('display.width', 1000)  # 터미널 출력 너비 설정

            print("\n--- [Test 1 & 2]: '출퇴근' 용도, 30~50만원, 키 170cm (1/2번 화면) ---")
            test_budget_min_1 = 300000
            test_budget_max_1 = 500000
            test_height_1 = 170
            test_purpose_1 = "출퇴근"

            # 2번 화면에 보여줄 4개 자전거
            recommendations_1 = recommend_by_query(test_budget_min_1, test_budget_max_1, test_height_1, test_purpose_1, top_n=4, df=df, cosine_sim=cosine_sim, purpose_map=purpose_map_fixed)

            selected_bikes_from_screen2 = []

            if isinstance(recommendations_1, pd.DataFrame):
                print(f"--- [2번 화면 결과] (상위 {len(recommendations_1)}개) ---")

                display_cols = ['제품명', '카테고리', '판매가', 'AI_Rank_Score']
                remaining_cols = [col for col in recommendations_1.columns if col not in display_cols]

                # display(recommendations_1[display_cols + remaining_cols])
                # 테스트를 위해 상위 2개 자전거 이름을 리스트로 저장
                selected_bikes_from_screen2 = recommendations_1['제품명'].head(2).tolist()
            else:
                print(f"--- [2번 화면 결과 없음] ---")
                print(recommendations_1)

            # --- 3번 화면 테스트 ---
            if selected_bikes_from_screen2:
                print(f"\n--- [Test 3]: 사용자가 '{', '.join(selected_bikes_from_screen2)}' 2개 선택 (3번 화면) ---")

                recommendations_2 = get_recommendations_from_selection(selected_bikes_from_screen2, top_n=4)

                if isinstance(recommendations_2, pd.DataFrame):
                    print(f"--- [3번 화면 AI 추천 결과] (상위 {len(recommendations_2)}개) ---")

                    display_cols_3 = ['제품명', '카테고리', '판매가', '유사도(%)']
                    remaining_cols = [col for col in recommendations_1.columns if col not in display_cols]

                    # display(recommendations_2[display_cols_3 + remaining_cols])
                else:
                    print(f"--- [3번 화면 추천 결과 없음] ---")
                    # display(recommendations_2)
            else:
                print("\n--- [Test 3]: 2번 화면 결과가 없어 3번 화면 테스트를 건너뜁니다. ---")












            #
            #
            #
            # log.info(f"데이터 로드 및 표준화 완료. 최종 데이터 형태: {df_std.shape}")
            # log.info("\n--- 표준화된 컬럼 목록 (샘플) ---")
            # log.info(df_std.columns.tolist())
            #
            # df_processed = preprocess_features(df_std)
            # log.info("Feature Engineering 완료.")
            #
            # # 한글 컬럼명을 표준 영문명으로 변경함
            # # 벡터화할 컬럼 정의
            # numeric_features = ['price', 'weight', 'height', 'motor_power', 'battery_capacity']
            # categorical_features = ['category', 'gears', 'brake_type', 'frame_material']
            # text_features = 'features'
            #
            # # 전처리 과정에서 생성된 컬럼들이 df_processed에 있는지 확인하고 없는 경우를 대비
            # existing_numeric = [col for col in numeric_features if col in df_processed.columns]
            # existing_categorical = [col for col in categorical_features if col in df_processed.columns]
            # if text_features not in df_processed.columns:
            #     text_features = None  # features 컬럼이 없으면 텍스트 처리를 건너뜀
            #
            # log.info(f"사용될 숫자형 특징: {existing_numeric}")
            # log.info(f"사용될 범주형 특징: {existing_categorical}")
            # if text_features:
            #     log.info(f"사용될 텍스트 특징: {text_features}")
            #
            # # 결측치 처리
            # for col in existing_numeric:
            #     df_processed[col].fillna(df_processed[col].median(), inplace=True)
            # for col in existing_categorical:
            #     df_processed[col].fillna('Unknown', inplace=True)
            # if text_features:
            #     df_processed[text_features].fillna('', inplace=True)
            #
            # # 파이프라인 구성
            # numeric_transformer = StandardScaler()
            # categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            # text_transformer = TfidfVectorizer(max_features=500)
            #
            # # ColumnTransformer의 transformer 리스트를 동적으로 구성
            # transformers = []
            # if existing_numeric:
            #     transformers.append(('num', numeric_transformer, existing_numeric))
            # if existing_categorical:
            #     transformers.append(('cat', categorical_transformer, existing_categorical))
            # if text_features:
            #     transformers.append(('tfidf', text_transformer, text_features))
            #
            # if not transformers:
            #     log.info("벡터화할 특징이 하나도 없습니다. 컬럼명을 확인해주세요.")
            # else:
            #     preprocessor = ColumnTransformer(
            #         transformers=transformers,
            #         remainder='drop')  # 위에서 정의하지 않은 컬럼은 버림
            #
            #     # 전체 모델링 파이프라인: 전처리 -> SVD
            #     svd_pipeline = Pipeline(steps=[
            #         ('preprocessor', preprocessor),
            #         ('svd', TruncatedSVD(n_components=30, random_state=42))])
            #
            #     # 데이터에 파이프라인 적용하여 잠재 벡터 생성
            #     X_svd = svd_pipeline.fit_transform(df_processed)
            #
            #     # 최종 유사도 행렬 계산
            #     similarity_matrix = cosine_similarity(X_svd)
            #
            #     # 모델명으로 인덱스 찾기 위한 딕셔너리 (이제 'model_name' 사용)
            #     indices = pd.Series(df_processed.index, index=df_processed['model_name']).drop_duplicates()
            #
            #     log.info("\n 특징 벡터화 및 SVD 모델링 완료.")
            #     log.info(f"  - 잠재 벡터 형태: {X_svd.shape}")
            #     log.info(f"  - 최종 유사도 행렬 형태: {similarity_matrix.shape}")
            #
            # # 함수 실행
            # df_processed['spec_score'] = df_processed.apply(calculate_total_spec_score, axis=1)
            # stat_scores_df = calculate_stat_scores_final(df_processed)
            # df_processed = pd.concat([df_processed, stat_scores_df], axis=1)
            # # df_processed.to_excel('df_processed.xlsx')
            #
            # log.info("최종 5각형 스탯 점수 계산 완료 ('유지보수 편의성' 지표 적용).")
            #
            # # np.max(df_processed['price'])
            # # np.min(df_processed['price'])
            #
            # user_usage = '출퇴근'
            # user_budget_min = 80000
            # user_budget_max = 500000
            # user_height = 180
            #
            # recommendations = recommend_alton_bikes(
            #     usage=user_usage,
            #     budget_min=user_budget_min,
            #     budget_max=user_budget_max,
            #     height=user_height,
            #     ranking_method='spec_score',
            #     top_n=4,
            #     df_processed=df_processed,
            #     indices=indices
            # )
            #
            # rcmdData = pd.merge(recommendations, df_std, on=['model_name'], how='left')
            # log.info(f"rcmdData : {rcmdData}")

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
