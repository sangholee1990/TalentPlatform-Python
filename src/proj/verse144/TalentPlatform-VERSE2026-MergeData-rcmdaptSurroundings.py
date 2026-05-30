# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt
# conda activate py39
# python 1_1_rcmdapt_req_surroundings_day1.py

# 백그라운드 실행
# nohup python 1_1_rcmdapt_req_surroundings_day1.py &
# tail -f nohup.out

# 절대경로로 직접 실행
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/1_1_rcmdapt_req_surroundings_day1.py

# 절대경로 백그라운드 실행
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/1_1_rcmdapt_req_surroundings_day1.py &
# tail -f nohup.out
# pkill -f 1_1_rcmdapt_req_surroundings_day1.py

# 로그 파일 확인
# tail -f /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/infra_collect.log

# =================================================
# 실행 전 확인사항
# =================================================
# 1. config 파일 존재 확인
# /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/config_infra.json

# 2. 입력 파일 존재 확인
# /HDD/DATA/OUTPUT/LSH0613/전처리/아파트실거래_경기도_의정부시.csv # 예시

# 3. 저장 경로 확인
# /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/dataset

# 4. 테스트 실행 시
# 현재 코드에는 PD_APTS.head(2) 가 들어가 있으므로 2개 좌표만 수행됨
# 전체 실행 시에는 아래 구문을 원복
# for index, row in PD_APTS.iterrows():

import os
import time
import numpy as np
import pandas as pd
import requests
import xmltodict
import json
import logging
from datetime import datetime
import argparse
import sys

# ============================================
# 유틸리티 함수
# ============================================
# 로그 설정
def init_logger(env):
    log_path = './logs' if env == 'local' else '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/logs'
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, f'infra_collect{datetime.now().strftime("%Y%m%d")}.log')

    logger = logging.getLogger('infra')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # 콘솔
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    # 파일
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

# 인증키 설정
def load_config():
    ENV = os.getenv('ENV', 'server').lower()

    if ENV == 'server':
        config_path = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/config_infra.json'
    else:
        config_path = './config_infra.json'

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config_infra 파일 없음: {config_path}')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


# =====================================================================================
# 공통 설정
# =====================================================================================
# 메타정보
sysOpt = {
    'case': {
        '1': {'gu_list': ['강서구', '강동구', '구로구', '동작구', '마포구', '서대문구', '광진구', '강북구', '금천구', '은평구'], 'sido': '서울특별시'},
        '2': {'gu_list': ['노원구', '강남구', '영등포구', '서초구', '도봉구', '송파구', '성북구', '양천구', '성동구', '동대문구'], 'sido': '서울특별시'},
        '3': {'gu_list': ['관악구', '중랑구', '용산구', '중구', '종로구'], 'sido': '서울특별시'},
        '4': {'gu_list': ['의정부시', '수원시', '성남시', '고양시', '용인시', '안산시', '남양주시', '화성시', '평택시', '시흥시'], 'sido': '경기도'},
        '5': {'gu_list': ['안성시', '포천시', '의왕시', '여주시', '광주시', '군포시', '오산시', '이천시', '양주시', '구리시'], 'sido': '경기도'},
        '6': {'gu_list': ['파주시', '광명시', '김포시', '양평군', '동두천시', '가평군', '연천군', '안양시', '과천시', '하남시'], 'sido': '경기도'},
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, required=False, default='1')
args = parser.parse_args()
metaInfo = sysOpt['case'][args.case]

# Variables
# sido = '서울특별시'
# sido = '경기도'
sido = metaInfo['sido']

ENV = os.getenv('ENV', 'server').lower()
log = init_logger(ENV)

config_infra = load_config()

KAKAO_API = 'https://dapi.kakao.com/v2/local/search/keyword.json?&query={}&category_group_code={}&y={}&x={}&radius={}'
KAKAO_API_KEY = config_infra['KAKAO_API_KEY']
# 1일 약 5천개 가능 (2일로 나누어 작업 필요)

BUS_API_BASE = 'http://ws.bus.go.kr/api/rest/stationinfo/getStationByPos?serviceKey={}&tmX={}&tmY={}&radius={}'
BUS_API_KEY = config_infra['BUS_API_KEY']
# 1일 약 1만개 가능

CATEGORIES = {
    '교통': [('지하철역', 'SW8')],    # 버스정류장 관련정보(X) -> 별도 API로 조회
    '교육': [('학교', 'SC4'), ('학원', 'AC5'), ('어린이집', 'PS3'), ('유치원', 'PS3')],
    '주거환경': [('주차장', 'PK6'), ('주유소', 'OL7'), ('충전소', 'OL7'), ('문화시설', 'CT1'), ('공공기관', 'PO3'), ('관광명소', 'AT4')],
    '편의시설': [('병원', 'HP8'), ('약국', 'PM9'), ('은행', 'BK9'), ('대형마트', 'MT1'), ('편의점', 'CS2'), ('음식점', 'FD6'), ('카페', 'CE7')]
}

# =====================================================================================
# ENV 설정

# -----------------------------------
# bus_api: 1일 1만개 가능
# kakao_api: 1일 약 5천개

if ENV == 'server':
    BASE_DIR = '/HDD/DATA/OUTPUT/LSH0613/전처리'
    SAVE_DIR = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/dataset'

    # 서버 테스트 (day0)
    # gu_list = [
    #     '의정부시',   # Kakao: 5256, Bus: 292
    # ]

    # 서버는 실제 실행 대상 (day1)
    # gu_list = [
    #     '수원시', '성남시', '고양시', '용인시',
    #     '안산시', '안양시', '남양주시', '화성시', '평택시',
    #     '시흥시', '파주시', '광명시', '김포시',
    # ]

    # 서버 day2가 필요하면 위 gu_list 대신 아래 사용
    # gu_list = [
    #     '군포시', '오산시', '이천시', '양주시', '구리시',
    #     '안성시', '포천시', '의왕시', '여주시', '광주시',
    #     '양평군', '동두천시', '가평군', '연천군'
    # ]

    # (day3)
    # gu_list = [
    #     '과천시', '하남시', 
    # ]

    # -----------------------------------
    # 서울 지역
    # bus_api: 1일 1만개 가능
    # kakao_api: 1일 약 5천개
    # gu_list = [
        # 1일
        # '강남구', '강동구', '강북구', '중구',                 # 456, 396, 114, 102
        # '강서구', '관악구', '광진구', '종로구',               # 518, 19, 226, 186, 103
        # '구로구', '중랑구', '금천구', '노원구',                # 446, 232, 135, 305
        # '도봉구', '동대문구', '동작구', '마포구',              # 205, 248, 210, 286,
        #
        # 2일
        # '서대문구', '서초구', '성동구', '성북구',              # 233, 488, 149, 171,
        # '송파구', '양천구', '영등포구',                       # 327, 513, 255,
        # '용산구', '은평구'                                  # 183, 424,
    # ]
    # gu_list = [
        # 1일
        # '강서구', '강동구', '구로구', '동작구',
        # '마포구', '서대문구', '광진구', '강북구', '금천구'
        
        # 2일
        # '노원구', '강남구', '영등포구', '서초구',
        # '도봉구', '은평구', '종로구' 

        # 3
        #'송파구', '성북구', '양천구', '성동구',
        #'동대문구', '관악구', '중랑구', '용산구', '중구'

    # ]

    gu_list = metaInfo['gu_list']
    log.info('[ENV] SERVER MODE')

else:
    # 로컬 테스트용
    log.info(f'[DEBUG] cwd: {os.getcwd()}')
    BASE_DIR = os.path.join(os.getcwd(), 'output', '전처리')
    SAVE_DIR = './dataset'

    # 로컬 소량 테스트
    gu_list = ['의정부시']  # 실제 존재하는 파일 기준으로 테스트

    log.info('[ENV] LOCAL MODE')

os.makedirs(SAVE_DIR, exist_ok=True)

log.info(f'[PATH] BASE_DIR: {BASE_DIR}')
log.info(f'[PATH] SAVE_DIR: {SAVE_DIR}')
log.info(f'[INFO] case: {args.case}')
log.info(f'[INFO] metaInfo: {metaInfo}')
log.info(f'[INFO] sido: {sido}')
log.info(f'[INFO] gu_list: {gu_list}')

# -----------------------------------
# API 호출 제한
# kakao_api: 1일 약 5천개.. / 2026년 현재 확인된바 10만건
# bus_api: 1일 약 1만개 가능
KAKAO_LIMIT = 95000
BUS_LIMIT = 9500

KAKAO_COUNT = 0
BUS_COUNT = 0

KAKAO_CALLS_PER_POINT = sum(len(v) for v in CATEGORIES.values())   # 현재 18
BUS_CALLS_PER_POINT = 1
# -----------------------------------


def load_apts(gu_name: str) -> pd.DataFrame:
    """
    최신 전처리 CSV에서 아파트명/위경도만 추출
    - 실거래 원본은 거래내역 단위라 같은 아파트가 여러 건 있을 수 있음
    - aptNm, lat, lon 기준으로 중복 제거하여 API 호출 대상을 축약
    """
    file_path = os.path.join(BASE_DIR, f'아파트실거래_{sido}_{gu_name}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'입력 파일 없음: {file_path}')

    df = pd.read_csv(file_path, low_memory=False)

    need_cols = ['aptNm', 'lat', 'lon']
    miss_cols = [c for c in need_cols if c not in df.columns]
    if miss_cols:
        raise ValueError(f'{gu_name} 파일에 필수 컬럼 없음: {miss_cols}')

    df = (
        df[need_cols]
        .rename(columns={'aptNm': '아파트', 'lat': 'latitude', 'lon': 'longitude'})
        .dropna(subset=['latitude', 'longitude'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df


def estimate_calls(pd_apts: pd.DataFrame) -> dict:
    """
    실행 전 예상 호출 수 계산
    - 같은 좌표는 실제 루프에서 1번만 호출하므로 좌표 기준 유니크 건수로 계산
    """
    unique_points = pd_apts[['latitude', 'longitude']].drop_duplicates().shape[0]
    kakao_expected = unique_points * KAKAO_CALLS_PER_POINT
    bus_expected = unique_points * BUS_CALLS_PER_POINT
    total_expected = kakao_expected + bus_expected

    return {
        'apt_rows': len(pd_apts),
        'unique_points': unique_points,
        'kakao_expected': kakao_expected,
        'bus_expected': bus_expected,
        'total_expected': total_expected
    }


# =====================================================================================
# 비즈니스 로직

for gu_name in gu_list:
    log.info('')
    log.info(f'===== {sido} {gu_name} =====')

    try:
        PD_APTS = load_apts(gu_name)
    except Exception as ex:
        log.error(f'{gu_name} 아파트 로딩 실패: {str(ex)}')
        continue

    est = estimate_calls(PD_APTS)

    log.info(f'{gu_name} rows(after aptNm/lat/lon dedup): {est["apt_rows"]}')
    log.info(f'{gu_name} unique coordinate points: {est["unique_points"]}')
    log.info(f'Kakao expected calls: {est["kakao_expected"]}')
    log.info(f'Bus expected calls: {est["bus_expected"]}')
    log.info(f'Total expected calls: {est["total_expected"]}')

    if est['kakao_expected'] > KAKAO_LIMIT:
        log.warning(f'{gu_name} 예상 Kakao 호출 수가 한도 초과: {est["kakao_expected"]}/{KAKAO_LIMIT}')
    if est['bus_expected'] > BUS_LIMIT:
        log.warning(f'{gu_name} 예상 Bus 호출 수가 한도 초과: {est["bus_expected"]}/{BUS_LIMIT}')

    log.info(f'[COUNT][START] Kakao: {KAKAO_COUNT}/{KAKAO_LIMIT}, Bus: {BUS_COUNT}/{BUS_LIMIT}')

    BROKEN = False

    # --------------------------------------------
    # Read data & query surroundings info
    # (위치 정보만으로 조회: '아파트', 'latitude', 'longitude')
    results = []
    xy_set = set()

    # for index, row in PD_APTS.head(2).iterrows(): # 2곳만 테스트
    for index, row in PD_APTS.iterrows():
        apt = row['아파트']
        y = row['latitude']
        x = row['longitude']
        radius = 1000
        x_y = '{}_{}'.format(x, y)

        # 중복조회 방지
        if x_y in xy_set:
            continue
        elif str(x) == 'nan' or str(y) == 'nan':
            continue
        else:
            xy_set.add(x_y)

        log.info(f'[{len(xy_set)}/{est["unique_points"]}] long: {x} ,lati: {y}')

        # ---------------------------------
        # kakao info
        # ---------------------------------
        for category, query_codes in CATEGORIES.items():
            # log.info(f'category: {category}')

            for query, group_code in query_codes:
                if KAKAO_COUNT >= KAKAO_LIMIT:
                    BROKEN = True
                    log.error(f'Kakao API local limit reached: {KAKAO_COUNT}/{KAKAO_LIMIT}')
                    break

                headers = {'Authorization': 'KakaoAK {}'.format(KAKAO_API_KEY)}
                req_url = KAKAO_API.format(query, group_code, y, x, radius)

                try:
                    time.sleep(0.2)
                    req_res = requests.get(req_url, headers=headers, timeout=30)
                    KAKAO_COUNT += 1

                    # log.info(f'[KAKAO] status={req_res.status_code} query={query} group={group_code}')

                    try:
                        res_json = req_res.json()
                    except Exception:
                        log.error(f'Kakao JSON parse 실패: {req_res.text[:300]}')
                        continue

                except Exception as ex:
                    log.error(f'Kakao API 호출 실패: {str(ex)}')
                    continue

                if req_res.status_code == 403:
                    BROKEN = True
                    log.error(f'Kakao 403 Forbidden: {str(res_json)[:500]}')
                    break

                if req_res.status_code != 200:
                    log.error(f'Kakao HTTP 오류: {req_res.status_code}, body={str(res_json)[:500]}')
                    continue

                if 'errorType' in res_json or 'message' in res_json:
                    log.error(f'Kakao API 응답 오류: {str(res_json)[:500]}')
                    continue

                if 'API limit has been exceeded' in str(res_json):
                    BROKEN = True
                    log.error(f'Kakao API provider limit exceeded: {KAKAO_COUNT}/{KAKAO_LIMIT}')
                    break

                docs = res_json.get('documents', [])
                # log.info(f'[KAKAO] docs={len(docs)} query={query}')

                for doc in docs:
                    c_g_code = doc.get('category_group_code')
                    c_g_name = doc.get('category_group_name')
                    p_name = doc.get('place_name')

                    try:
                        p_x = float(doc.get('x'))
                    except Exception:
                        p_x = np.nan

                    try:
                        p_y = float(doc.get('y'))
                    except Exception:
                        p_y = np.nan

                    p_dist = -1
                    try:
                        p_dist = int(doc.get('distance'))
                    except Exception:
                        pass

                    p_addr = doc.get('address_name')
                    p_road_addr = doc.get('road_address_name')
                    p_phone = doc.get('phone')
                    p_url = doc.get('place_url')

                    results.append([
                        gu_name, apt, x, y, radius, category, query, c_g_code, c_g_name,
                        p_name, p_dist, p_x, p_y, p_addr, p_road_addr, p_phone, p_url
                    ])

            if BROKEN:
                break

        if BROKEN:
            break

        # -------------------------------------------------------------------
        # bus-stop info (1000미터 거리내 버스정류장)
        # -------------------------------------------------------------------
        try:
            # log.info('category: 교통_버스정류장')

            if BUS_COUNT >= BUS_LIMIT:
                BROKEN = True
                log.error(f'BUS API local limit reached: {BUS_COUNT}/{BUS_LIMIT}')
                break

            BUS_API = BUS_API_BASE.format(BUS_API_KEY, x, y, 1000)

            time.sleep(0.2)
            bus_res = requests.get(BUS_API, timeout=30)
            BUS_COUNT += 1
            res_json = xmltodict.parse(bus_res.text)

            if 'LIMITED NUMBER OF SERVICE REQUESTS EXCEEDS' in str(res_json):
                BROKEN = True
                log.error(f'BUS API provider limit exceeded: {BUS_COUNT}/{BUS_LIMIT}')
                break

            items = res_json.get('ServiceResult', {}).get('msgBody', {}).get('itemList', [])

            if isinstance(items, dict):
                items = [items]

            for item in items:
                b_y = item.get('gpsY')
                b_x = item.get('gpsX')
                nodenm = item.get('stationNm')
                nodeno = item.get('stationId')

                b_dist = -1
                try:
                    b_dist = int(item.get('dist'))
                except Exception:
                    pass

                b_addr, b_road_addr, b_phone, b_url = '', '', '', ''

                if b_dist <= 1000:
                    results.append([
                        gu_name, apt, x, y, 1000, '교통', '버스정류장', '', '버스정류장',
                        '{}_{}'.format(nodeno, nodenm), b_dist, b_x, b_y, b_addr, b_road_addr, b_phone, b_url
                    ])

        except Exception as ex:
            log.error(f'버스정류장 조회 실패: {str(ex)}')

        log.info(f'[COUNT] Kakao: {KAKAO_COUNT}/{KAKAO_LIMIT}, Bus: {BUS_COUNT}/{BUS_LIMIT}')

    # ---------------------------------------------------
    # gu_name별로 저장
    # ---------------------------------------------------
    if BROKEN:
        log.warning(f'{gu_name} 처리 중 API 제한 등으로 중단됨. 저장 생략')
        log.warning(f'[COUNT][STOP] Kakao: {KAKAO_COUNT}/{KAKAO_LIMIT}, Bus: {BUS_COUNT}/{BUS_LIMIT}')
        continue

    pd_results = pd.DataFrame(results)
    if len(pd_results) == 0:
        log.warning(f'{gu_name} 결과 없음')
        continue

    pd_results.columns = [
        'gu_name', 'apt', 'apt_x', 'apt_y', 'radius', 'category', 'query', 'c_g_code', 'c_g_name',
        'place_name', 'distance', 'p_x', 'p_y', 'p_addr', 'p_road_addr', 'p_phone', 'p_url'
    ]

    log.info(f'result shape: {pd_results.shape}')

    save_path = os.path.join(SAVE_DIR, f'surroundings_{sido}_{gu_name}.xlsx')
    pd_results.to_excel(save_path, index=False)
    log.info(f'[SAVE] {save_path}')
    log.info(f'[COUNT][END] Kakao: {KAKAO_COUNT}/{KAKAO_LIMIT}, Bus: {BUS_COUNT}/{BUS_LIMIT}')
