# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt
# conda activate py39
# python upload_tb_infra.py

# 절대경로 백그라운드 실행
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/upload_tb_infra.py &
# tail -f nohup.out

import os
import glob
import shutil
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re

# =================================================
# 유틸
# =================================================
def extract_gu_name(file_path: str) -> str:
    """
    surroundings_관악구.xlsx -> 관악구
    surroundings_서울특별시_관악구.xlsx -> 관악구
    """
    base_name = os.path.basename(file_path).replace(".xlsx", "")
    parts = base_name.split("_")

    if len(parts) >= 3:
        return parts[-1]
    elif len(parts) == 2:
        return parts[-1]
    return ""

def standardize_filename(file_path: str, sido: str, gu_name: str) -> str:
    new_name = f"surroundings_{sido}_{gu_name}.xlsx"
    new_path = os.path.join(os.path.dirname(file_path), new_name)

    if os.path.abspath(file_path) == os.path.abspath(new_path):
        return file_path

    if not os.path.exists(new_path):
        shutil.copy2(file_path, new_path)

    return new_path

# =================================================
# 설정
# =================================================
JSON_KEY = "/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/project-p-32424-f1fe6277556d.json"
INPUT_DIR = "/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/dataset"
INPUT_PATTERN = os.path.join(INPUT_DIR, "surroundings_*_*.xlsx")
INPUT_PATTERN2 = os.path.join(INPUT_DIR, "surroundings_{TARGET_SIDO}_*.xlsx")
PROJECT_ID = "project-p-32424"
DATASET_ID = "DMS01"
TABLE_ID = "TB_INFRA"

FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# 서울 미적재 구만 업로드
# TARGET_SIDO = "서울특별시"
# TARGET_GU_LIST = [
#     "금천구", "노원구", "도봉구", "동대문구",
#     "마포구", "서대문구", "은평구",
#     "종로구", "중구", "중랑구"
# ]

# 필요 시 False로 두고 삭제만 안 할 수도 있음
DELETE_OLD_SEOUL = True

# 파일명이 아직 surroundings_관악구.xlsx 형태라면 True
# 이미 surroundings_서울특별시_관악구.xlsx 로 정리됐으면 False
RENAME_TO_STANDARD = False

# =================================================
# BigQuery 클라이언트
# =================================================
credentials = service_account.Credentials.from_service_account_file(JSON_KEY)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

# =================================================
# 컬럼 / 스키마 정의
# =================================================
expected_cols = [
    "gu_name", "apt", "apt_x", "apt_y", "radius",
    "category", "query", "c_g_code", "c_g_name",
    "place_name", "distance", "p_x", "p_y",
    "p_addr", "p_road_addr", "p_phone", "p_url",
    "aptGeo", "subGeo", "sgg", "구"
]

schema = [
    bigquery.SchemaField("gu_name", "STRING"),
    bigquery.SchemaField("apt", "STRING"),
    bigquery.SchemaField("apt_x", "FLOAT"),
    bigquery.SchemaField("apt_y", "FLOAT"),
    bigquery.SchemaField("radius", "INTEGER"),
    bigquery.SchemaField("category", "STRING"),
    bigquery.SchemaField("query", "STRING"),
    bigquery.SchemaField("c_g_code", "STRING"),
    bigquery.SchemaField("c_g_name", "STRING"),
    bigquery.SchemaField("place_name", "STRING"),
    bigquery.SchemaField("distance", "INTEGER"),
    bigquery.SchemaField("p_x", "FLOAT"),
    bigquery.SchemaField("p_y", "FLOAT"),
    bigquery.SchemaField("p_addr", "STRING"),
    bigquery.SchemaField("p_road_addr", "STRING"),
    bigquery.SchemaField("p_phone", "STRING"),
    bigquery.SchemaField("p_url", "STRING"),
    bigquery.SchemaField("aptGeo", "STRING"),
    bigquery.SchemaField("subGeo", "STRING"),
    bigquery.SchemaField("sgg", "STRING"),
    bigquery.SchemaField("구", "STRING"),
]

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    autodetect=False,
    schema=schema,
)

# =================================================
# 1. 서울 데이터만 삭제
# =================================================
fileList = sorted(glob.glob(INPUT_PATTERN))
if not fileList:
    raise Exception(f"업로드 대상 파일 목록 : {fileList}")

sidoDict = set()
for fileInfo in fileList:
    fileNameNotExt = os.path.basename(fileInfo).replace(".xlsx", "")
    partList = fileNameNotExt.split("_")
    if len(partList) >= 3 and re.compile(r'^[가-힣]+$').match(partList[1]):
        sidoDict.add(partList[1])

sidoList = list(sidoDict)
for sidoInfo in sidoList:
    # TARGET_SIDO = "서울특별시"
    TARGET_SIDO = sidoInfo
    print(f"[CHECK] TARGET_SIDO : {TARGET_SIDO}")

    if DELETE_OLD_SEOUL:
        delete_sql = f"""
        DELETE FROM `{FULL_TABLE_ID}`
        WHERE sgg LIKE '{TARGET_SIDO} %'
        """
        print(f"[CHECK] {TARGET_SIDO} 데이터 삭제 시작")
        delete_job = client.query(delete_sql)
        delete_job.result()
        print(f"[CHECK] {TARGET_SIDO} 데이터 삭제 완료")

    # =================================================
    # 2. 서울 파일 목록 확인
    # =================================================
    file_list = sorted(glob.glob(INPUT_PATTERN2.format(TARGET_SIDO=TARGET_SIDO)))
    if not file_list:
        raise FileNotFoundError(f"업로드 대상 파일 없음: {INPUT_PATTERN}")

    print(f"[CHECK] 업로드 대상 파일 수: {len(file_list)}")

    # =================================================
    # 3. 파일별 append
    # =================================================
    for file_path in file_list:
        gu_name = extract_gu_name(file_path)

        if RENAME_TO_STANDARD:
            file_path = standardize_filename(file_path, TARGET_SIDO, gu_name)

        print(f"\n[CHECK] file: {file_path}")

        df = pd.read_excel(file_path)

        # 파생 컬럼 생성
        df["aptGeo"] = df["apt_y"].astype(str) + ", " + df["apt_x"].astype(str)
        df["subGeo"] = df["p_y"].astype(str) + ", " + df["p_x"].astype(str)
        df["sgg"] = TARGET_SIDO + " " + df["gu_name"].astype(str)
        df["구"] = df["gu_name"]

        missing_cols = [col for col in expected_cols if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_cols]

        if missing_cols:
            print(f"[ERROR] 누락 컬럼 있음: {missing_cols}")
            continue

        if extra_cols:
            print(f"[WARN] 추가 컬럼은 업로드 제외됨: {extra_cols}")

        df = df[expected_cols].copy()

        # 타입 정리
        float_cols = ["apt_x", "apt_y", "p_x", "p_y"]
        int_cols = ["radius", "distance"]
        str_cols = [
            "gu_name", "apt", "category", "query", "c_g_code", "c_g_name",
            "place_name", "p_addr", "p_road_addr", "p_phone", "p_url",
            "aptGeo", "subGeo", "sgg", "구"
        ]

        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in str_cols:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({"nan": None, "None": None, "": None})

        print(f"[CHECK] rows: {len(df)}")

        load_job = client.load_table_from_dataframe(df, FULL_TABLE_ID, job_config=job_config)
        load_job.result()

        print(f"[CHECK] append 완료: {load_job.output_rows} rows")

    print(f"[CHECK] {TARGET_SIDO} 데이터 재적재 완료")
