# -*- coding: utf-8 -*-
# ============================================
# 요구사항
# ============================================
# [완료] LSH0359. Python을 이용한 GOSAT 및 OCO2 위성 자료처리 및 스캔 영역 시각화

# ============================================
# 라이브러리
# ============================================
import glob
import os
import platform
import warnings
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import Day

# ============================================
# 유틸리티 함수
# ============================================

# ============================================
# 주요 설정
# ============================================
serviceName = 'LSH0359'
contextPath = os.getcwd()
# contextPath = f"/SYSTEMS/PROG/PYTHON/IDE"

globalVar = {
    'ctxPath': f"{contextPath}"
    , 'inpPath1':  f"/DATA/INPUT/LSH0359/OCO2"
    , 'inpPath2': f"/DATA/INPUT/LSH0359"
    , 'outPath': f"/DATA/OUTPUT/LSH0359"
    , 'figPath': f"/DATA/FIG/LSH0359"
}

for key, val in globalVar.items():
    if key.__contains__('Path'):
        os.makedirs(val, exist_ok=True)
        print(f"[CHECK] {key} : {val}")

# 작업 경로 설정
os.chdir(f"{globalVar['ctxPath']}")
print(f"[CHECK] getcwd : {os.getcwd()}")


# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2018-01-01'
    , 'endDate': '2018-12-31'

    # 영역 설정 시 해상도
    # 2도 = 약 200 km
    , 'res': 2

    # 설정 정보
    , 'data': {
        'landList': None
        , 'landUse': None
        , 'stnList': None
    }

    # 특정 월 선택
    , 'selMonth': [3, 4, 5, 12]

    # 특정 시작일/종료일 선택
    , 'selSrtDate': '2018-01-21'
    , 'selEndDate': '2018-12-24'

    # 관심영역 설정
    , 'roi': {
        'ko': {
            # 'minLon' : 120
            # , 'maxLon' : 150
            # , 'minLat' : 30
            # , 'maxLat' : 40

            'minLon': 125
            , 'maxLon': 131
            , 'minLat': 33
            , 'maxLat': 42
        }
    }
}

# ============================================
# 비즈니스 로직
# ============================================
# 시작/종료일 설정
dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))