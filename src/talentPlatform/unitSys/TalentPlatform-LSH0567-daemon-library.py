# -*- coding: utf-8 -*-
# ============================================
# 요구사항
# ============================================
# [요청] LSH0567. Python을 이용한 태양 흑점주기 분석 및 미래 예측 시뮬레이션
# https://www.swpc.noaa.gov/products/solar-cycle-progression

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


# ============================================
# 주요 설정
# ============================================
serviceName = 'LSH0567'
# contextPath = os.getcwd()
contextPath = f"/SYSTEMS/PROG/PYTHON/IDE"

globalVar = {
    'ctxPath': f"{contextPath}"
    , 'inpPath':  f"/DATA/INPUT/{serviceName}"
    , 'outPath': f"/DATA/OUTPUT/{serviceName}"
    , 'figPath': f"/DATA/FIG/{serviceName}"
}

for key, val in globalVar.items():
    if key.__contains__('Path'):
        os.makedirs(val, exist_ok=True)
        print(f"[CHECK] {key} : {val}")

# 작업 경로 설정
# os.chdir(f"{globalVar['ctxPath']}")
print(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    'url' : {
        # 훈련 데이터
        'sunspot': 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots.json'

        # 테스트 데이터
        , 'flux': 'https://services.swpc.noaa.gov/json/solar-cycle//f10-7cm-flux.json'
    }
}

# ============================================
# 비즈니스 로직
# ============================================
import requests
import json
import pymysql
import requests
from urllib.parse import quote_plus

# 유틸리티 함수
# ============================================
def getUrlData(url):

    result = None

    try:
        res = requests.get(url)
        resCode = res.status_code

        if resCode != 200: return result
        jsonData = json.loads(res.content.decode('utf-8'))

        if len(jsonData) < 1: return result
        result = pd.DataFrame(jsonData)

    except Exception as e:
        print(f"Exception : {e}")

    return result


# =======================================================
# sunspotData 데이터
# =======================================================
sunspotData = getUrlData(sysOpt['url']['sunspot'])
if sunspotData is None or len(sunspotData) < 1: print(f"[ERROR] sunspotData : {'입력 자료를 확인해주세요.'}")

sunspotData['date'] = pd.to_datetime(sunspotData['time-tag'], format='%Y-%m')


plt.figure(figsize=(10, 8))
plt.plot(sunspotData['date'], sunspotData['ssn'], label='Sunspot')
plt.title('Monthly Sunspot Number')
plt.xlabel('Date')
plt.ylabel('SSN')
plt.legend()
plt.show()

from pmdarima import auto_arima

# 자동 ARIMA 모델 적합
model = auto_arima(sunspotData['ssn'], seasonal=False, stepwise=True)

# 모델 요약 출력
print(model.summary())

# 예측
forecast_steps = 24  # 예측할 기간 설정 (월 단위)
forecast = model.predict(n_periods=forecast_steps)

# 예측 날짜 생성
last_date = sunspotData.index[-1]
forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)]

# 실제 값과 예측 값 시각화
plt.figure(figsize=(15, 8))
plt.plot(sunspotData.index, sunspotData['ssn'], label='Actual')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Sunspot Number Forecast with Auto ARIMA')
plt.xlabel('Date')
plt.ylabel('SSN')
plt.legend()
plt.show()



# =======================================================
# fluxData
# =======================================================
fluxData = getUrlData(sysOpt['url']['flux'])
if fluxData is None or len(fluxData) < 1: print(f"[ERROR] fluxData : {'입력 자료를 확인해주세요.'}")
fluxData['date'] = pd.to_datetime(fluxData['time-tag'], format='%Y-%m')

