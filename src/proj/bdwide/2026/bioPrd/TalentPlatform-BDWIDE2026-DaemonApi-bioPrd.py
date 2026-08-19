# ============================================
# 요구사항
# ============================================
# 바이오 공정 타겟 도달 시간 예측 API (FastAPI)
# 명세1 http://localhost:8000/docs
# 인증키 20260221-bdwide

# =================================================
# 도움말
# =================================================
# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-bioPrd:app --reload --host=0.0.0.0 --port=8000 &

# ============================================
# 라이브러리
# ============================================
import os
import sys
import logging
import logging.handlers
import platform
import time
import tempfile
import warnings
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

# 모델 예측 로직 직접 내장 (import 대체)
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

def lognorm_pdf(x, A, mu, sigma, C):
    return C + A * np.exp(- (np.log(x + 1e-5) - mu)**2 / (2 * sigma**2))

def fit_survival_curve(ages, turbs):
    try:
        p0 = [turbs.max(), np.log(ages[np.argmax(turbs)]+1e-5), 1.0, 0.05]
        bounds = ([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        popt, _ = curve_fit(lognorm_pdf, ages, turbs, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except:
        return None

def find_target_time(ages, turbs, target=0.253):
    if len(turbs) == 0: return np.nan
    peak_idx = np.argmax(turbs)
    after_peak_turbs = turbs[peak_idx:]
    after_peak_ages = ages[peak_idx:]
    
    target_indices = np.where(after_peak_turbs <= target)[0]
    if len(target_indices) > 0:
        return after_peak_ages[target_indices[0]]
    return np.nan

def load_and_resample(filepath, interval=1.0):
    df = pd.read_csv(filepath, encoding='cp949', skiprows=[1,2])
    candidate_cols = ['Age', 'TURB', 'PRESS', 'AFOAM', 'BASE', 'pH', 'TEMP']
    cols_to_use = [c for c in candidate_cols if c in df.columns]
    df = df[cols_to_use].dropna(subset=['Age', 'TURB'])
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(axis=1, how='all').sort_values('Age').reset_index(drop=True)
    
    max_age = int(np.ceil(df['Age'].max()))
    ages_grid = np.arange(0, max_age + interval, interval)
    
    resampled_data = {'Age': ages_grid}
    for col in df.columns:
        if col != 'Age':
            s = df[col].ffill().bfill()
            if s.isnull().all():
                continue
            f_interp = interp1d(df['Age'], s, bounds_error=False, fill_value=(s.iloc[0], s.iloc[-1]))
            resampled_data[col] = f_interp(ages_grid)
            
    resampled_df = pd.DataFrame(resampled_data)
    return resampled_df

def predict_time_to_target(filepath, model, cutoff_age=None):
    if model is None:
        print("Error: Model is not loaded.")
        return None
        
    res_df = load_and_resample(filepath)
    
    if cutoff_age is None:
        cutoff_age = res_df['Age'].max()
    
    history = res_df[res_df['Age'] <= cutoff_age].copy()
    if len(history) < 2:
        print(f"Error: Not enough data points before cutoff {cutoff_age}h.")
        return None
        
    ages_hist = history['Age'].values
    turbs_hist = history['TURB'].values
    
    # 1. Biological Curve Prediction (Bio_Pred_Time)
    pred_time_B = 150.0 # Default if fail
    ages_future = np.arange(cutoff_age+1.0, 200.0, 1.0)
    popt = fit_survival_curve(ages_hist, turbs_hist)
    if popt is not None:
        pred_curve_B = lognorm_pdf(ages_future, *popt)
        full_ages_B = np.concatenate([ages_hist, ages_future])
        full_turbs_B = np.concatenate([turbs_hist, pred_curve_B])
        pred_t = find_target_time(full_ages_B, full_turbs_B, target=0.253)
        if pd.notnull(pred_t):
            pred_time_B = pred_t
            
    # 2. Extract Multivariate Features (Early Warning Indicators)
    turb_curr = history['TURB'].iloc[-1]
    turb_trend_1 = turb_curr - history['TURB'].iloc[-5] if len(history) >= 5 else 0
    turb_trend_2 = history['TURB'].iloc[-5] - history['TURB'].iloc[-10] if len(history) >= 10 else 0
    turb_accel = turb_trend_1 - turb_trend_2
    
    press_curr = history['PRESS'].iloc[-1] if 'PRESS' in history else 0
    afoam_curr = history['AFOAM'].iloc[-1] if 'AFOAM' in history else 0
    
    ph_curr = history['pH'].iloc[-1] if 'pH' in history else 7.0
    ph_trend = ph_curr - history['pH'].iloc[-5] if 'pH' in history and len(history) >= 5 else 0
    
    base_curr = history['BASE'].iloc[-1] if 'BASE' in history else 0
    base_trend = base_curr - history['BASE'].iloc[-5] if 'BASE' in history and len(history) >= 5 else 0
    
    # Create feature dataframe in same order as training
    features = [
        'Cutoff_Age', 'Bio_Pred_Time', 
        'TURB_current', 'TURB_trend', 'TURB_accel',
        'PRESS_current', 'AFOAM_current',
        'pH_current', 'pH_trend',
        'BASE_current', 'BASE_trend'
    ]
    
    X_test = pd.DataFrame([{
        'Cutoff_Age': cutoff_age,
        'Bio_Pred_Time': pred_time_B,
        'TURB_current': turb_curr,
        'TURB_trend': turb_trend_1,
        'TURB_accel': turb_accel,
        'PRESS_current': press_curr,
        'AFOAM_current': afoam_curr,
        'pH_current': ph_curr,
        'pH_trend': ph_trend,
        'BASE_current': base_curr,
        'BASE_trend': base_trend
    }], columns=features)
    
    # Predict
    predicted_time = model.predict(X_test)[0]
    return predicted_time

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
    log = logging.getLogger(prjName)
    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)
    log.setLevel(level=logging.INFO)

    return log

# 인증키 검사
def chkKey(key: str = Depends(APIKeyHeader(name="key"))):
    if key != '20260221-bdwide':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status,
        "code": code,
        "message": message,
        "cnt": cnt,
        "data": data
    }

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'BDWIDE2026'
prjName = 'bioPrd'

# 로그 파일 등 저장을 위한 컨텍스트 경로 (운영 시 환경에 맞게 수정 필요)
ctxPath = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026'
log = initLog(env, ctxPath, prjName)

sysOpt = {
    'oriList': ['*'],
    # 'modelInfo': f"{ctxPath}/bioPrd/stacking_lgbm_model.pkl"
    'modelInfo': "C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/bdwide/2026/bioPrd/stacking_lgbm_model.pkl"
}

try:
    modelInfo = joblib.load(sysOpt['modelInfo'])
    log.info(f"Model loaded successfully from {sysOpt['modelInfo']}")
except Exception as e:
    log.error(f"Exception during model load : {e}")
    sys.exit(1)

app = FastAPI(
    title="AI 바이오 공정 예측 API",
    description="타겟 시간 도달 시점을 예측하는 API",
    version="1.0",
    openapi_url='/api',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=sysOpt['oriList'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post(f"/api/prd")
async def prd(
        file: UploadFile = File(..., description='공정 데이터 CSV 파일')
):
    """
    기능\n
        바이오 공정 타겟(TURB <= 0.253) 도달 시점 예측 API\n
    파라미터\n
        file: 공정 데이터 첨부파일 (.CSV)\n
    """
    temp_file_path = None
    try:
        start_time = time.time()
        filename_lower = file.filename.lower()
        if not filename_lower.endswith('.csv'):
            return resResponse("fail", 400, "예측 실패, 유효한 CSV 첨부파일이 아님")
            
        # 클라이언트가 업로드한 파일을 임시 파일로 저장
        fd, temp_file_path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, 'wb') as f:
            content = await file.read()
            f.write(content)
            
        # 기존 모델 스크립트의 예측 함수 호출
        pred = predict_time_to_target(temp_file_path, model=modelInfo)
        
        if pred is None:
            return resResponse("fail", 500, "예측 실패, 데이터가 부족하거나 모델 연산 실패")
            
        output = {
            "prd": float(pred),
            "filename": file.filename
        }
        
        # log.info(f"predictTimeToTarget output : {output}")
        return resResponse("succ", 200, "처리 완료", 1, output)
        
    except Exception as e:
        log.error(f'Exception in predictTimeToTarget : {e}')
        return resResponse("fail", 400, f"예측 실패, {e}")
    finally:
        # 예측 완료 후 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)