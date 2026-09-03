# =================================================
# 도움말
# =================================================
# http://49.247.41.71:9931

# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026/bioPrd
# /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/python -m streamlit run TalentPlatform-BDWIDE2026-streamlit-bioPrd.py --server.port 9931
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/python -m streamlit run TalentPlatform-BDWIDE2026-streamlit-bioPrd.py --server.port 9931

# ============================================
# 라이브러리
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import tempfile
import os
import warnings
import os
import sys
import logging
import logging.handlers
import platform
import time
import tempfile
import altair as alt
import warnings
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

@st.cache_resource
def load_models(sysOpt):
    return joblib.load(sysOpt['modelInfo'])

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
        st.error("Error: Model is not loaded.")
        return None, None, None
        
    res_df = load_and_resample(filepath)
    
    if cutoff_age is None:
        cutoff_age = res_df['Age'].max()
    
    history = res_df[res_df['Age'] <= cutoff_age].copy()
    if len(history) < 2:
        st.error(f"Error: Not enough data points before cutoff {cutoff_age}h.")
        return None, None, None
        
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
    return predicted_time, cutoff_age, turb_curr

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'BDWIDE2026'
prjName = 'streamlit-bioPrd'

ctxPath = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026'
log = initLog(env, ctxPath, prjName)

sysOpt = {
    # 'modelInfo': "C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/src/proj/bdwide/2026/bioPrd/stacking_lgbm_model.pkl"
    'modelInfo': "/HDD/DATA/INPUT/BDWIDE2026/bioPrd/stacking_lgbm_model.pkl"
}

# ============================================
# 비즈니스 로직
# ============================================
try:
    models = load_models(sysOpt)
except Exception as e:
    st.error(f"'stacking_lgbm_model.pkl' 파일을 찾을 수 없거나 로드할 수 없습니다: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI 배양예측")

st.title("AI 배양예측 예측 시스템")
st.write("실시간 공정 데이터 (Age, TURB, PRESS 등) 및 생물학적 생존 곡선을 조합하여 예상시간 (TURB <= 0.25)을 예측합니다.")

# st.header("CSV 공정 데이터")

st.subheader("CSV 공정 데이터 직접 입력 또는 업로드")
uploaded_file = st.file_uploader("파일 업로드 시 신규 테이블 제공", type=['csv'])

if uploaded_file is not None:
    try:
        df_temp = pd.read_csv(uploaded_file, encoding='cp949', skiprows=[1,2])
    except:
        uploaded_file.seek(0)
        df_temp = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=[1,2])
        
    standard_cols = ['Age', 'TURB', 'PRESS', 'AFOAM', 'BASE', 'pH', 'TEMP']
    df_display = pd.DataFrame(columns=standard_cols)
    
    for col in standard_cols:
        if col in df_temp.columns:
            df_display[col] = pd.to_numeric(df_temp[col], errors='coerce')
        else:
            df_display[col] = np.nan
            
    # Age와 TURB가 없는 더미 데이터나 쓰레기값 제거
    df_display = df_display.dropna(subset=['Age', 'TURB']).reset_index(drop=True)
else:
    # 기본 제공 템플릿 테이블 (예시 데이터)
    df_display = pd.DataFrame({
        'Age': [0.0, 5.0, 10.0, 15.0, 20.0],
        'TURB': [0.45, 0.43, 0.40, 0.35, 0.28],
        'PRESS': [1.0, 1.0, 1.0, 1.0, 1.0],
        'AFOAM': [0.0, 0.0, 0.0, 0.0, 0.0],
        'BASE': [15.0, 15.1, 15.2, 15.3, 15.5],
        'pH': [6.8, 6.75, 6.7, 6.65, 6.6],
        'TEMP': [36.5, 36.5, 36.5, 36.5, 36.5]
    })

st.write("테이블에서 직접 입력 또는 업로드 기능을 지원합니다.")
edited_df = st.data_editor(df_display, num_rows="dynamic", use_container_width=True, key="csv_editor")

if st.button("AI 배양 예측"):
    temp_file_path = None
    try:
        # 편집된 DataFrame을 임시 CSV로 저장
        # (predict_time_to_target의 load_and_resample 함수가 skiprows=[1,2]를 수행하므로 더미 행 추가)
        fd, temp_file_path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, 'w', encoding='cp949', newline='') as f:
            cols = edited_df.columns.tolist()
            f.write(",".join(str(c) for c in cols) + "\n")
            f.write(",".join(["dummy"] * len(cols)) + "\n")
            f.write(",".join(["dummy"] * len(cols)) + "\n")
            edited_df.to_csv(f, index=False, header=False)
            
        pred_time, current_age, current_turb = predict_time_to_target(temp_file_path, models)
        
        if pred_time is not None:
            target_level = 0.253
            st.success("정상적으로 처리되었습니다.")

            col1, col2, col3 = st.columns(3)
            col1.metric(label=f"현재 시간", value=f"{current_age:.1f} hr")
            col2.metric(label=f"예측 시간 (TURB {target_level:.2f})", value=f"{pred_time:.1f} hr")
            col3.metric(label=f"남은 시간", value=f"{max(0, pred_time - current_age):.1f} hr")
            
            # 시각화 (CSV 기반) - Streamlit 내장(Altair) 웹 차트
            # st.subheader("배양 시간 추세 및 예측 타겟 시각화")
            df_res = load_and_resample(temp_file_path)
            
            # 1. 실제 추세선
            line = alt.Chart(df_res).mark_line(
                color='blue', 
                point=alt.OverlayMarkDef(color='blue', size=60, filled=True)
            ).encode(
                x=alt.X('Age', title='Age 배양 시간', scale=alt.Scale(domainMin=0)),
                y=alt.Y('TURB', title='TURB'),
                tooltip=[
                    alt.Tooltip('Age', title='Age 배양 시간', format='.1f'),
                    alt.Tooltip('TURB', title='TURB', format='.4f')
                ]
            )
            
            # 2. Target 라인
            target_rule = alt.Chart(pd.DataFrame({'y': [target_level]})).mark_rule(
                color='gray', strokeDash=[5, 5]
            ).encode(y='y')
            
            if current_turb > target_level:
                # 3. 예측 도착 지점
                point = alt.Chart(pd.DataFrame({'Age': [pred_time], 'TURB': [target_level]})).mark_circle(
                    color='red', size=200
                ).encode(
                    x='Age', y='TURB', 
                    tooltip=[
                        alt.Tooltip('Age', title='예측 도달 시간', format='.1f'), 
                        alt.Tooltip('TURB', title='Target', format='.4f')
                    ]
                )
                
                # 4. 예측 경로
                path = alt.Chart(pd.DataFrame({'Age': [current_age, pred_time], 'TURB': [current_turb, target_level]})).mark_line(
                    color='red', strokeDash=[5, 5]
                ).encode(
                    x='Age', y='TURB'
                )
                
                final_chart = line + target_rule + point + path
            else:
                final_chart = line + target_rule
                
            st.altair_chart(final_chart.interactive(), use_container_width=True)
            
            # results = []
            # if current_turb <= target_level:
            #     results.append({"목표 Turbidity": f"<= {target_level}", "상태": "이미 도달함", "예상 도달 시간": "-"})
            # else:
            #     results.append({"목표 Turbidity": f"<= {target_level}", "상태": "예측 완료", "예상 도달 시간": f"{pred_time:.1f} 시간"})
            #
            # st.table(pd.DataFrame(results))
            
    except Exception as e:
        st.error(f"예측 실패, {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)