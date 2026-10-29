import sys
import os
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xarray as xr

# Register NanumBarunGothic font
font_path = r'C:\SYSTEMS\PROG\WEB\NMSC\dms02_homepage\src\main\webapp\resources\homepage\fonts\NanumBarunGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False

# Add toolbox path
sys.path.append(r'C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox')
from nmsc_climate_toolbox import NMSCClimateToolbox

nct = NMSCClimateToolbox()

# Sample paths
doc_dir = r'C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\doc'
p_prod = os.path.join(doc_dir, 'multi_merge_le3_cdr-sst01m_ea020lc_202312010000.nc')
p_obs = os.path.join(doc_dir, 'multi_merge_le3_cdr-sst01m-obs_ea020lc_202312010000.nc')

print("Loading datasets...")
ds_prod = nct.open(p_prod)
ds_obs = nct.open(p_obs)
var_name = 'SST_L3_monthly'

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code_str, outputs=None):
    if outputs is None:
        outputs = []
    cells.append({
        "cell_type": "code",
        "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_str.strip().split("\n")]
    })

# 0. Title
add_md("""# (해양) 해수면온도(SST) 분석 툴박스 활용 가이던스

이 주피터 노트북은 `nmsc_climate_toolbox.py` 모듈을 활용하여 **핵심기후변수(ECV) 해수면온도(Sea Surface Temperature, SST)** 위성 기후자료의 전처리, 기후 평년값 및 편차 산출, 공간 분포도 시각화, 시계열 추세 분석, 그리고 **현장 관측자료(해양 기상부이)와의 시공간 매칭 및 검증 산점도 분석**까지 전 과정을 체계적으로 안내합니다.""")

# 1. Dataset Overview with fixed-layout HTML table
add_md("""## 1. 데이터셋 및 샘플자료 개요

본 가이던스에서 활용하는 위성 기후 생산자료와 현장 관측 검증자료의 상세 정보는 다음과 같습니다.

### 1.1. 주요 자료 설명

<table style="width: 100%; max-width: 100%; table-layout: fixed; border-collapse: collapse; margin: 15px 0;">
<thead>
<tr style="background-color: #f1f5f9;">
<th style="width: 24%; border: 1px solid #cbd5e1; padding: 8px 10px; font-weight: 700; text-align: center;">자료 구분</th>
<th style="width: 28%; border: 1px solid #cbd5e1; padding: 8px 10px; font-weight: 700; text-align: center;">파일명 (샘플)</th>
<th style="width: 22%; border: 1px solid #cbd5e1; padding: 8px 10px; font-weight: 700; text-align: center;">주요 변수명</th>
<th style="width: 15%; border: 1px solid #cbd5e1; padding: 8px 10px; font-weight: 700; text-align: center;">공간/지점 범위</th>
<th style="width: 11%; border: 1px solid #cbd5e1; padding: 8px 10px; font-weight: 700; text-align: center;">시간 범위 / 주기</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all;"><strong>해수면온도 생산자료<br>(L3 CDR)</strong></td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all; font-family: 'Consolas', monospace; font-size: 11.5px; color: #2563eb;">multi_merge_le3_cdr-sst01m_ea020lc_202312010000.nc</td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all; font-size: 12px;"><code>SST_L3_monthly</code> (°C)<br><code>analysis_uncertainty</code><br><code>observation_count</code></td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: keep-all; font-size: 12px;">동아시아 (15°~50°N, 105°~150°E)<br>해상도 0.05° (약 2km)</td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: keep-all; font-size: 12px;">2015.01 ~ 2023.12<br>(108개월, 월간)</td>
</tr>
<tr>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all;"><strong>해수면온도 검증자료<br>(해양기상부이)</strong></td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all; font-family: 'Consolas', monospace; font-size: 11.5px; color: #059669;">multi_merge_le3_cdr-sst01m-obs_ea020lc_202312010000.nc</td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: break-all; font-size: 12px;"><code>SST_buoy</code> (°C)<br><code>station_name</code> (관측소명)<br><code>lat</code>, <code>lon</code> (위경도)</td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: keep-all; font-size: 12px;">기상청 주요 해양기상부이 (30개소)</td>
<td style="border: 1px solid #cbd5e1; padding: 8px 10px; word-break: keep-all; font-size: 12px;">2022.01 ~ 2024.12<br>(매시간 / 월평균)</td>
</tr>
</tbody>
</table>

### 1.2. 관련 참조 문서
- **PUG (Product User Guide)**: NMSC 해수면온도 기후자료 레코드(CDR) 포맷 명세 및 좌표계 정의
- **ATBD (Algorithm Theoretical Basis Document)**: 천리안위성(COMS 및 GK2A) 융합 물리적 복사 알고리즘
- **Validation Report (품질검증 보고서)**: 전 지구 해양 관측 부이(In-situ Buoy) 비교 검증 결과

### 1.3. 핵심 용어 사전
- **NMSC (National Meteorological Satellite Center)**: 국가기상위성센터
- **ECV (Essential Climate Variable)**: 핵심기후변수 (기후 모니터링에 필수적인 관측 물리량)
- **CDR (Climate Data Record)**: 과거부터 장기간에 걸쳐 일관성 있게 생산된 기후 데이터 기록
- **Climatology (기후 평년값)**: 장기 기준 기간(예: 2015~2020년 등)의 월별/연도별 평균 기후 상태
- **Anomaly (편차)**: 특정 시점의 관측값이 기후 평년값에서 벗어난 차이(관측값 - 평년값)
- **Sen's Slope**: 시계열 자료의 이상치(Outlier)에 민감하지 않은 비모수적(Non-parametric) 장기 선형 경향성 추정치""")

# 2. Open Data
code_step2 = """import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os

# 한글 폰트 설정 (나눔바른고딕 / 맑은 고딕)
font_paths = [
    'resources/homepage/fonts/NanumBarunGothic.ttf',
    'C:/SYSTEMS/PROG/WEB/NMSC/dms02_homepage/src/main\webapp/resources/homepage/fonts/NanumBarunGothic.ttf',
    'C:/Windows/Fonts/malgun.ttf'
]
for fp in font_paths:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

from nmsc_climate_toolbox import NMSCClimateToolbox
nct = NMSCClimateToolbox()

# 1) 해수면온도 생산자료 샘플 경로
file_prod = 'doc/multi_merge_le3_cdr-sst01m_ea020lc_202312010000.nc'
ds_prod = nct.open(file_prod)
var_name = 'SST_L3_monthly'

print("=== [생산자료] 해수면온도 (L3 CDR) Dataset ===")
print(ds_prod)

# 2) 해수면온도 검증자료 샘플 경로
file_obs = 'doc/multi_merge_le3_cdr-sst01m-obs_ea020lc_202312010000.nc'
ds_obs = nct.open(file_obs)

print("\\n=== [검증자료] 해양기상부이 (Buoy) Dataset ===")
print(ds_obs)"""

out_step2 = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "Warning: xCDAT bounds generation failed: 'Dataset' object has no attribute 'bounds'\n",
            "=== [생산자료] 해수면온도 (L3 CDR) Dataset ===\n",
            str(ds_prod) + "\n\n",
            "Warning: xCDAT bounds generation failed: 'Dataset' object has no attribute 'bounds'\n",
            "=== [검증자료] 해양기상부이 (Buoy) Dataset ===\n",
            str(ds_obs) + "\n"
        ]
    }
]
add_md("## 2. 데이터 읽기\n\n`nmsc_climate_toolbox.py`의 `open()` 메서드를 통해 NetCDF 파일 포맷의 위성 생산자료와 현장 관측 검증자료를 로드합니다.")
add_code(code_step2, out_step2)

# 3. Filter
code_step3 = """# 분석 대상 시간 구간(2015-01 ~ 2020-12) 및 위경도 영역 필터링
ds_filtered = nct.filter(ds_prod, time_slice=slice('2015-01-01', '2020-12-31'))
print(f"필터링 후 시간 범위: {ds_filtered.time.min().values} ~ {ds_filtered.time.max().values}")
print(f"데이터셋 크기: {dict(ds_filtered.sizes)}")"""

ds_filtered = nct.filter(ds_prod, time_slice=slice('2015-01-01', '2020-12-31'))
out_step3 = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            f"필터링 후 시간 범위: {ds_filtered.time.min().values} ~ {ds_filtered.time.max().values}\n",
            f"데이터셋 크기: {dict(ds_filtered.sizes)}\n"
        ]
    }
]
add_md("## 3. 시공간 필터링\n\n`nct.filter()`를 사용하여 특정 기간(`time_slice`) 또는 특정 위경도 구역을 슬라이싱하여 분석 효율성을 극대화합니다.")
add_code(code_step3, out_step3)

# 4. Climatology & Anomaly
code_step4 = """# 기후 평년값(Climatology) 및 편차(Anomaly) 산출
# 기준 평년 기간: 2015-01-01 ~ 2020-12-31 (월별 주기 '1MS')
ds_resampled, ds_cli, ds_ano = nct.process_climate_data(
    dataset=ds_prod,
    freq='1MS',
    op='평균',
    start_yr='2015-01-01',
    end_yr='2020-12-31'
)
print("=== 기후 평년값 (ds_cli) ===")
print(ds_cli)

print("\\n=== 기후 편차 (ds_ano) ===")
print(ds_ano)"""

ds_resampled, ds_cli, ds_ano = nct.process_climate_data(
    dataset=ds_prod,
    freq='1MS',
    op='평균',
    start_yr='2015-01-01',
    end_yr='2020-12-31'
)
out_step4 = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "=== 기후 평년값 (ds_cli) ===\n",
            str(ds_cli) + "\n\n",
            "=== 기후 편차 (ds_ano) ===\n",
            str(ds_ano) + "\n"
        ]
    }
]
add_md("## 4. 기후 평년값 및 편차 산출\n\n`nct.process_climate_data()` 함수를 통해 설정된 기준 기간 동안의 월별 기후 평년값(Climatology)과 원자료 대비 기후 편차(Anomaly)를 일괄 계산합니다.")
add_code(code_step4, out_step4)

# 5. Image Maps
add_md("## 5. 이미지 영상 시각화\n\n`nct.resMap()` 메서드는 람베르트 정각원추도법(LCC) 투영 좌표계를 기반으로 해안선, 등치선(Contour Line) 및 레이블, 고해상도 컬러바를 포함한 분석 영상을 생성합니다.")

# 5.1 Original Map
code_map1 = """# 특정 시점(예: 2015년 1월)의 원자료 해수면온도 공간분포도 표출
nct.resMap(
    dataset=ds_prod,
    variable=var_name,
    time_idx=0,
    cmap_name='jet',
    custom_title="Monthly Mean ECV Sea Surface Temperature (%Y.%m)",
    custom_legend="Sea Surface Temperature (°C)",
    return_type='fig'
)
plt.show()"""

b64_map1 = nct.resMap(
    dataset=ds_prod,
    variable=var_name,
    time_idx=0,
    cmap_name='jet',
    custom_title="Monthly Mean ECV Sea Surface Temperature (%Y.%m)",
    custom_legend="Sea Surface Temperature (°C)",
    return_type='b64'
)

out_map1 = [
    {
        "data": {
            "image/png": b64_map1,
            "text/plain": ["<Figure size 800x666.667 with 2 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 5.1. 월평균 해수면온도 분포도 (Original Map)")
add_code(code_map1, out_map1)

# 5.2 Anomaly Map
code_map2 = """# 특정 시점의 해수면온도 평년 편차(Anomaly) 공간분포도 표출
nct.resMap(
    dataset=ds_prod,
    variable=var_name,
    time_idx=0,
    data_layer='anomaly',
    cmap_name='SST_ANOM (custom)',
    dataset_cli=ds_cli,
    custom_title="ECV Sea Surface Temperature Anomaly (%Y.%m - Climatology)",
    custom_legend="Sea Surface Temperature Anomaly (°C)",
    return_type='fig'
)
plt.show()"""

b64_map2 = nct.resMap(
    dataset=ds_prod,
    variable=var_name,
    time_idx=0,
    data_layer='anomaly',
    cmap_name='SST_ANOM (custom)',
    dataset_cli=ds_cli,
    custom_title="ECV Sea Surface Temperature Anomaly (%Y.%m - Climatology)",
    custom_legend="Sea Surface Temperature Anomaly (°C)",
    return_type='b64'
)

out_map2 = [
    {
        "data": {
            "image/png": b64_map2,
            "text/plain": ["<Figure size 800x666.667 with 2 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 5.2. 해수면온도 평년 편차 분포도 (Anomaly Map)")
add_code(code_map2, out_map2)

# 6. Time Series & Validation (Core enhancement)
add_md("""## 6. 시계열 분석 및 산점도 검증

이 장에서는 동아시아 전 해역의 장기 해수면온도 변화 경향을 분석하고, **기상청 해양 기상부이(30개 관측소) 실측 자료와 위성 생산자료 간의 시공간 일치(Spatial-Temporal Matching)를 수행하여 통계적 정확도(상관계수 R, 편차 Bias, 산점도)를 검증**합니다.""")

# 6.1 Spatial Mean & Trend
code_ts = """# 1) 동아시아 전 해역 공간 평균(Spatial Mean) 산출
ts_mean_da = nct.spaMean(ds_resampled, variable=var_name)

# 2) 선형 추세선(Linear Trend) 산출
trend_line, slope, p_value = nct.trend(ts_mean_da)
print(f"선형 추세 기울기 (Slope): {slope:.5f} °C/month, 유의수준 (p-value): {p_value:.4e}")

# 3) 시계열 및 추세선 동시 시각화
fig_ts = nct.timeGrp(ts_mean_da, trend_line=trend_line, variable_name="Sea Surface Temperature (°C)")
plt.show()"""

ts_mean_da = nct.spaMean(ds_resampled, variable=var_name)
trend_line, slope, p_value = nct.trend(ts_mean_da)
fig_ts = nct.timeGrp(ts_mean_da, trend_line=trend_line, variable_name="Sea Surface Temperature (°C)")
buf = io.BytesIO()
fig_ts.savefig(buf, format='png', bbox_inches='tight', dpi=150)
plt.close(fig_ts)
buf.seek(0)
b64_ts = base64.b64encode(buf.read()).decode('utf-8')

out_ts = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [f"선형 추세 기울기 (Slope): {slope:.5f} °C/month, 유의수준 (p-value): {p_value:.4e}\n"]
    },
    {
        "data": {
            "image/png": b64_ts,
            "text/plain": ["<Figure size 1000x500 with 1 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 6.1. 공간 평균 시계열 및 장기 추세 분석\n\n`nct.spaMean()`으로 공간 영역 평균 시계열을 도출하고, `nct.trend()`를 통해 선형 회귀 기울기 및 유의수준(p-value)을 계산한 뒤 `nct.timeGrp()`으로 시각화합니다.")
add_code(code_ts, out_ts)

# 6.2 Spatial-temporal matching
code_match = """# -------------------------------------------------------------
# 6.2. 위성 생산자료와 현장 관측(부이) 검증자료 시공간 일치 (Matching)
# -------------------------------------------------------------
stations = ds_obs['station_id'].values
matched_results = []

for stn in stations:
    lat_val = float(ds_obs['lat'].sel(station_id=stn).values)
    lon_val = float(ds_obs['lon'].sel(station_id=stn).values)
    stn_name = str(ds_obs['station_name'].sel(station_id=stn).values).strip()
    
    # 1) 부이 좌표와 최근접(nearest) 격자의 위성 데이터 추출
    sat_da = ds_prod['SST_L3_monthly'].sel(lat=lat_val, lon=lon_val, method='nearest')
    df_sat = sat_da.to_dataframe().reset_index()[['time', 'SST_L3_monthly']].rename(columns={'SST_L3_monthly': 'SST_Sat'})
    df_sat['time'] = pd.to_datetime(df_sat['time'])
    
    # 2) 부이 시계열 데이터를 월평균(Month Start, MS)으로 리샘플링
    buoy_da = ds_obs['SST_buoy'].sel(station_id=stn)
    df_buoy = buoy_da.to_dataframe().reset_index()
    df_buoy['time'] = pd.to_datetime(df_buoy['time'])
    df_buoy_m = df_buoy.set_index('time').resample('MS')['SST_buoy'].mean().reset_index().rename(columns={'SST_buoy': 'SST_Buoy'})
    
    # 3) 공통 관측 시점 기준 내부 조인 (Inner Join)
    merged_df = pd.merge(df_sat, df_buoy_m, on='time', how='inner').dropna()
    merged_df['station_id'] = stn
    merged_df['station_name'] = stn_name
    merged_df['lat'] = lat_val
    merged_df['lon'] = lon_val
    
    matched_results.append(merged_df)

# 전체 관측소 매칭 결과 통합
final_matched_df = pd.concat(matched_results, ignore_index=True)
print(f"총 매칭 관측 지점 수: {len(stations)}개소")
print(f"총 시공간 유효 일치 샘플 수: {len(final_matched_df)}건")
print("\\n[매칭 결과 샘플 데이터]")
print(final_matched_df[['time', 'station_id', 'station_name', 'SST_Sat', 'SST_Buoy']].head(8))"""

stations = ds_obs['station_id'].values
matched_results = []
for stn in stations:
    lat_val = float(ds_obs['lat'].sel(station_id=stn).values)
    lon_val = float(ds_obs['lon'].sel(station_id=stn).values)
    stn_name = str(ds_obs['station_name'].sel(station_id=stn).values).strip()
    sat_da = ds_prod['SST_L3_monthly'].sel(lat=lat_val, lon=lon_val, method='nearest')
    df_sat = sat_da.to_dataframe().reset_index()[['time', 'SST_L3_monthly']].rename(columns={'SST_L3_monthly': 'SST_Sat'})
    df_sat['time'] = pd.to_datetime(df_sat['time'])
    
    buoy_da = ds_obs['SST_buoy'].sel(station_id=stn)
    df_buoy = buoy_da.to_dataframe().reset_index()
    df_buoy['time'] = pd.to_datetime(df_buoy['time'])
    df_buoy_m = df_buoy.set_index('time').resample('MS')['SST_buoy'].mean().reset_index().rename(columns={'SST_buoy': 'SST_Buoy'})
    
    merged_df = pd.merge(df_sat, df_buoy_m, on='time', how='inner').dropna()
    merged_df['station_id'] = stn
    merged_df['station_name'] = stn_name
    merged_df['lat'] = lat_val
    merged_df['lon'] = lon_val
    matched_results.append(merged_df)

final_matched_df = pd.concat(matched_results, ignore_index=True)

out_match = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            f"총 매칭 관측 지점 수: {len(stations)}개소\n",
            f"총 시공간 유효 일치 샘플 수: {len(final_matched_df)}건\n\n",
            "[매칭 결과 샘플 데이터]\n",
            final_matched_df[['time', 'station_id', 'station_name', 'SST_Sat', 'SST_Buoy']].head(8).to_string() + "\n"
        ]
    }
]
add_md("### 6.2. 위성 생산자료와 현장 관측(부이) 시공간 일치 (Matching)\n\n위성 격자점과 지점 관측(In-situ Buoy) 데이터 간 공간적 오차를 최소화하기 위해 최근접 격자 추출(`method='nearest'`)을 수행하고, 위성 월별 시점에 맞추어 부이 자료를 월평균(MS)으로 리샘플링하여 시공간 매칭 데이터셋을 구축합니다.")
add_code(code_match, out_match)

# 6.3 Scatter plot validation (WITH KOREAN FONT APPLIED)
code_sca = """# -------------------------------------------------------------
# 6.3. 검증자료 비교 산점도 (nct.scaGrp) 표출 및 통계 검증
# -------------------------------------------------------------
ds_sat_matched = xr.Dataset({'SST_Sat': ('sample', final_matched_df['SST_Sat'].values)})
ds_buoy_matched = xr.Dataset({'SST_Buoy': ('sample', final_matched_df['SST_Buoy'].values)})

# 산점도 및 검증 통계 산출 (나눔바른고딕 한글 폰트 적용)
fig_sca = nct.scaGrp(ds_sat_matched, ds_buoy_matched, 'SST_Sat', 'SST_Buoy')
plt.show()

# 통계 지표 계산
r_val = np.corrcoef(final_matched_df['SST_Sat'], final_matched_df['SST_Buoy'])[0, 1]
bias_val = np.mean(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])
rmse_val = np.sqrt(np.mean((final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])**2))
mae_val = np.mean(np.abs(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy']))

print(f"=== 위성 생산자료 vs 해양기상부이 검증 결과 ===")
print(f"상관계수 (Correlation R) : {r_val:.4f}")
print(f"평균 오차 (Mean Bias)      : {bias_val:.4f} °C")
print(f"평균 제곱근 오차 (RMSE)  : {rmse_val:.4f} °C")
print(f"평균 절대 오차 (MAE)     : {mae_val:.4f} °C")"""

r_val = float(np.corrcoef(final_matched_df['SST_Sat'], final_matched_df['SST_Buoy'])[0, 1])
bias_val = float(np.mean(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy']))
rmse_val = float(np.sqrt(np.mean((final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])**2)))
mae_val = float(np.mean(np.abs(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])))

ds_sat_matched = xr.Dataset({'SST_Sat': ('sample', final_matched_df['SST_Sat'].values)})
ds_buoy_matched = xr.Dataset({'SST_Buoy': ('sample', final_matched_df['SST_Buoy'].values)})

fig_sca = nct.scaGrp(ds_sat_matched, ds_buoy_matched, 'SST_Sat', 'SST_Buoy')
fig_sca.axes[0].set_title(f"검증자료 비교 산점도 (Scatter Plot)\nR: {r_val:.3f}, Bias: {bias_val:.3f}", fontname=plt.rcParams['font.family'], fontsize=12, fontweight='bold')
buf = io.BytesIO()
fig_sca.savefig(buf, format='png', bbox_inches='tight', dpi=150)
plt.close(fig_sca)
buf.seek(0)
b64_sca = base64.b64encode(buf.read()).decode('utf-8')

r_val = float(np.corrcoef(final_matched_df['SST_Sat'], final_matched_df['SST_Buoy'])[0, 1])
bias_val = float(np.mean(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy']))
rmse_val = float(np.sqrt(np.mean((final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])**2)))
mae_val = float(np.mean(np.abs(final_matched_df['SST_Sat'] - final_matched_df['SST_Buoy'])))

out_sca = [
    {
        "data": {
            "image/png": b64_sca,
            "text/plain": ["<Figure size 600x600 with 2 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    },
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "=== 위성 생산자료 vs 해양기상부이 검증 결과 ===\n",
            f"상관계수 (Correlation R) : {r_val:.4f}\n",
            f"평균 오차 (Mean Bias)      : {bias_val:.4f} °C\n",
            f"평균 제곱근 오차 (RMSE)  : {rmse_val:.4f} °C\n",
            f"평균 절대 오차 (MAE)     : {mae_val:.4f} °C\n"
        ]
    }
]
add_md("### 6.3. 검증 산점도(Scatter Plot) 및 오차 검증 지표 표출\n\n`nct.scaGrp()`을 호출하여 Hexbin 밀도 분포, 1:1 일치선, 상관계수(R) 및 편차(Bias)를 산점도로 표출합니다.")
add_code(code_sca, out_sca)

# 6.4 Station Time Series Comparison (WITH KOREAN FONT)
code_stn_ts = """# 대표 관측소(예: 덕적도 22101 지점) 시계열 비교 그래프
target_stn = 22101
stn_df = final_matched_df[final_matched_df['station_id'] == target_stn].sort_values('time')
stn_name = stn_df['station_name'].iloc[0] if len(stn_df) > 0 else str(target_stn)

plt.figure(figsize=(10, 5))
plt.plot(stn_df['time'], stn_df['SST_Sat'], label='Satellite SST (L3 CDR)', color='#2563eb', marker='o', linewidth=2)
plt.plot(stn_df['time'], stn_df['SST_Buoy'], label=f'Buoy SST ({stn_name})', color='#ef4444', marker='s', linestyle='--', linewidth=2)

plt.title(f"SST Time Series Validation (Station: {stn_name} #{target_stn})", fontsize=13, fontweight='bold')
plt.xlabel("Time", fontsize=11)
plt.ylabel("Sea Surface Temperature (°C)", fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""

target_stn = 22101
stn_df = final_matched_df[final_matched_df['station_id'] == target_stn].sort_values('time')
stn_name = stn_df['station_name'].iloc[0] if len(stn_df) > 0 else str(target_stn)
fig_stn = plt.figure(figsize=(10, 5))
plt.plot(stn_df['time'], stn_df['SST_Sat'], label='Satellite SST (L3 CDR)', color='#2563eb', marker='o', linewidth=2)
plt.plot(stn_df['time'], stn_df['SST_Buoy'], label=f'Buoy SST ({stn_name})', color='#ef4444', marker='s', linestyle='--', linewidth=2)
plt.title(f"SST Time Series Validation (Station: {stn_name} #{target_stn})", fontsize=13, fontweight='bold')
plt.xlabel("Time", fontsize=11)
plt.ylabel("Sea Surface Temperature (°C)", fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

buf = io.BytesIO()
fig_stn.savefig(buf, format='png', bbox_inches='tight', dpi=150)
plt.close(fig_stn)
buf.seek(0)
b64_stn = base64.b64encode(buf.read()).decode('utf-8')

out_stn_ts = [
    {
        "data": {
            "image/png": b64_stn,
            "text/plain": ["<Figure size 1000x500 with 1 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 6.4. 주요 관측소별 시계열 비교 검증\n\n특정 해양기상부이 관측소(예: 덕적도 등)를 지정하여 위성 수온과 현장 관측 수온의 시간대별 일치도를 시각적으로 비교합니다.")
add_code(code_stn_ts, out_stn_ts)

# 7. Statistical & Extremes
add_md("## 7. 통계 시각화 및 극한 기후 분석\n\n`nmsc_climate_toolbox.py`의 `histGrp()`, `barGrp()`, `ext()` 함수를 활용하여 데이터 분포, 연도별 통계량, 95% 백분위수 기반의 고수온 극한 사상 빈도 및 Sen's Slope를 산출합니다.")

# 7.1 Histogram
code_hist = """# 특정 시점의 공간 격자 데이터 분포 및 확률밀도(KDE) 함수 표출
data_at_time = ds_prod[var_name].isel(time=0)
fig_hist = nct.histGrp(data_at_time, bins=30, variable_name="Sea Surface Temperature (°C)")
plt.show()"""

data_at_time = ds_prod[var_name].isel(time=0)
fig_hist = nct.histGrp(data_at_time, bins=30, variable_name="Sea Surface Temperature (°C)")
buf = io.BytesIO()
fig_hist.savefig(buf, format='png', bbox_inches='tight', dpi=150)
plt.close(fig_hist)
buf.seek(0)
b64_hist = base64.b64encode(buf.read()).decode('utf-8')

out_hist = [
    {
        "data": {
            "image/png": b64_hist,
            "text/plain": ["<Figure size 800x600 with 1 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 7.1. 데이터 분포 확인 (히스토그램 및 KDE)")
add_code(code_hist, out_hist)

# 7.2 Bar Graph
code_bar = """# 연도별 평균 해수면온도 산출 및 막대 그래프 표출
yearly_mean = ts_mean_da.groupby('time.year').mean(dim='time')
years = [str(y) for y in yearly_mean['year'].values]
values = yearly_mean.values

fig_bar = nct.barGrp(years, values, title="Yearly Mean Sea Surface Temperature (East Asia)", xlabel="Year", ylabel="Mean SST (°C)")
plt.show()"""

yearly_mean = ts_mean_da.groupby('time.year').mean(dim='time')
years = [str(y) for y in yearly_mean['year'].values]
values = yearly_mean.values

fig_bar = nct.barGrp(years, values, title="Yearly Mean Sea Surface Temperature (East Asia)", xlabel="Year", ylabel="Mean SST (°C)")
buf = io.BytesIO()
fig_bar.savefig(buf, format='png', bbox_inches='tight', dpi=150)
plt.close(fig_bar)
buf.seek(0)
b64_bar = base64.b64encode(buf.read()).decode('utf-8')

out_bar = [
    {
        "data": {
            "image/png": b64_bar,
            "text/plain": ["<Figure size 1000x500 with 1 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]
add_md("### 7.2. 연도별 통계량 비교 (막대 그래프)")
add_code(code_bar, out_bar)

# 7.3 Extreme events
code_ext = """# 95% 분위수(상위 5% 고수온) 임계값 산출 및 극한 기후 분석
ext_results = nct.ext(ts_mean_da, percentile=0.95)

print("=== 동아시아 해수면온도 극한 기후 지수 분석 결과 ===")
print(f"95% 백분위수 고수온 임계값 : {ext_results['threshold']:.3f} °C")
print(f"임계값 초과 극한 사상 발생 횟수 : {ext_results['extreme_count']}회")
print(f"Sen's Slope (견고한 장기 추세)    : {ext_results['sens_slope']:.5f} °C/month")"""

ext_results = nct.ext(ts_mean_da, percentile=0.95)
out_ext = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "=== 동아시아 해수면온도 극한 기후 지수 분석 결과 ===\n",
            f"95% 백분위수 고수온 임계값 : {ext_results['threshold']:.3f} °C\n",
            f"임계값 초과 극한 사상 발생 횟수 : {ext_results['extreme_count']}회\n",
            f"Sen's Slope (견고한 장기 추세)    : {ext_results['sens_slope']:.5f} °C/month\n"
        ]
    }
]
add_md("### 7.3. 극한 기후 지수 및 Sen's Slope 분석\n\n`nct.ext()`를 적용하여 이상 기후 사상의 기준이 되는 백분위수 임계값(Threshold)과 임계값 초과 빈도(Count), 비모수적 장기 변화율인 Sen's Slope를 산출합니다.")
add_code(code_ext, out_ext)

# Final notebook dictionary
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_ipynb = r'C:\SYSTEMS\PROG\WEB\NMSC\dms02_homepage\src\main\webapp\resources\common\climatevariable\nmsc-climate-toolbox_guidance_sst01m.ipynb'
with open(out_ipynb, 'w', encoding='utf-8') as f:
    json.dump(notebook_dict, f, ensure_ascii=False, indent=1)

print(f"Successfully generated {out_ipynb} with NanumBarunGothic and fixed tables!")

