# NMSC Climate Toolbox

**NMSC Climate Toolbox**는 핵심 기후 변수 데이터(NetCDF 형식)를 효율적으로 로드, 가공, 시각화 및 검증하기 위해 개발된 고성능 파이썬 라이브러리입니다.

## 주요 기능

*   **데이터 I/O (`open`, `search`)**: 복수의 NetCDF 기후 데이터 파일을 자동으로 묶어 xarray Dataset으로 로드합니다. (xcdat 규격을 자체 호환 처리)
*   **시공간 필터링 (`filter`)**: 필요한 기간(time) 및 위경도(lat, lon) 영역만 추출합니다.
*   **전처리 (`clean`, `weg`)**: 결측치를 제거하고 공간 가중치(위도 기반 면적 가중)를 적용합니다.
*   **통계 및 기후 연산 (`spaMean`, `timeMean`, `cli`, `ano`)**: 공간 평균, 시간 평균, 평년값(Climatology), 기후 편차(Anomaly)를 계산합니다.
*   **추세 분석 (`trend`)**: 시계열 데이터에 대한 선형 회귀 분석을 통해 기후 변화 추세(Slope) 및 신뢰도를 분석합니다.
*   **시각화 리포트 (`generate_static_map`, `timeGrp`, `scaGrp`)**: 정적 공간 분포 지도, 시계열 추세 그래프, 검증 데이터와의 1:1 산점도를 생성합니다.

## 설치 방법 (Installation)

이 라이브러리는 `pip` 또는 `conda` 패키지 관리자를 통해 설치할 수 있습니다. 
(※ 현재는 로컬 소스 코드 디렉토리에서의 빌드 방식을 기준으로 설명합니다.)

### 1. pip를 이용한 설치

프로젝트 최상위 디렉토리(setup.py가 위치한 곳)에서 아래 명령어를 실행하세요.
```bash
pip install .
```
> **참고:** 이 명령어는 GUI 요소를 제외한 핵심 데이터 처리 로직(`nmsc_climate_toolbox`)만 설치합니다.

### 2. Conda를 이용한 설치 (로컬 빌드)

Conda 레시피를 이용해 패키지를 빌드하고 설치할 수 있습니다.
```bash
# 1. 빌드 도구 설치
conda install -c conda-forge conda-build

# 2. 콘다 레시피 빌드
conda build conda_recipe/

# 3. 로컬에 빌드된 패키지 설치
conda install --use-local nmsc-climate-toolbox
```

## 사용 예시

파이썬 스크립트나 Jupyter Notebook에서 모듈을 임포트하여 분석에 활용할 수 있습니다.

```python
from nmsc_climate_toolbox import NMSCClimateToolbox

# 1. 파일 검색 및 열기
file_paths = NMSCClimateToolbox.search("./data/*.nc")
dataset = NMSCClimateToolbox.open(file_paths)

# 2. 특정 위경도 및 기간으로 데이터 필터링
filtered_ds = NMSCClimateToolbox.filter(dataset, 
                                        time_slice=('2020-01-01', '2022-12-31'), 
                                        lon_slice=(120, 150), 
                                        lat_slice=(20, 50))

# 3. 공간 평균 계산 후 평년값 대비 편차(Anomaly) 도출
spa_mean = NMSCClimateToolbox.spaMean(filtered_ds, variable='temperature')
anomaly = NMSCClimateToolbox.ano(spa_mean, variable='temperature')

# 4. 시계열 추세선 그리기
fig, ax = NMSCClimateToolbox.timeGrp(anomaly, variable_name="Temperature Anomaly")
fig.show()
```

## 라이선스
MIT License
