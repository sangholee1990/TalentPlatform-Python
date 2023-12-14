# -*- coding: utf-8 -*-

# 라이브러리 읽기
import pandas as pd
from datetime import datetime

# 파일 읽기
data = pd.read_csv('/DATA/INPUT/LSH0519/서울시 코로나19 확진자 발생동향.csv', encoding='EUC-KR')

# 컬럼 정보
data.columns

# 특정 컬럼 (서울시 기준일) 공백 제거
data['서울시 기준일'] = data['서울시 기준일'].str.replace(' ', '', regex=True)

# 특정 컬럼 (서울시 기준일)에서 문자열 길이를 통해 날짜형 변환
data['dtYmdH'] = data['서울시 기준일'].apply(lambda x: pd.to_datetime(x, format='%Y.%m.%d.%H') if len(x) == 13 else pd.to_datetime(x, format='%y.%m.%d.%H'))

# 날짜형에서 연월일 변환
data['dtYmd'] = data['dtYmdH'].dt.strftime('%Y%m%d')

# 날짜형에서 연월 변환
data['dtYm'] = data['dtYmdH'].dt.strftime('%Y%m')

# 월별에 따른 확진자 및 사망자 합계
dataL1 = data.groupby(data['dtYm'])[['전국 추가 확진', '전국 당일 사망자']].sum().reset_index(drop=False)

# ==========================================================================
# 확진자
# ==========================================================================
# 컬럼 선택
colInfo = '전국 추가 확진'

# 월간 최대 확진자
maxMonthInfo = dataL1.loc[dataL1[colInfo].idxmax()][['dtYm', colInfo]]
print(f'[CHECK] maxMonthInfo : {maxMonthInfo}')

# 일간 최대 확진자
dataL2 = data[data['dtYm'] == maxMonthInfo['dtYm']]
maxDayInfo = dataL2.loc[dataL2[colInfo].idxmax()][['dtYmd', colInfo]]
print(f'[CHECK] maxDayInfo : {maxDayInfo}')

# ==========================================================================
# 사망자
# ==========================================================================
# 컬럼 선택
colInfo = '전국 당일 사망자'

# 월간 최대 사망자
maxMonthInfo = dataL1.loc[dataL1[colInfo].idxmax()][['dtYm', colInfo]]
print(f'[CHECK] maxMonthInfo : {maxMonthInfo}')

# 일간 최대 사망자
dataL2 = data[data['dtYm'] == maxMonthInfo['dtYm']]
maxDayInfo = dataL2.loc[dataL2[colInfo].idxmax()][['dtYmd', colInfo]]
print(f'[CHECK] maxDayInfo : {maxDayInfo}')

# ==========================================================================
# 분석 결과
# ==========================================================================
# 월간/일간 확진자 통계 결과 2022년 03월 9,962,387명 확진자를 보이며 특히 2022년 3월 17일에서 가장 높은 확진자 (621,266명)를 보임
# 월간/일간 사망자 통계 결과 2022년 04월 6,564명 사망자를 보이며 특히 2022년 4월 8일에서 가장 높은 사망자 (373명)를 보임