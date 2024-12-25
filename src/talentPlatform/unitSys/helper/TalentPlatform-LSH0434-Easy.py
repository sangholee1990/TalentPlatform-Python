# -*- coding: utf-8 -*-
import glob
import os
import pandas as pd
from datetime import datetime
from sklearn import linear_model
import matplotlib.pyplot as plt

# ============================================
# 요구사항
# ============================================
# Python을 이용한 10년 대기 중금속 농도 예측 및 시각화

# 10년 대기 중금속 농도 측정 결과
# csv 파일을 토대로 예측 및 시각화

# ============================================
# 보조
# ============================================
# 날짜형을 10진수 변환
def decimalDate(dtDate):
    start = datetime(year=dtDate.year, month=1, day=1)
    end = datetime(year=dtDate.year+1, month=1, day=1)
    return dtDate.year + ((dtDate - start) / (end - start))

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0434'

# 옵션 설정
sysOpt = {
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

# 그림 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '10개년 대기 중금속 농도 측정.csv')
fileList = sorted(glob.glob(inpFile))
data = pd.read_csv(fileList[0], encoding='EUC-KR')

# 구분, 항목, 단위를 기준으로 spread to long 변환
dataL1 = pd.melt(data, id_vars=['구분', '항목', '단위', 'Unnamed: 121'])
dataL1['dtDate'] = pd.to_datetime(dataL1['variable'], format='%Y.%m 월')
dataL1['dtXran'] = dataL1['dtDate'].apply(lambda x: decimalDate(x))
dataL1['val'] = pd.to_numeric(dataL1['value'], errors='coerce')

dataL2 = dataL1[['구분', '항목', 'dtXran', 'val']].dropna().reset_index(drop=True)

# 선형 회귀모형 설정
lmModel = linear_model.LinearRegression()

grpList = sorted(set(dataL2['구분']))
typeList = sorted(set(dataL2['항목']))

for i, grpInfo in enumerate(grpList):
    for j, typeInfo in enumerate(typeList):

        dataL3 = dataL2.loc[(dataL2['구분'] == grpInfo) & (dataL2['항목'] == typeInfo)]
        if (len(dataL3) < 1): continue

        # print(f'[CHECK] grpInfo : {grpInfo} / typeInfo : {typeInfo}')

        # Converts the Series to 2D array
        X = dataL3['dtXran'].values.reshape(-1, 1)
        y = dataL3['val']

        if (len(X) < 1): continue

        # 학습 데이터를 통해 학습
        lmModel.fit(X, y)

        # 학습 데이터를 통해 예측
        prd = lmModel.predict(X)

        # 예측 결과 저장
        dataL2.loc[dataL3.index, 'prd'] = prd

        mainTitle = f'[{grpInfo}] {typeInfo} 연도별 대기 중금속 농도 비교'
        plt.title(mainTitle)
        saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.scatter(X, y, color='black')
        plt.plot(X, prd, color='blue', linewidth=3)
        plt.xlabel('연도')
        plt.ylabel('대기 중금속 농도')
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.close()
        # plt.show()
        print(f'[CHECK] saveImg : {saveImg}')

# 연도별 대기 중금속 농도 예측 시각화
for i, grpInfo in enumerate(grpList):
    # print(f'[CHECK] grpInfo : {grpInfo}')

    mainTitle = f'[{grpInfo}] 연도별 대기 중금속 농도 예측'
    plt.title(mainTitle)
    saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    for j, typeInfo in enumerate(typeList):

        dataL3 = dataL2.loc[(dataL2['구분'] == grpInfo) & (dataL2['항목'] == typeInfo)]
        if (len(dataL3) < 1): continue

        plt.plot(dataL3['dtXran'], dataL3['prd'], 'o-', label=typeInfo)
    plt.xlabel('연도')
    plt.ylabel('대기 중금속 농도')
    plt.legend()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.close()
    # plt.show()
    print(f'[CHECK] saveImg : {saveImg}')

# 자료 저장
saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, '10개년 대기 중금속 농도 예측')
os.makedirs(os.path.dirname(saveFile), exist_ok=True)
dataL2.to_csv(saveFile, index=False)
print(f'[CHECK] saveFile : {saveFile}')