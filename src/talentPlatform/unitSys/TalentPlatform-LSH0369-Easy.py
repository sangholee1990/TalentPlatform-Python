# -*- coding: utf-8 -*-
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# ============================================
# 보조
# ============================================


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0369'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2019-01-01'
    , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'HN_M.csv')
fileList = sorted(glob.glob(inpFile))

# 파일 읽기
data = pd.read_csv(fileList[0], encoding='EUC-KR')

# 전처리
data['성별'] = np.where(data['남성'] == 0, '남성', '여성')

data['체중'] = np.where(data['정상체중'] == 1, '1단계'
                   , np.where(data['비만전단계'] == 1, '2단계'
                   , np.where(data['비만'] == 1, '3단계'
                   , np.where(data['고도비만'] == 1, '4단계', None)
                   )))

data['칼로리'] = np.where(data['칼로리섭취1단계'] == 1, '1단계'
                   , np.where(data['칼로리섭취2단계'] == 1, '2단계'
                   , np.where(data['칼로리섭취3단계'] == 1, '3단계', None)
                   ))

data['연령'] = np.where(data['이십대'] == 1, '20대'
                  , np.where(data['삼십대'] == 1, '30대'
                  , np.where(data['사십대'] == 1, '40대'
                  , np.where(data['오십대'] == 1, '50대'
                  , np.where(data['육십대'] == 1, '60대'
                  , np.where(data['칠십대'] == 1, '70대', None)
                  )))))

# 전에 보내드린 HN_M 파일을 이용해서 종속 변수(BMI)는 그대로 두고 연령, 흡연, 성별, 신체 활동, 칼로리 섭취, 담뱃값 인상 데이터 (독립 변수)
# 각각 KNN 모델을 사용해서 분석해주시면 감사하겠습니다 (ex BMI & 연령, BMI & 성별 등등).
# colName = '칼로리'
for colName in data.columns:

    if (re.search('Unnamed: 0|체질량지수|십대|남성|여성|단계|정상체중|비만전단계|비만|고도비만', colName)): continue

    # print("[CHECK] colName : {}".format(colName))

    # 7:3에 대한 훈련/테스트 분류
    trainData, testData = train_test_split(data[[colName, '체질량지수']], test_size=0.3, random_state=123)

    # 카테고리형 변환 및 -1~1 스케일 조정
    trainColData = pd.factorize(trainData[colName])[0].reshape(-1, 1)
    testColData = pd.factorize(testData[colName])[0].reshape(-1, 1)

    # KNN 회귀모형 설정
    # knnModel = KNeighborsRegressor(n_neighbors=len(np.unique(data[colName])))
    knnModel = KNeighborsRegressor()

    # 훈련 데이터를 이용한 학습
    knnModel.fit(trainColData, trainData['체질량지수'])

    # 테스트 데이터를 이용한 예측
    testData['prd'] = knnModel.predict(testColData)

    # 테스트 데이터를 이용한 실측/예측의 평균제곱근오차 (RMSE)
    rmse = np.sqrt(mean_squared_error(testData['prd'], testData['체질량지수'].values))
    print("[CHECK] [{}] RMSE : {:.2f}".format(colName, rmse))

    # 시각화
    mainTitle = '{} 데이터를 이용한 체질량 실측 및 예측 산점도'.format(colName)
    saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    print("[CHECK] saveImg : {}".format(saveImg))

    # 시각화
    sns.scatterplot(data=testData, x='prd', y='체질량지수')
    plt.xlabel('체질량지수 예측')
    plt.ylabel('체질량지수 실측')
    plt.title('[{}] RMSE : {:.2f}'.format(colName, rmse))
    plt.tight_layout()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()