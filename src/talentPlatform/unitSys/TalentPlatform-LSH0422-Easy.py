# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ============================================
# 요구사항
# ============================================
# Python을 이용한 교통사고 데이터에 대한 인명 피해형태 예측

# 주어진 교통사고 데이터에 대해 인명 피해 형태를 예측하는 모델을 개발하고자 한다.
# 교통사고 자료에 대한 학습용데이터(Train.xlsx)를 통해 교통사고로 인한 인명 피해 형태를 예측하는 모델을 개발하고
# 평가용 입력데이터(Test_X.xlsx)를 이용하여 교통사고로 인한 인명 피해 형태를 예측하시오.

# ============================================
# 보조
# ============================================
# 함수 정의

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0422'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    # 'srtDate': '2019-01-01'
    # , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

# **************************************************
# 학습 데이터 읽기
# **************************************************
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Train.xlsx')
fileList = sorted(glob.glob(inpFile))
trainData = pd.read_excel(fileList[0])

# 결측치 제거
trainDataL1 = trainData.dropna()

# 종속변수 : 사고내용 (경상사고, 사망사고, 중상사고)
trainY = trainDataL1['사고내용']

# 독립변수
trainX = trainDataL1[['사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해차종', '가해성별', '가해연령']]
trainX = pd.get_dummies(trainX[['사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해차종', '가해성별', '가해연령']], columns=['사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해차종', '가해성별'])



# **************************************************
# SVM 분류
# **************************************************
# 분류 모델 생성 및 학습
svmModel = SVC(kernel='linear', C=1)
svmModel.fit(trainX, trainY)

# 예측
prdSvmY = svmModel.predict(trainX)

# 혼동행렬(분류표)
print(confusion_matrix(trainY, prdSvmY))

# 모형 적합도
print(classification_report(trainY, prdSvmY))

# **************************************************
# dct 분류
# **************************************************
# 분류 모델 생성 및 학습
dctModel = DecisionTreeClassifier()
dctModel.fit(trainX, trainY)

# 예측
prdDctY = dctModel.predict(trainX)

# 혼동행렬(분류표)
print(confusion_matrix(trainY, prdDctY))

# 모형 적합도
print(classification_report(trainY, prdDctY))

# **************************************************
# knn 모형
# **************************************************
# 옵션: k값(Voting 갯수), p값(거리측정 방법)
knnModel = KNeighborsClassifier(n_neighbors=5, p=2).fit(trainX, trainY)
prdKnnY = knnModel.predict(trainX)

# 혼동행렬(분류표)
print(confusion_matrix(trainY, prdKnnY))

# 모형 적합도
print(classification_report(trainY, prdKnnY))

# **************************************************
# 로지스틱 모형
# **************************************************
logModel = LogisticRegression(penalty='none').fit(trainX, trainY)
prdLogY = logModel.predict(trainX)

# 혼동행렬(분류표)
print(confusion_matrix(trainY, prdLogY))

# 모형 적합도
print(classification_report(trainY, prdLogY))

# **************************************************
# 테스트 데이터 읽기
# **************************************************
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Test_X.xlsx')
fileList = sorted(glob.glob(inpFile))
testData = pd.read_excel(fileList[0])
testDataL1 = testData.dropna()

# 독립변수
testX = testDataL1[['사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해차종', '가해성별', '가해연령']]
testX = pd.get_dummies(testX, columns=['사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해차종', '가해성별'])

# 예측 결과
# prdData = pd.DataFrame()
prdData = testDataL1
prdData['SVM'] = svmModel.predict(testX)
prdData['DCT'] = dctModel.predict(testX)
prdData['KNN'] = knnModel.predict(testX)
prdData['LOG'] = logModel.predict(testX)

# y_test_hat 엑셀자료로 내보내기
saveFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, 'y_test_hat')
os.makedirs(os.path.dirname(saveFile), exist_ok=True)
prdData.to_excel(saveFile, index=False)
print(f'[CHECK] saveFile : {saveFile}')