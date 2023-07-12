# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import pandas as pd

# from google.colab import files
# upload = files.upload()

# 파일 불러오기, 변수 선언 #


x = []
y = []
z = []
z1 = []
f = open('RawData.csv', encoding='utf-8')  # 가지고 있는 Data 파일명으로 변경 필요
data = csv.reader(f)
next(data)
data = list(data)

# 데이터 추출 #
for i in range(1, 10, 2):  # x 리스트에 연도를 담음(5개)
    x.append(int(data[1][i]))
for i in range(4, 21):  # y 리스트에 도시를 담음(17개)
    y.append(data[i][0])

print(x)

for i, findPlace in enumerate(y):
    print('[CHECK] findPlace : {}'.format(findPlace))

    # findPlace = input('도시를 입력하세요') # 입력받은 도시의 연도별 인구를 z 리스트에 담음(5개)
    findChart = y.index(findPlace)

    z = []
    if findPlace in y:
        for i in range(1, 10, 2):
            z.append(data[findChart + 4][i])
    else:
        print('해당 도시는 자료에 존재하지 않습니다')

    # 데이터 정리 #  -> z리스트의 값 수정하기
    Changed_z = ' '.join(z)  # 리스트를 문자열로 바꿈 (데이터 사이에 공백 추가)
    Changed2_z = Changed_z.replace(',', '')  # 문자열의 콤마 제거
    z = list(map(int, Changed2_z.split(' ')))  # 문자열을 공백기준으로 분리함 -> 리스트 됨

    print('[CHECK] z : {}'.format(z))

    # 그래프 그리기 #
    plt.rc('font', family='NanumGothic')
    plt.title('[{}] 도시별 인구증가 그래프'.format(findPlace))
    plt.ylim(min(z) - 100, max(z) + 100)
    plt.bar(x, z, color='r')
    plt.savefig('{} 도시별 인구증가 그래프.png'.format(findPlace), dpi=600, bbox_inches='tight')
    plt.show()

for i, findPlace in enumerate(x):
    print('[CHECK] findPlace : {}'.format(findPlace))

    # findPlace = input('도시를 입력하세요') # 입력받은 도시의 연도별 인구를 z 리스트에 담음(5개)
    findChart = (2 * x.index(findPlace)) + 1

    z = []
    nameList = []
    if findPlace in x:
        for j in range(0, 18, 1):
            z.append(data[4 + j][findChart])
            nameList.append(data[4 + j][0])
    else:
        print('해당 연도는 자료에 존재하지 않습니다')

    # 데이터 정리 #  -> z리스트의 값 수정하기
    Changed_z = ' '.join(z)  # 리스트를 문자열로 바꿈 (데이터 사이에 공백 추가)
    Changed2_z = Changed_z.replace(',', '')  # 문자열의 콤마 제거
    z = list(map(int, Changed2_z.split(' ')))  # 문자열을 공백기준으로 분리함 -> 리스트 됨

    # print('[CHECK] z : {}'.format(z))
    # print('[CHECK] nameList : {}'.format(nameList))

    dataL1 = pd.DataFrame(
        {
            'name': nameList
            , 'val': z
        }
    )

    # 그래프 그리기 #
    plt.rc('font', family='NanumGothic')
    dataL1.groupby(['name']).sum().plot(kind='pie', y='val')
    plt.savefig('{} 연도별 각 도시의 운구분포 그래프.png'.format(findPlace), dpi=600, bbox_inches='tight')
    plt.show()

    # 연도별 평균값
    meanVal = dataL1['val'].mean()
    print('[CHECK] [{}] 평균 : {}'.format(findPlace, meanVal))

    dataL2 = dataL1.loc[
        dataL1['val'] >= meanVal
        ]

    # 연도별 평균값보다 큰 도시
    print('[CHECK] [{}] 평균값보다 큰 도시 : {}'.format(findPlace, dataL2))
