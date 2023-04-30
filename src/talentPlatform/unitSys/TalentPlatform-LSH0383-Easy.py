# -*- coding: utf-8 -*-
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================
# 보조
# ============================================


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0383'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    # 'srtDate': '2019-01-01'
    # , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

# ********************************************************************
# 피지크 대회에서 특성별 퍼센트 시각화
# ********************************************************************
# 파일 설정
# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '피지크대회의 데이터분석.xlsx')
#
# # 파일 조회
# fileList = sorted(glob.glob(inpFile))
#
# # 파일 읽기
# data = pd.read_excel(fileList[0])
#
# # 행 합계
# data['sumVal'] = data.sum(axis = 'columns')
#
# # sumData = data.groupby(['특성'], as_index=False).sum()
#
# # 특성, 행 합계를 기준으로 spread to long 변환
# dataL1 = pd.melt(data, id_vars=['특성', 'sumVal'])
#
# # 퍼센트 계산
# dataL1['percent'] = (dataL1['value'] / dataL1['sumVal']) * 100.0
#
# dataL2 = dataL1.loc[dataL1['variable'] != 'sum']
#
# # 피지크 대회에서 특성별 퍼센트 시각화
# plt.rc('font', family='Malgun Gothic')
# plt.rc('axes', unicode_minus=False)
#
# # 그림 제목 설정
# mainTitle = '{}'.format('피지크 대회에서 특성별 퍼센트 시각화')
# saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
# os.makedirs(os.path.dirname(saveImg), exist_ok=True)
# print('[CHECK] saveImg : {}'.format(saveImg))
#
# # 특성을 기준으로 그룹화
# g = sns.FacetGrid(dataL2, col='특성', hue='variable', col_wrap=2)
# # x (variable) 및 y (percent)축을 기준으로 시각화
# g.map(sns.barplot, 'variable', 'percent', order=set(dataL2['variable'])).add_legend()
#
# # 범례 제목 제거
# g.legend.set_title(None)
#
# for ax in g.axes:
#     for p in ax.patches:
#              ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
#                  ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
#                  textcoords='offset points')
#
# # x축 제목 설정
# plt.xlabel('특성')
#
# plt.tight_layout()
# plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
# plt.show()
# plt.close()


# ********************************************************************
# 피지크 대회에서 특성별 꺾은선 시각화
# ********************************************************************
# 파일 설정
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '피지크대회의 데이터분석.xlsx')

# 파일 조회
fileList = sorted(glob.glob(inpFile))

# 파일 읽기
data = pd.read_excel(fileList[0], sheet_name='Sheet2')

# 특성, 행 합계를 기준으로 spread to long 변환
dataL1 = pd.melt(data, id_vars=['특성'])

# 퍼센트 계산
# dataL1['percent'] = (dataL1['value'] / dataL1['sumVal']) * 100.0

# dataL2 = dataL1.loc[dataL1['variable'] != 'sum']

# 피지크 대회에서 특성별 퍼센트 시각화
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 그림 제목 설정
mainTitle = '{}'.format('피지크 대회에서 특성별 꺾은선 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print('[CHECK] saveImg : {}'.format(saveImg))

# 특성을 기준으로 그룹화
sns.pointplot(data = dataL1, x='variable', y='value', hue='특성')

# x축 제목 설정
plt.xlabel('연도')

plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()
