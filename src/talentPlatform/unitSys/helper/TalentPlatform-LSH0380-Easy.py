# -*- coding: utf-8 -*-
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================
# 보조
# ============================================


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0380'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2019-01-01'
    , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

# ********************************************************************
# 파일 읽기
# ********************************************************************
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'BDA.csv')
fileList = sorted(glob.glob(inpFile))

# 파일 읽기
data = pd.read_csv(fileList[0])

# ********************************************************************
# 자료 전처리
# ********************************************************************
# data['담배값'] = np.where(data['price'] == 1, '이전', '이후')
# data['흡연자'] = np.where(data['smoke'] == 1, '비흡연자', '흡연자')
# data['성별'] = np.where(data['sex'] == 1, '남성', '여성')
# data['유산소'] = np.where(data['Physicalactivity'] == 1, '미수행', '수행')
# data['연도'] = np.where(data['price'] == 1, '2013~2014년', '2015~2016년')
# data['비만'] = np.where(data['bmi'] == 1, '~25'
#                        , np.where(data['bmi'] == 2, '25~30'
#                        , np.where(data['bmi'] == 3, '30~35'
#                        , np.where(data['bmi'] == 4, '35~', None)
#                        )))
# data['비만'] = pd.Categorical(data['비만'], categories=['~25', '25~30', '30~35', '35~'], ordered=False)
# data['칼로리'] =  data['calorie_per_day']
# data['연령'] = np.where(data['age'] == 1, '20대'
#                       , np.where(data['age'] == 2, '30대'
#                       , np.where(data['age'] == 3, '40대'
#                       , np.where(data['age'] == 4, '50대'
#                       , np.where(data['age'] == 5, '60대'
#                       , np.where(data['age'] == 6, '70대', None)
#                       )))))
#
# dataL1 = data[['담배값', '흡연자', '연령', '성별', '유산소', '비만', '연도', '칼로리']]

data['cigarettePrice'] = np.where(data['price'] == 1, 'before', 'after')
data['smoker'] = np.where(data['smoke'] == 1, 'non-smoker', 'smoker')
data['sex'] = np.where(data['sex'] == 1, 'male', 'female')
data['physical'] = np.where(data['Physicalactivity'] == 1, 'inaction', 'action')
data['year'] = np.where(data['price'] == 1, '2013~2014', '2015~2016')
data['cal'] =  data['calorie_per_day']
data['age'] = np.where(data['age'] == 1, '20'
                      , np.where(data['age'] == 2, '30'
                      , np.where(data['age'] == 3, '40'
                      , np.where(data['age'] == 4, '50'
                      , np.where(data['age'] == 5, '60'
                      , np.where(data['age'] == 6, '70', None)
                      )))))
dataL1 = data[['cigarettePrice', 'smoker', 'sex', 'physical', 'bmi', 'year', 'cal', 'age']]

# ********************************************************************
# 담배값 인상 이전에 따른 개수 시각화
# ********************************************************************
dataL2 = dataL1.loc[(dataL1['cigarettePrice'] == 'before')].reset_index(drop=True)

mainTitle = '{}'.format('담배값 인상 이전에 따른 개수 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print("[CHECK] saveImg : {}".format(saveImg))

fig, ax = plt.subplots(3, 2)
ax[2][1].set_visible(False)
axe = ax.ravel()

dataL2['smoker'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[0])
dataL2['age'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[1])
dataL2['sex'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[2])
dataL2['physical'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[3])
dataL2['bmi'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[4])

plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ********************************************************************
# 담배값 인상 이후에 따른 개수 시각화
# ********************************************************************
dataL2 = dataL1.loc[(dataL1['cigarettePrice'] == 'after')].reset_index(drop=True)

mainTitle = '{}'.format('담배값 인상 이후에 따른 개수 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print("[CHECK] saveImg : {}".format(saveImg))

fig, ax = plt.subplots(3, 2)
ax[2][1].set_visible(False)
axe = ax.ravel()

dataL2['smoker'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[0])
dataL2['age'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[1])
dataL2['sex'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[2])
dataL2['physical'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[3])
dataL2['bmi'].value_counts().plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[4])

plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ********************************************************************
# 담배값 인상 전후에 따른 퍼센트 시각화
# ********************************************************************
mainTitle = '{}'.format('담배값 인상 전후에 따른 퍼센트 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print("[CHECK] saveImg : {}".format(saveImg))

fig, ax = plt.subplots(3, 2)
ax[2][1].set_visible(False)
axe = ax.ravel()

dataL1[['cigarettePrice', 'smoker']].value_counts(normalize=True).reset_index(drop=False).pivot(index='smoker', columns=['cigarettePrice'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[0])
dataL1[['cigarettePrice', 'age']].value_counts(normalize=True).reset_index(drop=False).pivot(index='age', columns=['cigarettePrice'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[1])
dataL1[['cigarettePrice', 'sex']].value_counts(normalize=True).reset_index(drop=False).pivot(index='sex', columns=['cigarettePrice'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[2])
dataL1[['cigarettePrice', 'physical']].value_counts(normalize=True).reset_index(drop=False).pivot(index='physical', columns=['cigarettePrice'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[3])
dataL1[['cigarettePrice', 'bmi']].value_counts(normalize=True).reset_index(drop=False).pivot(index='bmi', columns=['cigarettePrice'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[4])

ax[0][0].get_legend().remove()
ax[0][1].get_legend().remove()
ax[1][0].get_legend().remove()
ax[1][1].get_legend().remove()
ax[2][0].legend(loc='right')

plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ********************************************************************
# BMI 지수에 따른 5개 변수 시각화
# ********************************************************************
mainTitle = '{}'.format('BMI 지수에 따른 5개 변수 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print("[CHECK] saveImg : {}".format(saveImg))

fig, ax = plt.subplots(3, 2)
ax[2][1].set_visible(False)
axe = ax.ravel()

dataL1[['bmi', 'age']].value_counts(normalize=False).reset_index(drop=False).pivot(index='age', columns=['bmi'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[0])
dataL1[['bmi', 'sex']].value_counts(normalize=False).reset_index(drop=False).pivot(index='sex', columns=['bmi'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[1])
dataL1[['bmi', 'physical']].value_counts(normalize=False).reset_index(drop=False).pivot(index='physical', columns=['bmi'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[2])
dataL1[['bmi', 'smoker']].value_counts(normalize=False).reset_index(drop=False).pivot(index='smoker', columns=['bmi'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[3])
dataL1[['bmi', 'year']].value_counts(normalize=False).reset_index(drop=False).pivot(index='year', columns=['bmi'])[0].plot(kind='bar', color=sns.color_palette(), rot=0, ax=axe[4])

ax[0][0].get_legend().remove()
ax[0][1].get_legend().remove()
ax[1][0].get_legend().remove()
ax[1][1].get_legend().remove()
ax[2][0].legend(loc='right')

plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ********************************************************************
# 담배값 인상 전후에 따른 칼로리 시각화
# ********************************************************************
mainTitle = '{}'.format('담배값 인상 전후에 따른 칼로리 시각화')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)
print("[CHECK] saveImg : {}".format(saveImg))

fig, ax = plt.subplots(1, 1)
sns.boxplot(x="cal", y="cigarettePrice", hue="cigarettePrice", dodge=False, data=dataL1)
plt.tight_layout()
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()