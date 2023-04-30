# -*- coding: utf-8 -*-
# ============================================
# 라이브러리
# ============================================
# -*- coding: utf-8 -*-
import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

import seaborn as sns

# ============================================
# 보조
# ============================================


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0365'

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

data = pd.read_csv(fileList[0], encoding='EUC-KR')

# =================================================================
# 담뱃값 인상 전후에 따른 성별 별 흡연자 비중
# =================================================================
# dataL1 = data[['담뱃값인상', '남성', '여성', '흡연']]
data['담배값'] = np.where(data['담뱃값인상'] == 1, '이후', '이전')
data['성별'] = np.where(data['남성'] == 0, '남성', '여성')

dataL1 = data.groupby(['담배값', '성별']).count().reset_index(drop=False)
# dataL1 = (data.groupby(['type', 'type2']).count() / data['흡연'].count()).reset_index(drop=False)
dataL2 = dataL1[['담배값', '성별', '흡연']]

mainTitle = '{}'.format('담배 인상 전후에 따른 성별별 흡연자 비중')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)

ax = sns.barplot(data=dataL2, x='성별', y='흡연', hue='담배값', ci=None)
for i in ax.containers:
    ax.bar_label(i, )
plt.ylabel('흡연 비중')
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()
plt.close()

# =================================================================
# 담뱃값 인상 전후에 따른 연령별 흡연자 비중을 퍼센트
# =================================================================
data['연령'] = np.where(data['이십대'] == 1, '20대'
                      , np.where(data['삼십대'] == 1, '30대'
                                 , np.where(data['사십대'] == 1, '40대'
                                            , np.where(data['오십대'] == 1, '50대'
                                                       , np.where(data['육십대'] == 1, '60대'
                                                                  , np.where(data['칠십대'] == 1, '70대', None)
                                                                  )))))

dataL1 = (data.groupby(['담배값', '연령']).count() / data['흡연'].count() * 100.0).reset_index(drop=False)
dataL2 = dataL1[['담배값', '연령', '흡연']]

mainTitle = '{}'.format('담배 인상 전후에 따른 연령별 흡연자 비율')
saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
os.makedirs(os.path.dirname(saveImg), exist_ok=True)

ax = sns.barplot(data=dataL2, x='연령', y='흡연', hue='담배값', ci=None)
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')
plt.ylabel('흡연 비율 [%]')
plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()
plt.close()
