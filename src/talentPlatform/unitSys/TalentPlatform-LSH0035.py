# -*- coding: utf-8 -*-

# ===============================================================================================
# Routine : Main R program
#
# Purpose : 재능상품 (크몽, 오투잡)
#
# Author : 해솔
#
# Revisions: V1.0 May 28, 2020 First release (MS. 해솔)
# ===============================================================================================

from src.talentPlatform.unitSysHelper.InitConfig import *
from plotnine import *
import logging
import logging.handlers
# from plotnine import *
# from plotnine.data import *
# import hydroeval
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform

import boto3
import pymysql
from datetime import datetime

from plotnine.data import *
import dfply as dfply

# =================================================
# 요구사항
# =================================================
# 가로축은 예시그래프와 같이 24시간으로 하고 싶은데
# 세로축은 해당 시간 갯수로 하고 싶습니다!
# 시간은 날짜나 분,초는 상관없이 시간으로만 하시면 될 것 같아요!
# 그렇게해서 이 파일이랑 이 파일이랑 예시그림처럼 한번에 비교될 수 있게 나오도록 만들고 싶은데
# 근데 보시면 첫번째 파일은 시간이 총 259개이고 두번째파일은 399개라서
# 두번째 파일 399개중에서 완전 랜덤으로 259개를 뽑아서
# 표로 만들 수 있을까요 ..?

# =================================================
# Set Env
# =================================================
# 작업환경 경로 설정
contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
prjName = 'test'
serviceName = 'LSH0035'

log = initLog(contextPath, prjName)
globalVar = initGlobalVar(contextPath, prjName)

# =================================================
# Main
# =================================================
try:
    log.info('[START] {}'.format('Main'))


    # breakpoint()
    # fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0167_dataL2.csv'))
    # dataL2 = pd.read_csv(fileInfo1[0], na_filter=False)
    #
    # saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 따른 히스토그램')

    fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0035_LIWC2015_Results_stress.xlsx'))[0]
    data1 = pd.read_excel(fileInfo1, sheet_name = 'Sheet0', usecols = "A,B,C,D").dropna()

    fileInfo2 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0035_LIWC2015_Results_non_stress.xlsx'))[0]
    data2 = pd.read_excel(fileInfo2, sheet_name = 'Sheet0', usecols = "A,B,C,D").dropna()

    data1['dtDateTime'] = pd.to_datetime(data1['Source (C)'], format='%Y-%m-%d %H:%M:%S')
    data2['dtDateTime'] = pd.to_datetime(data2['Source (C)'], format='%Y-%m-%d %H:%M:%S')

    data1Stat = (
            data1 >>
            dfply.mutate(
                hour=dfply.X.dtDateTime.dt.strftime("%H")
            ) >>
            dfply.group_by(dfply.X.hour) >>
            dfply.summarize(stress=dfply.n(dfply.X.hour))
    )

    data2Stat = (
            data2 >>
            dfply.sample(n=len(data1)) >>
            dfply.mutate(
                hour=dfply.X.dtDateTime.dt.strftime("%H")
            ) >>
            dfply.group_by(dfply.X.hour) >>
            dfply.summarize(non_stress=dfply.n(dfply.X.hour))
    )

    dataL1 = (
            data1Stat >>
            dfply.left_join(data2Stat, by='hour') >>
            dfply.gather('key', 'val', ['stress', 'non_stress'])
    )

    dodge_text = position_dodge(width=0.9)

    plot = (ggplot(dataL1, aes(x='hour', y='val', fill='key'))
            + geom_col(stat='identity', position='dodge')
            + geom_text(aes(label='val'),  # new
                        position=dodge_text,
                        size=8, va='bottom', format_string='{}%')
            + lims(y=(0, 30))
            )

    saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, 'Image_08.png')
    plot.save(saveImg, width=10, height=10, dpi=600, bbox_inches='tight')

except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] {}'.format('Main'))