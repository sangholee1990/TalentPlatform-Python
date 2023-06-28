# -*- coding: utf-8 -*-

from src.talentPlatform.unitSys.helper.InitConfig import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import glob
import pandas as pd
import dfply as dfply
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus":False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# =================================================
# Set Env
# =================================================
# 작업환경 경로 설정
contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
prjName = 'test'
serviceName = 'LSH0167'

log = initLog(contextPath, prjName)
globalVar = initGlobalVar(contextPath, prjName)


# =================================================
# Main
# =================================================
try:
    log.info('[START] {}'.format('Main'))

    # ***********************************************
    # 통합 데이터 전처리
    # ***********************************************
    fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0167_dataL2.csv'))
    dataL2 = pd.read_csv(fileInfo1[0], na_filter=False)

    # breakpoint()

    # ***********************************************
    # 데이터 요약 (요약통계량)
    # ***********************************************
    # 연소득당 거래금액 따른 기초 통계량
    dataL2.describe()

    # 법정동에 따른 연소득당 거래금액 따른 기초 통계량
    dataL3 = ((
            dataL2 >>
            dfply.group_by(dfply.X.d2) >>
            dfply.summarize(
                meanVal=dfply.mean(dfply.X.val)
            )
    ))

    # *******************************************************
    # 데이터 요약 (표/그래프 활용)
    # *******************************************************
    # 연소득당 거래금액 따른 히스토그램
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 따른 히스토그램')

    sns.distplot(dataL2['val'], kde=True, rug=False)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 법정동에 따른 연소득당 거래금액 히스토그램
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '법정동에 따른 연소득당 거래금액 히스토그램')

    sns.barplot(x='d2', y='meanVal', data=dataL3)
    plt.xticks(rotation = 45)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 연소득당 거래금액 따른 상자 그림
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 따른 상자 그림')

    sns.boxplot(y="val", data=dataL2)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 법정동에 따른 연소득당 거래금액 상자 그림
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '법정동에 따른 연소득당 거래금액 상자 그림')

    sns.boxplot(x = "d2", y="val", data=dataL2)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # 연소득당 거래금액 산점도
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '연소득당 거래금액 산점도')

    makeScatterPlot(dataL2['meanCost'], dataL2['거래금액'], saveImg, 3500, 100000)

    # *******************************************************
    # 데이터 분석 (데이터 분석 기법 활용)
    # *******************************************************
    # 주택 가격 결정 요인을 위한 회귀분석
    dataL4 = ((
            dataL2 >>
            dfply.select(dfply.X.건축년도, dfply.X.전용면적, dfply.X.층, dfply.X.val2, dfply.X.d2, dfply.X.val) >>
            dfply.rename(
                면적당거래금액 = dfply.X.val2
                , 연소득당거래금액 = dfply.X.val
            )
    ))

    # 주택 가격 결정 요인을 위한 관계성
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '주택 가격 결정 요인을 위한 관계성')

    sns.pairplot(dataL4)
    plt.show()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')

    # +++++++++++++++++++++++++++++++++++++++++++++++
    # 전체 아파트
    dataL5 = dataL4
    # +++++++++++++++++++++++++++++++++++++++++++++++
    # 모든 변수에 대한 다중선형회귀모형
    model = sm.OLS.from_formula('연소득당거래금액 ~ 건축년도 + 전용면적 + 층 + 면적당거래금액 + d2', dataL5)
    result = model.fit()
    result.summary()

    # 단계별 다중선형회귀모형
    # 그 결과 앞서 모든 변수 다중선형회귀모형과 동일한 결과를 보임
    bestModel = stepAic(smf.ols, ['건축년도', '전용면적', '층', '면적당거래금액', 'd2'], ['연소득당거래금액'], data=dataL5)
    bestModel.summary()

    # # 전체 아파트에 대한 주택가격 결정요인 (연소득당 거래금액) 예측 산점도
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '전체 아파트에 대한 주택가격 결정요인 예측 산점도')

    makeScatterPlot(bestModel.predict(), dataL5['연소득당거래금액'], saveImg, 0, 15)

except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] {}'.format('Main'))
