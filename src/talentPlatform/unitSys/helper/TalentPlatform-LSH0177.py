# -*- coding: utf-8 -*-

from src.talentPlatform.unitSys.helper.InitConfig import *
import matplotlib.pyplot as plt
import pandas as pd
# from pyramid.arima import auto_arima
import pmdarima as pm

#================================================
# 요구사항
#================================================
# Python을 이용한 코로나 예측 및 시각화 (보고서 포함)

# =================================================
# Set Env
# =================================================
# 작업환경 경로 설정
contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
prjName = 'test'
serviceName = 'LSH0177'

log = initLog(contextPath, prjName)
globalVar = initGlobalVar(contextPath, prjName)

# =================================================
# Set Fun
# =================================================
# 시각화를 위한 국가 선택
def get_covid_data_selected(covid_data, n, country_list):
    covid_data_selected = covid_data.sort_values(
        covid_data.columns[-1], ascending=False)[0:n]
    for country in country_list:
        covid_data_country = covid_data[covid_data['Country/Region']==country]
        covid_data_selected = pd.concat([covid_data_selected, covid_data_country], axis=0)
    return covid_data_selected

# 국가 및 날짜 데이터 변환
def get_covid_data_transposed(covid_data):
    covid_data = covid_data.T
    covid_data.columns =  covid_data.iloc[0]
    covid_data = covid_data[1:]
    covid_data.reset_index(inplace=True)
    covid_data.rename(columns={'index': 'date'},inplace=True)
    covid_data["date"] = pd.to_datetime(covid_data["date"])
    covid_data.set_index('date', inplace=True)
    return covid_data

# 날짜별 감영자 데이터 생성
def get_covid_data_diff(covid_data_transformed):
    covid_data_transformed_diff = covid_data_transformed.diff().fillna(covid_data_transformed.iloc[0])
    return covid_data_transformed_diff

# 분석을 위한 데이터 생성
def get_covid_data_transformed(covid_data, n, country_list):
    covid_data = get_covid_data_selected(covid_data, n, country_list)
    covid_data = get_covid_data_transposed(covid_data)
    return covid_data, get_covid_data_diff(covid_data)

# =================================================
# Main
# =================================================
try:
    log.info('[START] {}'.format('Main'))

    # ***************************************************
    # 코로나 19 데이터 수집
    # ***************************************************
    covid_data_org  = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

    covid_data = covid_data_org.copy()
    covid_data.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
    covid_data = covid_data.groupby(['Country/Region'], as_index=False).sum()

    # 대한민국 누적 확진자 및 일별 확진자 추출
    covid_data_total, covid_data_daily = get_covid_data_transformed(
        covid_data, 0, ["Korea, South"])

    # ***************************************************
    # 일별 확진자 누적건수 데이터셋 및 컬럼 할당
    # ***************************************************
    covid_data_total = covid_data_total.rename({'Korea, South': 'val'}, axis='columns')
    covid_data_total['dtDate'] = covid_data_total.index
    covid_data_total['dtDateDay'] = (covid_data_total.index - covid_data_total.index[0]).days

    # ***************************************************
    # 일별 확진자 건수 데이터셋 및 컬럼 할당
    # ***************************************************
    covid_data_daily = covid_data_daily.rename({'Korea, South': 'val'}, axis='columns')
    covid_data_daily['dtDate'] = covid_data_daily.index
    covid_data_daily['dtDateDay'] = (covid_data_daily.index - covid_data_daily.index[0]).days

    # ***************************************************
    # 일별 확진자 누적건수
    # ***************************************************
    lmFit = linregress(covid_data_total['dtDateDay'].astype(float), covid_data_total['val'].astype(float))
    slope = lmFit[0] ; intercept = lmFit[1] ; r = lmFit[2] ; pVal = lmFit[3]
    yHat = (slope * covid_data_total[['dtDateDay']]) + intercept

    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '대한민국 일별 확진자 누적건수')
    plt.plot(covid_data_total['val'], label='Korea, South')
    plt.plot(yHat, color='red', linewidth=2)
    plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=('2020-01-05', 130000),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (r, pVal), xy=('2020-01-05', 110000),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.title('대한민국 일별 확진자 누적건수')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

    # ***************************************************
    # 일별 확진자 건수
    # ***************************************************
    lmFit = linregress(covid_data_daily['dtDateDay'].astype(float), covid_data_daily['val'].astype(float))
    slope = lmFit[0] ; intercept = lmFit[1] ; r = lmFit[2] ; pVal = lmFit[3]
    yHat = (slope * covid_data_daily[['dtDateDay']]) + intercept

    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '대한민국 일별 확진자 건수')
    plt.plot(covid_data_daily['val'], label='Korea, South')
    plt.plot(yHat, color='red', linewidth=2)
    plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=('2020-01-05', 1250),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (r, pVal), xy=('2020-01-05', 1100),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.title('대한민국 일별 확진자 건수')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

    # ***************************************************
    # 회귀모형을 위한 데이터셋 및 컬럼 할당
    # ***************************************************
    data = covid_data_total

    #***************************************************
    # 2021-01-01년을 기준으로 룬련 및 테스트 데이터셋 분류
    # ***************************************************
    splitIdx = pd.Timestamp('01-01-2021')
    trainData = data.loc[:splitIdx, ]
    testData = data.loc[splitIdx:, ]

    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '대한민국 일별 확진자 누적건수에 대한 훈련-테스트 데이터 셋')
    plt.plot(trainData['val'], label='훈련 데이터셋')
    plt.plot(testData['val'], label='테스트 데이터셋')
    plt.title('대한민국 일별 확진자 누적건수에 대한 훈련-테스트 데이터 셋')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

    # ***************************************************
    # 시계열 예측
    # ***************************************************
    # ARIMA 모형
    model = pm.auto_arima(trainData[['val']])
    model.summary()

    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '시계열 세부정보')
    model.plot_diagnostics()
    plt.title('시계열 세부정보')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

    # 테스트 데이터셋을 통해 예측
    predData = pd.DataFrame()
    fitted, confint = model.predict(len(testData), return_conf_int=True)
    predData['val']  = fitted
    predData['lowConf']  = confint[:, 0]
    predData['uppConf']  = confint[:, 1]
    predData.set_index(testData.index, inplace = True)

    # 대한민국 일별 확진자 누적건수에 대한 시계열 예측
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '대한민국 일별 확진자 누적건수에 대한 시계열 예측')
    plt.plot(trainData['val'], label='훈련 데이터셋')
    plt.plot(testData['val'], label='테스트 데이터셋')
    plt.plot(predData['val'], label='시계열 예측')
    plt.title('대한민국 일별 확진자 누적건수에 대한 시계열 예측')
    plt.fill_between(predData.index,
                     predData['lowConf'],
                     predData['uppConf'],
                     color='k', alpha=0.15)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

    # 대한민국 일별 확진자 누적건수에 대한 실측-예측 산점도
    saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '대한민국 일별 확진자 누적건수에 대한 실측-예측 산점도')
    makeUserScatterPlot(testData['val'].astype(float), predData['val'].astype(float), '대한민국 일별 확진자 누적건수에 대한 실측-예측 산점도', saveImg, 60000, 10000)

except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] {}'.format('Main'))