# -*- coding: utf-8 -*-

if __name__ == '__main__':
    # 라이브러리 읽기

    import numpy as np
    import os
    import pandas as pd
    import datetime as dt
    from plotnine import *
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_squared_error

    print('[START] Main : {}'.format('Run Program'))
    print('[CHECK] Work Space : {}'.format(os.getcwd()))

    # 작업 환경 설정
    os.chdir(os.getcwd())

    # ==========================================================================
    # 회귀식 추정
    # ==========================================================================
    # 각각 하나의 독립변수와 종속변수로 이루어진 데이터 집합에 대해,
    # 두 변수의 관계를 가장 잘 설명할 수 있는 수학적 모형(1차 또는 2차 회귀식)을 가정하고
    # 에러를 최소화하는 모수 값을 최적화 알고리즘(Genetic Algorithm, Simulated Annealing)을 이용하여 추정하세요.

    # 1974년에 미국의 모터 트렌드 잡지에 실린 1973 ~ 1974년 자동차 모델의 연료 소비, 10가지 디자인 요소, 성능을 비교한 mtcars 데이터 사용
    data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv')

    # 무게 (wt)에 따른 연비 (mpg) 예측 수행
    # 독립변수 : 무게 (wt)
    # 종속변수 : 연비 (mpg)
    x = 'wt'
    y = 'mpg'

    # *******************************************************************************************************
    # Genetic Algorithm
    # *******************************************************************************************************
    print('[CHECK] Unit : {}'.format('Genetic Algorithm'))

    # 회귀모형 초기화
    model = smf.ols('mpg ~ wt', data=data)
    model = model.fit()

    # 요약
    # model.summary()

    # (mpg) = -5.344472 * (wt) + 37.285126
    print('[CHECK] params : {}'.format(model.params))

    # mpg 예측
    data['prd'] = model.predict()


    # 산점도 시각화
    mainTitle = '{}'.format('무게에 따른 연비 예측')
    saveImg = './{}'.format(mainTitle)

    plt.plot(data['wt'], data['mpg'], 'o')
    plt.plot(data['wt'], data['prd'], 'r', linewidth=2)
    plt.annotate('%s = %.2f x (%s) + %.2f' % ('mpg', model.params[1], 'wt', model.params[0]), xy=(3.5, 34), color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (np.sqrt(model.rsquared), model.f_pvalue), xy=(3.5, 32), color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('RMSE = %.2f' % (np.sqrt(mean_squared_error(data['mpg'], data['prd']))), xy=(3.5, 30), color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.xlabel('wt')
    plt.ylabel('mpg')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    # 산점도 시각화
    # mainTitle = '{}'.format('무게에 따른 연비 예측 test')
    # saveImg = './{}'.format(mainTitle)
    #
    # prd = (data['wt'] * -9.89999999999992) + 39.90000000000014
    # plt.plot(data['wt'], data['mpg'], 'o')
    # plt.plot(data['wt'], prd, 'r', linewidth=2)
    # plt.xlabel('wt')
    # plt.ylabel('mpg')
    # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    # plt.show()
    # plt.close()

    # *******************************************************************************************************
    # Simulated Annealing
    # *******************************************************************************************************
    print('[CHECK] Unit : {}'.format('Simulated Annealing'))
    slopeList = np.arange(-100, 100, 1)
    interceptList = np.arange(-100, 100, 1)

    # slopeList = np.arange(-5, 5, 0.01)
    # slopeList = np.arange(-10, 10, 0.1)
    # interceptList = np.arange(30, 40, 0.1)

    ref = data['mpg']

    dataL1 = pd.DataFrame()
    for i, slope in enumerate(slopeList):
        for j, intercept in enumerate(interceptList):

            prd = (data['wt'] * slope) + intercept
            rmse = np.sqrt(mean_squared_error(ref, prd))

            dict = {
                'slope' : [slope]
                , 'intercept' : [intercept]
                , 'rmse' : [rmse]
            }

            dataL1 = pd.concat([dataL1, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)

    idx = dataL1.idxmin()['rmse']

    # 앞선 회귀모형과 유사하게 근사값으로 출력
    # 보다 상세한 시뮬레이션을 위해서 slopeList, interceptList의 간격을 조밀하게 설정 (현재 0.1 간격)
    print('[CHECK] params : {}'.format(dataL1.iloc[idx, ]))

    # 산점도 시각화
    mainTitle = '{}'.format('시뮬레이션에 따른 RMSE 결과')
    saveImg = './{}'.format(mainTitle)

    plt.plot(dataL1['rmse'], 'o')
    plt.xlabel('index')
    plt.ylabel('RMSE')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    # # ==========================================================================
    # # 혹시 월 별 따릉이 이용건수 이런 것도 가능할까요??
    # # ==========================================================================
    # print('[CHECK] Unit : {}'.format('Visualization'))
    #
    # # 위 사이트에서 21년도 01월-7월 데이터로 하고싶습니다
    # visData = pd.read_csv('./서울특별시 공공자전거 일별 대여건수_21.07-21.12.csv', encoding='EUC-KR')
    #
    # visData['dtDate'] = pd.to_datetime(visData['대여일시'], format='%Y-%m-%d')
    # visData['월'] = visData['dtDate'].dt.strftime("%m")
    # visData['대여건수'] = visData['대여건수'].replace(',', '', regex=True).astype('float64')
    #
    # visDataL1 = visData.groupby(['월']).sum().reset_index()
    #
    # mainTitle = '{}'.format('월에 따른 따릉이 이용건수')
    # saveImg = './{}'.format(mainTitle)
    #
    # plot = (
    #         ggplot(data=visDataL1) +
    #         aes(x='월', y='대여건수', fill='대여건수') +
    #         theme_bw() +
    #         geom_bar(stat='identity') +
    #         labs(title=mainTitle, xlab='월', ylab='대여 건수') +
    #         theme(
    #             text=element_text(family="Malgun Gothic", size=18)
    #             # , axis_text_x=element_text(angle=45, hjust=1, size=6)
    #             # , axis_text_y=element_text(size=10)
    #         )
    # )
    #
    # fig = plot.draw()
    # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
    # fig.show()

    # 프로그램 종료
    print('[END] Main : {}'.format('Run Program'))
