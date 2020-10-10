# 안녕하세요 전문가님! 질문이 있어서 연락드렸습니다!
# 제가 회귀분석을 하고싶은데요.
# 케이스별로 데이터는 정리를 했습니다. 저는 해외에서 스마트 농업을 공부하면서 그 안에서 딥러닝으로 예측모델을 구현하려고 연구를 하고있습니다. 이 전에 데이터들간의 관계를 좀더 분석기법을 통해서 표현하고싶어서 이렇게 전문가님께 연락을 드렸습니다. PLS, PCR, PCA이런 기법들을 활용하고싶은데요..ㅠ
# 연구의 활용도로 쓰이겠지만 제일 큰 목적은 전문가님께 문의를 받고 앞으로도 계속 공부를 하는데 적용해서 하고싶은 마음이 더욱더 큽니다. 데이터도 참고해주셔서 상담 해주시면 감사하겠습니다.

# 라이브러리 읽기
import logging as log
import os
import sys
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import warnings
import seaborn as sns
from scipy import stats
import traceback
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics
from datetime import datetime
from sklearn.pipeline import Pipeline
from warnings import simplefilter
import glob
from sklearn.cross_decomposition import PLSRegression as PLS
import numpy as np
from sklearn.preprocessing import StandardScaler
import dfply as dfply
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# 로그 설정
log.basicConfig(stream=sys.stdout, level=log.INFO,
                format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
warnings.filterwarnings("ignore")
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


# 상관계수 행렬 시각화
def makeCorrPlot(data, savefigName):
    corr = data.corr(method='pearson')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, square=True, annot=False, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.savefig(savefigName, dpi=600, bbox_inches='tight')
    plt.show()


# 산점도 시각화
def makeScatterPlot(PredValY, valY, titleName, savefigName):
    X = PredValY
    Y = valY

    plt.scatter(X, Y)

    # arrVal = np.array([X, Y])
    # setMin = np.min(arrVal)
    # setMax = np.max(arrVal)
    # interval = (setMax - setMin) / 10

    setMin = 1.0
    setMax = 40
    interval = 2

    plt.title("")
    plt.xlabel('Val')
    plt.ylabel('Pred')
    # plt.xlim(0, setMax)
    # plt.ylim(0, setMax)
    plt.grid()

    # Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
    Bias = np.mean(X - Y)
    rBias = (Bias / np.mean(Y)) * 100.0
    RMSE = np.sqrt(np.mean((X - Y) ** 2))
    rRMSE = (RMSE / np.mean(Y)) * 100.0
    MAPE = np.mean(np.abs((X - Y) / X)) * 100.0

    slope, intercept, R, Pvalue, std_err = stats.linregress(X, Y)
    N = len(X)

    lmfit = (slope * X) + intercept
    plt.plot(X, lmfit, color='red', linewidth=2)
    plt.plot([0, setMax], [0, setMax], color='black')

    plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval), color='red',
                 fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R-square = %.2f  (p-value < %.2f)' % (R ** 2, Pvalue), xy=(setMin, setMax - interval * 2),
                 color='red',
                 fontweight='bold', xycoords='data',
                 horizontalalignment='left', verticalalignment='center')
    plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(setMin, setMax - interval * 3),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(setMin, setMax - interval * 4),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(setMin, setMax - interval * 5),
                 color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 6), color='black', fontweight='bold',
                 xycoords='data', horizontalalignment='left',
                 verticalalignment='center')
    plt.title(titleName)
    plt.savefig(savefigName, dpi=600, bbox_inches='tight')
    plt.show()


# 회귀모형을 위한 교차검증 수행
def gridsearch_cv_for_regression(model, param, kfold, train_input, train_target,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1, tracking=True):
    '''
    [Parameters]
    - model: A tuple like ('name', MODEL)
    - param
    - scoring: neg_mean_absolute_error, neg_mean_squared_error, neg_median_absolute_error, r2
               (http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
    - n_jobs: default as -1 (if it is -1, all CPU cores are used to train and validate models)
    - tracking: whether trained model's name and duration time are printed
    '''

    name = model[0]
    estimator = model[1]
    if tracking:
        start_time = datetime.now()
        print("[%s] Start parameter search for model '%s'" % (start_time, name))
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=kfold, scoring=scoring,
                                  n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target)
        end_time = datetime.now()
        duration_time = (end_time - start_time).seconds
        print(
            "[%s] Finish parameter search for model '%s' (time: %d seconds)" % (end_time, name, duration_time))
        print()
    else:
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=kfold, scoring=scoring,
                                  n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target)

    return gridsearch


try:
    log.info('[START] Main : {}'.format('Run Program'))

    # 작업환경 경로 설정
    # contextPath = os.getcwd()
    contextPath = 'D:/02. 블로그/PyCharm'
    # contextPath = 'E:/02. 블로그/PyCharm'

    # 전역 변수
    globalVar = {
        "config": {
            "imgContextPath": contextPath + '/resources/image/'
            , "csvConfigPath": contextPath + '/resources/data/csv/'
            , "xlsxConfigPath": contextPath + '/resources/data/xlsx/'
        }
    }

    log.info("[Check] globalVar : {}".format(globalVar))

    # ==============================================
    # 주 소스코드
    # ==============================================
    caseList = ['case{}'.format(str(i)) for i in range(1, 7)]

    case = 'case2'

    for (ind, case) in enumerate(caseList):
        log.info("[Check] case : {}".format(case))

        # 파일 읽기
        inFile = globalVar.get('config').get('csvConfigPath') + 'data/{}/data/*.csv'.format(case)

        dataL1 = pd.DataFrame()
        for i in glob.glob(inFile):
            # if ((i.__contains__('case1')) and not (
            #         (i.__contains__('Takemoto_2016_Field_4')) or (
            #         i.__contains__('Takemoto_2017_Field_2')))): continue
            # if ((i.__contains__('case2')) and not (
            #         (i.__contains__('Takemoto_2016_Field_4')) or (
            #         i.__contains__('Takemoto_2018_Field_3')))): continue

            log.info("[Check] inFile : {}".format(i))
            data = pd.read_csv(i, encoding="euc-kr")
            dataL1 = dataL1.append(data)

        valFile = globalVar.get('config').get('csvConfigPath') + 'data/{}/data_test/*.csv'.format(case)

        dataTestL1 = pd.DataFrame()
        for i in glob.glob(valFile):
            log.info("[Check] valFile : {}".format(i))

            dataTest = pd.read_csv(i, encoding="euc-kr")
            dataTestL1 = dataTestL1.append(dataTest)

        # 파일 정보 읽기
        # dataL1.head()

        # =========================================
        # 1. 탐색
        # =========================================
        # 각 자료에 대한 빈도분포
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_Hist.png' % (50, case, 'train')
        dataL1.hist()
        plt.savefig(savefigName, dpi=600, bbox_inches='tight')
        plt.show()

        # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_BoxPlot.png' % (50, case, 'train')
        dataL1.plot(kind='box', subplots=True, sharex=False, sharey=False)
        plt.savefig(savefigName, dpi=600, bbox_inches='tight')
        plt.show()

        # 상관분석 행렬 시각화 및 자료 저장
        dataCorr = dataL1.corr(method='pearson')
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_CorrMatrix.png' % (
        50, case, 'train')
        makeCorrPlot(dataCorr, savefigName)

        dataL1Summary = dataL1.describe()
        # dataTestL1Summary = dataTestL1.describe()
        log.info("[Check] Train dataL1 Summary : {}".format(dataL1Summary))

        # 각 자료에 대한 빈도분포
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_Hist.png' % (50, case, 'test')
        dataTestL1.hist()
        plt.savefig(savefigName, dpi=600, bbox_inches='tight')
        plt.show()

        # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_BoxPlot.png' % (50, case, 'test')
        dataTestL1.plot(kind='box', subplots=True, sharex=False, sharey=False)
        plt.savefig(savefigName, dpi=600, bbox_inches='tight')
        plt.show()

        # 상관분석 행렬 시각화 및 자료 저장
        dataCorr = dataTestL1.corr(method='pearson')
        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_CorrMatrix.png' % (
        50, case, 'test')
        makeCorrPlot(dataCorr, savefigName)

        dataTestL1Summary = dataTestL1.describe()
        log.info("[Check] Test dataL1 Summary : {}".format(dataTestL1Summary))

        # =========================================
        # 3. 목적에 맞는 분석
        # ========================================
        dataL1 = dataL1.rename({'Section5_Accumulated precipitation (mm)': 'Section5_Accumulated_precipitation'},
                               axis='columns')

        # 특정 행을 대상으로 NA값 삭제
        dataL2 = dataL1[['TD', 'SFV', 'N', 'Section4_Winkler scale(℃)', 'Section4_Accumulated precipitation (mm)',
                         'Section5_Winkler scale(℃)', 'Section5_Accumulated_precipitation', 'S1']].dropna(axis=0)

        # 강수 유무에 따른 별도 회귀계수 필요
        # 무강수
        dataL3 = (dataL2 >>
                  dfply.mask(dfply.X.Section5_Accumulated_precipitation < 10)
                  )
        # 유강수
        # dataL3 = (dataL2 >>
        #           dfply.mask(dfply.X.Section5_Accumulated_precipitation > 10)
        #           )

        # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

        # 트레이닝 셋에 대한 독립변수 설정
        X_train = dataL3[['TD', 'SFV', 'N', 'Section4_Winkler scale(℃)', 'Section4_Accumulated precipitation (mm)',
                          'Section5_Winkler scale(℃)', 'Section5_Accumulated_precipitation']]
        # X_train = dataL3[['TD', 'SFV']]

        # 트레이닝 셋에 대한 종속변수 설정
        Y_train = dataL3[['S1']]

        #  테스트 셋에 대한 독립변수 설정
        X_test = dataTestL1[
            ['TD', 'SFV', 'N', 'Section4_Winkler scale(℃)', 'Section4_Accumulated precipitation (mm)',
             'Section5_Winkler scale(℃)', 'Section5_Accumulated precipitation (mm)']]
        # X_test = dataTestL1[['TD', 'SFV']]

        #  테스트 셋에 대한 종속변수 설정
        Y_test = dataTestL1[['S1']]

        # X_train.hist()
        # plt.show()
        #
        # X_test.hist()
        # plt.show()
        #
        # Y_train.hist()
        # plt.show()
        #
        # Y_test.hist()
        # plt.show()

        # ========================================================
        # 주성분 회귀분석
        # ========================================================
        pca = PCA(n_components=len(X_train.columns))

        std = StandardScaler()
        std.fit(X_train)

        X_std_train = std.transform(X_train)
        X_std_test = std.transform(X_test)

        pca.fit(X_std_train)

        X_pca_train = pca.transform(X_std_train)
        X_pca_test = pca.transform(X_std_test)

        lr = LinearRegression()
        lr.fit(X_pca_train, Y_train)

        Y_pca_test_hat = lr.predict(X_pca_test)

        mean_squared_error(Y_pca_test_hat, Y_test)

        savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_PCA.png' % (51, case)
        makeScatterPlot(Y_pca_test_hat[:, 0], Y_test.values[:, 0], "PCA", savefigName)

        rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pca_test_hat))
        r2 = metrics.r2_score(Y_test, Y_pca_test_hat)
        log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % ("PCA", rmse, r2))

        # ====================================================
        # [표준회 X] 4종 회귀모형을 이용
        # ====================================================
        # 초기값 설정
        models = []
        params = []

        # 선형회귀모형 및 파라미터 설정
        model = ('Linear', linear_model.LinearRegression())
        param = {}

        models.append(model)
        params.append(param)

        # 릿지 회귀모형 및 파라미터 (알파 조정) 설정
        model = ('Ridge', linear_model.Ridge())
        param = {
            'alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
        }

        models.append(model)
        params.append(param)

        # 라쏘에 대해서 및 파라미터 (알파 조정) 설정
        model = ('Lasso', linear_model.Lasso())
        param = {
            'alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
        }

        models.append(model)
        params.append(param)

        # 엘라스틴에 대해서 및 파라미터 (알파 조정) 설정
        model = ('ElasticNet', linear_model.ElasticNet())
        param = {
            'alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
            'l1_ratio': [0.3, 0.5, 0.7]
        }

        models.append(model)
        params.append(param)

        # PLS Regression
        model = ('PLS', PLS())
        param = {
            # 'n_components': [1, 2, 3, 4, 5, 6, 7]
            'n_components': [1, 2]
        }

        models.append(model)
        params.append(param)

        # 모델에 대한 정보
        log.info("[Check] models | %s" % (models))

        # ====================================================
        # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
        # ====================================================
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

        # 교차 검증 수행
        kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

        # 트레이닝 셋을 이용한 학습
        results = []
        for i in range(len(models)):
            model = models[i]
            param = params[i]

            result = gridsearch_cv_for_regression(model=model, param=param, kfold=kfold
                                                  , train_input=X_train, train_target=Y_train)
            result.best_score_
            results.append(result)

        # 테스트 셋을 이용한 검증
        for i in range(len(results)):
            Y_test_hat = results[i].predict(X_test)
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s.png' % (i + 52, case)

            selModel = models[i][0]

            if (selModel.__contains__('Lasso') or selModel.__contains__('ElasticNet')):
                makeScatterPlot(Y_test_hat, Y_test.values[:, 0], selModel, savefigName)
            else:
                makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], selModel, savefigName)

            rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
            r2 = metrics.r2_score(Y_test, Y_test_hat)
            log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (selModel, rmse, r2))

        # ========================================================
        # [표준회 O] 4종 회귀모형을 이용
        # ========================================================
        # 초기값 설정
        models = []
        params = []

        # 선형회귀모형 및 파라미터 설정
        model = (
            'Scaled Linear', Pipeline([('Scaler', StandardScaler()), ('Linear', linear_model.LinearRegression())]))
        param = {}

        models.append(model)
        params.append(param)

        # 릿지 회귀모형 및 파라미터 (알파 조정) 설정
        model = ('Scaled Ridge', Pipeline([('Scaler', StandardScaler()), ('Ridge', linear_model.Ridge())]))
        param = {
            'Ridge__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
        }

        models.append(model)
        params.append(param)

        # 라쏘에 대해서 및 파라미터 (알파 조정) 설정
        model = ('Scaled Lasso', Pipeline([('Scaler', StandardScaler()), ('Lasso', linear_model.Lasso())]))
        param = {
            'Lasso__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
        }

        models.append(model)
        params.append(param)

        # 엘라스틴에 대해서 및 파라미터 (알파 조정) 설정
        model = (
            'Scaled ElasticNet',
            Pipeline([('Scaler', StandardScaler()), ('ElasticNet', linear_model.ElasticNet())]))
        param = {
            'ElasticNet__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
            'ElasticNet__l1_ratio': [0.3, 0.5, 0.7]
        }

        models.append(model)
        params.append(param)

        # PLS Regression
        model = [
            'Scaled PLS', Pipeline([('Scaler', StandardScaler()), ('PLS', PLS())])]
        param = {
            'PLS__n_components': [1, 2, 3, 4, 5, 6, 7]
            # 'PLS__n_components': [1, 2]
        }

        models.append(model)
        params.append(param)

        log.info("[Check] models | %s" % (models))
        log.info("[Check] params | %s" % (params))

        # ====================================================
        # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
        # ====================================================
        # 트레이닝 셋을 이용한 학습
        results = []
        for i in range(len(models)):
            model = models[i]
            param = params[i]
            result = gridsearch_cv_for_regression(model=model, param=param, kfold=kfold
                                                  , train_input=X_train, train_target=Y_train)
            result.best_score_
            results.append(result)

        # 테스트 셋을 이용한 검증
        for i in range(len(results)):
            Y_test_hat = results[i].predict(X_test)
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s.png' % (i + 57, case)

            selModel = models[i][0]

            if (selModel.__contains__('Scaled Lasso') or selModel.__contains__(
                    'Scaled ElasticNet')):
                makeScatterPlot(Y_test_hat, Y_test.values[:, 0], selModel, savefigName)
            else:
                makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], selModel, savefigName)

            rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
            r2 = metrics.r2_score(Y_test, Y_test_hat)
            log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (selModel, rmse, r2))

except Exception as e:
    log.error("Exception : {}".format(e))
    traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] Main : {}'.format('Run Program'))
