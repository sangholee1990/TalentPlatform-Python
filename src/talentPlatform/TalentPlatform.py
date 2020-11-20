# -*- coding: utf-8 -*-
import dfply


class sDtaProcess:

    def uPro01(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        from src.talentPlatform.util import central_limit_theorem as clt

        import numpy as np
        import logging as log
        import sys

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        # 이미지 읽기
        image = Image.open("/resources/image/lena_gray.gif")

        arrVal2D = np.array(image)
        arrVal1D = arrVal2D.flatten()

        log.info("=================== 과제 1 ===================")
        log.info("모집단 크기 : {%s}", arrVal1D.size)
        log.info("모집단 평균 : {%s}", round(np.mean(arrVal1D), 2))
        log.info("모집단 분산 : {%s}", round(np.var(arrVal1D), 2))
        log.info("모집단 최대값 : {%s}", np.max(arrVal1D))
        log.info("모집단 최소값 : {%s}", np.min(arrVal1D))
        log.info("모집단 중앙값 : {%s}", np.median(arrVal1D))

        log.info("=================== 과제 2 ===================")
        plt.hist(arrVal1D)
        plt.show()

        log.info("=================== 과제 3 ===================")
        binList = [10, 100, 1000]

        for i in binList:
            plt.hist(arrVal1D, bins=i)
            plt.show()

        log.info("=================== 과제 4 ===================")
        callClt = clt.CentralLimitTheorem(arrVal1D)
        sampleList = [5, 10, 20, 30, 50, 100]
        for sample in sampleList:
            callClt.run_sample(N=sample, plot=True, num_bins=None)

    def uPro02(self):
        import logging as log
        import sys
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        from urllib.request import urlopen
        from wordcloud import WordCloud
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from bs4 import BeautifulSoup
        import dfply as dfply

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
        contextPath = os.getcwd()

        # 제출할 내용 :
        # -파이썬 코드 파일
        # -단어 구름 시각화를위한 이미지 파일

        # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

        # 1) https://edition.cnn.com/2020/06/02/world/nodosaur-fossil-stomach-contents-scn-trnd/index.html에서 기사 내용을 스크랩하십시오.
        html = urlopen("https://edition.cnn.com/2020/06/02/world/nodosaur-fossil-stomach-contents-scn-trnd/index.html")
        # html = requests.get(url)
        soup = BeautifulSoup(html, 'html.parser')

        section = soup.select('section.zn-body-text')

        liGetText = []
        for i in section:
            getText = i.get_text()

            log.info("getText : {%s} : {%s}", len(getText), getText)

            # 단어 추출
            wordTokens = word_tokenize(getText)
            # 불용어
            stopWords = set(stopwords.words('english'))

            log.info("wordTokens : {%s} : {%s}", len(wordTokens), wordTokens)
            log.info("stopWords : {%s} : {%s}", len(stopWords), stopWords)

            # 2) 기사 내용을 사전 처리하여 불용어없이 단수 명사 목록을 얻습니다.
            for j in wordTokens:
                if j not in stopWords:
                    liGetText.append(j)

        log.info("liGetText : {%s} : {%s}", len(liGetText), liGetText)

        data = pd.DataFrame({
            'type': liGetText
        })

        # 3) 빈도분포 및 워드 클라우드 시각화
        dataL1 = (
            (data >>
             dfply.filter_by(
                 dfply.X.type != '.'
                 , dfply.X.type != ','
                 , dfply.X.type != "'"
                 , dfply.X.type != "''"
                 , dfply.X.type != "``"
                 , dfply.X.type != "'s"
             ) >>
             dfply.group_by(dfply.X.type) >>
             dfply.summarize(number=dfply.n(dfply.X.type)) >>
             dfply.ungroup() >>
             dfply.arrange(dfply.X.number, ascending=False)
             ))

        log.info("dataL1 : {%s} : {%s}", len(dataL1), dataL1)

        # 데이터 시각화를 위한 전처리
        objData = {}
        for i in dataL1.values:
            key = i[0]
            val = i[1]

            objData[key] = val

        log.info("objData : {%s} : {%s}", len(objData), objData)

        wordcloud = WordCloud(
            width=1000
            , height=1000
            , background_color="white"
        ).generate_from_frequencies(objData)

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        savefigName = contextPath + '/../resources/image/Image_01.png'
        plt.savefig(savefigName, width=1000, heiht=1000, dpi=600, bbox_inches='tight')
        plt.show()

    def uPro03(self):

        import logging as log
        import os
        import sys

        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import linregress
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from src.talentPlatform.util.forecasting_metrics import mase
        from sklearn.model_selection import KFold, GridSearchCV
        from sklearn import metrics
        from datetime import datetime

        # warnings.filterwarnings("ignore")

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        # 상관계수 행렬 시각화
        def makeCorrPlot(data, savefigName):

            corr = data.corr(method='pearson')
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            sns.heatmap(corr, square=True, annot=False, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=0.5)
            plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            plt.show()

        # 산점도 시각화
        def makeScatterPlot(valY, PredValY, savefigName):

            X = valY
            Y = PredValY

            plt.scatter(X, Y)

            # arrVal = np.array([X, Y])
            # setMax = np.max(arrVal)
            setMin = 200
            setMax = 8000
            interval = 500

            plt.title("")
            plt.xlabel('Val')
            plt.ylabel('Pred')
            plt.xlim(0, setMax)
            plt.ylim(0, setMax)
            plt.grid()

            ## Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
            Bias = np.mean(X - Y)
            rBias = (Bias / np.mean(Y)) * 100.0
            RMSE = np.sqrt(np.mean((X - Y) ** 2))
            rRMSE = (RMSE / np.mean(Y)) * 100.0
            MAPE = np.mean(np.abs((X - Y) / X)) * 100.0
            MASE = mase(X, Y)

            slope = linregress(X, Y)[0]
            intercept = linregress(X, Y)[1]
            R = linregress(X, Y)[2]
            Pvalue = linregress(X, Y)[3]
            N = len(X)

            lmfit = (slope * X) + intercept
            plt.plot(X, lmfit, color='red', linewidth=2)
            plt.plot([0, setMax], [0, setMax], color='black')

            plt.annotate('Pred = %.2f x (Val) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval), color='red',
                         fontweight='bold',
                         xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2), color='red',
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
            plt.annotate('MASE = %.2f' % (MASE), xy=(setMin, setMax - interval * 6),
                         color='black', fontweight='bold',
                         xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('N = %d' % N, xy=(setMin, setMax - interval * 7), color='black', fontweight='bold',
                         xycoords='data', horizontalalignment='left',
                         verticalalignment='center')
            plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            plt.show()

        # 선형회귀모형 트레이닝 모형
        def train_test_linreg(data, feature_cols):
            X = data[feature_cols]
            Y = data.total
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
            model = LinearRegression()
            model.fit(X_train, Y_train)

            # Make series using selected features and corresponding coefficients
            formula = pd.Series(model.coef_, index=feature_cols)

            # Save intercept
            intercept = model.intercept_

            # Calculate training RMSE and testing RMSE
            Y_pred_train = model.predict(X_train)
            Y_pred_test = model.predict(X_test)
            rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
            rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))

            # Calculate training R-square and testing R-square
            rsquared_train = model.score(X_train, Y_train)
            rsquared_test = model.score(X_test, Y_test)

            # Make result dictionary
            result = {'formula': formula, 'intercept': intercept, 'rmse_train': rmse_train, 'rmse_test': rmse_test,
                      'rsquared_train': rsquared_train, 'rsquared_test': rsquared_test,
                      'Y_train': Y_train, 'Y_pred_train': Y_pred_train, 'Y_test': Y_test, 'Y_pred_test': Y_pred_test,
                      'model': model
                      }

            return result

        # 릿지모형 트레이닝 모형
        def train_test_ridge(data, feature_cols, alpha_value):
            X = data[feature_cols]
            Y = data.total
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
            model = Ridge(alpha=alpha_value)
            model.fit(X_train, Y_train)

            # Make series using selected features and corresponding coefficients
            formula = pd.Series(model.coef_, index=list(X.columns.values))

            # Save intercept
            intercept = model.intercept_

            # Calculate training RMSE and testing RMSE
            Y_pred_train = model.predict(X_train)
            Y_pred_test = model.predict(X_test)
            rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
            rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))

            # Calculate training R-square and testing R-square
            rsquared_train = model.score(X_train, Y_train)
            rsquared_test = model.score(X_test, Y_test)

            # Make result dictionary
            result = {'formula': formula, 'intercept': intercept, 'rmse_train': rmse_train, 'rmse_test': rmse_test,
                      'rsquared_train': rsquared_train, 'rsquared_test': rsquared_test}

            return result

        # 교차검증을 통해 하이퍼 파라미터 찾기
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

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
        contextPath = os.getcwd()

        log.info("[START] : {%s}", "uProgram")

        # 파일 읽기
        inpData01 = pd.read_csv(contextPath + '/../resources/data/csv/inp_01.csv', na_filter=False)
        inpData02 = pd.read_csv(contextPath + '/../resources/data/csv/inp_02.csv', na_filter=False)
        inpData03 = pd.read_csv(contextPath + '/../resources/data/csv/inp_03.csv', na_filter=False)
        inpData04 = pd.read_csv(contextPath + '/../resources/data/csv/inp_04.csv', na_filter=False)
        inpData05 = pd.read_csv(contextPath + '/../resources/data/csv/inp_05.csv', na_filter=False)
        inpData06 = pd.read_csv(contextPath + '/../resources/data/csv/inp_06.csv', na_filter=False)

        # 기간 및 자치구에 따른 데이터 병합
        data = ((inpData01 >>
                 left_join(inpData02, by=('기간', '자치구')) >>
                 left_join(inpData03, by=('기간', '자치구')) >>
                 left_join(inpData04, by=('기간', '자치구')) >>
                 left_join(inpData05, by=('기간', '자치구')) >>
                 left_join(inpData06, by=('기간', '자치구')) >>
                 mask(X.자치구 != '합계') >>
                 drop(['합계_검거', '살인_발생', '살인_검거', '강도_발생', '강도_검거', '강간강제추행_발생', '강간강제추행_검거', '절도_발생', '절도_검거', '폭력_발생',
                       '폭력_검거', '합계', '소계'])
                 ))

        # 컬럼 개수 : 42개
        len(data.columns.values)

        # 컬럼 형태
        # dataStep1.dtypes

        # ======================================================
        #  범죄횟수를 기준으로 각 상관계수 행렬 시각화
        # ======================================================
        # data = pd.DataFrame(data.dropna(axis=0))

        tmpColY = data.iloc[:, 2]
        tmpColXStep1 = data.iloc[:, 3:21:1]
        tmpColXStep2 = data.iloc[:, 22:41:1]

        dataStep1 = pd.concat([tmpColY, tmpColXStep1], axis=1)
        dataStep1Corr = dataStep1.corr(method='pearson')
        savefigName = contextPath + '/../resources/image/Image_02.png'

        makeCorrPlot(dataStep1, savefigName)

        dataStep2 = pd.concat([tmpColY, tmpColXStep2], axis=1)
        dataStep2Corr = dataStep2.corr(method='pearson')
        savefigName = contextPath + '/../resources/image/Image_03.png'

        makeCorrPlot(dataStep2, savefigName)

        # ===================================================================
        #  전체 데이터셋 (기간, 자치구)을 이용한 독립변수 및 종속 변수 선정
        # ===================================================================
        dataL1 = ((data >>
                   drop(X.기간, X.자치구)
                   ))

        # ===================================================================
        #  [상관분석 > 유의미한 변수] 전체 데이터셋 (기간, 자치구)을 이용한 독립변수 및 종속 변수 선정
        # ===================================================================
        selCol = ['범죄횟수', '지구대파출소치안센터', '119안전센터', 'CCTV설치현황', '비거주용건물내주택', '계_사업체수', '계_종사자수']
        dataL1 = data[selCol]

        # 결측값에 대한 행 제거 (그에 따른 index 변화로 인해 pd.DataFrame 재변환)
        dataL2 = pd.DataFrame(dataL1.dropna(axis=0))
        dataL2.rename(columns={'범죄횟수': 'total'}, inplace=True)

        # 요약 통계량
        dataL2.describe()

        # 자치구 데이터셋 (기간 평균)을 이용한 독립변수 및 종속 변수 선정
        # selCol = ['기간', '자치구', '범죄횟수', '지구대파출소치안센터', 'CCTV설치현황', '전체세대', '비거주용건물내주택', '계_사업체수']
        # dataL1 = data[selCol]
        #
        # pd.plotting.scatter_matrix(dataL1)
        # plt.show()
        #
        # dataL2 = ((dataL1 >>
        #      group_by(X.자치구) >>
        #      summarize(
        #          total=X.범죄횟수.mean()
        #          , maenX1=X.지구대파출소치안센터.mean()
        #          , maenX2=X.CCTV설치현황.mean()
        #          , maenX3=X.전체세대.mean()
        #          , maenX4=X.비거주용건물내주택.mean()
        #          , maenX5=X.계_사업체수.mean()
        #          ) >>
        #         # arrange(X.number, ascending=False)
        #         drop(X.자치구)
        #      ))

        # ========================================
        #  회귀모형 수행
        # ========================================
        selVarList = list(dataL2.columns[~dataL2.columns.str.contains('total')])

        # 다중선형회귀 모형
        result = train_test_linreg(dataL2, selVarList)

        # 릿지 모형
        # result = train_test_ridge(dataL2, selVarList, 1.0)

        # =======================================
        #  시각화
        # ======================================
        # 트레이닝 데이터
        trainValY = result['Y_train'].values
        trainPredValY = result['Y_pred_train']
        savefigName = contextPath + '/../resources/image/Image_04.png'

        makeScatterPlot(trainValY, trainPredValY, savefigName)

        # 테스트 데이터
        testValY = result['Y_test'].values
        testPredValY = result['Y_pred_test']
        savefigName = contextPath + '/../resources/image/Image_05.png'

        makeScatterPlot(testValY, testPredValY, savefigName)

        # =======================================
        #  교차검증 수행
        # ======================================
        X = dataL2[selVarList]
        Y = dataL2.total
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

        # Pre-allocate models and corresponding parameter candidates
        models = []
        params = []

        model = ('Linear', LinearRegression())
        param = {}

        models.append(model)
        params.append(param)

        log.info("[CHECK] models : {%s}", models)
        log.info("[CHECK] params : {%s}", params)

        kfold = KFold(n_splits=10, shuffle=True)

        results = []

        # [교차검증] 트레이닝 데이터
        for i in range(1):
            model = models[i]
            param = params[i]
            result = gridsearch_cv_for_regression(model=model, param=param, kfold=kfold, train_input=X_train,
                                                  train_target=Y_train)
            result.best_score_
            results.append(result)

        # [교차검증] 테스트 데이터
        for i in range(len(results)):
            testValY = Y_test.values
            testPredValY = results[i].predict(X_test)

            savefigName = contextPath + '/../resources/image/Image_06.png'
            makeScatterPlot(testValY, testPredValY, savefigName)

        log.info("[END] : {%s}", "uProgram")

    def uPro04(self):

        import logging as log
        import os
        import sys

        import pandas as pd
        import re
        import matplotlib.pyplot as plt
        from nltk.corpus import stopwords
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        # import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        # callDtaProcess = sDtaProcess()
        # callDtaProcess.uPro01()

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        contextPath = os.getcwd()

        log.info("[START] : {%s}", "uProgram")

        data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
        airline_tweets = pd.read_csv(data_source_url)
        airline_tweets.head()

        plot_size = plt.rcParams["figure.figsize"]
        print(plot_size[0])
        print(plot_size[1])

        log.info("[CHECK] : {%s} : {%s}", len(plot_size), "plot_size")

        plot_size[0] = 8
        plot_size[1] = 6
        plt.rcParams["figure.figsize"] = plot_size
        airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')

        airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%',
                                                             colors=["red", "yellow", "green"])
        airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
        airline_sentiment.plot(kind='bar')

        log.info("[CHECK] : {%s} : {%s}", len(plot_size), "plot_size")

        sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence', data=airline_tweets)

        features = airline_tweets.iloc[:, 10].values
        labels = airline_tweets.iloc[:, 1].values

        processed_features = []
        for sentence in range(0, len(features)):
            processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
            processed_feature = re.sub(r'\s+[a-zA-Z]/s+', ' ', processed_feature)
            processed_feature = re.sub(r'\^[a-zA-Z]/s+', ' ', processed_feature)
            processed_feature = re.sub(r'/s+', ' ', processed_feature, flags=re.I)
            processed_feature = re.sub(r'^b/s+', '', processed_feature)
            processed_feature = processed_feature.lower()
            processed_features.append(processed_feature)

        vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        processed_features = vectorizer.fit_transform(processed_features).toarray()

        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

        text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        text_classifier.fit(X_train, y_train)
        predictions = text_classifier.predict(X_test)

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print(accuracy_score(y_test, predictions))

        # k-nearest neighbour 방법
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print(accuracy_score(y_test, predictions))

        clf = xgb.XGBClassifier()
        clf_cv = GridSearchCV(clf, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
        clf_cv.fit(X_train, y_train)
        print(clf_cv.best_params_, clf_cv.best_score_)

        clf = xgb.XGBClassifier(**clf_cv.best_params_)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print(accuracy_score(y_test, predictions))

        # clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000)
        #
        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # dtest = xgb.DMatrix(X_test, label=y_test)
        #
        # evals_result = {}
        # clf.fit(X_train, y_train, eval_metric='logloss'
        #         , eval_set=[
        #             (X_train, y_train),
        #             (X_test, y_test),
        #         ]
        #         , early_stopping_rounds=10
        #         , callbacks=[
        #             xgb.callback.record_evaluation(evals_result)
        #         ],
        #         )
        #
        # predictions = clf.predict(X_test)
        # print(confusion_matrix(y_test, predictions))
        # print(classification_report(y_test, predictions))
        # print(accuracy_score(y_test, predictions))

        log.info("[END] : {%s}", "uProgram")

    def uPro05(self):
        import pandas as pd
        # import moment

        # 파일 읽기
        data = pd.read_csv(contextPath + '/../resources/data/csv/KSG_KSG2019111100AMUS01a_20170600000000.csv',
                           na_filter=False, header=None)
        data.columns = ['TOC_CENTER', 'TOC_ID', 'DONG_CODE', 'yyyymmdd', 'hh', 'Hourly_power_consum', 'other']

        data = data.head(20000)
        data["dtDate"] = pd.to_datetime(data["yyyymmdd"], format='%Y%m%d')

        dataL1 = (
                data >>
                mutate(
                    month=X.dtDate.dt.strftime("%m")
                    , week=X.dtDate.dt.weekday
                    # , week2= X.dtDate.dt.strftime("%w")
                    , weekName=X.dtDate.dt.strftime("%a")
                )
        )

        # summary = dataL1.describe()

        # 전력소비 패턴 > 1) 가구별 1일 평균 전력 소비 패턴
        dataL2 = (
                dataL1 >>
                mask(
                    X.month == "06"
                    , X.Hourly_power_consum > 0
                ) >>
                group_by(X.TOC_ID, X.hh) >>
                summarize(
                    meanVal=X.Hourly_power_consum.mean()
                ) >>
                mutate(
                    meanValPerDay=X.meanVal * 24
                )
        )

        # 전력소비 패턴 > 2) 가구별 일주일 평균에서 요일별 피크인 요일
        dataL3 = (
                dataL1 >>
                mask(
                    X.month == "06"
                    , X.Hourly_power_consum > 0
                ) >>
                group_by(X.TOC_ID, X.week) >>
                summarize(
                    meanVal=X.Hourly_power_consum.mean()
                ) >>
                mutate(
                    meanValPerDay=X.meanVal * 24
                )
        )

        dataL3Peak = (
                dataL3 >>
                group_by(X.TOC_ID) >>
                mask(X.meanValPerDay == X.meanValPerDay.max())
        )

        savImgName = contextPath + '/../resources/image/Image_10.png'

        pn = (ggplot(dataL3, aes(x="week", y="meanValPerDay", colour="TOC_ID")) +
              geom_line() +
              geom_point(dataL3Peak, aes(x="week", y="meanValPerDay"), color='black')
              + theme_bw()
              + theme(legend_key=element_blank())
              + theme(legend_text=element_blank())
              + theme(legend_background=element_blank())
              + theme(legend_text=element_blank())
              + theme(legend_background=element_blank())
              + ggtitle("Peak day by day from weekly average by household")
              # scale_x_continuous(breaks=xtick, expand=(0,0)) +
              # scale_x_discrete(limits=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
              )
        ggsave(plot=pn, filename=savImgName, height=8, width=12, dpi=300)

        # 피크 전력 > 1) 가구별 1일 평균 전력 소비에서 피크인 시간
        dataL4 = (
                dataL1 >>
                mask(
                    X.month == "06"
                    , X.Hourly_power_consum > 0
                ) >>
                group_by(X.TOC_ID) >>
                mask(X.Hourly_power_consum == X.Hourly_power_consum.max())
        )

        # dataL2.merge(dataPeak, left_on = '')

        # mtcars3 = pd.merge(dataL1, data, how='left', left_on='maxVal', right_on='Hourly_power_consum')

        # dataL3 = (
        #     dataL1 >>
        #         left_join(data, by=[('Hourly_power_consum', 'maxVal'),('dtDate', 'dtDate'),('TOC_ID', 'TOC_ID')])
        # )

        # dataL2 = (
        #     dataL1 >>
        #         left_join(dataPeak, by = [('maxVal', 'Hourly_power_consum')])
        #         # group_by(X.TOC_ID, X.dtDate) >>
        #         # summarize(
        #         #     meanVal = X.Hourly_power_consum.mean()
        #         # )
        #
        # )
        # left_join(dataL1, by='TOC_ID')

        # dataL2.describe()

    def uPro06(self):

        # import pyproj
        # import geopandas as gpd
        import numpy as np
        # import pyproj
        # import geopandas as gpd
        import networkx as nx
        # import osmnx as ox
        import pandas as pd
        from sklearn.cluster import KMeans
        # import pysal

        # Get pickup points
        # from shapely.geometry import Point

        # warnings.filterwarnings("ignore")

        preContextPaath = contextPath + '/../resources/data/'

        blocks = gpd.read_file(preContextPaath + 'shapes2/부산/AL_26710_D198_20200714.shp')
        yards = gpd.read_file(preContextPaath + 'shapes2/부산/busan_bus_yards.shp')
        shelters = gpd.read_file(preContextPaath + 'shapes2/부산/busan_shelters.shp')
        census = pd.read_csv(preContextPaath + 'shapes2/부산/people.csv', encoding='cp949', sep=',')

        ax = shelters.plot()
        ax.set_axis_off()

        # Construct Elder Blocks from census data
        census_copiapo = census.query('FID1 == 1')[['FID2', 'People']]
        elder_pop = census_copiapo.set_index('FID2')
        blocks['A3'] = blocks['A3'].astype(np.int64)
        elder_pop_paipote = elder_pop
        blocks = blocks.merge(elder_pop_paipote, how='inner', left_on='A3', right_index=True)
        blocks_elder = blocks[blocks['People'] != 0]
        blocks_elder.plot(column='People', legend=True)

        X = np.array([blocks_elder.geometry.representative_point().x.values,
                      blocks_elder.geometry.representative_point().y.values]).T
        kmeans = KMeans(n_clusters=6)
        c = kmeans.fit_predict(X)
        blocks_elder['cluster'] = c
        blocks_elder.plot(column='cluster')

        blocks_elder.groupby('cluster')['People'].sum()

        # Upper bound for buses
        blocks_elder.groupby('cluster')['People'].sum().sum() / 30

        pickups = gpd.GeoSeries([Point(x, y) for x, y in kmeans.cluster_centers_])
        ax = blocks_elder.plot(column='cluster')
        pickups.plot(color='red', ax=ax)
        yards.plot(color='yellow', ax=ax)
        shelters.plot(color='green', ax=ax)
        ax.set_axis_off()

        G = ox.graph_from_place('부산시, 대한민국', network_type='drive', simplify=False, truncate_by_edge=True)
        # G = ox.graph_from_polygon(study_area['geometry'][0], network_type='drive')
        # ox.plot_graph(G)
        ox.plot_graph(G, node_size=1, edge_color='k', edge_linewidth=1, figsize=(12, 12))

        nodes = list(shelters.geometry)
        # list(pickups.geometry) +  + list(yards.geometry)
        nodes0 = list(blocks_elder.geometry.representative_point())
        ox_nodes = [ox.get_nearest_node(G, (node.x, node.y)) for node in nodes]
        ox_nodes0 = [ox.get_nearest_node(G, (node.x, node.y)) for node in nodes0]

        # 하버공식을 활용한 단거리 노드
        # ox_nodes0 = [ox.get_nearest_node(G, (node.x, node.y), method='haversine') for node in nodes0]
        # ox_nodes = [ox.get_nearest_node(G, (node.x, node.y), method='haversine') for node in nodes]

        # 유클리디안을 활용한 단거리 노드
        # ox_nodes0 = [ox.get_nearest_node(G, (node.x, node.y), method='euclidean') for node in nodes0]
        # ox_nodes = [ox.get_nearest_node(G, (node.x, node.y), method='euclidean') for node in nodes]

        from itertools import product
        import matplotlib.pyplot as plt
        import seaborn as sns

        # distance matrix
        distances = np.zeros((len(ox_nodes), len(ox_nodes)))
        for i, j in product(range(len(ox_nodes)), range(len(ox_nodes))):
            distances[i, j] = nx.shortest_path_length(G, source=ox_nodes[i], target=ox_nodes[j], weight='length')
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(distances / 1000, square=True, annot=True, fmt='.2f', ax=ax, cbar=False, cmap='viridis')

        fig.savefig('figures/distances.png', tight_layout=True, dpi=400)

        distances0 = np.zeros_like(blocks_elder.People.values)
        for i, src in enumerate(ox_nodes0):
            distances0[i] = nx.shortest_path_length(G, source=src,
                                                    target=ox_nodes[1 + blocks_elder.cluster.astype(int).values[i]],
                                                    weight='length')

        distances0

        np.savetxt('Busan/demands.txt', blocks_elder.People.values, fmt='%d')
        np.savetxt('Busan/distances0.txt', distances0)
        np.savetxt('Busan/buses.txt', np.array([20]), fmt='%d')
        np.savetxt('Busan/capacities.txt', np.array([250, 250, 250]).T, fmt='%d')
        np.savetxt('Busan/nodes.txt', [[node.x, node.y] for node in nodes])
        np.savetxt('Busan/source_nodes.txt', [[node.x, node.y] for node in nodes0])
        np.savetxt('Busan/distances.txt', distances)
        np.savetxt('Busan/clusters.txt', blocks_elder.cluster.astype(int).values, fmt='%d')

        ox_nodes[:1]

        # Instance Figure
        node_color = []
        node_size = []
        for node in G.nodes:
            if node in ox_nodes[:1]:
                node_color.append('#eaff2b')
                node_size.append(20)
            elif node in ox_nodes[1:7]:
                node_color.append('#ff2b2b')
                node_size.append(20)
            elif node in ox_nodes[7:]:
                node_color.append('#2bff59')
                node_size.append(20)
            elif node in ox_nodes0:
                node_color.append('#66ccff')
                node_size.append(10)
            else:
                node_color.append('#66ccff')
                node_size.append(0)
        fig, ax = ox.plot_graph(G, node_color=node_color, node_size=node_size, dpi=1500, save=True, node_zorder=3,
                                node_alpha=0.9,
                                bgcolor='#333333', edge_color='#a8a8a8', edge_linewidth=0.3)
        ax.scatter([], [], s=20, color='#eaff2b', alpha=0.9, label='Yard')
        ax.scatter([], [], s=20, color='#ff2b2b', alpha=0.9, label='Pickup Point')
        ax.scatter([], [], s=10, color='#66ccff', alpha=0.9, label='Source Node')
        ax.scatter([], [], s=20, color='#2bff59', alpha=0.9, label='Shelter')

        L = fig.legend(frameon=False, loc=(0.65, 0.35))
        for text in L.get_texts():
            text.set_color("white")
            text.set_fontsize(4)
        fig.savefig('Busan/figure.png', dpi=1500, facecolor='#333333')

        # Clustered Blocks

        fig, ax = ox.plot_graph(G, node_color=node_color, node_size=0, dpi=100, save=True, node_zorder=3,
                                node_alpha=0.9,
                                bgcolor='#333333', edge_color='#a8a8a8', edge_linewidth=0.2)
        blocks_elder.plot(column='cluster', ax=ax)
        ax.set_axis_off()
        fig = ax.get_figure()
        fig.savefig('clustered_blocks.png', dpi=1000, facecolor='#333333')

        route = [0, 1, 3, 8]
        ox_route = [ox_nodes[i] for i in route]
        nx_route = [ox_route[0]]
        for i in range(len(ox_route[:-1])):
            nx_route += nx.shortest_path(G, source=ox_route[i], target=ox_route[i + 1])[1:]

        node_color = []
        node_size = []
        for node in G.nodes:
            if node in ox_nodes[:1]:
                node_color.append('#eaff2b')
                node_size.append(20)
            elif node in ox_nodes[1:7]:
                node_color.append('#ff2b2b')
                node_size.append(20)
            elif node in ox_nodes[7:]:
                node_color.append('#2bff59')
                node_size.append(20)
            elif node in ox_nodes0:
                node_color.append('#66ccff')
                node_size.append(0)
            else:
                node_color.append('#66ccff')
                node_size.append(0)
        fig, ax = ox.plot_graph_route(G, nx_route, node_color=node_color, node_size=node_size, dpi=100, save=True,
                                      node_zorder=3, node_alpha=0.9,
                                      bgcolor='#333333', edge_color='#a8a8a8', edge_linewidth=0.2,
                                      route_linewidth=1,
                                      route_color='#66ccff')
        fig.savefig('Busan/route0.png', dpi=1500, facecolor='#333333')

        x = []
        with open('Busan/sol_min-max.txt', 'r') as sol_file:
            lines = sol_file.readlines()
            for line in lines:
                if 'x_' in line:
                    x.append(tuple(line.split('=')[0].split('_')[1:]))
        x

        routes = {}
        for m in range(20):
            routes[m] = []
            for t in range(1, 7):
                for i in range(10):
                    for j in range(10):
                        if (str(i), str(j), str(m), str(t)) in x:
                            routes[m].append((i, j))
        for i in routes:
            routes[i] = [routes[i][0][0]] + [dest for src, dest in routes[i]]

            node_color = []
            node_size = []
            for node in G.nodes:
                if node in ox_nodes[:1]:
                    node_color.append('#eaff2b')
                    node_size.append(20)
                elif node in ox_nodes[1:7]:
                    node_color.append('#ff2b2b')
                    node_size.append(20)
                elif node in ox_nodes[7:]:
                    node_color.append('#2bff59')
                    node_size.append(20)
                elif node in ox_nodes0:
                    node_color.append('#66ccff')
                    node_size.append(0)
                else:
                    node_color.append('#66ccff')
                    node_size.append(0)

            nx_routes = []
            for route in routes:
                ox_route = [ox_nodes[i] for i in routes[route]]
                nx_route = [ox_route[0]]
                for i in range(len(ox_route[:-1])):
                    nx_route += nx.shortest_path(G, source=ox_route[i], target=ox_route[i + 1])[1:]

                fig, ax = ox.plot_graph_route(G, nx_route, node_color=node_color, node_size=node_size, dpi=1500,
                                              save=True,
                                              node_zorder=3, node_alpha=0.9,
                                              bgcolor='#333333', edge_color='#a8a8a8', edge_linewidth=0.2,
                                              route_linewidth=1, route_color='#66ccff',
                                              filename='route_{}'.format(route))
                fig.savefig('Busan/route0.png', dpi=1500, facecolor='#333333')

        nx_routes = []
        for route in routes:
            ox_route = [ox_nodes[i] for i in routes[route]]
            nx_route = [ox_route[0]]
            for i in range(len(ox_route[:-1])):
                nx_route += nx.shortest_path(G, source=ox_route[i], target=ox_route[i + 1])[1:]
            nx_routes.append(nx_route)

        fig, ax = ox.plot_graph_routes(G, nx_routes, node_color=node_color, node_size=node_size, dpi=1500,
                                       save=True,
                                       node_zorder=3, node_alpha=0.9,
                                       bgcolor='#333333', edge_color='#a8a8a8', edge_linewidth=0.2,
                                       route_linewidth=1,
                                       route_color='#66ccff', filename='route_{}'.format(route))

    def uPro06(self):
        import logging as log
        import os
        import sys
        import pandas as pd
        # import pvlib

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        contextPath = os.getcwd()

        log.info("[START] : %s", "Run SPA SZA")

        data = pd.read_csv(contextPath + '/../resources/data/csv/ECMWF_To_Csv_201907010000.csv', na_filter=False)

        sDateTime = "20190701000000"
        dtDateTime = pd.to_datetime(sDateTime, format='%Y%m%d%H%M%S')

        dataL1 = data

        for i in data.index:
            lat = data._get_value(i, 'lat')
            lon = data._get_value(i, 'lon')
            sp = data._get_value(i, 'sp')
            t2m = data._get_value(i, 't2m')

            solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=sp, temperature=t2m,
                                                               method='nrel_numpy')
            dataL1._set_value(i, 'spaSza', solPosInfo['zenith'].values)

        dataL1.to_csv(contextPath + '/../resources/data/csv/ECMWF_To_Csv_Output_201907010000.csv', sep=',',
                      na_rep='NaN')

        log.info("[END] : %s", "Run SPA SZA")

    def uPro07(self):
        import logging as log
        import os
        import sys
        import pandas as pd
        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        contextPath = os.getcwd()

        log.info("[START] : %s", "Run Program")

        # 가로축은 예시그래프와 같이 24시간으로 하고 싶은데
        # 세로축은 해당 시간 갯수로 하고 싶습니다!
        # 시간은 날짜나 분,초는 상관없이 시간으로만 하시면 될 것 같아요!
        # 그렇게해서 이 파일이랑 이 파일이랑 예시그림처럼 한번에 비교될 수 있게 나오도록 만들고 싶은데
        # 근데 보시면 첫번째 파일은 시간이 총 259개이고 두번째파일은 399개라서
        # 두번째 파일 399개중에서 완전 랜덤으로 259개를 뽑아서
        # 표로 만들 수 있을까요 ..?

        data1 = pd.read_excel(contextPath + '/../resources/data/xlsx/LIWC2015_Results_stress.xlsx')
        data2 = pd.read_excel(contextPath + '/../resources/data/xlsx/LIWC2015_Results_non_stress.xlsx')

        data1['dtDateTime'] = pd.to_datetime(data1['dateTime'], format='%Y-%m-%d %H:%M:%S')
        data2['dtDateTime'] = pd.to_datetime(data2['dateTime'], format='%Y-%m-%d %H:%M:%S')

        data1Stat = (
                data1 >>
                mutate(
                    hour=X.dtDateTime.dt.strftime("%H")
                ) >>
                group_by(X.hour) >>
                summarize(stress=n(X.hour))
        )

        data2Stat = (
                data2 >>
                sample(n=len(data1)) >>
                mutate(
                    hour=X.dtDateTime.dt.strftime("%H")
                ) >>
                group_by(X.hour) >>
                summarize(non_stress=n(X.hour))
        )

        dataL1 = (
                data1Stat >>
                left_join(data2Stat, by='hour') >>
                gather('key', 'val', ['stress', 'non_stress'])
        )

        dodge_text = position_dodge(width=0.9)

        plot = (ggplot(dataL1, aes(x='hour', y='val', fill='key'))
                + geom_col(stat='identity', position='dodge')
                + geom_text(aes(label='val'),  # new
                            position=dodge_text,
                            size=8, va='bottom', format_string='{}%')
                + lims(y=(0, 30))
                )

        # plot.save(contextPath + '/../resources/image/Image_07.png', width=10, height=10, dpi=600)
        plot.save(contextPath + '/../resources/image/Image_08.png', width=10, height=10, dpi=600)

        log.info("[END] : %s", "Run Program")

    def uPro06(self):
        import logging as log
        import os
        import sys
        import dateutil
        import pandas as pd
        import email
        import imaplib
        import configparser
        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        contextPath = os.getcwd()

        log.info("[START] : %s", "Run Program")

        # imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
        # 다운받은 파일을 db에 넣어서 사용할 예정이라
        # 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
        # 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
        # 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
        # 일정은 차주 수요일정두요
        # 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
        # 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
        # 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

        # email_address = raw_input("Enter email address: ") if not LOGIN_USERNAME else LOGIN_USERNAME='sangho.lee.1990@gmail.com'
        # email_password = getpass("Enter email password: ") if not LOGIN_PASSWORD else LOGIN_PASSWORD

        # Email 설정정보 불러오기
        config = configparser.ConfigParser()
        config['Gmail'] = {}
        config['Gmail']['ID'] = '아이디'
        config['Gmail']['Password'] = '비밀번호'

        # 시작/종료 시간 설정
        startDate = "20190801"
        endDate = "20200801"

        # saveFile = contextPath + '/../resources/data/csv/Gmail_Info_{0}_{1}.csv'.format(startDate, endDate)
        saveFile = contextPath + '/../resources/data/xlsx/Gmail_Info_{0}_{1}.xlsx'.format(startDate, endDate)

        # gmail imap 세션 생성
        session = imaplib.IMAP4_SSL('imap.gmail.com', 993)

        # 로그인
        session.login(config['Gmail']['ID'], config['Gmail']['Password'])

        # 받은편지함
        session.select('Inbox')

        # 받은 편지함 내 모든 메일 검색
        dtStartDate = pd.to_datetime(startDate, format='%Y%m%d').strftime("%d-%b-%Y")
        dtEndDate = pd.to_datetime(endDate, format='%Y%m%d').strftime("%d-%b-%Y")

        # 특정 날짜 검색
        searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(dtStartDate, dtEndDate)

        #  전체 검색
        # searchOpt = 'ALL'

        log.info("[Check] searchOpt : %s", searchOpt)
        result, data = session.search(None, searchOpt)

        if (result != 'OK'):
            log.error("[Check] 조회 실패하였습니다.")
            exit(1)

        if (len(data) <= 0):
            log.error("[Check] 조회 목록이 없습니다.")
            exit(1)

        log.info("[Check] result : %s", result)

        # 메일 읽기
        emailList = data[0].split()

        log.info("[Check] emailList : %s", len(emailList))

        # 최근 메일 읽기
        # all_email.reverse()

        dataL1 = pd.DataFrame()

        # for i in range(1, len(emailList)):
        for i in range(1, 100):

            mail = emailList[i]
            msgFrom = ''
            msgSender = ''
            msgTo = ''
            msgDate = ''
            subject = ''
            message = ''
            msgDateFmt = ''
            title = ""

            log.info("[Check] mail : %s", mail)

            try:
                result, data = session.fetch(mail, '(RFC822)')
                raw_email = data[0][1]
                raw_email_string = raw_email.decode('utf-8')
                msg = email.message_from_string(raw_email_string)

                # 메일 정보
                msgFrom = msg.get('From')
                msgSender = msg.get('Sender')
                msgTo = msg.get('To')
                msgDate = msg.get('Date')
                msgDateFmt = dateutil.parser.parse(msg.get('Date')).strftime("%Y-%m-%d %H:%M:%S")
                subject = email.header.decode_header(msg.get('Subject'))
                msgEncoding = subject[0][1]
                title = subject[0][0].decode(msgEncoding)

                # for sub in subject:
                #     if isinstance(sub[0], bytes):
                #         title += sub[0].decode(msgEncoding)
                #     else:
                #         title += sub[0]

                if msg.is_multipart():
                    for part in msg.get_payload():
                        if part.get_content_type() == 'text/plain':
                            bytes = part.get_payload(decode=True)
                            encode = part.get_content_charset()
                            message = message + str(bytes, encode)

                            print, bytes, encode, message
                else:
                    if msg.get_content_type() == 'text/plain':
                        bytes = msg.get_payload(decode=True)
                        encode = msg.get_content_charset()
                        message = str(bytes, encode)

                # 딕션너리 정의
                dataInfo = {
                    'MailID': [mail]
                    , 'From': [msgFrom]
                    , 'Sender': [msgSender]
                    , 'To': [msgTo]
                    , 'Date': [msgDate]
                    , 'DateFmt': [msgDateFmt]
                    , 'Title': [title]
                    , 'Message': [message]
                }

                data = pd.DataFrame(dataInfo)
                dataL1 = data >> bind_rows(dataL1)

            except Exception as e:
                print("Exception : ", e)

        session.close()
        session.logout()

        # dataL1.to_csv(saveFile, sep=',', na_rep='NaN', index=False)
        dataL1.to_excel(saveFile)

        log.info("[END] : %s", "Run Program")

    def uPro07(self):
        import logging as log
        import os
        import sys
        import datetime
        import pandas as pd
        import numpy as np
        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *
        from scipy.stats import linregress
        import hydroeval
        from mizani.breaks import date_breaks
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        def makeScatterPlot(valY, PredValY, savefigName):
            X = valY
            Y = PredValY

            plt.scatter(X, Y, alpha=0.3, edgecolors='none')

            # arrVal = np.array([X, Y])
            # setMax = np.max(arrVal)
            setMin = 50
            setMax = 4000
            interval = 200

            plt.title("")
            plt.xlabel('Observed (EL.m)')
            plt.ylabel('Simulated (EL.m)')
            plt.xlim(0, setMax)
            plt.ylim(0, setMax)
            plt.grid(False)

            # Bias (relative Bias), RMSE (relative RMSE), R, slope, intercept, pvalue
            Bias = np.mean(X - Y)
            rBias = (Bias / np.mean(Y)) * 100.0
            RMSE = np.sqrt(np.mean((X - Y) ** 2))
            rRMSE = (RMSE / np.mean(Y)) * 100.0
            MAPE = np.mean(np.abs((X - Y) / X)) * 100.0

            slope = linregress(X, Y)[0]
            intercept = linregress(X, Y)[1]
            R = linregress(X, Y)[2]
            Pvalue = linregress(X, Y)[3]
            N = len(X)

            tmpX = np.linspace(0, setMax)

            lmfit = (slope * tmpX) + intercept
            plt.plot(tmpX, lmfit, color='red', linewidth=2, linestyle=':')
            plt.plot([0, setMax], [0, setMax], color='black')

            plt.annotate('Sim = %.2f x (Obs) + %.2f' % (slope, intercept), xy=(setMin, setMax - interval), color='red',
                         fontweight='bold',
                         xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2), color='red',
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

            rect = plt.Rectangle((25, 2700), 2000, 1250, linewidth=1, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)

            plt.savefig(savefigName, dpi=600, bbox_inches='tight')

            plt.show()

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
        # warnings.filterwarnings("ignore")

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        # 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False

        # 설치된 글꼴 목록
        # fontList = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Pub' in f.name]
        # path = 'C:\\Users\\jmpark\\AppData\\Local\\Microsoft\\Windows\\Fonts\\KoPubWorld Dotum Medium.ttf'
        # fontprop = fm.FontProperties(fname=path)

        contextPath = os.getcwd()

        log.info("[START] : %s", "Run Program")

        # 안녕하세요 문의드리려고하는데요 파이썬으로 데이터 시각화를 하려고 하는데 가능한지 여쭤보려합니다
        # 파일보내드립니다 파이썬 공부하면서 하려니 잘 안되서 도움을 받아야할꺼 같아서 연락드렸어요
        # 일별 자료이고 그림은 2가지이고 통계값 계산하는거 한가지 이렇게 됩니다
        # 그림1같이 모의값-관측값 비교하고 위에 강우량 그려넣으려고 하구요
        # 그림2같이 모의,관측값 1:1 선 그러넣구 중쳡하려구합니다
        # 3번은 관측값 모의값 RMSE하고 NSE 구하려고해요, 일별자료로하는거하고, 월별 평균값 만들어하는거 하구요

        # saveFile = contextPath + '/../resources/data/xlsx/Gmail_Info_{0}_{1}.xlsx'.format(startDate, endDate)

        # 강우, 용수량, 관측값, 모의값 (val1, val2, val3, val4)
        data = pd.read_excel(contextPath + '/../resources/data/xlsx/데이터샘플.xlsx')

        data['dtDateTime'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

        dataL1 = (
                data >>
                mutate(
                    year=X.dtDateTime.dt.strftime("%Y")
                    , month=X.dtDateTime.dt.strftime("%m")
                    , day=X.dtDateTime.dt.strftime("%d")
                )
        )

        # ===========================================
        # 그림 1. 데이터 그림
        # ===========================================
        startDateTime = '1994-01-01 00:00:00'
        centerDateTime = '2002-01-01 00:00:00'
        endDateTime = '2011-01-01 00:00:00'
        dtStartDateTime = datetime.datetime.strptime(startDateTime, '%Y-%m-%d %H:%M:%S')
        dtCenterDateTime = datetime.datetime.strptime(centerDateTime, '%Y-%m-%d %H:%M:%S')
        dtEndDateTime = datetime.datetime.strptime(endDateTime, '%Y-%m-%d %H:%M:%S')

        # 이중 축으로 시각화
        fig, ax1 = plt.subplots()

        line1 = ax1.plot(data['dtDateTime'], data['val2'], 'r-', label='용수량')
        line3 = ax1.plot(data['dtDateTime'], data['val3'], 'g-', label='관측값')
        line4 = ax1.plot(data['dtDateTime'], data['val4'], 'y-', label='모의값')

        ax1.set_ylabel('')
        ax1.set_xlim(dtStartDateTime, dtEndDateTime)
        ax1.set_ylim(0, 7000)

        ax2 = ax1.twinx()
        line2 = ax2.plot(data['dtDateTime'], data['val1'], 'b-', label='강우')
        ax2.set_ylabel('')
        ax2.set_ylim(0, 700)
        ax2.invert_yaxis()

        # ax1.hlines(4000, dtStartDateTime, dtEndDateTime, color='gray')
        ax1.vlines(dtCenterDateTime, 0, 7000, color='gray')

        ax1.annotate('', xy=(dtStartDateTime, 4000), xycoords='data', xytext=(dtCenterDateTime, 4000),
                     textcoords='data',
                     arrowprops=dict(facecolor='gray', arrowstyle='<->'))
        ax1.annotate('', xy=(dtCenterDateTime, 4000), xycoords='data', xytext=(dtEndDateTime, 4000), textcoords='data',
                     arrowprops=dict(facecolor='gray', arrowstyle='<->'))

        ax1.annotate('Calibration', xy=(datetime.datetime.strptime('1998-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 4250),
                     xycoords='data', textcoords='data', color='black', horizontalalignment='center',
                     verticalalignment='center')
        ax1.annotate('Validation', xy=(datetime.datetime.strptime('2007-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 4250),
                     xycoords='data', textcoords='data', color='black', horizontalalignment='center',
                     verticalalignment='center')

        line = line1 + line2 + line3 + line4
        label = [l.get_label() for l in line]
        ax1.legend(line, label, loc="center", mode="expand", ncol=4)

        # ax1.grid(True)
        plt.savefig(contextPath + '/../resources/image/Image_11.png', width=10, height=6, dpi=600, bbox_inches='tight')
        plt.show()

        # 단일 축으로 시각화 (1)
        plotDataL1 = (
                dataL1 >>
                gather('key', 'val', ['val1', 'val2'])
        )

        plot = (ggplot(plotDataL1, aes(x='dtDateTime', y='val', color='key'))
                + geom_line()
                + scale_x_datetime(expand=(0, 0), limits=(startDateTime, endDateTime), breaks=date_breaks('1 years'))
                + scale_y_continuous(expand=(0, 0), limits=(0, 300), breaks=range(0, 400, 50))
                + theme(axis_text_x=element_text(angle=45, hjust=1)
                        , text=element_text(family="Malgun Gothic")
                        )
                + scale_color_discrete(name='Color', labels=("강우", "용수량"))
                + labs(x='Date [Year-Month-Day]', y='')
                )
        plot.save(contextPath + '/../resources/image/Image_08.png', bbox_inches='tight', width=10, height=6, dpi=600)

        # 단일 축으로 시각화 (2)
        plotDataL2 = (
                dataL1 >>
                gather('key', 'val', ['val3', 'val4'])
        )

        plot = (ggplot(plotDataL2, aes(x='dtDateTime', y='val', color='key'))
                + geom_line()
                + scale_x_datetime(expand=(0, 0), limits=(startDateTime, endDateTime), breaks=date_breaks('1 years'))
                + scale_y_continuous(expand=(0, 0), limits=(0, 3500), breaks=range(0, 4000, 500))
                + theme(axis_text_x=element_text(angle=45, hjust=1)
                        , text=element_text(family="Malgun Gothic")
                        )
                + scale_color_discrete(name='Color', labels=("관측값", "모의값"))
                + labs(x='Date [Year-Month-Day]', y='')
                )
        plot.save(contextPath + '/../resources/image/Image_09.png', bbox_inches='tight', width=10, height=6, dpi=600)

        # ===========================================
        # 그림 2. 데이터 그림
        # ===========================================
        xAxis = dataL1['val3'].values
        yAxis = dataL1['val4'].values
        savefigName = contextPath + '/../resources/image/Image_10.png'

        makeScatterPlot(xAxis, yAxis, savefigName)

        # =============================================================================
        # 그림 3. the root mean square error (RMSE), Nash–Sutcliffe efficiency (NSE)
        # =============================================================================
        sDate = '2005-05-04'
        dtDate = pd.to_datetime(sDate, format='%Y-%m-%d')

        tmpDataL1 = (
                dataL1 >>
                mutate(isFlag=if_else(X.dtDateTime > dtDate, True, False))
        )

        flagList = [True, False]

        for flag in flagList:
            # 일별 데이터
            dataL2 = (
                    tmpDataL1 >>
                    mask(X.isFlag == flag) >>
                    group_by(X.year, X.month, X.day) >>
                    summarize(
                        meanVal3=mean(X.val3)
                        , meanVal4=mean(X.val4)
                    )
            )

            # 월별 자료
            dataL3 = (
                    tmpDataL1 >>
                    mask(X.isFlag == flag) >>
                    group_by(X.year, X.month) >>
                    summarize(
                        meanVal3=mean(X.val3)
                        , meanVal4=mean(X.val4)
                    )
            )

            xAxis = dataL2['meanVal3'].values
            yAxis = dataL2['meanVal4'].values

            log.info("[Check] sDate : %s", sDate)
            log.info("[Check] isFlag : %s", flag)
            log.info("[Check] Number : %s", len(xAxis))

            log.info("[Check] Daily RMSE : %s", round(hydroeval.rmse(xAxis, yAxis), 2))
            log.info("[Check] Daily NSE : %s", round(hydroeval.nse(xAxis, yAxis), 2))

            xAxis = dataL3['meanVal3'].values
            yAxis = dataL3['meanVal4'].values

            log.info("[Check] Monthly RMSE : %s", round(hydroeval.rmse(xAxis, yAxis), 2))
            log.info("[Check] Monthly NSE : %s", round(hydroeval.nse(xAxis, yAxis), 2))

        log.info("[END] : %s", "Run Program")

    def uPro08(self):
        import logging as log
        import os
        import sys
        import dateutil
        import pandas as pd
        import email
        import imaplib
        import configparser
        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

        contextPath = os.getcwd()

        log.info("[START] : %s", "Run Program")

        # imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
        # 다운받은 파일을 db에 넣어서 사용할 예정이라
        # 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
        # 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
        # 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
        # 일정은 차주 수요일정두요
        # 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
        # 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
        # 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

        # email_address = raw_input("Enter email address: ") if not LOGIN_USERNAME else LOGIN_USERNAME='sangho.lee.1990@gmail.com'
        # email_password = getpass("Enter email password: ") if not LOGIN_PASSWORD else LOGIN_PASSWORD

        # Email 설정정보 불러오기
        config = configparser.ConfigParser()
        config['Gmail'] = {}
        config['Gmail']['ID'] = 'sangho.lee.1990@gmail.com'
        config['Gmail']['Password'] = 'cjswo123!Q'

        # config['Gmail']['ID'] = '아이디'
        # config['Gmail']['Password'] = '비밀번호'

        # 시작/종료 시간 설정
        startDate = "20190801"
        endDate = "20200801"

        # saveFile = contextPath + '/../resources/data/csv/Gmail_Info_{0}_{1}.csv'.format(startDate, endDate)
        saveFile = contextPath + '/../resources/data/xlsx/Gmail_Info_{0}_{1}.xlsx'.format(startDate, endDate)

        # gmail imap 세션 생성
        session = imaplib.IMAP4_SSL('imap.gmail.com', 993)

        # 로그인
        session.login(config['Gmail']['ID'], config['Gmail']['Password'])

        # 받은편지함
        session.select('Inbox')

        # 받은 편지함 내 모든 메일 검색
        dtStartDate = pd.to_datetime(startDate, format='%Y%m%d').strftime("%d-%b-%Y")
        dtEndDate = pd.to_datetime(endDate, format='%Y%m%d').strftime("%d-%b-%Y")

        # 특정 날짜 검색
        searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(dtStartDate, dtEndDate)

        #  전체 검색
        # searchOpt = 'ALL'

        log.info("[Check] searchOpt : %s", searchOpt)
        result, data = session.search(None, searchOpt)

        if (result != 'OK'):
            log.error("[Check] 조회 실패하였습니다.")
            exit(1)

        if (len(data) <= 0):
            log.error("[Check] 조회 목록이 없습니다.")
            exit(1)

        log.info("[Check] result : %s", result)

        # 메일 읽기
        emailList = data[0].split()

        log.info("[Check] emailList : %s", len(emailList))

        # 최근 메일 읽기
        # all_email.reverse()

        dataL1 = pd.DataFrame()

        # for i in range(1, len(emailList)):
        for i in range(1, 100):

            mail = emailList[i]
            msgFrom = ''
            msgSender = ''
            msgTo = ''
            msgDate = ''
            subject = ''
            message = ''
            msgDateFmt = ''
            title = ""

            log.info("[Check] mail : %s", mail)

            try:
                result, data = session.fetch(mail, '(RFC822)')
                raw_email = data[0][1]
                raw_email_string = raw_email.decode('utf-8')
                msg = email.message_from_string(raw_email_string)

                # 메일 정보
                msgFrom = msg.get('From')
                msgSender = msg.get('Sender')
                msgTo = msg.get('To')
                msgDate = msg.get('Date')
                msgDateFmt = dateutil.parser.parse(msg.get('Date')).strftime("%Y-%m-%d %H:%M:%S")
                subject = email.header.decode_header(msg.get('Subject'))
                msgEncoding = subject[0][1]
                title = subject[0][0].decode(msgEncoding)

                # for sub in subject:
                #     if isinstance(sub[0], bytes):
                #         title += sub[0].decode(msgEncoding)
                #     else:
                #         title += sub[0]

                if msg.is_multipart():
                    for part in msg.get_payload():
                        if part.get_content_type() == 'text/plain':
                            bytes = part.get_payload(decode=True)
                            encode = part.get_content_charset()
                            message = message + str(bytes, encode)
                            # print, bytes, encode, message
                else:
                    if msg.get_content_type() == 'text/plain':
                        bytes = msg.get_payload(decode=True)
                        encode = msg.get_content_charset()
                        message = str(bytes, encode)

                # 딕션너리 정의
                dataInfo = {
                    'MailID': [mail]
                    , 'From': [msgFrom]
                    , 'Sender': [msgSender]
                    , 'To': [msgTo]
                    , 'Date': [msgDate]
                    , 'DateFmt': [msgDateFmt]
                    , 'Title': [title]
                    , 'Message': [message]
                }

                data = pd.DataFrame(dataInfo)
                # dataL1 = data >> bind_rows(dataL1)
                dataL1 = dataL1.append(data, ignore_index=True)

            except Exception as e:
                print("Exception : ", e)

        session.close()
        session.logout()

        # dataL1.to_csv(saveFile, sep=',', na_rep='NaN', index=False)
        dataL1.to_excel(saveFile)

        log.info("[END] : %s", "Run Program")

    def uPro09(self):
        # 회귀분석을 파이썬으로 작성해서 제출합니다. 파이썬 파일만 제가 열어보았을 때 무엇을 어떻게 했는지 알 수 있도록 설명을 더해서 파이썬 파일을 제출합니다.
        # 포함되어야 할 사항은
        # 회귀분석을 패키지의 명령어를 이용해서 분석한다.
        # 패키지의 정해진 명령어를 사용하지 않고 최소해를 직접 찾아나가도록 경사하강법 등의 방법을 이용해서 해를 구한 후 1과 같아 지는지 확인한다.
        # 네이버 등등 많은 곳에 이미 나와 있는 내용들입니다. 파이썬에 대한 숙련도가 각기 다르기 때문에 제가 한번 보고자 함이니 각자의 위치에서 최선을 다해 공부하셔서 제출해주시기 바랍니다.
        # 과제를 통해 느낀 점을 솔직하게 적어주시기 바랍니다.
        # 회귀분석에 대해서 전혀 몰랐다 등등
        # 과제를 통해 제 수업 방향을 수정해나갈 수 있도록 적극적으로 함께 해주시기 바랍니다.
        # 제출기한: 9월 10일 오후 4시까지

        # 라이브러리 읽기
        import logging as log
        import os
        import sys
        import numpy as np
        import random
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
        # warnings.filterwarnings("ignore")

        # 프로그램 시작
        log.info("[START] : %s", "Run Program")

        # 작업환경 경로 설정
        contextPath = os.getcwd()
        log.info("[Check] contextPath : %s", contextPath)

        # ========================================
        # 기본값 설정 (독립변수, 종속변수)
        # ========================================
        # 독립변수 설정
        x = [i for i in range(50)]
        log.info("[Check] x : %s", x)

        # 종속변수 설정
        y = [2 * j + random.normalvariate(0, 10) for j in range(50)]
        log.info("[Check] y : %s", y)

        # 이미지 저장 파일명 설정
        savefigName = contextPath + '/../resources/image/Image_12.png'

        # 산점도에 따른 그림 시각화
        plt.scatter(x, y)
        plt.xlabel('input variable X')
        plt.ylabel('output variable Y')
        plt.title('artificial dataset')
        plt.savefig(savefigName, width=1000, heiht=1000, dpi=600, bbox_inches='tight')
        plt.show()

        # ==============================================
        # 1. 함수를 통해 선형회귀모형
        # ==============================================
        # 평균 함수 정의
        def mean(values):
            return sum(values) / len(values)

        # 선형회귀모형에서 절편 및 기울기 함수 정의
        def beta(x, y):
            # 회귀계수 계산 (beta_1)
            covariance = 0
            variance_x = 0
            num_points = len(x)
            for i in range(num_points):
                covariance += (x[i] - mean(x)) * (y[i] - mean(y))
                variance_x += pow(x[i] - mean(x), 2)

            beta_1 = covariance / variance_x

            # 절편 계산 (beta_0)
            beta_0 = mean(y) - beta_1 * mean(x)

            return [beta_0, beta_1]

        # 선형회귀모형에서 절편 및 기울기 함수 호출
        beta_v1 = beta(x, y)

        log.info("[Check] model coef using function : %s", beta_v1[1])
        log.info("[Check] model intercept using function : %s", beta_v1[0])

        # ==============================================
        # 2. numpy 라이브러리를 통해 선형회귀모형
        # ==============================================
        np.cov(x, y, ddof=0)[0, 1]

        # 회귀계수 계산 (beta_1)
        beta_1 = np.cov(x, y, ddof=0)[0, 1] / np.var(x)

        # 절편 계산 (beta_0)
        beta_0 = np.mean(y) - beta_1 * np.mean(x)

        beta_v2 = [beta_0, beta_1]

        log.info("[Check] model coef using numpy library : %s", beta_1)
        log.info("[Check] model intercept using numpy library : %s", beta_0)

        # ==============================================
        # 3. scikit-learn 라이브러리를 통해 선형회귀모형
        # ==============================================
        trainX = np.array(x)
        trainY = np.array(y)

        # 선형회귀모형 선언
        model = LinearRegression()

        # 선형회귀모형에 대한 학습 (독립변수: trainX, 종속변수: trainY)
        model.fit(trainX.reshape(-1, 1), trainY)

        # 회귀계수 계산 (model.coef_)
        log.info("[Check] model coef using scikit-learn library : %s", model.coef_)

        # 절편 계산 (model.intercept_)
        log.info("[Check] model intercept using scikit-learn library : %s", model.intercept_)

        # 선형회귀모형에 대한 예측 (독립변수: trainX)
        predY = model.predict(trainX.reshape(-1, 1))
        log.info("[Check] x : %s", predY)

        # 이미지 저장 파일명 설정
        savefigName = contextPath + '/../resources/image/Image_13.png'

        # 산점도에 따른 그림 시각화
        plt.scatter(trainX, trainY, color='black', alpha=0.3)
        plt.xlabel('input variable X')
        plt.ylabel('output variable Y')
        plt.title('artificial dataset')
        plt.plot(trainX, predY, color='red', linewidth=3, linestyle='--')
        plt.savefig(savefigName, width=1000, heiht=1000, dpi=600, bbox_inches='tight')
        plt.show()

        # 프로그램 종료
        log.info("[END] : %s", "Run Program")

    def uPro10(self):
        # 라이브러리 읽기
        import logging as log
        import os
        import sys

        import pandas as pd

        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] %(message)s")
        # warnings.filterwarnings("ignore")

        # 작업환경 경로 설정
        contextPath = os.getcwd()
        log.info('[Check] contextPath : {}'.format(contextPath))

        xlsxContextPath = contextPath + '/../resources/data/xlsx/'
        log.info('[Check] xlsxContextPath : {}'.format(xlsxContextPath))

        imgContextPath = contextPath + '/../resources/image/'
        log.info('[Check] imgContextPath : {}'.format(imgContextPath))

        log.info('[START] main : {}'.format('Run Program'))

        # ==============================================
        # 주 소스코드
        # ==============================================
        # 안녕하세요~ 저 저번에 해솔님께 파이썬을 활용한 막대그래프 작업 요청했었는데 똑같은거 그대로 또 요청하려구요!
        # 대신 저번보다 데이터가 많아요! 이번에도 두 파일의 갯수 차가 나는데 제가 보낸 파일 보시면 첫번째 파일은 총 181074개이고 두번째 파일은 총 45410개입니다!
        # 그래서 저번처럼 첫번째 파일에서 랜덤으로 45410개를 뽑아서 두번째 파일과의 시간을 비교할 수 있을까요?
        # 우선 수요일 자정 전까지 작업이 가능한지 여쭙고 싶습니다 ㅠㅠ..! 그리고 금액대도 알려주시면 감사하겠습니다!

        data1 = pd.read_excel(xlsxContextPath + 'Jan_stress.xlsx')
        data2 = pd.read_excel(xlsxContextPath + 'Jan_nonstress.xlsx')

        data1['dtDateTime'] = pd.to_datetime(data1['dateTime'], format='%Y-%m-%d %H:%M:%S')
        data2['dtDateTime'] = pd.to_datetime(data2['dateTime'], format='%Y-%m-%d %H:%M:%S')

        data1Stat = (
                data1 >>
                mutate(
                    hour=X.dtDateTime.dt.strftime("%H")
                ) >>
                group_by(X.hour) >>
                summarize(stress=n(X.hour))
        )

        data2Stat = (
                data2 >>
                sample(n=len(data1)) >>
                mutate(
                    hour=X.dtDateTime.dt.strftime("%H")
                ) >>
                group_by(X.hour) >>
                summarize(non_stress=n(X.hour))
        )

        dataL1 = (
                data1Stat >>
                left_join(data2Stat, by='hour') >>
                gather('key', 'val', ['stress', 'non_stress']) >>
                mutate(perVal=(X.val / len(data1)) * 100)
        )

        dodge_text = position_dodge(width=0.9)

        # 빈도분포 (개수)
        plot = (ggplot(dataL1, aes(x='hour', y='val', fill='key'))
                + geom_col(stat='identity', position='dodge')
                + geom_text(aes(label='val'),  # new
                            position=dodge_text,
                            size=8, va='bottom', format_string='{}')
                + lims(y=(0, 3000))
                )

        plot.save(imgContextPath + 'Image_14.png', width=12, height=10, dpi=600)

        # 빈도분포 (% 퍼센트)
        plot = (ggplot(dataL1, aes(x='hour', y='perVal', fill='key'))
                + geom_col(stat='identity', position='dodge')
                + geom_text(aes(label='perVal'),  # new
                            position=dodge_text,
                            size=8, va='bottom', format_string="{:.1f} %")
                + lims(y=(0, 8))
                )

        plot.save(imgContextPath + 'Image_15.png', width=12, height=10, dpi=600)

        # 프로그램 종료
        log.info('[END] main : {}'.format('Run Program'))

    def uPro11(self):
        # 라이브러리 읽기
        import logging as log
        import os
        import sys
        import datetime
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        import warnings
        import seaborn as sns
        from scipy import stats
        import traceback
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import KFold, GridSearchCV
        from sklearn import metrics
        from datetime import datetime
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from warnings import simplefilter

        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(filename)s > %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
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
        def makeScatterPlot(PredValY, valY, savefigName):

            X = PredValY
            Y = valY

            plt.scatter(X, Y)

            # arrVal = np.array([X, Y])
            # setMin = np.min(arrVal)
            # setMax = np.max(arrVal)
            # interval = (setMax - setMin) / 20

            setMin = 200
            setMax = 20000
            interval = 1000

            plt.title("")
            plt.xlabel('Val')
            plt.ylabel('Pred')
            plt.xlim(0, setMax)
            plt.ylim(0, setMax)
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
            log.info('[START] main : {}'.format('Run Program'))

            # 작업환경 경로 설정
            contextPath = os.getcwd()

            # 전역 변수
            globalVar = {
                "config": {
                    "imgContextPath": contextPath + '/../resources/image/'
                    , "csvConfigPath": contextPath + '/../resources/data/csv/'
                    , "xlsxConfigPath": contextPath + '/../resources/data/xlsx/'
                }
            }

            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스코드
            # ==============================================
            # 파일 읽기
            diamonds = pd.read_csv(globalVar.get('config').get('csvConfigPath') + 'diamonds.csv')

            # 파일 정보 읽기
            diamonds.head()

            # =========================================
            # 1. 탐색
            # =========================================
            # Unnamed에 대한 컬럼 삭제
            diamonds = diamonds.drop(['Unnamed: 0'], axis=1)
            diamonds.head()

            # 해당 컬럼 (cut, color, clarity)에 대한 타입 확인
            print(diamonds.cut.unique())
            print(diamonds.clarity.unique())
            print(diamonds.color.unique())

            # =========================================
            # 2. 데이터 정제 (Missing value 등의 처리)
            # =========================================
            # 각 열에 대한 연속적 변수로 변환
            categorical_features = ['cut', 'color', 'clarity']
            le = LabelEncoder()
            # Convert the variables to numerical
            for i in range(3):
                new = le.fit_transform(diamonds[categorical_features[i]])
                diamonds[categorical_features[i]] = new
            diamonds.head()

            # 각 자료에 대한 빈도분포
            diamonds.hist()
            plt.show()

            # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
            diamonds.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
            plt.show()

            # 상관분석 행렬 시각화 및 자료 저장
            dataCorr = diamonds.corr(method='pearson')
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_16.png'
            makeCorrPlot(dataCorr, savefigName)

            # 독립변수 (x, y, z, carat) 간의 상관성이 높기 때문에 다중공선성 발생

            # =========================================
            # 3. 목적에 맞는 분석
            # =========================================
            # Create features and target matrixes
            # 전체 독립변수 설정
            # X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
            # 상관분석을 통해 유의미한 변수 선택
            X = diamonds[['carat', 'x', 'y', 'z', 'color', 'clarity']]

            # 종속변수 설정
            Y = diamonds[['price']]

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

            # 4종 모델에 대한 정보
            log.info("[Check] models | %s" % (models))

            # 4종 모형에 대한 파라메타
            log.info("[Check] params | %s" % (params))

            # ========================================================
            # 트레이닝 및 테스트 셋 구분없이 전체 데이터셋으로 학습
            # ========================================================
            # 교차 검증 수행
            kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

            # 전체 데이터셋을 이용한 학습
            results = []
            for i in range(len(models)):
                model = models[i]
                param = params[i]

                # 모델 및 파라미터에 따라 교차검증 수행
                result = gridsearch_cv_for_regression(model=model, param=param, kfold=kfold
                                                      , train_input=X, train_target=Y)

                # 교차검증 결과 중에서 최적 하이퍼 파라미터를 찾기
                result.best_score_

                # 각 모형에 대한 결과값을 저장
                results.append(result)

            # 전체 데이터셋을 이용한 검증
            for i in range(len(results)):

                # 예측 결과
                Y_hat = results[i].predict(X)

                # 산점도를 위한 이미지 파일명 설정
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 17)

                # 각 모형에 따라 입력 파일 변환
                if i >= 2:
                    makeScatterPlot(Y_hat, Y.values[:, 0], savefigName)
                else:
                    makeScatterPlot(Y_hat[:, 0], Y.values[:, 0], savefigName)

                # 검증 수치 (RMSE, R2) 계산
                rmse = np.sqrt(metrics.mean_squared_error(Y_hat, Y))
                r2 = metrics.r2_score(Y_hat, Y)
                log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (models[i][0], rmse, r2))

            # ====================================================
            # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
            # ====================================================
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 20)

                if i >= 2:
                    makeScatterPlot(Y_test_hat, Y_test.values[:, 0], savefigName)
                else:
                    makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], savefigName)

                rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
                r2 = metrics.r2_score(Y_test, Y_test_hat)
                log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (models[i][0], rmse, r2))

            # ========================================================
            # [표준회 O] 4종 회귀모형을 이용
            # ========================================================
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 24)

                if i >= 2:
                    makeScatterPlot(Y_test_hat, Y_test.values[:, 0], savefigName)
                else:
                    makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], savefigName)

                rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
                r2 = metrics.r2_score(Y_test, Y_test_hat)
                log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (models[i][0], rmse, r2))

            # ========================================================
            # [표준화 Normalization] 4종 회귀모형을 이용
            # ========================================================
            # 초기값 설정
            models = []
            params = []

            # 선형회귀모형 및 파라미터 설정
            model = (
                'Normal Linear',
                Pipeline([('Normal', MinMaxScaler(feature_range=(0, 1))), ('Linear', linear_model.LinearRegression())]))
            param = {}

            models.append(model)
            params.append(param)

            # 릿지 회귀모형 및 파라미터 (알파 조정) 설정
            model = (
                'Normal Ridge',
                Pipeline([('Normal', MinMaxScaler(feature_range=(0, 1))), ('Ridge', linear_model.Ridge())]))
            param = {
                'Ridge__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
            }
            param = {}

            models.append(model)
            params.append(param)

            # 라쏘에 대해서 및 파라미터 (알파 조정) 설정
            model = (
                'Normal Lasso',
                Pipeline([('Normal', MinMaxScaler(feature_range=(0, 1))), ('Lasso', linear_model.Lasso())]))
            param = {
                'Lasso__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
            }
            param = {}

            models.append(model)
            params.append(param)

            # 엘라스틴에 대해서 및 파라미터 (알파 조정) 설정
            model = (
                'Normal ElasticNet',
                Pipeline([('Normal', MinMaxScaler(feature_range=(0, 1))), ('ElasticNet', linear_model.ElasticNet())]))
            param = {
                'ElasticNet__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
                'ElasticNet__l1_ratio': [0.3, 0.5, 0.7]
            }
            param = {}

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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 28)

                if i >= 2:
                    makeScatterPlot(Y_test_hat, Y_test.values[:, 0], savefigName)
                else:
                    makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], savefigName)

                rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
                r2 = metrics.r2_score(Y_test, Y_test_hat)
                log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (models[i][0], rmse, r2))

        except Exception as e:
            log.error("Exception : {}".format(e))
            traceback.print_exc()
            # sys.exit(1)

        finally:
            log.info('[END] main : {}'.format('Run Program'))

    def uPro13(self):
        # imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
        # 다운받은 파일을 db에 넣어서 사용할 예정이라
        # 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
        # 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
        # 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
        # 일정은 차주 수요일정두요
        # 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
        # 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
        # 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

        # 라이브러리 읽기
        import logging as log
        import os
        import sys
        import dateutil
        import pandas as pd
        import email
        import imaplib
        import configparser
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import traceback
        from email.header import decode_header
        from datetime import datetime

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
        # warnings.filterwarnings("ignore")
        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        # 그래프에서 글꼴 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False

        try:
            log.info('[START] main : {}'.format('Run Program'))

            # 작업환경 경로 설정
            contextPath = os.getcwd()

            # 전역 변수
            globalVar = {
                "contextPath": {
                    "img": contextPath + '/../../resources/image/'
                    , "csv": contextPath + '/../../resources/data/csv/'
                    , "xlsx": contextPath + '/../../resources/data/xlsx/'
                }
                , "config": {
                    "system": contextPath + '/../../resources/config/system.cfg'
                }
            }

            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스코드
            # ==============================================
            log.info("[START] Main : %s", "Run Program")

            # 메일 및 시작/종료 시간 설정
            # Gmail 정보
            # mailType = 'gmail'

            # Naver 정보
            # mailType = 'naver'

            # Line Work 정보
            mailType = 'lineWorks'

            # 시작/종료 시간 설정
            startDate = "20180101"
            endDate = "20200801"

            # Email 설정정보 불러오기
            systemConfigName = globalVar.get('config').get('system')

            config = configparser.ConfigParser()
            config.read(systemConfigName, encoding='utf-8')

            if mailType.__contains__('gmail'):
                imap4Info = 'imap.gmail.com'
            elif mailType.__contains__('naver'):
                imap4Info = 'imap.naver.com'
            elif mailType.__contains__('lineWorks'):
                imap4Info = 'imap.worksmobile.com'
            else:
                mailType = 'gmail'
                imap4Info = 'imap.gmail.com'

            # 받은 편지함 내 모든 메일 검색
            dtStartDate = pd.to_datetime(startDate, format='%Y%m%d')
            dtEndDate = pd.to_datetime(endDate, format='%Y%m%d')

            sStartDate = dtStartDate.strftime("%d-%b-%Y")
            sEndDate = dtEndDate.strftime("%d-%b-%Y")

            saveFile = globalVar.get('contextPath').get('xlsx') + '{}_Info_{}_{}.xlsx'.format(mailType, startDate,
                                                                                              endDate)

            # ==================================================
            # 메일 로그인
            # ==================================================
            id = config.get(mailType, 'id')
            password = config.get(mailType, 'password')

            session = imaplib.IMAP4_SSL(imap4Info, 993)
            session.login(id, password)

            # 받은편지함
            session.select('Inbox')

            if mailType.__contains__('gmail'):
                # 특정 날짜 검색
                searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(sStartDate, sEndDate)
            else:
                #  전체 검색
                searchOpt = 'ALL'

            log.info("[Check] searchOpt : %s", searchOpt)
            result, data = session.search(None, searchOpt)

            if (result != 'OK'):
                log.error("[Check] 조회 실패하였습니다.")

            log.info("[Check] result : %s", result)

            # 메일 읽기
            emailList = data[0].split()

            if (len(emailList) <= 0):
                log.error("[Check] 조회 목록이 없습니다.")

            log.info("[Check] emailList : %s", len(emailList))

            dataL1 = pd.DataFrame()

            # i = 1000
            # i = 1
            # for i in range(1, len(emailList)):
            for i in range(1, 100):

                msgFrom = ''
                msgSender = ''
                msgTo = ''
                msgDate = ''
                subject = ''
                message = ''
                msgDateFmt = ''
                title = ""

                try:
                    mail = emailList[i]
                    log.info("[Check] mail : %s", mail)

                    result, data = session.fetch(mail, '(RFC822)')
                    raw_email = data[0][1]
                    raw_email_string = raw_email.decode('utf-8')
                    msg = email.message_from_string(raw_email_string)

                    # 메일 정보
                    msgFrom = msg.get('From')
                    msgSender = msg.get('Sender')
                    msgTo = msg.get('To')
                    msgDate = msg.get('Date')
                    msgDateFmt = dateutil.parser.parse(msg.get('Date')).strftime("%Y-%m-%d %H:%M:%S")
                    subject = email.header.decode_header(msg.get('Subject'))
                    msgEncoding = subject[0][1]
                    title = subject[0][0].decode(msgEncoding)

                    dtDateUnix = datetime.timestamp(pd.to_datetime(msgDate))
                    dtStartDateUnix = datetime.timestamp(dtStartDate)
                    dtEndDateUnix = datetime.timestamp(dtEndDate)

                    if (dtDateUnix < dtStartDateUnix) or (dtEndDateUnix < dtDateUnix):
                        continue

                    if msg.is_multipart():
                        for part in msg.get_payload():
                            if part.get_content_type() == 'text/plain':
                                bytes = part.get_payload(decode=True)
                                encode = part.get_content_charset()
                                if (encode == None): encode = 'UTF-8'
                                message = message + str(bytes, encode)

                    else:
                        if msg.get_content_type() == 'text/plain':
                            bytes = msg.get_payload(decode=True)
                            encode = msg.get_content_charset()
                            if (encode == None): encode = 'UTF-8'
                            message = str(bytes, encode)

                    # 딕션너리 정의
                    dataInfo = {
                        'MailID': [mail]
                        , 'From': [msgFrom]
                        , 'Sender': [msgSender]
                        , 'To': [msgTo]
                        , 'Date': [msgDate]
                        , 'DateFmt': [msgDateFmt]
                        , 'Title': [title]
                        , 'Message': [message]
                    }

                    data = pd.DataFrame(dataInfo)
                    dataL1 = dataL1.append(data)

                except Exception as e:
                    log.error("Exception : {}".format(e))

            session.close()
            session.logout()

            dataL1.to_excel(saveFile)

        except Exception as e:
            log.error("Exception : {}".format(e))
            # traceback.print_exc()
            # sys.exit(1)

        finally:
            log.info('[END] Main : {}'.format('Run Program'))

    def uPro14(self):
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
            plt.xlim(0, setMax)
            plt.ylim(0, setMax)
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
            contextPath = os.getcwd()

            # 전역 변수
            globalVar = {
                "config": {
                    "imgContextPath": contextPath + '/../../resources/image/'
                    , "csvConfigPath": contextPath + '/../../resources/data/csv/'
                    , "xlsxConfigPath": contextPath + '/../../resources/data/xlsx/'
                }
            }

            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스코드
            # ==============================================
            # 파일 읽기
            inFile = globalVar.get('config').get('csvConfigPath') + 'data/case2/data/*.csv'

            dataL1 = pd.DataFrame()
            for i in glob.glob(inFile):
                if ((i.__contains__('case1')) and not (
                        (i.__contains__('Takemoto_2016_Field_4')) or (
                        i.__contains__('Takemoto_2017_Field_2')))): continue
                if ((i.__contains__('case2')) and not (
                        (i.__contains__('Takemoto_2016_Field_4')) or (
                        i.__contains__('Takemoto_2018_Field_3')))): continue

                log.info("[Check] i : {}".format(i))
                data = pd.read_csv(i, encoding="euc-kr")
                dataL1 = dataL1.append(data)

            valFile = globalVar.get('config').get('csvConfigPath') + 'data/case1/data_test/*.csv'

            dataTestL1 = pd.DataFrame()
            for i in glob.glob(valFile):
                dataTest = pd.read_csv(i, encoding="euc-kr")
                dataTestL1 = dataTestL1.append(dataTest)

            # 파일 정보 읽기
            dataL1.head()

            # =========================================
            # 1. 탐색
            # =========================================
            # 각 자료에 대한 빈도분포
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_Hist.png' % (50)
            dataL1.hist()
            plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            plt.show()

            # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_BoxPlot.png' % (50)
            dataL1.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
            plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            plt.show()

            # 상관분석 행렬 시각화 및 자료 저장
            dataCorr = dataL1.corr(method='pearson')
            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_50.png'
            makeCorrPlot(dataCorr, savefigName)

            # =========================================
            # 3. 목적에 맞는 분석
            # ========================================
            # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

            # 트레이닝 셋에 대한 독립변수 설정
            # X_train = dataL1[['TD', 'SFV', 'N', 'Section4_Winkler scale(℃)', 'Section4_Accumulated precipitation (mm)', 'Section5_Winkler scale(℃)', 'Section5_Accumulated precipitation (mm)']]
            X_train = dataL1[['TD', 'SFV', 'N', ]]

            # 트레이닝 셋에 대한 종속변수 설정
            Y_train = dataL1[['S1']]

            #  테스트 셋에 대한 독립변수 설정
            # X_test = dataTestL1[['TD', 'SFV', 'N', 'Section4_Winkler scale(℃)', 'Section4_Accumulated precipitation (mm)', 'Section5_Winkler scale(℃)', 'Section5_Accumulated precipitation (mm)']]
            X_test = dataTestL1[['TD', 'SFV', 'N', ]]

            #  테스트 셋에 대한 종속변수 설정
            Y_test = dataTestL1[['S1']]

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

            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_PCA.png' % (51)
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
                # 'n_components': [len(X_train.columns)]
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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 52)

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
            model = (
                'Scaled PLS', Pipeline([('Scaler', StandardScaler()), ('PLS', linear_model.ElasticNet())]))
            param = {
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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s.png' % (i + 57)

                selModel = models[i][0]

                if (selModel.__contains__('Scaled Lasso') or selModel.__contains__(
                        'Scaled ElasticNet') or selModel.__contains__('Scaled PLS')):
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

    def uPro15(self):
        # imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
        # 다운받은 파일을 db에 넣어서 사용할 예정이라
        # 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
        # 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
        # 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
        # 일정은 차주 수요일정두요
        # 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
        # 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
        # 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

        # 라이브러리 읽기
        import logging as log
        import os
        import sys
        import dateutil
        import pandas as pd
        import email
        import imaplib
        import configparser
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import traceback
        from email.header import decode_header
        from datetime import datetime

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(name)s | %(lineno)5.5s | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
        # warnings.filterwarnings("ignore")
        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        # 그래프에서 글꼴 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False

        try:
            log.info('[START] Main : {}'.format('Run Program'))

            # 작업환경 경로 설정
            contextPath = os.getcwd()

            # 전역 변수
            globalVar = {
                "contextPath": {
                    "img": contextPath + '/../../resources/image/'
                    , "csv": contextPath + '/../../resources/data/csv/'
                    , "xlsx": contextPath + '/../../resources/data/xlsx/'
                }
                , "config": {
                    "system": contextPath + '/../../resources/config/system.cfg'
                }
            }

            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스코드
            # ==============================================
            log.info("[START] Main : %s", "Run Program")

            # 메일 및 시작/종료 시간 설정
            # Gmail 정보
            # mailType = 'gmail'

            # Naver 정보
            # mailType = 'naver'

            # Line Work 정보
            mailType = 'lineWorks'

            # 시작/종료 시간 설정
            startDate = "20180101"
            endDate = "20200801"

            # Email 설정정보 불러오기
            systemConfigName = globalVar.get('config').get('system')

            config = configparser.ConfigParser()
            config.read(systemConfigName, encoding='utf-8')

            if mailType.__contains__('gmail'):
                imap4Info = 'imap.gmail.com'
            elif mailType.__contains__('naver'):
                imap4Info = 'imap.naver.com'
            elif mailType.__contains__('lineWorks'):
                imap4Info = 'imap.worksmobile.com'
            else:
                mailType = 'gmail'
                imap4Info = 'imap.gmail.com'

            # 받은 편지함 내 모든 메일 검색
            dtStartDate = pd.to_datetime(startDate, format='%Y%m%d')
            dtEndDate = pd.to_datetime(endDate, format='%Y%m%d')

            sStartDate = dtStartDate.strftime("%d-%b-%Y")
            sEndDate = dtEndDate.strftime("%d-%b-%Y")

            saveFile = globalVar.get('contextPath').get('xlsx') + '{}_Info_{}_{}.xlsx'.format(mailType, startDate,
                                                                                              endDate)

            # ==================================================
            # 메일 로그인
            # ==================================================
            id = config.get(mailType, 'id')
            password = config.get(mailType, 'password')

            session = imaplib.IMAP4_SSL(imap4Info, 993)
            session.login(id, password)

            # 받은편지함
            session.select('Inbox')

            if mailType.__contains__('gmail'):
                # 특정 날짜 검색
                searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(sStartDate, sEndDate)
            else:
                #  전체 검색
                searchOpt = 'ALL'

            log.info("[Check] searchOpt : %s", searchOpt)
            result, data = session.search(None, searchOpt)

            if (result != 'OK'):
                log.error("[Check] 조회 실패하였습니다.")

            log.info("[Check] result : %s", result)

            # 메일 읽기
            emailList = data[0].split()

            if (len(emailList) <= 0):
                log.error("[Check] 조회 목록이 없습니다.")

            log.info("[Check] emailList : %s", len(emailList))

            dataL1 = pd.DataFrame()

            # i = 1000
            # i = 1
            # for i in range(1, len(emailList)):
            for i in range(1, 100):

                msgFrom = ''
                msgSender = ''
                msgTo = ''
                msgDate = ''
                subject = ''
                message = ''
                msgDateFmt = ''
                title = ""

                try:
                    mail = emailList[i]
                    log.info("[Check] mail : %s", mail)

                    result, data = session.fetch(mail, '(RFC822)')
                    raw_email = data[0][1]
                    raw_email_string = raw_email.decode('utf-8')
                    msg = email.message_from_string(raw_email_string)

                    # 메일 정보
                    msgFrom = msg.get('From')
                    msgSender = msg.get('Sender')
                    msgTo = msg.get('To')
                    msgDate = msg.get('Date')
                    msgDateFmt = dateutil.parser.parse(msg.get('Date')).strftime("%Y-%m-%d %H:%M:%S")
                    subject = email.header.decode_header(msg.get('Subject'))
                    msgEncoding = subject[0][1]
                    title = subject[0][0].decode(msgEncoding)

                    dtDateUnix = datetime.timestamp(pd.to_datetime(msgDate))
                    dtStartDateUnix = datetime.timestamp(dtStartDate)
                    dtEndDateUnix = datetime.timestamp(dtEndDate)

                    if (dtDateUnix < dtStartDateUnix) or (dtEndDateUnix < dtDateUnix):
                        continue

                    if msg.is_multipart():
                        for part in msg.get_payload():
                            if part.get_content_type() == 'text/plain':
                                bytes = part.get_payload(decode=True)
                                encode = part.get_content_charset()
                                if (encode == None): encode = 'UTF-8'
                                message = message + str(bytes, encode)

                    else:
                        if msg.get_content_type() == 'text/plain':
                            bytes = msg.get_payload(decode=True)
                            encode = msg.get_content_charset()
                            if (encode == None): encode = 'UTF-8'
                            message = str(bytes, encode)

                    # 딕션너리 정의
                    dataInfo = {
                        'MailID': [mail]
                        , 'From': [msgFrom]
                        , 'Sender': [msgSender]
                        , 'To': [msgTo]
                        , 'Date': [msgDate]
                        , 'DateFmt': [msgDateFmt]
                        , 'Title': [title]
                        , 'Message': [message]
                    }

                    data = pd.DataFrame(dataInfo)
                    dataL1 = dataL1.append(data)

                except Exception as e:
                    log.error("Exception : {}".format(e))

            session.close()
            session.logout()

            dataL1.to_excel(saveFile)

        except Exception as e:
            log.error("Exception : {}".format(e))
            # traceback.print_exc()
            # sys.exit(1)

        finally:
            log.info('[END] Main : {}'.format('Run Program'))

    def uPro16(self):
        import AMD_Tools3 as AMD
        import time, re
        import numpy as np
        import pandas as pd
        from datetime import timedelta, date
        from pytest import collect

        # =======================================================
        #  입력 파라메타 정의
        # =======================================================
        # TMP_mea, APCP RH WIND
        elements = ['TMP_mea', 'APCP', 'RH', 'WIND']
        # places = ['place']
        srtDate = '2020-07-20'
        endDate = '2020-07-21'

        # =======================================================
        #  자료 자동 수집
        # =======================================================
        data = pd.read_csv("2020_ohtani_田植え_12-2_raw.csv", encoding="UTF-8")
        dateList = pd.date_range(start=srtDate, end=endDate, freq="D")

        # 결측값 제거
        dataL1 = data.dropna(axis=0)

        dataL3 = pd.DataFrame()
        for i in dateList:
            sDate1 = i.strftime("%Y-%m-%d")
            sDate2 = (i + timedelta(days=1)).strftime("%Y-%m-%d")

            inDate = [sDate1, sDate2]

            for i in data.index:
                # for j in range(0, 3):

                lon = dataL1._get_value(j, 'x')
                lat = dataL1._get_value(j, 'y')
                inGeo = [lat, lat, lon, lon]

                # time = dataL1._get_value(i, 'time')
                # dtDate = pd.to_datetime(time, format='%Y/%m/%d %H:%M')
                # sDate1 = dtDate.strftime("%Y-%m-%d")
                # sDate2 = (dtDate + timedelta(days=1)).strftime("%Y-%m-%d")

                # 자료 처리를 위한 딕션너리 정의
                dict = {
                    'date': [sDate1]
                    , 'lon': [lon]
                    , 'lat': [lat]
                    , 'TD': dataL1._get_value(j, 'TD')
                    , 'SFV': dataL1._get_value(j, 'SFV')
                    , 'FA': dataL1._get_value(j, 'FA')
                }

                for element in elements:
                    print('[INFO] element : {}'.format(element))
                    print('[INFO] inDate : {}'.format(inDate))
                    print('[INFO] inGeo : {}'.format(inGeo))

                    Msh, tim, tmpLat, tmpLon = AMD.GetMetData(element, inDate, inGeo)
                    getVal = Msh.ravel()

                    # 1번째 요소에 대해서만 자료 저장
                    dict[element] = [getVal[0]]

                dataL2 = pd.DataFrame.from_dict(dict)
                dataL3 = dataL3.append(dataL2)

        fname = 'place.csv'
        dataL3.to_csv(fname, index=None)

        print('완료')

    def uPro17(self):

        # 라이브러리 읽기
        import configparser
        import logging as log
        import os
        import sys
        import dateutil
        import pandas as pd
        import GetOldTweets3 as got
        from datetime import datetime
        from bs4 import BeautifulSoup
        import dfply as dfply
        import numpy as np
        import pandas as pd
        import time
        from dateutil import tz
        import traceback
        from requests_oauthlib import OAuth1Session
        import warnings
        from haversine import haversine
        from requests_oauthlib import OAuth1Session
        import json

        # 작업환경 경로 설정
        # contextPath = os.getcwd()
        contextPath = 'D:/04. 재능플랫폼/Github/TalentPlatform-Python'

        # 전역 변수
        globalVar = {
            "config": {
                "imgContextPath": contextPath + '/resources/image/'
                , "csvConfigPath": contextPath + '/resources/data/csv/'
                , "xlsxConfigPath": contextPath + '/resources/data/xlsx/'
                , "system": contextPath + '/resources/config/system.cfg'
            }
        }

        log.info("[Check] globalVar : {}".format(globalVar))

        systemConfigName = globalVar.get('config').get('system')
        config = configparser.ConfigParser()
        config.read(systemConfigName, encoding='utf-8')

        log.info("[Check] systemConfigName : {}".format(systemConfigName))

        # ==============================================
        # 주 소스코드
        # ==============================================
        log.info("[START] Main : %s", "Run Program")

        # setKeyword = "#TEST"
        # setKeyword = "@noradio"
        setKeyword = "우한폐렴 OR 코로나"
        setDate = "2015-07-19"

        # 위/경도 설정
        srtLat = 40.44
        endLat = 41.12

        srtLon = -74.93
        endLon = -72.63

        cenLat = (srtLat + endLat) / 2.0
        cenLon = (srtLon + endLon) / 2.0

        dist = haversine((srtLat, srtLon), (cenLat, cenLon), unit='km')
        setGeocode = "{},{},{}km".format(cenLat, cenLon, 500)

        # https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
        # 등급에 따른 API 종류 존재 (Standard search API, Enterprise search APIs, Premium search API)
        # 표준 검색 API 사용

        oath = OAuth1Session(
            config.get('twitter', 'consumerKey')
            , config.get('twitter', 'consumerSecret')
            , config.get('twitter', 'accessToken')
            , config.get('twitter', 'accessTokenSecret')
        )

        # 현재를 기준으로 7일 전
        params = {
            "q": setKeyword
            # , "lang": "ko"
            , "result_type": "mixed"
            , "count": "100"
            # , "until": setDate
            # , "geocode": setGeocode
            , "retryonratelimit": True
        }

        url = "https://api.twitter.com/1.1/search/tweets.json?"
        responce = oath.get(url, params=params)

        if responce.status_code != 200:
            raise Exception("Error code: %d" % (responce.status_code))

        tweets = json.loads(responce.text)
        data = pd.DataFrame(tweets['statuses'])

    def uPro18(self):
        import glob
        import logging as log
        import sys
        import warnings

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from scipy import stats

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
        warnings.filterwarnings("ignore")

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
        # 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False

        def stepAic(model, exog, endog, **kwargs):
            """
            This select the best exogenous variables with AIC
            Both exog and endog values can be either str or list.
            (Endog list is for the Binomial family.)

            Note: This adopt only "forward" selection

            Args:
                model: model from statsmodels.formula.api
                exog (str or list): exogenous variables
                endog (str or list): endogenous variables
                kwargs: extra keyword argments for model (e.g., data, family)

            Returns:
                model: a model that seems to have the smallest AIC
            """

            exog = np.r_[[exog]].flatten()
            endog = np.r_[[endog]].flatten()
            remaining = set(exog)
            selected = []

            formula_head = ' + '.join(endog) + ' ~ '
            formula = formula_head + '1'
            aic = model(formula=formula, **kwargs).fit().aic
            print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

            current_score, best_new_score = np.ones(2) * aic

            while remaining and current_score == best_new_score:
                scores_with_candidates = []
                for candidate in remaining:
                    formula_tail = ' + '.join(selected + [candidate])
                    formula = formula_head + formula_tail
                    aic = model(formula=formula, **kwargs).fit().aic
                    print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

                    scores_with_candidates.append((aic, candidate))

                scores_with_candidates.sort()
                scores_with_candidates.reverse()
                best_new_score, best_candidate = scores_with_candidates.pop()

                if best_new_score < current_score:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score

            formula = formula_head + ' + '.join(selected)

            print('The best formula: {}'.format(formula))

            return model(formula, **kwargs).fit()

        try:
            log.info('[START] Main : {}'.format('Run Program'))

            # 작업환경 경로 설정
            # contextPath = os.getcwd()
            contextPath = 'E:/04. 재능플랫폼/Github/TalentPlatform-Python'

            globalVar = {
                "contextPath": {
                    "img": contextPath + '/resources/image/'
                    , "movie": contextPath + '/resources/movie/'
                    , "csv": contextPath + '/resources/data/csv/'
                    , "xlsx": contextPath + '/resources/data/xlsx/'
                }
                , "config": {
                    "system": contextPath + '/resources/config/system.cfg'
                }
            }

            # 전역 변수
            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스 코드
            # ==============================================

            # ===============================================
            # 교재 연습문제 8장
            # ===============================================
            # 2번 문제
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/nutrient2.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            # (1)
            dataL1 = data.replace(to_replace=0, value=np.nan)
            dataL1.isnull().sum()

            # (2)
            dataL1.describe()

            # (3)
            # 상자 그림
            dataL1.boxplot()
            plt.show()

            # 히스토그램
            dataL1.hist()
            plt.show()

            # 3번 문제
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/pima2.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            # (1)
            dataL1 = data['diabetes'].value_counts(normalize=False)

            dataL1.plot.bar()
            plt.show()

            dataL1.plot.pie()
            plt.show()

            # (2)
            data.groupby(['diabetes']).describe()

            data.groupby(['diabetes']).boxplot()
            plt.show()

            data.groupby(['diabetes']).hist()
            plt.show()

            # (3)
            gpAge = pd.cut(data['age'], bins=[20, 30, 40, 50, np.inf], labels=['20-30', '30-40', '41-50', '50+'])

            # 분할표
            crossAge = pd.crosstab(gpAge, data['diabetes'])
            crossAge

            crossAge.plot.bar()
            plt.show()

            # (4)
            gpPregnant = pd.cut(data['pregnant'], bins=[0, 5, 10, np.inf], labels=['0-5', '6-10', '10+'])

            # 분할표
            crossPregnant = pd.crosstab(gpPregnant, data['diabetes'])

            crossPregnant.plot.bar()
            plt.show()

            # (5)
            dataL1 = pd.merge(data.reset_index(), gpPregnant.reset_index(), on='index')

            selCol = ['glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'diabetes', 'pregnant_y']

            # 평균
            dataL1[selCol].groupby(['diabetes', 'pregnant_y']).mean()

            # 표준편차
            dataL1[selCol].groupby(['diabetes', 'pregnant_y']).std()

            # ===============================================
            # 교재 연습문제 9장
            # ===============================================
            # 연습문제 2번
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/bateriasoap.csv')

            data = dict(
                {
                    'Placebo': [105, 119, 100, 97, 96, 101, 94, 95, 98]
                    , 'Caffeine': [96, 99, 94, 89, 96, 93, 88, 105, 88]
                }
            )

            # F 테스트
            # P값이 0.38으로서 귀무가설 기각하지 못함(두 캡슐의 분산 차이가 없다)
            # 따라서 등분산 조건 (equal_var=True)
            stats.bartlett(data['Placebo'], data['Caffeine'])

            # T 테스트
            # P값이 0.063로서 귀무가설 기각 (두 캡슐의 차이가 있다)
            stats.ttest_ind(data['Placebo'], data['Caffeine'], equal_var=True)

            # 연습문제 2번
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/mtcars.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            # F 테스트
            # P값이 0.001 이하로서 귀무가설 기각 (두 캡슐의 분산 차이가 있다)
            # 따라서 상이한 분산 조건 (equal_var=False)
            stats.bartlett(data['am'], data['mpg'])

            # T 테스트
            # P값이 0.001 이하로서 귀무가설 기각 (두 캡슐의 차이가 있다)
            stats.ttest_ind(data['am'], data['mpg'], equal_var=False)

            # ==========================================
            # 교재 연습문제 10장
            # ==========================================

            # 연습문제 1
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/computer.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            selCol = ['erp ', 'myct', 'mmax', 'cach', 'chmin', 'chmax', 'prpe']
            dataL1 = data[selCol]

            # 산점도
            sns.pairplot(dataL1)
            plt.show()

            # 상관계수 행렬
            dataL1.corr(method='pearson')

            # 다중 선형 회귀모형
            xCol = ['myct', 'mmax', 'cach', 'chmin', 'chmax']
            yCol = ['erp ']

            xAxis = dataL1[xCol]
            yAxis = dataL1[yCol]

            model = sm.OLS(yAxis, xAxis)

            result = model.fit()
            result.summary()

            # 연습문제 2번
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/mtcars.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            xCol = ['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
            yCol = ['mpg']

            bestModel = stepAic(smf.ols, xCol, yCol, data=data)
            bestModel.summary()

            # ==========================================
            # 교재 연습문제 11장
            # ==========================================
            # 연습문제 1번
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/bateriasoap.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            fit = smf.ols('data["BacterialCounts"] ~ C(data["Method"])', data=data).fit()
            fit.summary()

            aovFit = sm.stats.anova_lm(fit)
            aovFit

            # 연습문제 2번
            fileInfo = glob.glob(globalVar.get('contextPath').get('csv') + 'rpy/downloading.csv')
            data = pd.read_csv(fileInfo[0], encoding="UTF-8")

            fit = smf.ols('data["Time(Sec)"] ~ C(data["TimeofDay"])', data=data).fit()
            fit.summary()

            aovFit = sm.stats.anova_lm(fit)
            aovFit

        except Exception as e:
            log.error("Exception : {}".format(e))
            # traceback.print_exc()
            # sys.exit(1)

        finally:
            log.info('[END] Main : {}'.format('Run Program'))

    def uPro19(self):
        pass

    def uPro20(self):
        pass

    def uPro21(self):
        pass

    def uPro22(self):
        pass

    def uProDefault(self):
        import logging as log
        import os
        import sys
        # from plotnine import *
        # from plotnine.data import *
        # from dfply import *
        # import hydroeval
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import warnings

        # 로그 설정
        log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
        warnings.filterwarnings("ignore")

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
        # 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False

        try:
            log.info('[START] Main : {}'.format('Run Program'))

            # 작업환경 경로 설정
            # contextPath = os.getcwd()
            contextPath = 'E:/04. 재능플랫폼/Github/TalentPlatform-Python'

            globalVar = {
                "contextPath": {
                    "img": contextPath + '/resources/image/'
                    , "movie": contextPath + '/resources/movie'
                    , "csv": contextPath + '/resources/data/csv/'
                    , "xlsx": contextPath + '/resources/data/xlsx/'
                }
                , "config": {
                    "system": contextPath + '/resources/config/system.cfg'
                }
            }

            # 전역 변수
            log.info("[Check] globalVar : {}".format(globalVar))

            # ==============================================
            # 주 소스코드
            # ==============================================

        except Exception as e:
            log.error("Exception : {}".format(e))
            # traceback.print_exc()
            # sys.exit(1)

        finally:
            log.info('[END] Main : {}'.format('Run Program'))


if __name__ == '__main__':

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
    import dfply as dfply
    from matplotlib import rcParams
    import matplotlib.pylab as pylab
    from sklearn.preprocessing import minmax_scale
    from sklearn.preprocessing import MinMaxScaler

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # 로그 설정
    log.basicConfig(stream=sys.stdout, level=log.INFO,
                    format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
    warnings.filterwarnings("ignore")
    plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 10),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

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

        pylab.rcParams['figure.figsize'] = (6, 6)
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
        plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(setMin, setMax - interval * 2),
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
        # plt.savefig(savefigName, dpi=600)
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
        # contextPath = 'D:/04. 재능플랫폼/Github/TalentPlatform-Python'
        contextPath = 'E:/04. 재능플랫폼/Github/TalentPlatform-Python'

        # 전역 변수
        globalVar = {
            "config": {
                "imgContextPath": contextPath + '/resources/image/TMP3/'
                , "csvConfigPath": contextPath + '/resources/data/csv/'
                , "xlsxConfigPath": contextPath + '/resources/data/xlsx/'
            }
        }

        log.info("[Check] globalVar : {}".format(globalVar))

        # ==============================================
        # 주 소스코드
        # ==============================================
        # caseList = ['case{}'.format(str(i)) for i in range(1, 7)]
        # caseList = ['new_case5']
        caseList = ['new_case1']

        # case = 'case2'

        for (ind, case) in enumerate(caseList):
            log.info("[Check] case : {}".format(case))

            # 파일 읽기
            inFile = globalVar.get('config').get('csvConfigPath') + 'data/{}/data/*.csv'.format(case)

            dataL1 = pd.DataFrame()
            for i in glob.glob(inFile):
                log.info("[Check] inFile : {}".format(i))
                fileNameInfo = os.path.splitext(os.path.basename(i))[0]

                data = pd.read_csv(i, encoding="euc-kr")
                # dataL1 = data
                dataL1 = dataL1.append(data)

            valFile = globalVar.get('config').get('csvConfigPath') + 'data/{}/data_test/*.csv'.format(case)

            dataTestL1 = pd.DataFrame()
            for j in glob.glob(valFile):
                log.info("[Check] valFile : {}".format(j))

                dataTest = pd.read_csv(j, encoding="euc-kr")
                dataTestL1 = dataTestL1.append(dataTest)

                # 파일 정보 읽기
                # dataL1.head()

            # =========================================
            # 1. 탐색
            # =========================================
            # # 각 자료에 대한 빈도분포
            # pylab.rcParams['figure.figsize'] = (14, 10)
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_Hist.png' % (fileNameInfo, 50, case, 'train')
            # dataL1.hist()
            # # plt.savefig(savefigName, dpi=600)
            # plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            # plt.show()
            #
            # # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
            # pylab.rcParams['figure.figsize'] = (14, 10)
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_BoxPlot.png' % (fileNameInfo, 50, case, 'train')
            # # dataL1.plot(kind='box', subplots=True, sharex=False, sharey=False)
            # sns.boxplot(data=dataL1, orient="h", palette="Set2")
            # plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            # plt.show()
            #
            # # plt.show()
            #  # ax = sns.boxplot(x="day", y="total_bill", data=tips)
            #
            # # 상관분석 행렬 시각화 및 자료 저장
            # pylab.rcParams['figure.figsize'] = (10, 10)
            # dataCorr = dataL1.corr(method='pearson')
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_CorrMatrix.png' % (fileNameInfo, 50, case, 'train')
            # makeCorrPlot(dataCorr, savefigName)
            #
            # dataL1Summary = dataL1.describe()
            # # dataTestL1Summary = dataTestL1.describe()
            # log.info("[Check] Train dataL1 Summary : {}".format(dataL1Summary))
            #
            # # 각 자료에 대한 빈도분포
            # pylab.rcParams['figure.figsize'] = (14, 10)
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_Hist.png' % (fileNameInfo, 50, case, 'test')
            # dataTestL1.hist()
            # plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            # plt.show()
            #
            # # 각 자료에 대한 상자그림 (자료에 대한 상대범위 확인)
            # pylab.rcParams['figure.figsize'] = (14, 10)
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_BoxPlot.png' % (fileNameInfo, 50, case, 'test')
            # sns.boxplot(data=dataL1, orient="h", palette="Set2")
            # # dataTestL1.plot(kind='box', subplots=True, sharex=False, sharey=False)
            # plt.savefig(savefigName, dpi=600, bbox_inches='tight')
            # plt.show()
            #
            # # 상관분석 행렬 시각화 및 자료 저장
            # pylab.rcParams['figure.figsize'] = (10, 10)
            # dataCorr = dataTestL1.corr(method='pearson')
            # savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_%s_CorrMatrix.png' % (fileNameInfo, 50, case, 'test')
            # makeCorrPlot(dataCorr, savefigName)
            #
            # dataTestL1Summary = dataTestL1.describe()
            # log.info("[Check] Test dataL1 Summary : {}".format(dataTestL1Summary))

            # =========================================
            # 3. 목적에 맞는 분석
            # ========================================
            dataL1 = dataL1.rename({'Section4_Winkler scale': 'Section4_Winkler_scale'}, axis='columns')
            dataL1 = dataL1.rename({'Section5_Accumulated precipitation': 'Section5_Accumulated_precipitation'},
                                   axis='columns')
            dataL1 = dataL1.rename({'Section5_Winkler scale': 'Section5_Winkler_scale'}, axis='columns')

            # 특정 행을 대상으로 NA값 삭제
            # dataL2 = dataL1[['TD', 'SFV', 'N', 'Section4_Winkler scale', 'Section4_Accumulated precipitation',
            #                   'Section5_Winkler scale', 'Section5_Accumulated_precipitation', 'S1']].dropna(axis=0)
            dataL2 = dataL1[['TD', 'SFV', 'N', 'Section4_Winkler_scale', 'Section4_precipitation',
                             'Section5_Winkler_scale', 'Section5_precipitation', 'S1']].dropna(axis=0)

            tmpDataL2 = ((dataL2 >>
                          dfply.group_by(dfply.X.Section4_precipitation, dfply.X.Section5_precipitation) >>
                          dfply.summarize(
                              meanTD=dfply.X.TD.mean()
                              , meanSFV=dfply.X.SFV.mean()
                              , meanN=dfply.X.N.mean()
                              , meanSec4Wink=dfply.X.Section4_Winkler_scale.mean()
                              , meanSec4Prec=dfply.X.Section4_precipitation.sum()
                              , meanSec5Wink=dfply.X.Section5_Winkler_scale.mean()
                              , meanSec5Prec=dfply.X.Section5_precipitation.sum()
                              , meanS1=dfply.X.S1.mean()
                              , cnt=dfply.n(dfply.X.Section4_precipitation)
                          )
                          ))

            # 전체
            dataL3 = tmpDataL2

            # 강수 유무에 따른 별도 회귀계수 필요
            # 무강수
            # dataL3 = (dataL2 >>
            #          dfply.mask(dfply.X.Section5_Accumulated_precipitation < 10)
            # )
            # dataL3 = (dataL2 >>
            #          dfply.mask(dfply.X.Section5_precipitation < 10)
            # )
            # dataL3 = (dataL2 >>
            #          dfply.mask(dfply.X.Section5_Accumulated_precipitation == 0)
            # )
            # dataL3 = (tmpDataL2 >>
            #          dfply.mask(dfply.X.Section5_precipitation == 0)
            # )

            # 유강수
            # dataL3 = (dataL2 >>
            #           dfply.mask(dfply.X.Section5_Accumulated_precipitation > 10)
            #           )
            # dataL3 = (dataL2 >>
            #           dfply.mask(dfply.X.Section5_precipitation > 10)
            #           )

            # 트레이닝 및 테스트 셋을 각각 75% 및 25% 선정
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

            # 트레이닝 셋에 대한 독립변수 설정
            # X_train = dataL3[['TD', 'SFV', 'N', 'Section4_Winkler scale', 'Section4_Accumulated precipitation',
            #                   'Section5_Winkler scale', 'Section5_Accumulated_precipitation']]
            # X_train = dataL3[['TD', 'SFV', 'N', 'Section4_Winkler scale', 'Section4_precipitation',
            #                   'Section5_Winkler scale', 'Section5_precipitation']]
            X_train = dataL3[['meanTD', 'meanSFV', 'meanN', 'meanSec4Wink', 'meanSec4Prec',
                              'meanSec5Wink', 'meanSec5Prec']]

            # 트레이닝 셋에 대한 종속변수 설정
            # Y_train = dataL3[['S1']]
            Y_train = dataL3[['meanS1']]

            #  테스트 셋에 대한 독립변수 설정
            # X_test = dataTestL1[['TD', 'SFV', 'N', 'Section4_Winkler scale', 'Section4_Accumulated precipitation',
            #      'Section5_Winkler scale', 'Section5_Accumulated precipitation']]
            X_test = dataTestL1[['TD', 'SFV', 'N', 'Section4_Winkler scale', 'Section4_precipitation',
                                 'Section5_Winkler scale', 'Section5_precipitation']]
            # X_test = dataTestL1[['meanTD', 'meanSFV', 'maenN', 'meanSec4Wink', 'meanSec4Prec',
            #                   'meanSec5Wink', 'meanSec5Prec']]
            # X_test = dataTestL1[['TD', 'SFV']]

            #  테스트 셋에 대한 종속변수 설정
            Y_test = dataTestL1[['S1']]
            # Y_test = dataTestL1[['meanS1']]

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

            if len(X_train) == 0 or len(Y_train) == 0:
                continue

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

            savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s_PCA.png' % (
            fileNameInfo, 51, case)
            makeScatterPlot(Y_pca_test_hat[:, 0], Y_test.values[:, 0], "PCA", savefigName)

            rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pca_test_hat))
            r2 = metrics.r2_score(Y_test, Y_pca_test_hat)
            log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % ("PCA", rmse, r2))

            # ====================================================
            # [표준화 X] 4종 회귀모형을 이용
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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s.png' % (
                fileNameInfo, i + 52, case)

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
                savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s.png' % (
                fileNameInfo, i + 57, case)

                selModel = models[i][0]

                if (selModel.__contains__('Scaled Lasso') or selModel.__contains__(
                        'Scaled ElasticNet')):
                    makeScatterPlot(Y_test_hat, Y_test.values[:, 0], selModel, savefigName)
                else:
                    makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], selModel, savefigName)

                rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
                r2 = metrics.r2_score(Y_test, Y_test_hat)
                log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (selModel, rmse, r2))

                # ========================================================
                # [표준화 0-1] 4종 회귀모형을 이용
                # ========================================================
                # 초기값 설정
                models = []
                params = []

                # 선형회귀모형 및 파라미터 설정
                model = (
                    'Scaled Linear',
                    Pipeline([('ScalerMinMax', MinMaxScaler(feature_range=(0, 1))),
                              ('Linear', linear_model.LinearRegression())]))
                param = {}

                models.append(model)
                params.append(param)

                # 릿지 회귀모형 및 파라미터 (알파 조정) 설정
                model = ('ScalerMinMax Ridge',
                         Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('Ridge', linear_model.Ridge())]))
                param = {
                    'Ridge__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
                }

                models.append(model)
                params.append(param)

                # 라쏘에 대해서 및 파라미터 (알파 조정) 설정
                model = ('ScalerMinMax Lasso',
                         Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('Lasso', linear_model.Lasso())]))
                param = {
                    'Lasso__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
                }

                models.append(model)
                params.append(param)

                # 엘라스틴에 대해서 및 파라미터 (알파 조정) 설정
                model = (
                    'ScalerMinMax ElasticNet',
                    Pipeline(
                        [('Scaler', MinMaxScaler(feature_range=(0, 1))), ('ElasticNet', linear_model.ElasticNet())]))
                param = {
                    'ElasticNet__alpha': [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
                    'ElasticNet__l1_ratio': [0.3, 0.5, 0.7]
                }

                models.append(model)
                params.append(param)

                # PLS Regression
                model = [
                    'ScalerMinMax PLS', Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('PLS', PLS())])]
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
                    savefigName = globalVar.get('config').get('imgContextPath') + 'Image_%s_%s_%s.png' % (
                        fileNameInfo, i + 70, case)

                    selModel = models[i][0]

                    if (selModel.__contains__('ScalerMinMax Lasso') or selModel.__contains__(
                            'ScalerMinMax ElasticNet')):
                        makeScatterPlot(Y_test_hat, Y_test.values[:, 0], selModel, savefigName)
                    else:
                        makeScatterPlot(Y_test_hat[:, 0], Y_test.values[:, 0], selModel, savefigName)

                    rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_hat))
                    r2 = metrics.r2_score(Y_test, Y_test_hat)
                    log.info("[Check] %s | RMSE : %.4f // R-square: %.4f" % (selModel, rmse, r2))


    except Exception as e:
        log.error("Exception : {}".format(e))
        # traceback.print_exc()
        # sys.exit(1)

    finally:
        log.info('[END] Main : {}'.format('Run Program'))
